# Merge Push 生命周期与 GPU 执行模型分析

> **文档修订说明**：Section 1.4–1.5（依赖链修正）、Section 4.4（优化时间线）、
> Section 7（新增：Sonnet 的额外分析）由 Claude Sonnet 4.6 修订/撰写（2026-03-12）。
> 核心修正：GPU unmap → Copy **不是必要依赖**，当前代码中此依赖是保守写法。

## 1. UVM 页面迁移的完整流水线

### 1.1 每个 VA Block 的四个阶段

每个 VA Block 的处理在 `uvm_va_block_make_resident_copy` + `uvm_va_block_service_finish`
中完成，分为四个阶段：

```
阶段 1: Unmap  ─ 撤销旧映射（"断开旧的连接"）
阶段 2: Alloc  ─ 分配 GPU 物理内存
阶段 3: Copy   ─ CE DMA: 数据从 CPU DRAM → GPU VRAM
阶段 4: Map    ─ 更新 GPU PTE: 虚拟地址 → 新物理地址
```

### 1.2 为什么是 Unmap → Copy → Map 这个顺序

```
Before:
  GPU SM ──虚拟地址──→ [GPU PTE] ──→ (无映射/旧映射)
  CPU    ──虚拟地址──→ [CPU PTE] ──→ CPU DRAM (数据在这里)

After:
  GPU SM ──虚拟地址──→ [GPU PTE] ──→ GPU VRAM (数据迁移到这里)
  CPU    ──虚拟地址──→ [CPU PTE] ──→ (映射已撤销)
```

顺序是 **Unmap → Copy → Map**。先撤旧映射，再搬数据，再建新映射。

### 1.3 各阶段的执行方式与 Merge 关系

**阶段 1: Unmap**（`uvm_va_block_make_resident_copy` line 4965-4988）

- **CPU unmap**（`block_unmap_cpu`）：修改 CPU 页表 + TLB flush (IPI)。
  同步操作，不用 GPU push。
- **GPU unmap**（`block_unmap_gpu`，line 8173-8248）：如果 GPU 已有旧映射，
  创建**独立 push**（自己的 push_begin + push_end），**立即提交给 GPU**。
- ⚠️ **GPU unmap 不走 merge 共享 push。每个 block 独立提交。**
这意味着 merge 只合并了 Copy 和 Map 的 push。Unmap 仍然是每个 block 独立的，且因为立即提交，GPU 可以在 CPU 处理后续 block 时并行执行前面 block 的 unmap——这部分是有 CPU-GPU 重叠的。


**阶段 2: Alloc**（line 4999-5004）

- `block_populate_pages`：在 GPU VRAM 分配物理页。纯 CPU 操作。

**阶段 3: Copy** (CE DMA)（line 5006-5015）

- CE 使用**物理地址** DMA：`src_phys(CPU DRAM) → dst_phys(GPU VRAM)`
- CE 不经过 GPU MMU，不需要 PTE。
- 通过 `uvm_push_acquire_tracker(push, &block->tracker)` 在 GPU 命令流中
  插入 semaphore wait，确保 GPU unmap 完成后才执行 copy。
- **Merge 时**：写入共享 `merge_ext.copy_push`，**延迟提交**（push_end 在 batch 结束后）。
- **无 Merge 时**：独立 push_begin/push_end，立即提交。

**阶段 4: Map** (PTE Update)（line 12516）

- 更新 GPU 页表: `GPU_virtual_addr → dst_phys(GPU VRAM)`
- 通过 `uvm_push_acquire_tracker` 等待 copy 完成。
- **Merge 时**：写入共享 `merge_ext.map_push`，**延迟提交**。
- **无 Merge 时**：独立 push_begin_acquire/push_end，立即提交。

### 1.4 Tracker / Semaphore 如何保证顺序

每个 `uvm_va_block_t` 有一个 `tracker`，记录所有影响该 block 的 GPU 操作。

```
Block N 的依赖链（GPU 侧，当前代码的实际行为）：

  GPU Unmap push (独立, push_end 立即提交)
    → tracker 记录 unmap 的 semaphore entry
      → Copy cmd 写入时: push_acquire_tracker(copy_push, &block->tracker)
        → GPU 命令流中插入 "等待 block->tracker 里所有 semaphore" 指令
        ⚠️ tracker 里包含 unmap entry，所以 copy 等了 unmap——但这是保守写法，
           见 Section 1.5
          → Copy push 完成后: tracker_add_push_safe(&block->tracker, copy_push)
            → block->tracker 现在包含 copy push 的 semaphore
              → Map cmd 写入时: push_acquire_tracker(map_push, &block->tracker)
                → GPU 命令流中插入 "等待 copy semaphore" 指令  ← 真正必要
```

具体代码路径（merge 路径）：
1. `block_copy_resident_pages_between` (line 4245):
   `uvm_push_acquire_tracker(merge_copy_push, &block->tracker)`
   — 写入 "等 block->tracker 中所有挂起操作" 的 semaphore wait（含 unmap，保守）
2. `block_copy_resident_pages` (line 4420):
   `uvm_tracker_add_push_safe(&local_tracker, merge_copy_push)`
   然后 (line 4906): `uvm_tracker_add_tracker_safe(&block->tracker, &local_tracker)`
   — 将 copy push 的 semaphore 加入 block->tracker
3. `block_map_gpu_to` (line 8716):
   `uvm_push_acquire_tracker(merge_map_push, &va_block->tracker)`
   — 写入 "等 copy push" 的 semaphore wait（真正必要的依赖）

**merge 路径的 Map 和非 merge 路径一样等待 Copy 完成**，机制相同。
"一起发送 GPU" 的意思是 coordinator 同时 push_end 两个 push，但 GPU 通过
semaphore 保证先执行完所有 Copy 命令，再执行 Map 命令。

### 1.5 真正的最小依赖链（[Sonnet 修订]）

当前代码让 copy 等待 `block->tracker` 中的所有操作，其中包括 GPU unmap。
这是**保守写法**——不是严格必要的。分析：

```
真正需要的最小依赖:

  GPU Unmap ──(semaphore)──→ Map
      原因: Map 建立新 PTE 前，旧 PTE 必须已撤销，否则同一虚拟地址有两个有效映射

  CPU Unmap ──(在 push_end 之前完成)──→ Copy 执行
      原因: CPU PTE 必须在 DMA 实际读取数据之前清除，防止 CPU 并发写入脏数据
      关键: DMA 直到 push_end 之后才开始，所以 CPU unmap 只需在 push_end 前完成即可

  Copy ──(semaphore)──→ Map
      原因: GPU PTE 必须在数据真正到位后才能建立

不需要的依赖（当前代码保守添加的）:
  GPU Unmap ──→ Copy  ✗
      CE 使用物理地址 DMA（src: CPU DRAM 物理地址，dst: 新分配的 GPU VRAM 物理地址）
      CE 完全不经过 GPU MMU，GPU 旧 PTE 指向的是 old_GPU_PA（旧位置），
      CE 写入的是 new_GPU_PA（新分配），两者物理地址不同，互不干扰
```

**各场景分析**：

| 场景 | GPU unmap 存在? | copy 等 unmap 是否必要 |
|------|----------------|----------------------|
| Cold read（首次访问） | 不存在 | tracker 为空，push_acquire_tracker 是 no-op |
| Warm（block 曾 GPU-resident，后迁回 CPU） | 存在 | 不必要——CE 写 new_GPU_PA，与 old_GPU_PA 无关 |
| 有其他 GPU 先前操作 | 可能 | tracker 等待的是"所有先前 GPU 操作"，unmap 只是其中一项 |

**实际最常见的 replayable fault 场景是 cold read，GPU unmap 根本不存在。**

---

## 2. 当前 Merge 设计与 Push 计数

### 2.1 Push 计数对比

以 200 个 thin blocks 为例（cold read，无 GPU unmap）：

| Push 类型 | 无 Merge | 有 Merge | 说明 |
|-----------|---------|---------|------|
| GPU Unmap (MEMOPS) | 0 | 0 | cold read 无旧映射 |
| CE Copy (CPU_TO_GPU) | 200 个独立 push | **1 个共享 push** | ★ merge 节省 199 个 |
| GPU Map (MEMOPS) | 200 个独立 push | **1 个共享 push** | ★ merge 节省 199 个 |
| **总计** | **400** | **2** | |

非 cold read 场景（有 GPU unmap），GPU unmap push 两者一样多（每 block 一个，
不受 merge 影响）。

### 2.2 Push 空间溢出保护

```c
// uvm_va_block.c line 4159-4161 (copy path)
if (use_ext_push &&
    !uvm_push_has_space(block_context->merge_ext.copy_push, UVM_MAX_PUSH_SIZE / 4))
    use_ext_push = false;   // 回退到独立 push

// uvm_va_block.c line 8691-8694 (map path)
if (use_ext_push) {
    if ((gpu_state->pte_is_2m && !new_pte_state->pte_is_2m) ||
        !uvm_push_has_space(block_context->merge_ext.map_push, UVM_MAX_PUSH_SIZE / 4))
        use_ext_push = false;
}
```

共享 push 剩余空间不足 25% 时自动回退到独立 push。不会溢出。

---

## 3. 当前设计的 CPU-GPU 流水线问题

### 3.1 无 Merge：有 CPU-GPU 流水线

```
          CPU                          GPU CE            GPU MEMOPS
          ───                          ──────            ──────────
t=0   block 0: unmap + copy/map push     (idle)            (idle)
t=5us push_end(copy_0, map_0) ────→   开始 copy_0        等 semaphore
      开始处理 block 1                copy_0 完成 → 开始 map_0
t=10us push_end(copy_1, map_1) ───→   开始 copy_1        map_0 完成
       开始处理 block 2               copy_1 完成 → 开始 map_1
      ...                              ...               ...
```

CPU 处理 block N+1 时，GPU 执行 block N 的 copy/map。**有重叠。**
但每 block 有 2 次 push_begin/push_end 开销（~2-4us）。

### 3.2 分段 Merge（已实现）：恢复 CPU-GPU 流水线

将 thin blocks 按每 16 个一组分段处理。每个 segment 有独立的 copy_push
和 map_push，完成后立即 push_end。Coordinator 立即开始下一个 segment。

```
          CPU                            GPU CE            GPU MEMOPS
          ───                            ──────            ──────────
          push_begin(A_copy, A_map)
Seg 0:  block 0-15: unmap + write cmds  (idle)            (idle)
t=16us  push_end(A) ────────────────→  开始 copy_0       等 semaphore
        push_begin(B_copy, B_map)       copy_0→map_0, copy_1
Seg 1:  block 16-31: write cmds        copy_1→map_1, copy_2
t=32us  push_end(B) ────────────────→  ...A 执行中        ...
        push_begin(C_copy, C_map)       开始执行 B
Seg 2:  block 32-47: write cmds        B 执行中, A 完成
        ...
```

**CPU 写下一个 segment 时，GPU 执行上一个 segment。双缓冲流水线。**

Fat blocks 单独 dispatch，不走 merge，使用独立 push。

---

## 4. 优化方向分析

### 4.1 分段 Merge + 双缓冲（已实现）

**实现方式**：Coordinator 管理 segment 边界。

1. Phase 1 将 batch_groups[] 原地分区：thin blocks 在前，fat blocks 在后
2. Fat blocks 一次性 dispatch（独立 push，无 merge）
3. Thin blocks 按 segment_size（默认 16）分段 dispatch：
   - Coordinator 为每个 segment 做 push_begin → 唤醒 workers →
     wait_for_completion → push_end
   - push_end 后 GPU 立即开始执行，Coordinator 立即开始下一个 segment

```
收益曲线（push 开销节省 vs segment 大小）:
N=1:   0%  节省（无合并）
N=8:   87% 节省
N=16:  94% 节省  ← 默认 segment_size
N=200: 99% 节省  ← 旧设计，失去流水线
```

**segment_size 通过 `uvm_merge_segment_size` 模块参数可调。**

每 segment 有一次 barrier 开销（wake_up + wait_for_completion, ~1-5us）。
200 blocks / 16 = ~12 segments → 12-60us 额外开销。但换来 CPU-GPU 流水线。

### 4.3 Unmap 延迟依赖：已有的和可以进一步优化的

**GPU Unmap 的依赖延迟——已经实现**

当前代码中，GPU unmap push 立即提交（push_end），但 CPU 写 copy 命令时
**并不等待 GPU unmap 执行完成**。依赖关系通过 semaphore 嵌入命令流：

```c
// block_copy_resident_pages_between (line 4242-4247)
if (use_ext_push) {
    spin_lock(push_lock);
    uvm_push_acquire_tracker(push, &block->tracker); // 写入 semaphore wait
    spin_unlock(push_lock);
    // CPU 继续执行，不等 GPU unmap 完成
}
```

GPU 在执行 copy 时才真正等待 unmap semaphore。所以 GPU unmap 和 CPU 处理后续
blocks 之间**已经有重叠**。

**CPU Unmap 的 TLB Flush 延迟——可以进一步优化**

当前：每个 block 做 CPU unmap 时，`unmap_mapping_range` 会同步清除 PTE +
flush TLB（发 IPI）。这是串行开销。

关键约束：**CPU PTE 必须在 DMA 开始前被清除**（防止 CPU 在迁移过程中写旧页）。
但 DMA 不在 push_end 之前开始。

所以 CPU TLB flush 只需在 **push_end 之前**完成，不需要在写 copy 命令之前完成：

```
当前（per-block flush）：
  block 0: clear PTE + flush TLB (IPI)     ← 等 IPI 返回
           write copy cmd
  block 1: clear PTE + flush TLB (IPI)     ← 等 IPI 返回
           write copy cmd
  ...
  push_end

优化（per-segment batch flush）：
  block 0: clear PTE (不 flush)             ← 只改页表，无 IPI
           write copy cmd
  block 1: clear PTE (不 flush)
           write copy cmd
  ...block 7: clear PTE + write cmd
  batch TLB flush (一次 IPI 覆盖所有 8 blocks)  ← 一次 IPI
  push_end                                       ← 此时 TLB 已 flush，DMA 安全
```

这将每 segment 的 IPI 开销从 N 次减少到 1 次。

⚠️ 这需要修改 CPU unmap 路径，将 PTE clearing 和 TLB flush 分离。
当前 `unmap_mapping_range` 将两者绑定在一起。要实现这个优化需要使用
`unmap_mapping_range_noflush` 变体（需要内核级修改，在之前的 batch_unmap
讨论中已提到）。

### 4.4 分段 + 双缓冲 + Unmap 延迟 的综合时间线（[Sonnet 修订]）

将三个优化结合，200 blocks，segment_size=8。
**关键修正**：基于 Section 1.5，copy 不再等待 GPU unmap。
GPU unmap 只需在 Map 之前完成（通过 Map push 的 push_acquire_tracker）。

```
          CPU                                 GPU CE              GPU MEMOPS
          ───                                 ──────              ──────────
          push_begin(A_copy, A_map)
Seg A:  block 0-7:
          GPU unmap (独立 push, 立即提交) ──→ unmap_0 开始执行    (idle)
          clear CPU PTE (不 flush)            unmap_1 开始执行
          Alloc + write copy/map cmds         unmap_2...7 执行中
        batch TLB flush (1 次 IPI)
t=10us  push_end(A_copy, A_map) ──────────→ copy_0 开始（无需等 unmap）
        push_begin(B_copy, B_map)            copy_1, copy_2...    等 copy 完成 semaphore
Seg B:  block 8-15:                          copy_7 完成          ↓
          GPU unmap (独立 push) ────────→    (CE 继续执行 B 的    map_0..7 开始执行
          clear CPU PTE (不 flush)            copy 命令)           (等 unmap sema 已满足
          write cmds                                               因 unmap_0..7 早已完成)
        batch TLB flush
t=20us  push_end(B) ──────────────────────→ B_copy 开始          A_map 执行中...
        ...
```

**修正后的效果**：
- Push 开销：每 8 blocks 2 次 push（vs 16 次），节省 87%
- CPU-GPU 流水线：完全重叠，GPU 不闲置
- CPU IPI：每 segment 1 次（vs 8 次），节省 87%
- GPU unmap 与 CE copy **真正并行**：copy 无需等 unmap semaphore，
  两者在不同 GPU 引擎上同时执行（CE 引擎 vs MEMOPS 引擎）

⚠️ **per-segment 粒度的 semaphore 粗化**（见 Section 7.1）：
map push 等待的是整个 segment 的 copy 完成，不是 per-block。
block 0 的 map 要等到 block 7 的 copy 完成后才能开始执行。

---

## 5. 当前 Random Read 场景的实际数据

从 profiling 数据（k4, merge=ON）：
- 每 batch ~200 thin blocks，每 block ~1 fault = 1 page (4KB)
- GPU CE 工作量/block: DMA 4KB ≈ 0.3-1us
- GPU MEMOPS 工作量/block: 写 PTE ≈ 0.2us
- CPU 处理时间/block (merge): ~1us (写 cmd + spinlock)

在这种场景下：
- CPU 200us, GPU 300us → merge 总时间约 500us
- 无 merge: CPU 1000us (含 push 开销), GPU 300us → 约 1000us (CPU-bound)
- **当前 merge 已经有效**（500us vs 1000us）

如果用分段+双缓冲 (segment=8, 25 segments)：
- 每 segment: CPU 10us, GPU 12us → 流水线稳态 max(10, 12) = 12us/segment
- 总时间: 25 × 12us = 300us (+ 启动开销)
- **理论上可以从 500us 优化到 ~300-350us**

但对于当前的 random read 场景，GPU 工作量极小（0.3-1us/page），
分段双缓冲的收益主要来自减少 CPU 侧的 segment barrier 等待。
实际收益可能比理论值小，需要 profiling 验证。

---

## 6. 总结

| 问题 | 回答 |
|------|------|
| Copy 怎么知道 GPU 地址？ | CE 用物理地址，不经过 GPU MMU。GPU 物理内存在 copy 前已分配 |
| 顺序是什么？ | Copy 和 GPU unmap **可以并行**；Map 需要等两者都完成 |
| Unmap 走 merge push 吗？ | **不走**。GPU unmap 独立 push 立即提交。仅 Copy 和 Map 被 merge |
| Copy 命令执行时需要等 GPU unmap 吗？ | **不需要**（Section 1.5）。CE 用物理地址，与 GPU PTE 无关。当前代码保守等待 |
| Copy 命令执行时需要等 CPU unmap 吗？ | **需要**，但只需在 push_end 前完成，不需要在写 copy 命令前完成 |
| 200 blocks 全合并合理吗？ | **已改为分段**。默认 16 blocks/segment，94% push 节省 + CPU-GPU 流水线 |
| 双缓冲 push 可行吗？ | **已实现**。Coordinator 循环分段 dispatch，push_end 后立即开始下一 segment |
| per-segment map 要等整段 copy 吗？ | **是的**（Section 7.1），这是 shared push 的内在代价，segment 小时影响小 |
| Push buffer 会溢出吗？ | 不会。有 `uvm_push_has_space` 检查 + 自动回退 |

### 下一步行动建议

1. **短期（当前 random read 场景）**：当前整批 merge 已有效（k4: 22% 加速），
   可以先跑公平 A/B 对比确认实际收益。

2. **中期（分段 merge + 双缓冲）**：实现 segment_size=8-16 的分段提交 +
   双缓冲 push。Coordinator 管理 segment 边界，每 16 个 thin blocks 做一次
   push_end → GPU 立即开始执行。CPU 立即开始下一个 segment 的命令写入。

3. **长期（batch TLB flush）**：将 CPU unmap 的 PTE clearing 和 TLB flush
   分离，per-segment 做一次 batch flush。需要内核级修改。

4. **依赖链精简（GPU unmap → Copy 依赖移除）**：见 Section 7.3。
   在分段方案实现后，移除 copy push 中的 unmap tracker acquire，
   让 CE copy 和 GPU unmap 在 GPU 侧真正并行。

---

## 7. Sonnet 的额外分析（Claude Sonnet 4.6，2026-03-12）

### 7.1 Per-Segment Semaphore 粒度粗化问题

在分段双缓冲方案中，一个 segment 共享一对 (copy_push, map_push)。
由于 copy_push 只有一个 completion semaphore（在 push 末尾），
map_push 对整个 segment 只写一条 wait 指令：

```
copy_push: [tracker_0_wait][copy_0_cmds][tracker_1_wait][copy_1_cmds]...[copy_7_cmds][sem_done_A]
map_push:  [wait sem_done_A][map_0_cmds][map_1_cmds]...[map_7_cmds]
```

**问题**：block 0 的 map 命令需要等到 copy_7 完成后才能开始执行，
而语义上 map_0 只需要等 copy_0 完成。这是 shared push 的内在代价。

**影响评估**：
- 每 page 4KB 的 CE copy 时间 ≈ 0.3-1 µs
- segment_size=8：block 0 的 map 最多多等 7 × 1 µs = 7 µs
- 但 GPU MEMOPS engine 和 CE engine 可以并行，等待期间 CE 在执行后续 segments
- 对于小页面（4KB）场景，实际影响很小
- 对于大页面（2MB，copy time ~100+ µs/block），应减小 segment_size 到 2-4

**结论**：segment_size 应当与典型的 CE copy 时间成反比，
而不是固定值。可考虑基于 fault 中观察到的 page size 自适应。

### 7.2 GPU Unmap 本身没有被 Merge

当前设计中，GPU unmap 每个 block 独立 push_begin/push_end。如果一个 batch
中有 200 个 warm blocks（均有旧 GPU 映射），就有 200 次独立的 MEMOPS push。
这和 CE copy/map 的情况完全一样——只是 unmap 没有被纳入 merge。

**为什么没有 merge unmap**：unmap 发生在 `block_unmap_gpu` 中，
在每个 block 的处理开始时，此时 shared push 可能还没建立（或尚未 open）。
在 kthread 设计中，unmap 在 `service_block_faults_kthread` 里逐 block 执行，
和 copy/map 的 merge 时机不同。

**潜在优化**：若场景中 warm blocks 占比高，可以在处理一个 segment 的所有
blocks 之前，先统一发出所有 GPU unmap 命令到一个共享的 unmap_push，
然后再处理 copy/map。这样 unmap 的 push_begin/end 开销也能被摊薄。
但这需要两遍遍历 segment，实现复杂度较高。对于主要是 cold read 的场景，
价值有限。

### 7.3 移除 Copy → GPU Unmap 的保守依赖（可量化收益）

基于 Section 1.5 的分析，copy_push 中的 `push_acquire_tracker` 等待了 GPU unmap，
但这不是必要的。若移除这个依赖：

**修改方式**：在 `block_copy_resident_pages_between` 的 merge 路径，
不再对 copy push 做完整的 tracker acquire，仅等待 copy 所真正需要的前驱
（即前一次对 *目标* GPU VRAM 物理页的写操作，如果有的话）：

```c
// 当前（保守）：
uvm_push_acquire_tracker(merge_copy_push, &block->tracker); // 等所有先前操作

// 优化（最小依赖）：
// copy push 不需要等 GPU unmap
// 只需 map push 通过 push_acquire_tracker 等待 copy + unmap
// → 在 map push 写入时，tracker 里同时包含 unmap 和 copy 的 semaphore
```

**收益**：在 warm 场景下（有 GPU unmap），GPU 的 CE engine 不再被
unmap semaphore 阻塞，可以与 MEMOPS engine（执行 unmap）真正并行。
对于 cold read（最常见），此改动无差异（tracker 里本来没有 unmap entry）。

### 7.4 Merge 停止策略与递减收益（呼应 batch-level 分析）

当前 merge 在 batch 结束时由 coordinator 关闭 push，没有中途停止机制。
实际上 `UVM_KTHREAD_MERGE_TIMEOUT_NS = 500` 和 `UVM_KTHREAD_MERGE_MAX_FAULTS = 64`
都已定义但未实现。

**递减收益曲线**：
```
合并第 k 个 thin block 的边际收益 ≈ push_open_cost / k
合并第 k 个 thin block 的边际成本 = 等待该 block 加入的延迟 + push_lock 竞争
```

从 k=16 开始，边际收益 < 1 µs，而 push_lock 竞争和等待开销可能已超过此值。

**合理的自适应策略**：
1. 在 merge_state 里维护一个 `atomic_t accum_faults` 计数器
2. 每个 thin block worker append 完后：`total = atomic_add_return(num_faults, &accum_faults)`
3. 若 `total >= UVM_KTHREAD_MERGE_MAX_FAULTS`：该 worker 负责 push_end，
   state 重置为 0，下一个 thin block 开启新的 merge push pair
4. 若到 batch 结束 state 仍为 2：coordinator 负责收尾的 push_end

这样每 64 个 faults 就提交一次，GPU 可以更早开始执行，
而不是等整个 batch（可能 200+ faults）都写完才提交。

### 7.5 Segment 大小与 Page Size 的关系（自适应建议）

| Page Size | CE copy time/block | 建议 segment_size | 理由 |
|---|---|---|---|
| 4KB | 0.3–1 µs | 16–32 | copy 快，可以攒更多 block 再提交 |
| 64KB | 5–15 µs | 4–8 | 攒太多会让 map 等很久 |
| 2MB | 100–300 µs | 1–2 | 基本退化为 per-block，merge 意义不大 |

实现时可以在 segment flush 逻辑里根据 `item->num_faults × page_size` 估算
当前 segment 积累的 CE 工作量，动态决定是否提前 flush。
