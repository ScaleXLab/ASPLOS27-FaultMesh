# Cross-VA-Block Fault Merging: 性能分析

## 1. 我们到底优化了什么

### 1.1 原始 UVM 处理流程

原始 UVM 对每个 VA Block 的 fault 处理流程：

```
main loop: 对每个 VA Block 调用 service_fault_batch_dispatch
  -> 查找 VA Block（树遍历 + block 创建）
  -> service_fault_batch_block(gpu, va_block, ...)
       -> mutex_lock(&va_block->lock)               // 加锁
       -> service_fault_batch_block_locked(...)
            -> 遍历属于这个 block 的 fault，计算迁移决策
            -> uvm_va_block_service_locked(...)
                 -> uvm_va_block_service_copy(...)
                      -> block_copy_resident_pages_between(...)
                           -> push_begin()   // 创建 CE copy push  ← push #1
                           -> 写入 CE copy 命令（实际的数据拷贝）
                           -> push_end()     // 关闭并提交 push
                 -> uvm_va_block_service_finish(...)
                      -> block_map_gpu_to(...)
                           -> push_begin()   // 创建 MEMOPS map push ← push #2
                           -> 写入 PTE 映射命令（页表更新）
                           -> push_end()     // 关闭并提交 push
       -> mutex_unlock(&va_block->lock)              // 解锁
```

每个 VA Block 产生 2 个 GPU push。处理 8 个 block = 16 个 push。

### 1.2 修改后的流程（v3: copy + map 双共享 push）

在 `service_fault_batch_merged` 中同时开好**两个**共享 push：一个 CE copy push
和一个 MEMOPS map push，分别通过 `merge_ext.copy_push` 和 `merge_ext.map_push`
传递给底层函数。

```
service_fault_batch_merged:
  -> uvm_push_begin(&copy_push, CE channel)       // 共享 CE push（只开一次）
  -> uvm_push_begin(&map_push, MEMOPS channel)    // 共享 MEMOPS push（只开一次）
  -> merge_ext.copy_push = &copy_push
  -> merge_ext.map_push  = &map_push

  -> 对每个 VA Block 调用 service_fault_batch_block：
       -> mutex_lock(&va_block->lock)
       -> ... 遍历 fault，计算迁移决策 ...
       -> block_copy_resident_pages_between(...)
            -> 检测到 merge_ext.copy_push != NULL
            -> 跳过 push_begin()
            -> uvm_push_acquire_tracker(copy_push, &block->tracker)  // 同步
            -> 直接往共享 copy push 里写 CE copy 命令
            -> 跳过 push_end()
            -> uvm_tracker_add_push_safe(copy_tracker, copy_push)    // 传播依赖
       -> block_map_gpu_to(...)
            -> 检测到 merge_ext.map_push != NULL
            -> 跳过 push_begin_acquire()
            -> uvm_push_acquire_tracker(map_push, &va_block->tracker)  // 等待 copy
            -> 直接往共享 map push 里写 PTE 映射命令
            -> 跳过 push_end()
            -> uvm_tracker_add_push_safe(out_tracker, map_push)  // 传播依赖
       -> mutex_unlock(&va_block->lock)

  -> merge_ext.copy_push = NULL
  -> merge_ext.map_push  = NULL
  -> uvm_push_end(&copy_push)    // 最后才关闭共享 CE push
  -> uvm_push_end(&map_push)     // 最后才关闭共享 MEMOPS push
```

**同步机制**：map push 中每个 block 的 PTE 命令之前都会插入一条
semaphore wait（通过 `uvm_push_acquire_tracker`），等待共享 copy push
中对应的 CE 操作完成。即使 copy push 尚未 `push_end`，这也能正确工作，
因为 semaphore entry 在 `push_begin` 时就已经确定。

### 1.3 实际效果

| | Copy push 数量 | Map push 数量 | 加锁/解锁次数 | 总 push 数 |
|---|---|---|---|---|
| 原始（8 个 block）| 8 | 8 | 8 | **16** |
| v2 合并（仅 copy）| **1（共享）** | 8（没变）| 8（没变）| **9** |
| v3 合并（copy + map）| **1（共享）** | **1（共享）** | 8（没变）| **2** |

**v3 将 16 个 push 降到 2 个，省了 14 个 push_begin/push_end 开销。锁仍然没省。**

## 2. 为什么 service_fault_batch_block 不能处理不同 VA Block 的 fault

这是 UVM 架构的根本限制。`service_fault_batch_block` 以及它内部的所有函数，
都围绕**单个 va_block 对象**设计。原因如下：

### 2.1 锁的作用域

```c
// service_fault_batch_block (line 1566)
uvm_mutex_lock(&va_block->lock);
// ... 所有操作都在这个锁的保护下 ...
uvm_mutex_unlock(&va_block->lock);
```

锁住的是**一个特定的 va_block**。你持有 block A 的锁，只能修改 block A 的
数据。如果你想同时修改 block B，你需要另外获取 block B 的锁。UVM 不允许在
持有 block A 的锁时，去操作 block B 的页表、residency 等数据。

### 2.2 页面索引是 block 内部的相对索引

```c
// service_fault_batch_block_locked (line 1386)
uvm_page_index_t page_index = uvm_va_block_cpu_page_index(va_block, current_entry->fault_address);
```

`page_index` 是 fault 地址在 **这个 block 内部**的偏移量（0~511，因为一个
2MB block 有 512 个 4KB 页）。如果一个 fault 的地址不在
`[va_block->start, va_block->end]` 范围内，这个 page_index 就是非法的。

### 2.3 所有数据结构都是 per-block 的

每个 `uvm_va_block_t` 拥有自己独立的：

- **resident_mask**: 哪些页面驻留在哪个处理器（CPU/GPU）上
- **pte_bits**: 当前页表权限状态（读/写/原子）
- **mapped**: 哪些处理器映射了这个 block
- **tracker**: 记录影响这个 block 的 GPU 操作的依赖关系

这些 mask 和 tracker 的大小都是 `PAGES_PER_UVM_VA_BLOCK`（512）。它们只能
描述**这一个 block** 的 512 个页面。你无法用 block A 的 resident_mask 去表示
block B 的页面状态。

### 2.4 fault 遍历循环的终止条件

```c
// service_fault_batch_block_locked (line 1376-1379)
for (i = first_fault_index;
     i < batch_context->num_coalesced_faults &&
     ordered_fault_cache[i]->va_space == va_space &&
     ordered_fault_cache[i]->fault_address <= end;  // end = va_block->end
     ++i) {
```

循环在 `fault_address > va_block->end` 时停止。这意味着只处理地址落在
**这个 block 范围内**的 fault。超出范围的 fault 属于下一个 block，会在
主循环的下一次迭代中处理。

### 2.5 结论：VA Block 是不可跨越的处理边界

UVM 的整个 fault 处理流程——从加锁、页面索引、迁移决策、数据拷贝、页表更新
到 tracker 同步——都以单个 VA Block 为边界。要真正跨 block 合并处理，需要：

1. 同时持有多个 block 的锁
2. 有一种跨 block 的页面索引方式
3. 有跨 block 的 residency/PTE 数据结构
4. 重写遍历循环使其能跨越 block 边界

这相当于重写 UVM 的核心 ~12000 行代码，风险极高。

## 3. Overhead 分析

### Overhead 1: Pre-scan（已通过 v2 优化缓解）

v1 版本：每次 dispatch 都调用 `find_va_block_for_fault()`（树遍历）+
`count_block_faults()`（数组扫描），即使第一个 block 就已经很大。

v2 优化：增加了快速 2MB 检查。比较位置 `i` 和 `i+7` 的 fault 地址是否在
同一个 2MB block 内（1 次数组读取 + 1 次位运算比较）。如果是，直接跳过
合并逻辑，零开销。

### Overhead 2: 每个 block 独立加锁/解锁

原始路径处理 1 个 block 内的 8 个 fault：1 次 mutex_lock/unlock。
合并路径处理 8 个 block 各 1 个 fault：8 次 mutex_lock/unlock。
mutex 操作在内核中有非平凡的开销（原子操作 + 可能的调度）。

### Overhead 3: 8 个 map push 没有被合并（v3 已解决）

~~Copy 用 CE 通道，map 用 MEMOPS 通道——两个不同的 GPU 通道。一个 push
只能在一个通道上。~~ v3 中同时打开两个共享 push（分别在 CE 和 MEMOPS 通道），
在每个 block 处理期间交替向两个 push 写入命令。通过在 map push 中插入
`uvm_push_acquire_tracker` 来保证 map 命令在对应 copy 命令之后执行。
这不需要两遍处理，也不需要保存中间状态。

### Overhead 4: Tracker 同步

使用外部共享 push 时，每个 block 需要额外调用
`uvm_push_acquire_tracker(push, &block->tracker)` 来建立依赖关系，
以及 `uvm_tracker_add_push_safe(copy_tracker, push)` 来传播依赖。

## 4. map push 共享的实现（v3 方案）

之前认为要共享 map push 必须做"两遍处理"（先全部 copy，再全部 map），
因此受限于 `block_service_context` 单例问题。

v3 的突破：**不需要两遍处理**。两个 push 可以同时打开，在单遍处理中交替写入。
关键发现：`uvm_push_t` 的 semaphore entry 在 `push_begin` 时就确定了，
`push_end` 只是信号触发。因此即使 copy push 还没有 `push_end`，map push
就可以通过 `uvm_push_acquire_tracker` 获取 copy push 的 semaphore entry，
GPU 硬件会在执行时保证正确的顺序。

修改点：
1. `service_fault_batch_merged`：额外开一个 `map_push`（MEMOPS 通道），
   设置 `merge_ext.map_push = &map_push`，循环结束后 `push_end` 两个 push。
2. `block_map_gpu_to`：当 `use_ext_push` 为 true 时，不再创建新 push，
   而是调用 `uvm_push_acquire_tracker(push, &va_block->tracker)` 来获取
   copy 完成的依赖，然后直接往共享 map push 写入 PTE 命令。

## 5. 数据对比

### 随机读（Phase1-rand）

| block_faults | 原始 avg/fault | 合并后 avg/fault |
|-------------:|---------------:|-----------------:|
|            1 | ~4.5 us        | ~4.5 us (极少)   |
|            8 | ~0.5 us        | ~1.5 us          |
|           16 | ~0.3 us        | ~0.8 us          |

v2 合并后 bf=8 比原始 bf=8 慢 3 倍。v3 将 map push 也合并（16→2 push），
预期在 push 建立开销占主导的场景下获得显著提升。剩余开销来自
per-block 加锁和 tracker 同步。

### 收敛速度

- 原始：per-fault cost 在 bf=16 时收敛到 ~0.3 us
- 合并后：需要 bf=30 才收敛到同样水平

## 6. 关于页面连续性的备注

从原始直方图数据看，即使是随机访问（Phase1-rand），dispatch 时间也随
block_faults 数量增加而亚线性下降。这说明在**原始路径**中，push 建立开销
（而非页面连续性）是主要瓶颈。

v3 合并路径将 push 数量从 2N 降到 2，剩余的 per-block 开销主要是
加锁/解锁和 tracker 同步。如果 v3 仍无显著效果，说明 push 建立开销
占总时间比例较小，瓶颈在其他环节（锁、per-page 处理、GPU 执行本身），
此时需要重新考虑优化方向。
