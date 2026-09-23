# UVM Parallel Fault Processing Architecture

## Overview

This document explains the complete data flow of the parallel fault processing
pipeline in `uvm_gpu_replayable_faults_parallel.c`.  The system has three
layers that execute in sequence:

```
GPU fault buffer
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  Phase 1 (serial, coordinator thread)               │
│  Group faults by VABlock                            │
│  Input:  ordered_fault_cache[0..N-1]                │
│  Output: groups[0..num_groups-1]                    │
│          each group = { va_block, first_fault, cnt } │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│  Segment Formation (serial, coordinator)            │
│  uvm_kthread_dispatch_and_wait()                    │
│  Walk groups[], accumulate faults, cut segments     │
│  when running_faults >= threshold (default 16)      │
└───────────────────┬─────────────────────────────────┘
                    │  (for each segment)
                    ▼
┌─────────────────────────────────────────────────────┐
│  dispatch_and_wait_inner()                          │
│  Wake K workers, each processes N/K items           │
│  Workers call service_block_faults_kthread() per    │
│  item (= per VABlock)                               │
└─────────────────────────────────────────────────────┘
```

---

## Layer 1: Phase 1 — Fault Grouping (serial)

**Location**: `uvm_gpu_replayable_faults.c`, inside `service_fault_batch_parallel()`

**Input**: `batch_context->ordered_fault_cache[]` — an array of N individual
fault entries, sorted by (va_space, fault_address).

**Output**: `groups[]` — an array of `uvm_block_fault_group_t`, where each
entry represents one VABlock:

```c
typedef struct {
    uvm_va_block_t *va_block;
    NvU32 first_fault_index;  // index into ordered_fault_cache
    NvU32 num_faults;         // consecutive faults belonging to this block
} uvm_block_fault_group_t;
```

**How it works**: The code walks through ordered faults.  When the fault
address falls outside the current VABlock, or the va_space changes, the
accumulated group is flushed to `groups[]` and a new group begins.  This is
why the "flush" code appears three times:

1. **va_space changed** — must release the old va_space lock before acquiring
   the new one, so flush the current group first.
2. **New VABlock** — the fault address belongs to a different va_block, so
   flush the previous group and start a new one.
3. **Loop end** — flush whatever remains in the last group.

**Example**: If the fault cache has 256 faults spanning 180 VABlocks, Phase 1
produces `num_groups = 180`, with each group's `num_faults` telling how many
of the 256 faults belong to that VABlock (e.g., one block might have 3 faults,
another might have 1, etc.).

---

## Layer 2: Segment Formation (serial, coordinator)

**Location**: `uvm_kthread_dispatch_and_wait()` in
`uvm_gpu_replayable_faults_parallel.c`

**Input**: `groups[0..num_items-1]` from Phase 1.

**Purpose**: Decide how many groups go into each "segment".  A segment is a
batch of VABlocks that share a single pair of GPU push buffers (copy + map).

**Algorithm**: Walk groups left-to-right, accumulating their fault counts.
When the running total reaches the threshold (default 16), close the segment
and start a new one.  If the remaining faults after closing would be less
than the threshold, extend the current segment to absorb them (avoids a
tiny tail segment).

```
groups:   [B0:3f] [B1:1f] [B2:7f] [B3:2f] [B4:5f] [B5:20f] [B6:1f] [B7:2f]
           ├─────────────────────┤ ├──────────────────────────────────────────┤
           segment 0 (13+5=18f)    segment 1 (20+1+2=23f, absorbed tail)
```

Two variables track the scan:
- `running_faults`: faults accumulated in the *current* segment (reset to 0
  when a segment is closed)
- `faults_so_far`: faults accumulated across *all* groups scanned so far
  (never reset; used to compute `remaining_faults = batch_total - faults_so_far`)

For each segment, the coordinator:
1. Opens a shared push pair: `uvm_push_begin(copy)` + `uvm_push_begin(map)`
2. Sets `pool->merge_dispatch_active = true` so workers know to use the
   shared push
3. Calls `dispatch_and_wait_inner()` with the segment's groups
4. After workers finish: `uvm_push_end(copy)` + `uvm_push_end(map)` → submits
   all accumulated GPU commands in one shot

**If merge is disabled** (`uvm_merge_dispatch = 0`): All groups are dispatched
in a single call to `dispatch_and_wait_inner()` without shared pushes.  Each
worker opens its own independent push for each VABlock.

---

## Layer 3: Worker Dispatch — `dispatch_and_wait_inner()`

**Location**: `dispatch_and_wait_inner()` in
`uvm_gpu_replayable_faults_parallel.c`

**Input**: A segment of `seg_num_items` groups (VABlocks).

**Mechanism**:

```c
pool->batch_num_active = num_active;   // e.g., 4 workers
pool->batch_num_groups = seg_num_items; // e.g., 20 groups in this segment
wake_up_nr(&pool->worker_waitq, num_active);  // wake 4 workers
wait_for_completion_timeout(&pool->all_done, 30s);  // wait until all done
```

This wakes up `num_active` persistent kthreads.  Each kthread was sleeping in
a `wait_event_interruptible_exclusive()` loop.  When woken, a worker:

1. Claims a worker ID via `atomic_add_return` on `batch_worker_counter`
2. Uses **static interleaved partitioning** to pick its items:

```c
for (idx = my_batch_id; idx < num_items; idx += num_active) {
    // Worker 0 processes items 0, 4, 8, 12, ...
    // Worker 1 processes items 1, 5, 9, 13, ...
    // Worker 2 processes items 2, 6, 10, 14, ...
    // Worker 3 processes items 3, 7, 11, 15, ...
    service_block_faults_kthread(item, worker_context);
}
```

This is NOT "one worker per item".  It's "each worker processes every K-th
item", where K = `num_active`.  So with 4 workers and 20 items, each worker
processes 5 items sequentially.

When a worker finishes one item, it does
`atomic_dec_and_test(&pool->items_remaining)`.  When all items are done,
`complete(&pool->all_done)` signals the coordinator to proceed.

**Why interleaved and not contiguous blocks?**  Interleaving distributes the
workload more evenly when some VABlocks are faster to process than others.
If blocks were assigned contiguously (e.g., worker 0 gets items 0-4, worker 1
gets 5-9), one worker could get all the "heavy" blocks and become a
bottleneck.

---

## Layer 4: Per-VABlock Processing — `service_block_faults_kthread()`

**Input**: One `uvm_kthread_work_item_t`, which represents one VABlock with
its associated faults.

**Key fields**:
- `item->va_block`: the VABlock to process
- `item->first_fault_index`: index into `ordered_fault_cache` where this
  block's faults start
- `item->num_faults`: how many faults belong to this block

**The per-fault loop (line 458)** iterates over the faults within this single
VABlock.  This is NOT iterating over different VABlocks — it's iterating over
the individual fault entries that all belong to the same VABlock.

**What the loop does for each fault**:

1. **Duplicate check** (line 480-486): If this fault has the same address as
   the previous fault and the previous wasn't fatal → skip (already handled).

2. **Authorization check** (line 493-499): If the GPU already has permission
   for this page → skip (no migration needed).

3. **Thrashing check** (line 501-508): If thrashing detection says to
   throttle → skip.

4. **Residency decision** (line 521-530): Determine where the page should
   reside (CPU? GPU? which GPU?).

5. **Set page mask bits**: Mark which pages need migration and what access
   type they require.

After the loop, **one call** to `uvm_va_block_service_locked()` (line 549)
processes all the marked pages together — this is where the actual unmap,
alloc, copy, and map GPU operations happen for the entire VABlock at once.

**So the per-fault loop is NOT doing the heavy work per-fault.**  It's doing
lightweight per-fault filtering and classification.  The heavy work (GPU
commands, DMA transfers, page table updates) happens once for the entire
VABlock in `uvm_va_block_service_locked()`.

---

## Push Sharing (Merge)

When `pool->merge_dispatch_active == true` (set by the coordinator before
dispatch), each worker's `block_context->merge_ext` points to the shared push
pair:

```c
wctx->block_context->merge_ext.copy_push = &pool->merge_state.copy_push[0];
wctx->block_context->merge_ext.map_push  = &pool->merge_state.map_push[0];
wctx->block_context->merge_ext.push_lock = &pool->merge_state.push_lock;
```

Inside `uvm_va_block_service_locked()`, when the code needs to write GPU
commands (copy, map), it checks `merge_ext.copy_push != NULL`.  If set, it
acquires `push_lock`, writes commands into the shared push buffer, and
releases the lock.  If the shared push doesn't have enough space (or the
operation requires an incompatible push type like 2M→4K split), it falls
back to opening its own independent push.

This means multiple VABlocks' GPU commands end up in the same push buffer,
submitted to the GPU in one batch → amortizes the per-push overhead
(push_begin ≈ 0.8µs, push_end ≈ 1.6µs).

---

## Complete Flow Example

```
Fault buffer: 256 faults across 180 VABlocks
seg_fault_threshold = 16

Phase 1 (serial):
  Group into 180 entries: groups[0]={blkA, 3 faults}, groups[1]={blkB, 1 fault}, ...

Segment Formation (serial, coordinator):
  Segment 0: groups[0..5], total 18 faults
  Segment 1: groups[6..22], total 21 faults
  ... (more segments)
  Segment N: groups[170..179], total 14 faults → absorbed into segment N-1

For each segment:
  Coordinator: push_begin(copy), push_begin(map)
  Coordinator: dispatch_and_wait_inner(seg_count=6)
    → wake 4 workers
    → Worker 0 processes groups[0], [4]       → 2 VABlocks
    → Worker 1 processes groups[1], [5]       → 2 VABlocks
    → Worker 2 processes groups[2]            → 1 VABlock
    → Worker 3 processes groups[3]            → 1 VABlock
    Each worker, for each assigned VABlock:
      Lock va_block
      Iterate faults (per-fault filter/classify)
      Call uvm_va_block_service_locked() once → writes GPU cmds to shared push
      Unlock va_block
      atomic_dec items_remaining
    All done → complete(&all_done)
  Coordinator: push_end(copy), push_end(map)  → submit all GPU cmds at once
  (next segment...)
```

---

## Summary of Key Confusions Addressed

| Question | Answer |
|----------|--------|
| What is `item`? | One VABlock + its associated faults. One item = one group from Phase 1. |
| Why iterate faults inside one VABlock? | To filter (duplicates, authorized, thrashing) and classify pages BEFORE the single heavy `uvm_va_block_service_locked()` call. The loop is lightweight. |
| What does `idx += num_active` mean? | Static interleaved work partitioning. Worker K processes items K, K+N, K+2N, ... This balances load across workers. |
| What is `my_batch_id`? | A worker's sequence number (0, 1, 2, ...) claimed via atomic increment when woken. Determines which items it processes. |
| Why does the kthread have a `while` loop? | It's a persistent thread — it sleeps waiting for work, processes a batch, then sleeps again. This avoids thread creation/destruction overhead per batch. |
| What does `dispatch_and_wait_inner` do? | Wakes K workers, they process N items in parallel (each doing N/K items), then signals completion. |
| Why wake K workers for N items? | K workers process N items total, not 1 item each. Each worker handles multiple items via the interleaved loop. |
| What do `running_faults`/`faults_so_far` track? | `running_faults` = faults in current segment (resets per segment). `faults_so_far` = cumulative faults scanned (never resets, used to compute how many faults remain). |
