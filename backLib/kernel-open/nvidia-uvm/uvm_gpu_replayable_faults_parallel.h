/*******************************************************************************
    Copyright (c) 2024-2025 NVIDIA Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

*******************************************************************************/

#ifndef __UVM_GPU_REPLAYABLE_FAULTS_PARALLEL_H__
#define __UVM_GPU_REPLAYABLE_FAULTS_PARALLEL_H__

#include "uvm_linux.h"
#include "uvm_forward_decl.h"
#include "uvm_lock.h"
#include "uvm_push.h"
#include "uvm_tracker.h"
#include "uvm_va_block_types.h"
#include <linux/kthread.h>
#include <linux/completion.h>

// Maximum number of blocks that can be processed in parallel per batch.
// Must be large enough to handle all faults in a single GPU fault buffer
// drain (~512-1024 unique blocks for random access patterns).
#define UVM_PARALLEL_FAULT_MAX_BLOCKS 1024

// Maximum number of persistent kthreads in the pool
#define UVM_KTHREAD_MAX_WORKERS 32

// Queue size for the kthread work queue (must be >= MAX_BLOCKS)
#define UVM_KTHREAD_QUEUE_SIZE  UVM_PARALLEL_FAULT_MAX_BLOCKS

// Dynamic merge: if a block has fewer faults than this, the kthread will
// try to merge its GPU push with subsequent thin blocks in the queue.
#define UVM_KTHREAD_MERGE_FAULT_THRESHOLD 8

// Default fault-count threshold per merge segment.  Blocks are accumulated
// into a segment until total faults >= this value, then the segment is closed.
#define UVM_KTHREAD_SEGMENT_SIZE          16

// Module parameter (defined in uvm_gpu_replayable_faults.c)
extern unsigned uvm_merge_dispatch;

// Module parameter (defined in uvm_gpu_replayable_faults_parallel.c)
extern unsigned uvm_merge_profile_enable;

// Forward declarations
struct uvm_fault_service_batch_context_struct;
struct uvm_fault_buffer_entry_struct;

// Snapshot of va_space state needed during block service.
// This allows us to release va_space lock while still having access to
// necessary configuration.
//
// Note: We keep this minimal to avoid large stack allocations.
// The va_space pointer is safe to use because:
// 1. The va_space is reference counted through file descriptor
// 2. The mm is retained during parallel processing
typedef struct {
    // The gpu_va_space for the faulting GPU
    uvm_gpu_va_space_t *gpu_va_space;

    // Pointer to the va_space (not a copy - va_space outlives the batch)
    uvm_va_space_t *va_space;

    // The mm_struct retained during the batch
    struct mm_struct *mm;
} uvm_va_space_snapshot_t;

// Forward declaration for the parallel context
typedef struct uvm_parallel_fault_context_struct uvm_parallel_fault_context_t;

// Group info for parallel processing - tracks faults for a single block
typedef struct {
    uvm_va_block_t *va_block;
    NvU32 first_fault_index;
    NvU32 num_faults;
} uvm_block_fault_group_t;

// Pre-allocated service context for each worker
// This avoids malloc/free per worker, reducing overhead significantly
typedef struct {
    uvm_service_block_context_t *service_context;  // Pointer to avoid incomplete type
    uvm_va_block_context_t *block_context;         // Allocated once during init
    bool in_use;
} uvm_parallel_worker_context_t;

// Kthread work item: one per VA block.
// Workers may attach the block to a shared merge push or keep it on a private
// push depending on the active experimental dispatch policy and module params.
typedef struct {
    uvm_va_block_t              *va_block;
    uvm_gpu_t                   *gpu;
    uvm_va_space_snapshot_t      snapshot;
    struct uvm_fault_service_batch_context_struct *batch_context;
    NvU32                        first_fault_index;
    NvU32                        num_faults;
    bool                         hmm_migratable;
    // Output
    NV_STATUS                    status;
    uvm_tracker_t                tracker;
    NvU32                        faults_serviced;
    bool                         has_fatal_faults;
    // Timing (for FPD logging)
    NvU64                        dispatch_start_ns;
    NvU64                        dispatch_end_ns;
    // Per-dispatch breakdown copied from block_context after servicing
    struct {
        NvU64 unmap_ns;
        NvU64 alloc_ns;
        NvU64 copy_ns;
        NvU64 map_ns;
        NvU32 subregion_count;
        NvU32 pages_migrated;
    } breakdown_ns;
    struct {
        NvU32 copy_ext_used;
        NvU32 copy_ext_fallback;
        NvU32 map_ext_used;
        NvU32 map_ext_fallback_2m;
        NvU32 map_ext_fallback_space;
    } merge_profile;

    // Async copy-map pipeline state (optional)
    bool                         pipeline_async;
    uvm_service_block_context_t *pipeline_service_context;
    uvm_va_block_context_t      *pipeline_block_context;
} uvm_kthread_work_item_t;

// Persistent kthread pool for parallel fault processing
typedef struct {
    struct task_struct  *threads[UVM_KTHREAD_MAX_WORKERS];
    NvU32                num_workers;

    // Per-item output array (status, tracker, profiling); inputs are read
    // from batch_groups[] and the batch-level shared fields below.
    uvm_kthread_work_item_t  *queue;        // dynamically allocated
    NvU32                     queue_capacity;

    // Batch-level shared state set once by coordinator before dispatch.
    // Workers read these instead of per-item copies.
    uvm_gpu_t                            *batch_gpu;
    uvm_va_space_snapshot_t               batch_snapshot;
    struct uvm_fault_service_batch_context_struct *batch_ctx;
    uvm_block_fault_group_t              *batch_groups;
    NvU32                                 batch_num_groups;
    NvU32                                 batch_num_active;
    NvU32                                 batch_total_faults;
    NvU32                                 batch_max_group_faults;

    // Synchronization
    atomic_t                 batch_worker_counter; // workers claim batch-local ID
    atomic_t                 items_remaining;
    struct completion        all_done;
    wait_queue_head_t        worker_waitq;
    bool                     batch_ready;    // flag checked by workers

    // Pre-allocated worker contexts (one per worker)
    uvm_parallel_worker_context_t  worker_contexts[UVM_KTHREAD_MAX_WORKERS];
    bool                           contexts_initialized;

    // Coordinator sets this before each inner dispatch to tell workers whether
    // the current epoch has an open shared merge push pair available.
    // Individual workers may still keep some blocks private depending on
    // uvm_merge_shared_push_max_faults.
    bool                     merge_dispatch_active;

    // Base index into pool->queue[] for the current segment dispatch.
    // Workers store results at queue[queue_base + idx].
    NvU32                    queue_base;

    // Pool-level shared pushes for merge optimization.
    // State: 0=CLOSED, 2=OPEN.
    struct {
        spinlock_t   copy_push_lock;
        spinlock_t   map_push_lock;
        uvm_push_t   copy_push[UVM_KTHREAD_MAX_WORKERS];
        uvm_push_t   map_push[UVM_KTHREAD_MAX_WORKERS];
        atomic_t     state;
        int          active;    // 0 or 1: which push pair workers should use (legacy)
        atomic_t     worker_pushed_end; // 1 = last worker already called push_end
    } merge_state;

    // Per-batch merge profiling stats (gated by uvm_merge_profile_enable)
    struct {
        atomic_t    thin_count;
        atomic_t    fat_count;
        atomic_t    used;
        atomic64_t  spin_ns;
        atomic_t    contended_workers;
        atomic_t    total_segments;
        atomic_t    copy_ext_used;
        atomic_t    copy_ext_fallback;
        atomic_t    map_ext_used;
        atomic_t    map_ext_fallback_2m;
        atomic_t    map_ext_fallback_space;
        atomic64_t  group_ns;          // Phase 1: VA block grouping loop time
        atomic64_t  classify_ns;       // thin/fat partition sort time (after Phase 1)
        atomic64_t  fat_dispatch_ns;   // wall time of Phase A (fat blocks)
        atomic64_t  thin_dispatch_ns;  // wall time of Phase B (all thin segments)
        atomic64_t  seg_push_begin_ns; // cumulative push_begin time (copy+map) in segments
        atomic64_t  seg_push_end_ns;   // cumulative push_end time (copy+map) in segments
        atomic64_t  seg_dispatch_ns;   // cumulative dispatch_and_wait_inner time in segments
        atomic_t    reclassed_count;   // batches where num_thin < seg_size → reclassed to fat
        atomic_t    tiny_seg_count;    // segments with total_faults < 8 (should be 0 after fix)
        atomic_t    effective_seg_threshold;
        atomic_t    max_group_faults;
        atomic_t    merged_block_count;
        atomic_t    solo_block_count;
        atomic_t    heavy_block_count;
        atomic64_t  heavy_block_fault_sum;
        // Timeline metrics for overlap analysis (Phase 3)
        atomic64_t  batch_start_ns;    // start of dispatch_and_wait
        atomic64_t  first_submit_ns;   // first push_end timestamp
        atomic64_t  last_submit_ns;    // last push_end timestamp
        atomic64_t  batch_complete_ns; // end of dispatch_and_wait
    } merge_stats;

    // Continuous-workers state: workers stay awake across segments within a
    // batch, spinning on seg_generation instead of going through wait_event /
    // wake_up_nr for each segment.
    struct {
        bool        active;         // set before first segment, cleared after last
        atomic_t    seg_generation; // coordinator increments when new segment state is ready
        atomic_t    seg_finished;   // set to 1 after all segments are done
        atomic_t    batch_gen;      // incremented each batch; workers use to detect stale state
    } continuous;

    // Persistent cross-batch spin: workers spin-wait between batches instead
    // of going back to wait_event.  Eliminates ~30-160us scheduler wake-up
    // latency per batch.  Workers spin for up to persistent_timeout_us after
    // each batch; if no new batch arrives, they fall back to wait_event.
    struct {
        atomic_t    batch_gen;      // coordinator increments when new batch state is ready
    } persistent;

    // Segment-parallel merge: workers are grouped by segment, each segment
    // gets its own pre-opened push pair on a different CE (round-robin).
    // Multiple workers within a segment use strided block distribution.
    struct {
        bool        active;
        NvU32       num_segments;
        NvU32       seg_start[UVM_KTHREAD_MAX_WORKERS];
        NvU32       seg_count[UVM_KTHREAD_MAX_WORKERS];
        spinlock_t  push_locks[UVM_KTHREAD_MAX_WORKERS];

        NvU32       workers_per_seg[UVM_KTHREAD_MAX_WORKERS];
        NvU32       merged_blocks_per_seg[UVM_KTHREAD_MAX_WORKERS];
        NvU32       worker_seg_id[UVM_KTHREAD_MAX_WORKERS];
        NvU32       worker_local_id[UVM_KTHREAD_MAX_WORKERS];
        NvU32       total_workers_dispatched;

        atomic_t    copy_done_counters[UVM_KTHREAD_MAX_WORKERS];
        atomic_t    copy_pushed_end[UVM_KTHREAD_MAX_WORKERS];
    } seg_parallel;

    // Dynamic dispatch: workers atomically claim blocks from a global cursor.
    // Segment membership is pre-computed per-block so push pair selection is
    // independent of scheduling order — segments control GPU push batching,
    // not CPU work assignment.
    struct {
        bool        active;
        atomic_t    work_cursor;
        NvU32       num_segments;
        NvU32       block_seg_id[UVM_PARALLEL_FAULT_MAX_BLOCKS];
        spinlock_t  copy_push_locks[UVM_KTHREAD_MAX_WORKERS];
        spinlock_t  map_push_locks[UVM_KTHREAD_MAX_WORKERS];
        // copy_map_split: last copy worker calls push_end(copy) early so GPU
        // can begin executing copies while map commands are still appended.
        NvU32       merged_blocks_per_seg[UVM_KTHREAD_MAX_WORKERS];
        atomic_t    copy_done_counters[UVM_KTHREAD_MAX_WORKERS];
        atomic_t    copy_pushed_end[UVM_KTHREAD_MAX_WORKERS];
    } dynamic;

    // Per-worker timing accumulators (gated by uvm_worker_timing_debug)
    struct {
        u64 total_wake_to_first_lock_ns;
        u64 total_lock_acquire_ns;
        u64 total_block_service_ns;
        u64 total_item_setup_ns;
        u64 total_batches;
        u64 total_items;
        // Unmap sub-phase: time spent in block_unmap_cpu (includes IPI if enabled).
        // Only populated when uvm_fpd_profile_enable=1 (sets capture_breakdown in
        // service_block_faults_kthread, which fills item->breakdown_ns.unmap_ns).
        u64 total_unmap_ns;
    } worker_timing[UVM_KTHREAD_MAX_WORKERS];

    // Shutdown flag
    bool                     shutting_down;

} uvm_kthread_pool_t;

// Context for a parallel fault batch
struct uvm_parallel_fault_context_struct {
    // Array of block groups (populated by Phase 1 grouping)
    uvm_block_fault_group_t groups[UVM_PARALLEL_FAULT_MAX_BLOCKS];

    // Kthread pool for parallel dispatch
    uvm_kthread_pool_t *kthread_pool;
};

// Initialize the parallel fault context for a batch
void uvm_parallel_fault_context_init(uvm_parallel_fault_context_t *ctx);

// Create a snapshot of va_space state while holding the read lock
void uvm_va_space_snapshot_init(
    uvm_va_space_snapshot_t *snapshot,
    uvm_va_space_t *va_space,
    uvm_gpu_va_space_t *gpu_va_space,
    struct mm_struct *mm);

// Check if parallel fault processing is enabled
// This can be controlled via module parameter
bool uvm_parallel_fault_enabled(void);

// Initialize the parallel fault subsystem (create kthread pool)
NV_STATUS uvm_parallel_fault_init(void);

// Deinitialize the parallel fault subsystem (destroy kthread pool)
void uvm_parallel_fault_exit(void);

// Kthread pool lifecycle
NV_STATUS uvm_kthread_pool_create(uvm_kthread_pool_t **out_pool, NvU32 num_workers);
void uvm_kthread_pool_destroy(uvm_kthread_pool_t *pool);
uvm_kthread_pool_t *uvm_kthread_pool_get_global(void);

// Dispatch num_items from the kthread pool queue and wait for completion.
// When merge is enabled, blocks are grouped into segments by accumulating
// faults until total >= UVM_KTHREAD_SEGMENT_SIZE.  Each segment shares a
// single pair of GPU pushes (copy + map).  No thin/fat distinction — all
// blocks participate in fault-count-based segmentation.
NV_STATUS uvm_kthread_dispatch_and_wait(uvm_kthread_pool_t *pool,
                                        NvU32 num_items,
                                        NvU64 group_ns,
                                        uvm_tracker_t *out_tracker);

NV_STATUS uvm_gpu_async_copy_init(uvm_gpu_t *gpu);
void uvm_gpu_async_copy_deinit(uvm_gpu_t *gpu);

// Service faults for a single VA block using a pre-allocated worker context.
// Caller MUST hold va_space read lock for the block's va_space.
NV_STATUS service_block_faults_kthread(uvm_kthread_work_item_t *item,
                                       uvm_parallel_worker_context_t *worker_ctx);

// Module parameter to enable/disable parallel fault processing
// 0 = disabled, 2 = kthread pool
extern int uvm_parallel_fault_processing;

// Adaptive pre-scan profiling counters (defined in uvm_gpu_replayable_faults.c)
extern atomic_t g_prescan_bypass_count;
extern atomic_t g_prescan_parallel_count;
extern atomic64_t g_prescan_bypass_blocks_sum;
extern atomic64_t g_prescan_parallel_blocks_sum;

#define PRESCAN_HIST_MAX 64
extern atomic_t g_prescan_block_hist[PRESCAN_HIST_MAX + 1];

// Module parameter: number of kthread workers (default 8)
extern unsigned uvm_kthread_workers;

// Module parameter: enable FPD profiling (0=off, 1=on)
extern unsigned uvm_fpd_profile_enable;
// Module parameter: log per-batch fault count to dmesg (0=off, 1=on)
extern unsigned uvm_batch_profile_enable;
// Module parameter: log [PAGE_REP] raw=N unique=M per batch (0=off, 1=on)
extern unsigned uvm_page_rep_profile_enable;
// Module parameter: log [PAGE_REP_DETAIL] residual duplicate-page detail lines (0=off, 1=on)
extern unsigned uvm_page_rep_detail_enable;
// Module parameter: enable verbose [UVM_FAULT_DBG] per-fault dump (0=off, 1=on)
extern unsigned uvm_fault_dbg_enable;

// Module parameters for async copy-map pipeline.
extern unsigned uvm_parallel_async_copy_map_enable;
extern unsigned uvm_parallel_async_copy_map_workers;
extern unsigned uvm_parallel_async_copy_map_queue_depth;
extern unsigned uvm_parallel_async_copy_map_prefetch_safe;

#endif // __UVM_GPU_REPLAYABLE_FAULTS_PARALLEL_H__
