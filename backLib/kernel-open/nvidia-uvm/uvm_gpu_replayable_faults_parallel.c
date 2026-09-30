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

#include "uvm_gpu_replayable_faults_parallel.h"
#include "uvm_gpu_replayable_faults.h"
#include "uvm_va_space.h"
#include "uvm_va_block.h"
#include "uvm_va_range.h"
#include "uvm_gpu.h"
#include "uvm_tracker.h"
#include "uvm_kvmalloc.h"
#include "uvm_perf_thrashing.h"
#include "uvm_tools.h"
#include "uvm_hmm.h"
#include "uvm_common.h"
#include "uvm_linux.h"
#include "uvm_channel.h"
#include "uvm_extern_decl.h"
#include "uvm_push.h"
#include "uvm_rm_mem.h"
#include "uvm_hal.h"

#include <linux/mmap_lock.h>
#include <linux/pgtable.h>
#include <linux/smp.h>
#include <linux/swap.h>
#include <linux/pagemap.h>

// 0 = disabled, 2 = kthread pool
int uvm_parallel_fault_processing = 0;
module_param(uvm_parallel_fault_processing, int, 0644);
MODULE_PARM_DESC(uvm_parallel_fault_processing,
                 "Parallel fault processing: 0=disabled, 2=kthread pool");

// Workqueue mode: 0 = pinned to cores (round-robin), 1 = unbound (kernel decides)
static unsigned uvm_parallel_fault_unbound = 0;
module_param(uvm_parallel_fault_unbound, uint, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_parallel_fault_unbound,
                 "Workqueue mode: 0=pinned cores (default), 1=unbound");

// Maximum concurrent workers (0 = no limit, uses all available)
static unsigned uvm_parallel_fault_max_workers = 0;
module_param(uvm_parallel_fault_max_workers, uint, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_parallel_fault_max_workers,
                 "Max concurrent workers (0=no limit)");

// Number of persistent kthread workers (default 8)
unsigned uvm_kthread_workers = 8;
module_param(uvm_kthread_workers, uint, 0644);
MODULE_PARM_DESC(uvm_kthread_workers,
                 "Number of persistent kthread workers for parallel fault processing (default 8)");

unsigned uvm_merge_profile_enable = 0;
module_param(uvm_merge_profile_enable, uint, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_merge_profile_enable,
                 "Enable per-batch merge statistics logging (0=off, 1=on)");

static unsigned uvm_merge_seg_timing_enable = 0;
module_param(uvm_merge_seg_timing_enable, uint, S_IRUGO | S_IWUSR);

static unsigned uvm_dispatch_timing_debug = 0;
module_param(uvm_dispatch_timing_debug, uint, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_dispatch_timing_debug,
                 "Log fine-grained dispatch overhead breakdown per batch (0=off, 1=on)");

static unsigned uvm_persistent_spin_workers = 0;
module_param(uvm_persistent_spin_workers, uint, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_persistent_spin_workers,
                 "Workers spin-wait indefinitely between batches instead of sleeping (0=off, 1=on)");

unsigned uvm_worker_timing_debug = 0;
module_param(uvm_worker_timing_debug, uint, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_worker_timing_debug,
                 "Accumulate per-worker timing breakdown; print on pool destroy (0=off, 1=on)");

unsigned uvm_parallel_async_copy_map_enable = 0;
module_param(uvm_parallel_async_copy_map_enable, uint, 0644);
MODULE_PARM_DESC(uvm_parallel_async_copy_map_enable,
                 "Enable async copy-map pipeline for no-merge parallel faults (0=off, 1=on)");

unsigned uvm_parallel_async_copy_map_workers = 1;
module_param(uvm_parallel_async_copy_map_workers, uint, 0644);
MODULE_PARM_DESC(uvm_parallel_async_copy_map_workers,
                 "Legacy compatibility knob for the old queue pipeline; ignored by the GPU-local dual-slot path");

unsigned uvm_parallel_async_copy_map_queue_depth = 2;
module_param(uvm_parallel_async_copy_map_queue_depth, uint, 0644);
MODULE_PARM_DESC(uvm_parallel_async_copy_map_queue_depth,
                 "Legacy compatibility knob for the old queue pipeline; ignored by the GPU-local dual-slot path");

unsigned uvm_parallel_async_copy_map_prefetch_safe = 1;
module_param(uvm_parallel_async_copy_map_prefetch_safe, uint, 0644);
MODULE_PARM_DESC(uvm_parallel_async_copy_map_prefetch_safe,
                 "Allow GPU-local async copy-map when prefetch is enabled (default 1)");

// Global kthread pool (new path)
static uvm_kthread_pool_t *g_uvm_kthread_pool = NULL;

static bool async_copy_map_enabled_for_item(uvm_kthread_work_item_t *item,
                                            uvm_service_block_context_t *service_context)
{
    uvm_va_space_t *va_space = NULL;

    if (!uvm_parallel_async_copy_map_enable)
        return false;

    if (!item || !service_context || !service_context->block_context)
        return false;

    if (!item->gpu || !item->gpu->async_copy_initialized)
        return false;

    va_space = item->snapshot.va_space;
    if (!va_space)
        return false;

    if (uvm_va_block_is_hmm(item->va_block) || item->hmm_migratable)
        return false;

    if (!uvm_parallel_async_copy_map_prefetch_safe && uvm_perf_prefetch_enabled(va_space))
        return false;

    return true;
}

static void clear_pipeline_item_state(uvm_kthread_work_item_t *item)
{
    if (!item)
        return;

    item->pipeline_async = false;
    item->pipeline_service_context = NULL;
    item->pipeline_block_context = NULL;
}

static void async_copy_slot_reset(uvm_gpu_async_copy_slot_t *slot)
{
    if (!slot)
        return;

    slot->processor_id = UVM_ID_INVALID;
    slot->va_block = NULL;
    slot->va_space = NULL;
    slot->tracker = NULL;
    slot->status = NULL;
    slot->occupied = false;
    memset(&slot->service_context, 0, sizeof(slot->service_context));
    slot->service_context.block_context = slot->block_context;
}

static NV_STATUS clone_pipeline_contexts_to_slot(uvm_gpu_async_copy_slot_t *slot,
                                                 uvm_kthread_work_item_t *item,
                                                 uvm_service_block_context_t *src_service_context)
{
    uvm_page_mask_t **saved_node_masks;
    size_t index;

    UVM_ASSERT(slot);
    UVM_ASSERT(slot->block_context);
    UVM_ASSERT(item);
    UVM_ASSERT(src_service_context);
    UVM_ASSERT(src_service_context->block_context);

    saved_node_masks = slot->block_context->make_resident.cpu_pages_used.node_masks;

    memcpy(slot->block_context, src_service_context->block_context, sizeof(*slot->block_context));
    slot->block_context->mm = item->snapshot.mm;
    slot->block_context->make_resident.cpu_pages_used.node_masks = saved_node_masks;
    for (index = 0; index < num_possible_nodes(); index++) {
        uvm_page_mask_copy(saved_node_masks[index],
                           src_service_context->block_context->make_resident.cpu_pages_used.node_masks[index]);
    }

    memcpy(&slot->service_context, src_service_context, sizeof(slot->service_context));
    slot->service_context.block_context = slot->block_context;
    slot->processor_id = item->gpu->id;
    slot->va_block = item->va_block;
    slot->va_space = item->snapshot.va_space;
    slot->tracker = &item->tracker;
    slot->status = &item->status;
    slot->occupied = true;
    item->pipeline_async = true;

    return NV_OK;
}

static void move_async_slot_payload(uvm_gpu_async_copy_slot_t *dst,
                                    uvm_gpu_async_copy_slot_t *src)
{
    uvm_page_mask_t **saved_node_masks;
    size_t index;

    UVM_ASSERT(dst);
    UVM_ASSERT(src);
    UVM_ASSERT(dst->block_context);
    UVM_ASSERT(src->block_context);

    saved_node_masks = dst->block_context->make_resident.cpu_pages_used.node_masks;

    memcpy(dst->block_context, src->block_context, sizeof(*dst->block_context));
    dst->block_context->make_resident.cpu_pages_used.node_masks = saved_node_masks;
    for (index = 0; index < num_possible_nodes(); index++) {
        uvm_page_mask_copy(saved_node_masks[index],
                           src->block_context->make_resident.cpu_pages_used.node_masks[index]);
    }

    memcpy(&dst->service_context, &src->service_context, sizeof(dst->service_context));
    dst->service_context.block_context = dst->block_context;
    dst->processor_id = src->processor_id;
    dst->va_block = src->va_block;
    dst->va_space = src->va_space;
    dst->tracker = src->tracker;
    dst->status = src->status;
    dst->occupied = src->occupied;
}

static int async_copy_map_worker_fn(void *data)
{
    uvm_gpu_t *gpu = data;
    uvm_gpu_async_copy_slot_t *slot;
    unsigned long flags;

    while (!kthread_should_stop()) {
        NV_STATUS status = NV_ERR_INVALID_STATE;
        NV_STATUS tracker_status = NV_OK;

        wait_event_interruptible(gpu->async_copy_waitq,
                                 gpu->async_copy_stop ||
                                 gpu->cd.occupied ||
                                 gpu->pd_copy.occupied ||
                                 kthread_should_stop());

        if (gpu->async_copy_stop || kthread_should_stop())
            break;

        spin_lock_irqsave(&gpu->async_copy_lock, flags);
        if (!gpu->cd.occupied && gpu->pd_copy.occupied) {
            move_async_slot_payload(&gpu->cd, &gpu->pd_copy);
            async_copy_slot_reset(&gpu->pd_copy);
        }

        if (!gpu->cd.occupied) {
            spin_unlock_irqrestore(&gpu->async_copy_lock, flags);
            continue;
        }

        slot = &gpu->cd;
        spin_unlock_irqrestore(&gpu->async_copy_lock, flags);

        if (slot->va_space)
            uvm_va_space_down_read(slot->va_space);

        if (slot->va_block) {
            uvm_mutex_lock(&slot->va_block->lock);
            status = uvm_va_block_service_copy_finish(slot->processor_id,
                                                      slot->va_block,
                                                      &slot->service_context);
            if (slot->tracker)
                tracker_status = uvm_tracker_add_tracker_safe(slot->tracker,
                                                              &slot->va_block->tracker);
            uvm_mutex_unlock(&slot->va_block->lock);
        }

        if (slot->va_space)
            uvm_va_space_up_read(slot->va_space);

        if (status == NV_OK)
            status = tracker_status;
        if (slot->status)
            *slot->status = status;

        spin_lock_irqsave(&gpu->async_copy_lock, flags);
        async_copy_slot_reset(&gpu->cd);
        wake_up_all(&gpu->async_copy_waitq);
        spin_unlock_irqrestore(&gpu->async_copy_lock, flags);
    }

    wake_up_all(&gpu->async_copy_waitq);
    return 0;
}

static bool submit_async_pipeline_item(uvm_kthread_work_item_t *item,
                                       uvm_service_block_context_t *src_service_context)
{
    uvm_gpu_t *gpu;
    uvm_gpu_async_copy_slot_t *slot = NULL;
    unsigned long flags;
    NV_STATUS status;

    if (!item || !item->gpu)
        return false;

    gpu = item->gpu;
    if (!gpu->async_copy_initialized)
        return false;

    spin_lock_irqsave(&gpu->async_copy_lock, flags);
    if (!gpu->pd_copy.occupied)
        slot = &gpu->pd_copy;
    else if (!gpu->cd.occupied)
        slot = &gpu->cd;

    if (!slot) {
        spin_unlock_irqrestore(&gpu->async_copy_lock, flags);
        return false;
    }

    status = clone_pipeline_contexts_to_slot(slot, item, src_service_context);
    if (status != NV_OK) {
        async_copy_slot_reset(slot);
        spin_unlock_irqrestore(&gpu->async_copy_lock, flags);
        item->status = status;
        return false;
    }
    spin_unlock_irqrestore(&gpu->async_copy_lock, flags);

    wake_up_all(&gpu->async_copy_waitq);
    return true;
}

static NV_STATUS flush_gpu_async_copy(uvm_gpu_t *gpu)
{
    if (!uvm_parallel_async_copy_map_enable || !gpu || !gpu->async_copy_initialized)
        return NV_OK;

    if (!wait_event_timeout(gpu->async_copy_waitq,
                            !gpu->pd_copy.occupied && !gpu->cd.occupied,
                            msecs_to_jiffies(30000))) {
        pr_err("UVM: async copy-map flush timed out after 30s\n");
        return NV_ERR_TIMEOUT;
    }

    return NV_OK;
}

static NV_STATUS flush_pipeline_items(uvm_kthread_pool_t *pool)
{
    return flush_gpu_async_copy(pool ? pool->batch_gpu : NULL);
}

NV_STATUS uvm_gpu_async_copy_init(uvm_gpu_t *gpu)
{
    if (!gpu || gpu->async_copy_initialized || !uvm_parallel_async_copy_map_enable)
        return NV_OK;

    gpu->pd_copy.block_context = uvm_va_block_context_alloc(NULL);
    if (!gpu->pd_copy.block_context)
        return NV_ERR_NO_MEMORY;

    gpu->cd.block_context = uvm_va_block_context_alloc(NULL);
    if (!gpu->cd.block_context) {
        uvm_va_block_context_free(gpu->pd_copy.block_context);
        gpu->pd_copy.block_context = NULL;
        return NV_ERR_NO_MEMORY;
    }

    async_copy_slot_reset(&gpu->pd_copy);
    async_copy_slot_reset(&gpu->cd);
    gpu->async_copy_stop = false;
    gpu->async_copy = kthread_create(async_copy_map_worker_fn,
                                     gpu,
                                     "uvm_gacopy/%u",
                                     uvm_id_value(gpu->id));
    if (IS_ERR(gpu->async_copy)) {
        gpu->async_copy = NULL;
        uvm_va_block_context_free(gpu->pd_copy.block_context);
        uvm_va_block_context_free(gpu->cd.block_context);
        gpu->pd_copy.block_context = NULL;
        gpu->cd.block_context = NULL;
        return NV_ERR_NO_MEMORY;
    }

    gpu->async_copy_initialized = true;
    wake_up_process(gpu->async_copy);
    return NV_OK;
}

void uvm_gpu_async_copy_deinit(uvm_gpu_t *gpu)
{
    if (!gpu || !gpu->async_copy_initialized)
        return;

    flush_gpu_async_copy(gpu);
    gpu->async_copy_stop = true;
    wake_up_all(&gpu->async_copy_waitq);

    if (gpu->async_copy) {
        kthread_stop(gpu->async_copy);
        gpu->async_copy = NULL;
    }

    if (gpu->pd_copy.block_context) {
        uvm_va_block_context_free(gpu->pd_copy.block_context);
        gpu->pd_copy.block_context = NULL;
    }

    if (gpu->cd.block_context) {
        uvm_va_block_context_free(gpu->cd.block_context);
        gpu->cd.block_context = NULL;
    }

    async_copy_slot_reset(&gpu->pd_copy);
    async_copy_slot_reset(&gpu->cd);
    gpu->async_copy_initialized = false;
    gpu->async_copy_stop = false;
}

bool uvm_parallel_fault_enabled(void)
{

    if (uvm_parallel_fault_processing == 2)
        return g_uvm_kthread_pool != NULL;
    return false;
}

uvm_kthread_pool_t *uvm_kthread_pool_get_global(void)
{
    return g_uvm_kthread_pool;
}

// Forward declarations for kthread
static int uvm_kthread_worker_fn(void *data);

// ============================================================================
// Kthread Pool Implementation
// ============================================================================

typedef struct {
    uvm_kthread_pool_t *pool;
    int worker_id;
} uvm_kthread_worker_arg_t;

NV_STATUS uvm_kthread_pool_create(uvm_kthread_pool_t **out_pool, NvU32 num_workers)
{
    uvm_kthread_pool_t *pool;
    NvU32 i;
    NvU32 num_cpus = num_online_cpus();
    int cpu;

    pool = uvm_kvmalloc_zero(sizeof(*pool));
    if (!pool)
        return NV_ERR_NO_MEMORY;

    num_workers = min(num_workers, (NvU32)UVM_KTHREAD_MAX_WORKERS);
    if (num_workers == 0)
        num_workers = min(num_cpus, (NvU32)UVM_KTHREAD_MAX_WORKERS);

    pool->queue = uvm_kvmalloc_zero(sizeof(uvm_kthread_work_item_t) * UVM_KTHREAD_QUEUE_SIZE);
    if (!pool->queue) {
        uvm_kvfree(pool);
        return NV_ERR_NO_MEMORY;
    }
    pool->queue_capacity = UVM_KTHREAD_QUEUE_SIZE;

    pool->num_workers = num_workers;
    init_waitqueue_head(&pool->worker_waitq);
    init_completion(&pool->all_done);
    atomic_set(&pool->batch_worker_counter, 0);
    atomic_set(&pool->items_remaining, 0);
    pool->batch_ready = false;
    pool->shutting_down = false;
    pool->contexts_initialized = false;
    atomic_set(&pool->persistent.batch_gen, 0);
    for (i = 0; i < num_workers; i++) {
        uvm_parallel_worker_context_t *wctx = &pool->worker_contexts[i];

        wctx->service_context = uvm_kvmalloc(sizeof(*wctx->service_context));
        if (!wctx->service_context)
            goto cleanup;
        memset(wctx->service_context, 0, sizeof(*wctx->service_context));

        wctx->block_context = uvm_va_block_context_alloc(NULL);
        if (!wctx->block_context) {
            uvm_kvfree(wctx->service_context);
            wctx->service_context = NULL;
            goto cleanup;
        }

        wctx->service_context->block_context = wctx->block_context;
        wctx->in_use = false;
    }
    pool->contexts_initialized = true;

    cpu = cpumask_first(cpu_online_mask);
    for (i = 0; i < num_workers; i++) {
        pool->threads[i] = kthread_create(uvm_kthread_worker_fn,
                                          pool, "uvm_kfault/%u", i);
        if (IS_ERR(pool->threads[i])) {
            pr_err("UVM: Failed to create kthread %u\n", i);
            pool->threads[i] = NULL;
            goto cleanup_threads;
        }
        kthread_bind(pool->threads[i], cpu);
        wake_up_process(pool->threads[i]);
        pr_info("UVM: kthread %u bound to CPU %d\n", i, cpu);
        cpu = cpumask_next(cpu, cpu_online_mask);
        if (cpu >= nr_cpu_ids)
            cpu = cpumask_first(cpu_online_mask);
    }

    pr_info("UVM: Kthread pool created with %u workers on %u CPUs\n",
            num_workers, num_cpus);
    *out_pool = pool;
    return NV_OK;

cleanup_threads:
    pool->shutting_down = true;
    smp_wmb();
    wake_up_all(&pool->worker_waitq);
    for (i = 0; i < num_workers; i++) {
        if (pool->threads[i]) {
            kthread_stop(pool->threads[i]);
            pool->threads[i] = NULL;
        }
    }

cleanup:
    for (i = 0; i < num_workers; i++) {
        uvm_parallel_worker_context_t *wctx = &pool->worker_contexts[i];
        if (wctx->block_context)
            uvm_va_block_context_free(wctx->block_context);
        if (wctx->service_context)
            uvm_kvfree(wctx->service_context);
        wctx->block_context = NULL;
        wctx->service_context = NULL;
    }
    uvm_kvfree(pool->queue);
    uvm_kvfree(pool);
    return NV_ERR_NO_MEMORY;
}

void uvm_kthread_pool_destroy(uvm_kthread_pool_t *pool)
{
    NvU32 i;

    if (!pool)
        return;

    pool->shutting_down = true;
    smp_wmb();
    wake_up_all(&pool->worker_waitq);
    for (i = 0; i < pool->num_workers; i++) {
        if (pool->threads[i]) {
            kthread_stop(pool->threads[i]);
            pool->threads[i] = NULL;
        }
    }

    for (i = 0; i < pool->num_workers; i++) {
        uvm_parallel_worker_context_t *wctx = &pool->worker_contexts[i];
        if (wctx->block_context)
            uvm_va_block_context_free(wctx->block_context);
        if (wctx->service_context)
            uvm_kvfree(wctx->service_context);
        wctx->block_context = NULL;
        wctx->service_context = NULL;
    }

    for (i = 0; i < pool->queue_capacity; i++)
        clear_pipeline_item_state(&pool->queue[i]);

    if (uvm_worker_timing_debug) {
        for (i = 0; i < pool->num_workers; i++) {
            u64 batches = pool->worker_timing[i].total_batches;
            u64 items   = pool->worker_timing[i].total_items;
            if (batches == 0)
                continue;
            pr_info("UVM: [WORKER_TIMING] worker=%u batches=%llu items=%llu "
                    "avg_wake_to_lock=%lluns avg_lock=%lluns "
                    "avg_service=%lluns avg_setup=%lluns\n",
                    i, batches, items,
                    pool->worker_timing[i].total_wake_to_first_lock_ns / batches,
                    items ? pool->worker_timing[i].total_lock_acquire_ns / items : 0,
                    items ? pool->worker_timing[i].total_block_service_ns / items : 0,
                    items ? pool->worker_timing[i].total_item_setup_ns / items : 0);
        }
    }

    pr_info("UVM: Kthread pool destroyed (%u workers)\n", pool->num_workers);
    uvm_kvfree(pool->queue);
    uvm_kvfree(pool);
}

// Worker main loop: sleep → wake → process batch items → signal done → repeat
//
// Holds va_space_down_read across all items from the same va_space, matching
// the serial path's locking granularity.
static int uvm_kthread_worker_fn(void *data)
{
    uvm_kthread_pool_t *pool = (uvm_kthread_pool_t *)data;
    int worker_id;
    NvU32 i;

    worker_id = -1;
    for (i = 0; i < pool->num_workers; i++) {
        if (pool->threads[i] == current) {
            worker_id = i;
            break;
        }
    }
    if (worker_id < 0) {
        pr_err("UVM: kthread worker could not find its ID\n");
        return -EINVAL;
    }

    while (!kthread_should_stop()) {
        uvm_va_space_t *held_va_space = NULL;
        int my_batch_id;
        NvU32 num_items, num_active, q_base, idx;
        uvm_parallel_worker_context_t *wctx = &pool->worker_contexts[worker_id];

        wait_event_interruptible_exclusive(pool->worker_waitq,
            pool->batch_ready || pool->shutting_down || kthread_should_stop());

        if (pool->shutting_down || kthread_should_stop())
            break;

new_batch_from_spin:
        {
        ktime_t __wt_wake = uvm_worker_timing_debug ? ktime_get() : 0;
        ktime_t __wt_first_lock = 0;
        u64     __wt_service_ns = 0;
        u64     __wt_setup_ns = 0;
        bool    __wt_first_lock_set = false;

        my_batch_id = atomic_add_return(1, &pool->batch_worker_counter) - 1;
        num_items = pool->batch_num_groups;
        num_active = pool->batch_num_active;
        q_base = pool->queue_base;

        if ((NvU32)my_batch_id >= num_active || num_items == 0)
            goto segment_done_timing;

        for (idx = (NvU32)my_batch_id; idx < num_items; idx += num_active) {
            uvm_kthread_work_item_t *item = &pool->queue[q_base + idx];
            uvm_block_fault_group_t *grp = &pool->batch_groups[idx];
            uvm_va_space_t *item_vs;
            ktime_t __wt_setup_start = 0;

            if (uvm_worker_timing_debug)
                __wt_setup_start = ktime_get();

            item->va_block          = grp->va_block;
            item->gpu               = pool->batch_gpu;
            item->snapshot          = pool->batch_snapshot;
            item->batch_context     = pool->batch_ctx;
            item->first_fault_index = grp->first_fault_index;
            item->num_faults        = grp->num_faults;
            item->hmm_migratable    = true;
            item->status            = NV_OK;
            uvm_tracker_init(&item->tracker);
            item->faults_serviced   = 0;
            item->has_fatal_faults  = false;

            if (uvm_worker_timing_debug)
                __wt_setup_ns += ktime_to_ns(ktime_sub(ktime_get(), __wt_setup_start));

            item_vs = item->va_block
                      ? uvm_va_block_get_va_space_maybe_dead(item->va_block)
                      : NULL;
            if (item_vs != held_va_space) {
                ktime_t __wt_lock_start = 0;
                if (held_va_space)
                    uvm_va_space_up_read(held_va_space);
                held_va_space = item_vs;
                if (held_va_space) {
                    if (uvm_worker_timing_debug)
                        __wt_lock_start = ktime_get();
                    uvm_va_space_down_read(held_va_space);
                    if (uvm_worker_timing_debug) {
                        u64 __lock_ns = ktime_to_ns(ktime_sub(ktime_get(), __wt_lock_start));
                        __wt_service_ns += 0; // placeholder
                        pool->worker_timing[worker_id].total_lock_acquire_ns += __lock_ns;
                        if (!__wt_first_lock_set) {
                            __wt_first_lock = ktime_get();
                            __wt_first_lock_set = true;
                        }
                    }
                }
            }

            if (!held_va_space) {
                item->status = NV_WARN_MORE_PROCESSING_REQUIRED;
                if (atomic_dec_and_test(&pool->items_remaining))
                    complete(&pool->all_done);
                continue;
            }

            {
                ktime_t __wt_svc_start = 0;
                if (uvm_worker_timing_debug)
                    __wt_svc_start = ktime_get();

                item->status = service_block_faults_kthread(item, wctx);

                if (uvm_worker_timing_debug)
                    __wt_service_ns += ktime_to_ns(ktime_sub(ktime_get(), __wt_svc_start));
            }

            if (atomic_dec_and_test(&pool->items_remaining)) {
                complete(&pool->all_done);
            }
        }

segment_done_timing:
        if (uvm_worker_timing_debug && (NvU32)my_batch_id < num_active) {
            pool->worker_timing[worker_id].total_wake_to_first_lock_ns +=
                __wt_first_lock_set
                ? ktime_to_ns(ktime_sub(__wt_first_lock, __wt_wake))
                : 0;
            pool->worker_timing[worker_id].total_block_service_ns += __wt_service_ns;
            pool->worker_timing[worker_id].total_item_setup_ns += __wt_setup_ns;
            pool->worker_timing[worker_id].total_batches++;
            pool->worker_timing[worker_id].total_items +=
                (num_items + num_active - 1 - (NvU32)my_batch_id) / num_active;
        }
        }
        if (held_va_space)
            uvm_va_space_up_read(held_va_space);

        // Persistent cross-batch spin: workers spin indefinitely between
        // batches, only exiting on new batch or shutdown.  Eliminates all
        // scheduler wake-up latency at the cost of CPU burn (acceptable
        // for research benchmarking).
        if (uvm_persistent_spin_workers &&
            !pool->shutting_down && !kthread_should_stop()) {
            int last_pgen = atomic_read(&pool->persistent.batch_gen);
            unsigned __pspin = 0;

            while (atomic_read(&pool->persistent.batch_gen) == last_pgen) {
                if (kthread_should_stop() || pool->shutting_down)
                    break;
                cpu_relax();
                if (++__pspin >= 10000) {
                    cond_resched();
                    __pspin = 0;
                }
            }

            if (atomic_read(&pool->persistent.batch_gen) != last_pgen &&
                !kthread_should_stop() && !pool->shutting_down) {
                smp_rmb();
                held_va_space = NULL;
                goto new_batch_from_spin;
            }
        }
    }

    return 0;
}

// Service faults for a single block - called from kthread worker.
// Caller MUST hold va_space_down_read for the block's va_space.
NV_STATUS service_block_faults_kthread(uvm_kthread_work_item_t *item,
                                       uvm_parallel_worker_context_t *worker_ctx)
{
    NV_STATUS status = NV_OK;
    NV_STATUS tracker_status;
    bool capture_breakdown = uvm_merge_profile_enable;
    uvm_va_block_t *va_block = item->va_block;
    uvm_gpu_t *gpu = item->gpu;
    uvm_va_block_retry_t va_block_retry;
    uvm_fault_service_batch_context_t *batch_context = item->batch_context;
    uvm_fault_buffer_entry_t **ordered_fault_cache = batch_context->ordered_fault_cache;
    uvm_service_block_context_t *block_context;
    NvU32 i;
    NvU32 first_fault_index = item->first_fault_index;
    NvU32 num_faults = item->num_faults;
    uvm_page_index_t first_page_index = PAGES_PER_UVM_VA_BLOCK;
    uvm_page_index_t last_page_index = 0;
    const uvm_va_policy_t *policy;
    uvm_va_space_t *va_space;

    if (capture_breakdown)
        item->dispatch_start_ns = NV_GETTIME();

    if (!va_block)
        return NV_WARN_MORE_PROCESSING_REQUIRED;

    va_space = uvm_va_block_get_va_space_maybe_dead(va_block);
    if (!va_space)
        return NV_WARN_MORE_PROCESSING_REQUIRED;

    if (!uvm_va_block_is_hmm(va_block) && !va_block->va_range)
        return NV_WARN_MORE_PROCESSING_REQUIRED;

    if (!worker_ctx || !worker_ctx->service_context || !worker_ctx->block_context)
        return NV_ERR_INVALID_STATE;

    block_context = worker_ctx->service_context;
    worker_ctx->block_context->mm = item->snapshot.mm;
    item->pipeline_async = false;
    item->pipeline_service_context = NULL;
    item->pipeline_block_context = NULL;

    if (capture_breakdown) {
        worker_ctx->block_context->breakdown_ns.unmap_ns = 0;
        worker_ctx->block_context->breakdown_ns.alloc_ns = 0;
        worker_ctx->block_context->breakdown_ns.copy_ns  = 0;
        worker_ctx->block_context->breakdown_ns.map_ns   = 0;
        worker_ctx->block_context->breakdown_ns.subregion_count = 0;
        worker_ctx->block_context->breakdown_ns.pages_migrated  = 0;
    }

    block_context->operation = UVM_SERVICE_OPERATION_REPLAYABLE_FAULTS;
    block_context->num_retries = 0;

    if (uvm_va_block_is_hmm(va_block))
        uvm_hmm_migrate_begin_wait(va_block);

    uvm_mutex_lock(&va_block->lock);

    va_space = uvm_va_block_get_va_space_maybe_dead(va_block);
    if (!va_space || (!uvm_va_block_is_hmm(va_block) && !va_block->va_range)) {
        status = NV_WARN_MORE_PROCESSING_REQUIRED;
        goto unlock;
    }

    uvm_processor_mask_zero(&block_context->resident_processors);
    block_context->thrashing_pin_count = 0;
    block_context->read_duplicate_count = 0;

    if (uvm_va_block_is_hmm(va_block)) {
        status = NV_WARN_MORE_PROCESSING_REQUIRED;
        goto unlock;
    }

    policy = uvm_va_range_get_policy(va_block->va_range);

    for (i = 0; i < num_faults; i++) {
        NvU32 fault_index = first_fault_index + i;
        uvm_fault_buffer_entry_t *current_entry = ordered_fault_cache[fault_index];
        const uvm_fault_buffer_entry_t *previous_entry =
            (fault_index > 0) ? ordered_fault_cache[fault_index - 1] : NULL;
        uvm_page_index_t page_index;
        uvm_perf_thrashing_hint_t thrashing_hint;
        uvm_processor_id_t new_residency;
        bool read_duplicate = false;
        uvm_fault_access_type_t service_access_type;
        NvU32 service_access_type_mask;

        if (current_entry->fault_address < va_block->start ||
            current_entry->fault_address > va_block->end)
            break;

        if (current_entry->is_fatal) {
            item->has_fatal_faults = true;
            item->faults_serviced++;
            continue;
        }

        if (previous_entry &&
            current_entry->va_space == previous_entry->va_space &&
            current_entry->fault_address == previous_entry->fault_address &&
            !previous_entry->is_fatal) {
            item->faults_serviced++;
            continue;
        }

        page_index = uvm_va_block_cpu_page_index(va_block, current_entry->fault_address);

        service_access_type = current_entry->fault_access_type;
        service_access_type_mask = current_entry->access_type_mask;

        if (uvm_va_block_page_is_gpu_authorized(va_block,
                                                page_index,
                                                gpu->id,
                                                uvm_fault_access_type_to_prot(service_access_type))) {
            item->faults_serviced++;
            continue;
        }

        thrashing_hint = uvm_perf_thrashing_get_hint(va_block,
                                                     block_context->block_context,
                                                     current_entry->fault_address,
                                                     gpu->id);
        if (thrashing_hint.type == UVM_PERF_THRASHING_HINT_TYPE_THROTTLE) {
            item->faults_serviced++;
            continue;
        }

        if (thrashing_hint.type == UVM_PERF_THRASHING_HINT_TYPE_PIN) {
            if (block_context->thrashing_pin_count++ == 0)
                uvm_page_mask_zero(&block_context->thrashing_pin_mask);
            uvm_page_mask_set(&block_context->thrashing_pin_mask, page_index);
        }

        if (page_index < first_page_index)
            first_page_index = page_index;
        if (page_index > last_page_index)
            last_page_index = page_index;

        new_residency = uvm_va_block_select_residency(va_block,
                                                      block_context->block_context,
                                                      page_index,
                                                      gpu->id,
                                                      service_access_type_mask,
                                                      policy,
                                                      &thrashing_hint,
                                                      UVM_SERVICE_OPERATION_REPLAYABLE_FAULTS,
                                                      item->hmm_migratable,
                                                      &read_duplicate);

        if (!uvm_processor_mask_test_and_set(&block_context->resident_processors, new_residency))
            uvm_page_mask_zero(&block_context->per_processor_masks[uvm_id_value(new_residency)].new_residency);

        uvm_page_mask_set(&block_context->per_processor_masks[uvm_id_value(new_residency)].new_residency, page_index);
        block_context->access_type[page_index] = service_access_type;

        if (read_duplicate) {
            if (block_context->read_duplicate_count++ == 0)
                uvm_page_mask_zero(&block_context->read_duplicate_mask);
            uvm_page_mask_set(&block_context->read_duplicate_mask, page_index);
        }

        item->faults_serviced++;
    }

    if (first_page_index <= last_page_index) {
        block_context->region = uvm_va_block_region(first_page_index, last_page_index + 1);
        if (async_copy_map_enabled_for_item(item, block_context)) {
            status = UVM_VA_BLOCK_RETRY_LOCKED(va_block, &va_block_retry,
                uvm_va_block_service_locked_populate(gpu->id, va_block, &va_block_retry, block_context));

            if (status == NV_OK) {
                if (!submit_async_pipeline_item(item, block_context)) {
                    status = uvm_va_block_service_copy_finish(gpu->id, va_block, block_context);
                    clear_pipeline_item_state(item);
                }
                else {
                    goto unlock;
                }
            }
            else if (status == NV_ERR_NOT_SUPPORTED) {
                status = UVM_VA_BLOCK_RETRY_LOCKED(va_block, &va_block_retry,
                    uvm_va_block_service_locked(gpu->id, va_block, &va_block_retry, block_context));
            }
        }
        else {
            status = UVM_VA_BLOCK_RETRY_LOCKED(va_block, &va_block_retry,
                uvm_va_block_service_locked(gpu->id, va_block, &va_block_retry, block_context));
        }
    }

    tracker_status = uvm_tracker_add_tracker_safe(&item->tracker, &va_block->tracker);
    if (status == NV_OK)
        status = tracker_status;

unlock:
    uvm_mutex_unlock(&va_block->lock);

    if (status != NV_WARN_MORE_PROCESSING_REQUIRED && uvm_va_block_is_hmm(va_block))
        uvm_hmm_migrate_finish(va_block);

    if (capture_breakdown) {
        item->breakdown_ns.unmap_ns        = worker_ctx->block_context->breakdown_ns.unmap_ns;
        item->breakdown_ns.alloc_ns        = worker_ctx->block_context->breakdown_ns.alloc_ns;
        item->breakdown_ns.copy_ns         = worker_ctx->block_context->breakdown_ns.copy_ns;
        item->breakdown_ns.map_ns          = worker_ctx->block_context->breakdown_ns.map_ns;
        item->breakdown_ns.subregion_count = worker_ctx->block_context->breakdown_ns.subregion_count;
        item->breakdown_ns.pages_migrated  = worker_ctx->block_context->breakdown_ns.pages_migrated;
        item->dispatch_end_ns = NV_GETTIME();
    }

    return status;
}

// ============================================================================
// Kthread dispatch and wait
// ============================================================================

// Start workers for seg_num_items items.  Returns the number of active
// workers that were woken.  Does NOT wait for completion.
static NvU32 dispatch_start_inner(uvm_kthread_pool_t *pool,
                                  NvU32 seg_num_items)
{
    unsigned max_w_val = uvm_parallel_fault_max_workers;
    NvU32 num_active = pool->num_workers;

    if (seg_num_items == 0)
        return 0;

    if (max_w_val > 0)
        num_active = min_t(NvU32, max_w_val, num_active);
    num_active = min_t(NvU32, num_active, seg_num_items);

    pool->batch_num_active = num_active;
    pool->batch_num_groups = seg_num_items;

    atomic_set(&pool->batch_worker_counter, 0);
    atomic_set(&pool->items_remaining, seg_num_items);
    reinit_completion(&pool->all_done);

    pool->batch_ready = true;
    smp_wmb();

    if (uvm_persistent_spin_workers)
        atomic_inc(&pool->persistent.batch_gen);

    wake_up_nr(&pool->worker_waitq, num_active);
    return num_active;
}

// Wait for workers to finish, then collect trackers from completed items.
static NV_STATUS dispatch_wait_inner(uvm_kthread_pool_t *pool,
                                     NvU32 seg_num_items,
                                     NvU32 num_active,
                                     uvm_tracker_t *out_tracker)
{
    NvU32 i;
    NV_STATUS combined_status = NV_OK;
    ktime_t tw0 = 0, tw1 = 0, tw2 = 0;

    if (seg_num_items == 0)
        return NV_OK;

    if (uvm_dispatch_timing_debug)
        tw0 = ktime_get();

    if (!wait_for_completion_timeout(&pool->all_done, msecs_to_jiffies(30000))) {
        pr_err("UVM: kthread dispatch timed out after 30s (%u items, %u active)\n",
               seg_num_items, num_active);
        pool->batch_ready = false;
        smp_wmb();
        return NV_ERR_TIMEOUT;
    }

    if (uvm_dispatch_timing_debug)
        tw1 = ktime_get();

    pool->batch_ready = false;

    {
        NV_STATUS flush_status = flush_pipeline_items(pool);
        if (flush_status != NV_OK && combined_status == NV_OK)
            combined_status = flush_status;
    }

    for (i = 0; i < seg_num_items; i++) {
        uvm_kthread_work_item_t *item = &pool->queue[pool->queue_base + i];
        NV_STATUS ts;

        if (item->status != NV_OK && combined_status == NV_OK)
            combined_status = item->status;

        ts = uvm_tracker_add_tracker_safe(out_tracker, &item->tracker);
        if (ts != NV_OK && combined_status == NV_OK)
            combined_status = ts;

        uvm_tracker_deinit(&item->tracker);
    }

    if (uvm_dispatch_timing_debug) {
        tw2 = ktime_get();
        pr_info("[DISPATCH_WAIT] completion_ns=%lld tracker_merge_ns=%lld items=%u active=%u\n",
                (long long)ktime_to_ns(ktime_sub(tw1, tw0)),
                (long long)ktime_to_ns(ktime_sub(tw2, tw1)),
                seg_num_items, num_active);
    }

    return combined_status;
}

// Combined dispatch-and-wait for callers that don't need overlap.
static NV_STATUS dispatch_and_wait_inner(uvm_kthread_pool_t *pool,
                                         NvU32 seg_num_items,
                                         uvm_tracker_t *out_tracker)
{
    NvU32 num_active;
    NV_STATUS ret;
    ktime_t t0 = 0, t1 = 0, t2 = 0;

    if (seg_num_items == 0)
        return NV_OK;

    if (uvm_dispatch_timing_debug)
        t0 = ktime_get();

    num_active = dispatch_start_inner(pool, seg_num_items);

    if (uvm_dispatch_timing_debug)
        t1 = ktime_get();

    ret = dispatch_wait_inner(pool, seg_num_items, num_active, out_tracker);

    if (uvm_dispatch_timing_debug) {
        t2 = ktime_get();
        pr_info("[DISPATCH_TIMING] setup_wake_ns=%lld wait_tracker_ns=%lld items=%u active=%u\n",
                (long long)ktime_to_ns(ktime_sub(t1, t0)),
                (long long)ktime_to_ns(ktime_sub(t2, t1)),
                seg_num_items, num_active);
    }

    return ret;
}

NV_STATUS uvm_kthread_dispatch_and_wait(uvm_kthread_pool_t *pool,
                                        NvU32 num_items,
                                        NvU64 group_ns,
                                        uvm_tracker_t *out_tracker)
{
    NV_STATUS status = NV_OK;
    NvU32 batch_max_group_faults = 0;
    uvm_block_fault_group_t *all_groups = pool->batch_groups;

    if (num_items == 0)
        return NV_OK;

    {
        NvU32 i;
        for (i = 0; i < num_items; i++) {
            if (all_groups[i].num_faults > batch_max_group_faults)
                batch_max_group_faults = all_groups[i].num_faults;
        }
    }

    pool->batch_max_group_faults = batch_max_group_faults;

    if (uvm_merge_profile_enable) {
        atomic_set(&pool->merge_stats.thin_count, 0);
        atomic_set(&pool->merge_stats.fat_count, 0);
        atomic_set(&pool->merge_stats.used, 0);
        atomic64_set(&pool->merge_stats.spin_ns, 0);
        atomic_set(&pool->merge_stats.contended_workers, 0);
        atomic_set(&pool->merge_stats.total_segments, 0);
        atomic_set(&pool->merge_stats.copy_ext_used, 0);
        atomic_set(&pool->merge_stats.copy_ext_fallback, 0);
        atomic_set(&pool->merge_stats.map_ext_used, 0);
        atomic_set(&pool->merge_stats.map_ext_fallback_2m, 0);
        atomic_set(&pool->merge_stats.map_ext_fallback_space, 0);
        atomic64_set(&pool->merge_stats.group_ns, (long long)group_ns);
        atomic64_set(&pool->merge_stats.classify_ns, 0);
        atomic64_set(&pool->merge_stats.fat_dispatch_ns, 0);
        atomic64_set(&pool->merge_stats.thin_dispatch_ns, 0);
        atomic64_set(&pool->merge_stats.seg_push_begin_ns, 0);
        atomic64_set(&pool->merge_stats.seg_push_end_ns, 0);
        atomic64_set(&pool->merge_stats.seg_dispatch_ns, 0);
        atomic_set(&pool->merge_stats.reclassed_count, 0);
        atomic_set(&pool->merge_stats.tiny_seg_count, 0);
        atomic_set(&pool->merge_stats.effective_seg_threshold, UVM_KTHREAD_SEGMENT_SIZE);
        atomic_set(&pool->merge_stats.max_group_faults, batch_max_group_faults);
        atomic_set(&pool->merge_stats.merged_block_count, 0);
        atomic_set(&pool->merge_stats.solo_block_count, 0);
        atomic_set(&pool->merge_stats.heavy_block_count, 0);
        atomic64_set(&pool->merge_stats.heavy_block_fault_sum, 0);
        atomic64_set(&pool->merge_stats.batch_start_ns, ktime_to_ns(ktime_get()));
        atomic64_set(&pool->merge_stats.first_submit_ns, 0);
        atomic64_set(&pool->merge_stats.last_submit_ns, 0);
        atomic64_set(&pool->merge_stats.batch_complete_ns, 0);
    }

    pool->batch_groups = &all_groups[0];
    pool->queue_base = 0;

    status = dispatch_and_wait_inner(pool, num_items, out_tracker);

    /* Per-block breakdown histogram: when profiling is on, collect per-block
     * breakdown bucketed by each block's num_faults.
     * Format: [BL_BLOCK_BD_HIST] faults:cnt:wall_ns:unmap_ns:alloc_ns:copy_ns:map_ns
     */
    if (uvm_merge_profile_enable &&
        uvm_merge_seg_timing_enable && num_items > 0 && status == NV_OK) {
        s64 *bl_wall  = kzalloc(128 * sizeof(s64), GFP_KERNEL);
        s64 *bl_unmap = kzalloc(128 * sizeof(s64), GFP_KERNEL);
        s64 *bl_alloc = kzalloc(128 * sizeof(s64), GFP_KERNEL);
        s64 *bl_copy  = kzalloc(128 * sizeof(s64), GFP_KERNEL);
        s64 *bl_map   = kzalloc(128 * sizeof(s64), GFP_KERNEL);
        int *bl_cnt   = kzalloc(128 * sizeof(int), GFP_KERNEL);
        if (bl_wall && bl_unmap && bl_alloc && bl_copy && bl_map && bl_cnt) {
            NvU32 i;
            int sc;
            for (i = 0; i < num_items; i++) {
                uvm_kthread_work_item_t *it = &pool->queue[i];
                int bucket = it->num_faults;
                s64 item_ns;
                if (bucket <= 0 || bucket >= 128) continue;
                item_ns = (it->dispatch_end_ns > it->dispatch_start_ns)
                        ? (s64)(it->dispatch_end_ns - it->dispatch_start_ns) : 0;
                bl_wall[bucket]  += item_ns;
                bl_unmap[bucket] += it->breakdown_ns.unmap_ns;
                bl_alloc[bucket] += it->breakdown_ns.alloc_ns;
                bl_copy[bucket]  += it->breakdown_ns.copy_ns;
                bl_map[bucket]   += it->breakdown_ns.map_ns;
                bl_cnt[bucket]++;
            }
            for (sc = 1; sc < 128; sc++) {
                if (bl_cnt[sc] == 0) continue;
                pr_info("[BL_BLOCK_BD_HIST] %d:%d:%lld:%lld:%lld:%lld:%lld\n",
                        sc, bl_cnt[sc], bl_wall[sc],
                        bl_unmap[sc], bl_alloc[sc],
                        bl_copy[sc], bl_map[sc]);
            }
        }
        kfree(bl_wall);
        kfree(bl_unmap);
        kfree(bl_alloc);
        kfree(bl_copy);
        kfree(bl_map);
        kfree(bl_cnt);
    }

    if (uvm_merge_profile_enable) {
        s64 bs_ns, fs_ns, ls_ns, bc_ns;
        atomic64_set(&pool->merge_stats.batch_complete_ns, ktime_to_ns(ktime_get()));
        bs_ns = atomic64_read(&pool->merge_stats.batch_start_ns);
        fs_ns = atomic64_read(&pool->merge_stats.first_submit_ns);
        ls_ns = atomic64_read(&pool->merge_stats.last_submit_ns);
        bc_ns = atomic64_read(&pool->merge_stats.batch_complete_ns);
        pr_info("[MERGE_STATS] gpu=%s batch=%u thin=%d fat=%d used=%d spin_ns=%lld contended=%d"
                " segments=%d copy_ok=%d copy_fb=%d map_ok=%d map_fb2m=%d map_fbsp=%d"
                " group_ns=%lld classify_ns=%lld"
                " fat_ns=%lld thin_ns=%lld"
                " seg_pb_ns=%lld seg_disp_ns=%lld seg_pe_ns=%lld"
                " reclass=%d tiny_segs=%d seg_threshold=%d max_group_faults=%d"
                " merged_blocks=%d solo_blocks=%d heavy_blocks=%d heavy_faults=%lld"
                " first_submit_delay=%lld submit_span=%lld gpu_tail=%lld\n",
                uvm_gpu_name(pool->batch_gpu),
                pool->batch_ctx ? pool->batch_ctx->batch_id : 0,
                atomic_read(&pool->merge_stats.thin_count),
                atomic_read(&pool->merge_stats.fat_count),
                atomic_read(&pool->merge_stats.used),
                (long long)atomic64_read(&pool->merge_stats.spin_ns),
                atomic_read(&pool->merge_stats.contended_workers),
                atomic_read(&pool->merge_stats.total_segments),
                atomic_read(&pool->merge_stats.copy_ext_used),
                atomic_read(&pool->merge_stats.copy_ext_fallback),
                atomic_read(&pool->merge_stats.map_ext_used),
                atomic_read(&pool->merge_stats.map_ext_fallback_2m),
                atomic_read(&pool->merge_stats.map_ext_fallback_space),
                (long long)atomic64_read(&pool->merge_stats.group_ns),
                (long long)atomic64_read(&pool->merge_stats.classify_ns),
                (long long)atomic64_read(&pool->merge_stats.fat_dispatch_ns),
                (long long)atomic64_read(&pool->merge_stats.thin_dispatch_ns),
                (long long)atomic64_read(&pool->merge_stats.seg_push_begin_ns),
                (long long)atomic64_read(&pool->merge_stats.seg_dispatch_ns),
                (long long)atomic64_read(&pool->merge_stats.seg_push_end_ns),
                atomic_read(&pool->merge_stats.reclassed_count),
                atomic_read(&pool->merge_stats.tiny_seg_count),
                atomic_read(&pool->merge_stats.effective_seg_threshold),
                atomic_read(&pool->merge_stats.max_group_faults),
                atomic_read(&pool->merge_stats.merged_block_count),
                atomic_read(&pool->merge_stats.solo_block_count),
                atomic_read(&pool->merge_stats.heavy_block_count),
                (long long)atomic64_read(&pool->merge_stats.heavy_block_fault_sum),
                (long long)(fs_ns > bs_ns ? fs_ns - bs_ns : 0),
                (long long)(ls_ns > fs_ns ? ls_ns - fs_ns : 0),
                (long long)(bc_ns > ls_ns ? bc_ns - ls_ns : 0));
    }

    pool->batch_groups = all_groups;
    return status;
}

void uvm_parallel_fault_context_init(uvm_parallel_fault_context_t *ctx)
{
    memset(ctx->groups, 0, sizeof(ctx->groups));
}

void uvm_va_space_snapshot_init(uvm_va_space_snapshot_t *snapshot,
                                uvm_va_space_t *va_space,
                                uvm_gpu_va_space_t *gpu_va_space,
                                struct mm_struct *mm)
{
    snapshot->va_space = va_space;
    snapshot->gpu_va_space = gpu_va_space;
    snapshot->mm = mm;
}

// Initialize the parallel fault subsystem
NV_STATUS uvm_parallel_fault_init(void)
{
    NV_STATUS status;



    status = uvm_kthread_pool_create(&g_uvm_kthread_pool, uvm_kthread_workers);
    if (status != NV_OK) {
        pr_warn("UVM: Failed to create kthread pool, kthread mode unavailable\n");
        // Non-fatal: workqueue path is still available
    }

    pr_info("UVM: Parallel fault subsystem initialized (kt=%s)\n",
            g_uvm_kthread_pool ? "ok" : "fail");
    return NV_OK;
}

void uvm_parallel_fault_exit(void)
{
    if (g_uvm_kthread_pool) {
        uvm_kthread_pool_destroy(g_uvm_kthread_pool);
        g_uvm_kthread_pool = NULL;
    }

    pr_info("UVM: Parallel fault subsystem destroyed\n");
}

// ============================================================================
// CE Parallelism Profile Test v2 (ioctl-based)
//
// Part 1: Cross-CE parallelism at 128KB / 1MB / 4MB
//   A: 1 push baseline on CPU_TO_GPU (CE 2)
//   B: 2 pushes both on CPU_TO_GPU (same CE, serial)
//   C: 2 pushes on CPU_TO_GPU + GPU_INTERNAL (CE 2 + CE 4, parallel)
//   D: 2 pushes on CPU_TO_GPU + GPU_TO_CPU  (CE 2 + CE 3, parallel)
//
// Part 2: CPU-side pipeline simulation (4 blocks)
//   E: Serial   — push+wait per block on CE 2
//   F: Pipeline  — push all 4 then wait, all on CE 2
//   G: Pipeline cross-CE — push round-robin CE 2 / CE 4
// ============================================================================

#include "uvm_test.h"
#include "uvm_test_ioctl.h"
#include "uvm_global.h"

#define CE_ITERS             10
#define CE_PIPELINE_BLOCKS    4
#define CE_MAX_ALLOC_SIZE    (256 * 1024)

typedef struct {
    s64 sum, mn, mx;
    int count;
} ce_stats_t;

static void ce_stats_init(ce_stats_t *s)
{
    s->sum = 0; s->mn = S64_MAX; s->mx = 0; s->count = 0;
}

static void ce_stats_add(ce_stats_t *s, s64 val)
{
    if (val < 0) return;
    s->sum += val; s->count++;
    if (val < s->mn) s->mn = val;
    if (val > s->mx) s->mx = val;
}

static s64 ce_stats_avg(ce_stats_t *s)
{
    return s->count > 0 ? s->sum / s->count : -1;
}

static void ce_stats_print(const char *label, ce_stats_t *s)
{
    if (s->count == 0) {
        pr_info("UVM:CE_PROFILE:   %-44s SKIPPED\n", label);
        return;
    }
    pr_info("UVM:CE_PROFILE:   %-44s avg=%7lld  min=%7lld  max=%7lld ns\n",
            label, ce_stats_avg(s), s->mn, s->mx);
}

static void ce_stats_print_perpg(const char *label, ce_stats_t *s, NvU32 num_pages)
{
    if (s->count == 0) {
        pr_info("UVM:CE_PROFILE:   %-38s SKIPPED\n", label);
        return;
    }
    pr_info("UVM:CE_PROFILE:   %-38s avg=%8lld  min=%8lld  max=%8lld  perpg=%5lld ns\n",
            label, ce_stats_avg(s), s->mn, s->mx,
            ce_stats_avg(s) / num_pages);
}

static NV_STATUS do_single_push_test(uvm_gpu_t *gpu, uvm_channel_manager_t *mgr,
                                     uvm_channel_type_t type, NvU64 va, NvU32 size,
                                     ce_stats_t *stats)
{
    int iter;
    ce_stats_init(stats);
    for (iter = 0; iter < CE_ITERS; iter++) {
        uvm_push_t push;
        ktime_t t0;
        NV_STATUS s;

        t0 = ktime_get();
        s = uvm_push_begin(mgr, type, &push, "ce_prof single");
        if (s != NV_OK) continue;
        uvm_push_set_flag(&push, UVM_PUSH_FLAG_NEXT_MEMBAR_NONE);
        gpu->parent->ce_hal->memset_v_4(&push, va, 0xAA, size);
        uvm_push_end_and_wait(&push);
        ce_stats_add(stats, ktime_to_ns(ktime_sub(ktime_get(), t0)));
    }
    return NV_OK;
}

static NV_STATUS do_dual_push_test(uvm_gpu_t *gpu, uvm_channel_manager_t *mgr,
                                   uvm_channel_type_t type_a, uvm_channel_type_t type_b,
                                   NvU64 va, NvU32 size, ce_stats_t *stats)
{
    int iter;
    NvU32 half = size / 2;

    ce_stats_init(stats);
    for (iter = 0; iter < CE_ITERS; iter++) {
        uvm_push_t pa, pb;
        uvm_tracker_t tracker = UVM_TRACKER_INIT();
        ktime_t t0;
        NV_STATUS s;

        t0 = ktime_get();

        s = uvm_push_begin(mgr, type_a, &pa, "ce_prof dual A");
        if (s != NV_OK) continue;
        uvm_push_set_flag(&pa, UVM_PUSH_FLAG_NEXT_MEMBAR_NONE);
        gpu->parent->ce_hal->memset_v_4(&pa, va, 0xBB, half);
        uvm_push_end(&pa);
        uvm_tracker_add_push(&tracker, &pa);

        s = uvm_push_begin(mgr, type_b, &pb, "ce_prof dual B");
        if (s != NV_OK) { uvm_tracker_wait(&tracker); uvm_tracker_deinit(&tracker); continue; }
        uvm_push_set_flag(&pb, UVM_PUSH_FLAG_NEXT_MEMBAR_NONE);
        gpu->parent->ce_hal->memset_v_4(&pb, va + half, 0xCC, half);
        uvm_push_end(&pb);
        uvm_tracker_add_push(&tracker, &pb);

        uvm_tracker_wait(&tracker);
        ce_stats_add(stats, ktime_to_ns(ktime_sub(ktime_get(), t0)));
        uvm_tracker_deinit(&tracker);
    }
    return NV_OK;
}

static NV_STATUS do_pipeline_serial_test(uvm_gpu_t *gpu, uvm_channel_manager_t *mgr,
                                         uvm_channel_type_t type,
                                         NvU64 va, NvU32 block_size, int nblocks,
                                         ce_stats_t *stats)
{
    int iter;
    ce_stats_init(stats);
    for (iter = 0; iter < CE_ITERS; iter++) {
        int b;
        ktime_t t0 = ktime_get();

        for (b = 0; b < nblocks; b++) {
            uvm_push_t push;
            NV_STATUS s = uvm_push_begin(mgr, type, &push, "ce_prof serial %d", b);
            if (s != NV_OK) break;
            uvm_push_set_flag(&push, UVM_PUSH_FLAG_NEXT_MEMBAR_NONE);
            gpu->parent->ce_hal->memset_v_4(&push, va + (NvU64)b * block_size,
                                            0xDD, block_size);
            uvm_push_end_and_wait(&push);
        }
        ce_stats_add(stats, ktime_to_ns(ktime_sub(ktime_get(), t0)));
    }
    return NV_OK;
}

static NV_STATUS do_pipeline_async_test(uvm_gpu_t *gpu, uvm_channel_manager_t *mgr,
                                        uvm_channel_type_t type,
                                        NvU64 va, NvU32 block_size, int nblocks,
                                        ce_stats_t *stats)
{
    int iter;
    ce_stats_init(stats);
    for (iter = 0; iter < CE_ITERS; iter++) {
        uvm_tracker_t tracker = UVM_TRACKER_INIT();
        int b;
        ktime_t t0 = ktime_get();

        for (b = 0; b < nblocks; b++) {
            uvm_push_t push;
            NV_STATUS s = uvm_push_begin(mgr, type, &push, "ce_prof pipe %d", b);
            if (s != NV_OK) break;
            uvm_push_set_flag(&push, UVM_PUSH_FLAG_NEXT_MEMBAR_NONE);
            gpu->parent->ce_hal->memset_v_4(&push, va + (NvU64)b * block_size,
                                            0xEE, block_size);
            uvm_push_end(&push);
            uvm_tracker_add_push(&tracker, &push);
        }
        uvm_tracker_wait(&tracker);
        ce_stats_add(stats, ktime_to_ns(ktime_sub(ktime_get(), t0)));
        uvm_tracker_deinit(&tracker);
    }
    return NV_OK;
}

static NV_STATUS do_pipeline_cross_ce_test(uvm_gpu_t *gpu, uvm_channel_manager_t *mgr,
                                           uvm_channel_type_t type_a,
                                           uvm_channel_type_t type_b,
                                           NvU64 va, NvU32 block_size, int nblocks,
                                           ce_stats_t *stats)
{
    int iter;
    uvm_channel_type_t types[2];
    types[0] = type_a;
    types[1] = type_b;

    ce_stats_init(stats);
    for (iter = 0; iter < CE_ITERS; iter++) {
        uvm_tracker_t tracker = UVM_TRACKER_INIT();
        int b;
        ktime_t t0 = ktime_get();

        for (b = 0; b < nblocks; b++) {
            uvm_push_t push;
            NV_STATUS s = uvm_push_begin(mgr, types[b % 2], &push,
                                         "ce_prof xce %d", b);
            if (s != NV_OK) break;
            uvm_push_set_flag(&push, UVM_PUSH_FLAG_NEXT_MEMBAR_NONE);
            gpu->parent->ce_hal->memset_v_4(&push, va + (NvU64)b * block_size,
                                            0xFF, block_size);
            uvm_push_end(&push);
            uvm_tracker_add_push(&tracker, &push);
        }
        uvm_tracker_wait(&tracker);
        ce_stats_add(stats, ktime_to_ns(ktime_sub(ktime_get(), t0)));
        uvm_tracker_deinit(&tracker);
    }
    return NV_OK;
}

// ============================================================================
// Part 3: CPU Unmap (TLB Shootdown) test helpers
//
// Uses UVM's native unmap_mapping_range(va_space->mapping, ...) — the same
// function called by block_unmap_cpu() in uvm_va_block.c.
//
// NOTE: These unmaps remove CPU PTEs without updating UVM's internal
// pte_bits tracking. This is safe in a benchmark context because:
//   - cudaFree cleanup doesn't require valid CPU PTEs
//   - Any subsequent CPU access triggers a fault that re-maps correctly
// ============================================================================

#define UNMAP_ITERS         20

typedef struct {
    struct address_space *mapping;
    NvU64                 start_offset;
    NvU32                 num_pages;
    s64                   elapsed_ns;
    struct completion    *barrier;
    atomic_t             *ready_count;
    struct completion     done;
} unmap_thread_data_t;

static int unmap_thread_fn(void *arg)
{
    unmap_thread_data_t *data = arg;
    NvU32 p;
    ktime_t t0;

    atomic_inc(data->ready_count);
    wait_for_completion(data->barrier);

    t0 = ktime_get();
    for (p = 0; p < data->num_pages; p++) {
        unmap_mapping_range(data->mapping,
                            data->start_offset + (NvU64)p * PAGE_SIZE,
                            PAGE_SIZE, 1);
    }
    data->elapsed_ns = ktime_to_ns(ktime_sub(ktime_get(), t0));
    complete(&data->done);
    return 0;
}

static NV_STATUS do_cpu_unmap_individual_test(struct address_space *mapping,
                                              NvU64 base, NvU32 num_pages,
                                              int iters, ce_stats_t *stats)
{
    int iter;
    ce_stats_init(stats);
    for (iter = 0; iter < iters; iter++) {
        NvU64 iter_base = base + (NvU64)(iter * num_pages) * PAGE_SIZE;
        NvU32 p;
        ktime_t t0 = ktime_get();
        for (p = 0; p < num_pages; p++) {
            unmap_mapping_range(mapping,
                                iter_base + (NvU64)p * PAGE_SIZE,
                                PAGE_SIZE, 1);
        }
        ce_stats_add(stats, ktime_to_ns(ktime_sub(ktime_get(), t0)));
    }
    return NV_OK;
}

static NV_STATUS do_cpu_unmap_batch_test(struct address_space *mapping,
                                         NvU64 base, NvU32 num_pages,
                                         int iters, ce_stats_t *stats)
{
    int iter;
    ce_stats_init(stats);
    for (iter = 0; iter < iters; iter++) {
        NvU64 iter_base = base + (NvU64)(iter * num_pages) * PAGE_SIZE;
        ktime_t t0 = ktime_get();
        unmap_mapping_range(mapping, iter_base,
                            (loff_t)num_pages * PAGE_SIZE, 1);
        ce_stats_add(stats, ktime_to_ns(ktime_sub(ktime_get(), t0)));
    }
    return NV_OK;
}

static NV_STATUS do_cpu_unmap_parallel_test(struct address_space *mapping,
                                            NvU64 base, NvU32 num_pages,
                                            int nthreads, int iters,
                                            ce_stats_t *stats)
{
    int iter;

    ce_stats_init(stats);
    for (iter = 0; iter < iters; iter++) {
        NvU64 iter_base = base + (NvU64)(iter * num_pages) * PAGE_SIZE;
        NvU32 per_thread = num_pages / nthreads;
        struct completion barrier;
        atomic_t ready_count = ATOMIC_INIT(0);
        unmap_thread_data_t td[4];
        struct task_struct *tasks[4];
        int t;
        ktime_t t0;
        bool failed = false;

        init_completion(&barrier);

        for (t = 0; t < nthreads; t++) {
            td[t].mapping      = mapping;
            td[t].start_offset = iter_base + (NvU64)(t * per_thread) * PAGE_SIZE;
            td[t].num_pages    = per_thread;
            td[t].elapsed_ns   = 0;
            td[t].barrier      = &barrier;
            td[t].ready_count  = &ready_count;
            init_completion(&td[t].done);

            tasks[t] = kthread_run(unmap_thread_fn, &td[t], "uvm_unmap_%d", t);
            if (IS_ERR(tasks[t])) {
                pr_warn("UVM:CE_PROFILE: kthread_run failed\n");
                complete_all(&barrier);
                while (--t >= 0)
                    wait_for_completion(&td[t].done);
                failed = true;
                break;
            }
        }
        if (failed)
            continue;

        while (atomic_read(&ready_count) < nthreads)
            cpu_relax();

        t0 = ktime_get();
        complete_all(&barrier);

        for (t = 0; t < nthreads; t++)
            wait_for_completion(&td[t].done);

        ce_stats_add(stats, ktime_to_ns(ktime_sub(ktime_get(), t0)));
    }
    return NV_OK;
}

// Persistent kthread variant: threads created once, reused across iterations.
// Eliminates kthread_run/kthread_stop overhead per iteration, simulating the
// persistent kthread pool architecture planned for UVM parallel fault processing.
typedef struct {
    struct address_space *mapping;
    NvU64                 start_offset;
    NvU32                 num_pages;
    atomic_t             *ready_count;
    struct completion     go;
    struct completion     done;
    bool                  stop;
} persist_unmap_data_t;

static int persist_unmap_fn(void *arg)
{
    persist_unmap_data_t *data = arg;

    while (!data->stop) {
        NvU32 p;
        atomic_inc(data->ready_count);
        wait_for_completion(&data->go);
        if (data->stop)
            break;

        for (p = 0; p < data->num_pages; p++) {
            unmap_mapping_range(data->mapping,
                                data->start_offset + (NvU64)p * PAGE_SIZE,
                                PAGE_SIZE, 1);
        }
        complete(&data->done);
    }
    return 0;
}

// thread_stride: bytes between each thread's start address.
//   0 = contiguous (all threads in same 2MB PTE page → same ptl spinlock)
//   2MB = each thread in a different PTE page → different ptl spinlocks
static NV_STATUS do_cpu_unmap_persistent_test(struct address_space *mapping,
                                              NvU64 base, NvU32 num_pages,
                                              int nthreads, int iters,
                                              NvU64 thread_stride,
                                              ce_stats_t *stats)
{
    persist_unmap_data_t td[4];
    struct task_struct *tasks[4];
    atomic_t ready_count = ATOMIC_INIT(0);
    int t, iter;
    NvU32 per_thread = num_pages / nthreads;
    NvU64 iter_span;

    if (thread_stride > 0)
        iter_span = (NvU64)(nthreads - 1) * thread_stride + (NvU64)per_thread * PAGE_SIZE;
    else
        iter_span = (NvU64)num_pages * PAGE_SIZE;

    ce_stats_init(stats);

    for (t = 0; t < nthreads; t++) {
        td[t].mapping = mapping;
        td[t].ready_count = &ready_count;
        td[t].stop = false;
        init_completion(&td[t].go);
        init_completion(&td[t].done);

        tasks[t] = kthread_run(persist_unmap_fn, &td[t], "uvm_punmap_%d", t);
        if (IS_ERR(tasks[t])) {
            int j;
            for (j = 0; j < t; j++) {
                td[j].stop = true;
                complete(&td[j].go);
                kthread_stop(tasks[j]);
            }
            return NV_ERR_NO_MEMORY;
        }
    }

    for (iter = 0; iter < iters; iter++) {
        NvU64 iter_base = base + (NvU64)iter * iter_span;
        ktime_t t0;

        while (atomic_read(&ready_count) < nthreads)
            cpu_relax();
        atomic_set(&ready_count, 0);

        for (t = 0; t < nthreads; t++) {
            if (thread_stride > 0)
                td[t].start_offset = iter_base + (NvU64)t * thread_stride;
            else
                td[t].start_offset = iter_base + (NvU64)(t * per_thread) * PAGE_SIZE;
            td[t].num_pages = per_thread;
            reinit_completion(&td[t].done);
        }

        t0 = ktime_get();
        for (t = 0; t < nthreads; t++)
            complete(&td[t].go);

        for (t = 0; t < nthreads; t++)
            wait_for_completion(&td[t].done);

        ce_stats_add(stats, ktime_to_ns(ktime_sub(ktime_get(), t0)));
    }

    while (atomic_read(&ready_count) < nthreads)
        cpu_relax();
    for (t = 0; t < nthreads; t++) {
        td[t].stop = true;
        complete(&td[t].go);
    }
    for (t = 0; t < nthreads; t++)
        kthread_stop(tasks[t]);

    return NV_OK;
}

// ============================================================================
// Part 4: Sparse (non-contiguous) unmap — single TLB shootdown via
// selective PTE clearing.
//
// Compares three strategies for unmapping N non-contiguous pages
// (every-other-page pattern in a 2N range):
//   1) Individual: N × unmap_mapping_range(PAGE_SIZE)  →  N IPIs
//   2) Superset:   1 × unmap_mapping_range(2N*PAGE_SIZE) → 1 IPI (over-unmaps)
//   3) Selective:  clear N PTEs manually + 1 on_each_cpu IPI
// ============================================================================

struct sparse_flush_info {
    unsigned long *addrs;
    int            count;
};

static void ipi_invlpg_selective(void *info)
{
    struct sparse_flush_info *fi = info;
    int i;
    for (i = 0; i < fi->count; i++)
        asm volatile("invlpg (%0)" :: "r"(fi->addrs[i]) : "memory");
}

static pmd_t *walk_to_pmd(struct mm_struct *mm, unsigned long addr)
{
    pgd_t *pgd;
    p4d_t *p4d;
    pud_t *pud;
    pmd_t *pmd;

    pgd = pgd_offset(mm, addr);
    if (pgd_none(*pgd) || pgd_bad(*pgd))
        return NULL;

    p4d = p4d_offset(pgd, addr);
    if (p4d_none(*p4d) || p4d_bad(*p4d))
        return NULL;

    pud = pud_offset(p4d, addr);
    if (pud_none(*pud) || pud_bad(*pud))
        return NULL;

    pmd = pmd_offset(pud, addr);
    if (pmd_none(*pmd) || pmd_trans_huge(*pmd))
        return NULL;

    return pmd;
}

// Sparse individual: N × unmap_mapping_range(PAGE_SIZE) for non-contiguous pages
static NV_STATUS do_sparse_individual_test(struct address_space *mapping,
                                           NvU64 base, NvU32 num_pages,
                                           NvU32 stride, int iters,
                                           ce_stats_t *stats)
{
    int iter;
    NvU32 range = num_pages * stride;

    ce_stats_init(stats);
    for (iter = 0; iter < iters; iter++) {
        NvU64 iter_base = base + (NvU64)(iter * range) * PAGE_SIZE;
        NvU32 p;
        ktime_t t0 = ktime_get();

        for (p = 0; p < num_pages; p++) {
            unmap_mapping_range(mapping,
                                iter_base + (NvU64)(p * stride) * PAGE_SIZE,
                                PAGE_SIZE, 1);
        }
        ce_stats_add(stats, ktime_to_ns(ktime_sub(ktime_get(), t0)));
    }
    return NV_OK;
}

// Sparse superset: 1 × unmap_mapping_range covering the full 2N range
static NV_STATUS do_sparse_superset_test(struct address_space *mapping,
                                         NvU64 base, NvU32 num_pages,
                                         NvU32 stride, int iters,
                                         ce_stats_t *stats)
{
    int iter;
    NvU32 range = num_pages * stride;

    ce_stats_init(stats);
    for (iter = 0; iter < iters; iter++) {
        NvU64 iter_base = base + (NvU64)(iter * range) * PAGE_SIZE;
        ktime_t t0 = ktime_get();

        unmap_mapping_range(mapping, iter_base,
                            (loff_t)range * PAGE_SIZE, 1);
        ce_stats_add(stats, ktime_to_ns(ktime_sub(ktime_get(), t0)));
    }
    return NV_OK;
}

// Selective PTE clear with full page accounting + single batched IPI.
//
// Per-PTE work is identical to kernel's zap_present_folio_ptes():
//   ptep_get_and_clear, dirty/accessed tracking, RSS update, rmap, put_page.
// The ONLY difference from baseline (N × unmap_mapping_range): all N pages'
// TLB entries are flushed in a single IPI round instead of N rounds.
static NV_STATUS do_sparse_selective_test(struct address_space *mapping,
                                          NvU64 base, NvU32 num_pages,
                                          NvU32 stride, int iters,
                                          ce_stats_t *stats)
{
    struct mm_struct *mm = current->mm;
    NvU32 range = num_pages * stride;
    unsigned long *addrs;
    int iter;
    struct sparse_flush_info finfo;

    addrs = kmalloc(sizeof(unsigned long) * num_pages, GFP_KERNEL);
    if (!addrs)
        return NV_ERR_NO_MEMORY;

    ce_stats_init(stats);

    for (iter = 0; iter < iters; iter++) {
        NvU64 iter_base = base + (NvU64)(iter * range) * PAGE_SIZE;
        unsigned long first_addr = (unsigned long)iter_base;
        unsigned long last_addr  = first_addr +
                                   (NvU64)((num_pages - 1) * stride) * PAGE_SIZE;
        NvU32 p, count = 0;
        ktime_t t0;

        mmap_read_lock(mm);

        t0 = ktime_get();

        if ((first_addr & PMD_MASK) == (last_addr & PMD_MASK)) {
            pmd_t *pmd = walk_to_pmd(mm, first_addr);
            if (pmd) {
                spinlock_t *ptl = pte_lockptr(mm, pmd);
                spin_lock(ptl);
                for (p = 0; p < num_pages; p++) {
                    unsigned long addr = first_addr +
                                         (NvU64)(p * stride) * PAGE_SIZE;
                    pte_t *ptep = pte_offset_kernel(pmd, addr);
                    pte_t ptent = ptep_get_and_clear(mm, addr, ptep);

                    if (pte_present(ptent)) {
                        struct page *page = pte_page(ptent);
                        struct folio *folio = page_folio(page);

                        if (pte_dirty(ptent))
                            folio_mark_dirty(folio);
                        if (pte_young(ptent))
                            folio_mark_accessed(folio);
                        percpu_counter_add(&mm->rss_stat[mm_counter(page)], -1);
                        if (atomic_add_negative(-1, &page->_mapcount)) {
                            if (folio_test_large(folio))
                                atomic_dec(&folio->_nr_pages_mapped);
                        }
                        put_page(page);
                        addrs[count++] = addr;
                    }
                }
                spin_unlock(ptl);
            }
        } else {
            for (p = 0; p < num_pages; p++) {
                unsigned long addr = first_addr +
                                     (NvU64)(p * stride) * PAGE_SIZE;
                pmd_t *pmd = walk_to_pmd(mm, addr);
                spinlock_t *ptl;
                pte_t *ptep;
                pte_t ptent;

                if (!pmd)
                    continue;
                ptl = pte_lockptr(mm, pmd);
                spin_lock(ptl);
                ptep = pte_offset_kernel(pmd, addr);
                ptent = ptep_get_and_clear(mm, addr, ptep);
                if (pte_present(ptent)) {
                    struct page *page = pte_page(ptent);
                    struct folio *folio = page_folio(page);

                    if (pte_dirty(ptent))
                        folio_mark_dirty(folio);
                    if (pte_young(ptent))
                        folio_mark_accessed(folio);
                    percpu_counter_add(&mm->rss_stat[mm_counter(page)], -1);
                    if (atomic_add_negative(-1, &page->_mapcount)) {
                        if (folio_test_large(folio))
                            atomic_dec(&folio->_nr_pages_mapped);
                    }
                    put_page(page);
                    addrs[count++] = addr;
                }
                spin_unlock(ptl);
            }
        }

        // Single batched IPI — THE ONLY DIFFERENCE from baseline.
        // Baseline does N × tlb_finish_mmu() = N IPI rounds.
        // We do 1 × smp_call_function_many() = 1 IPI round.
        if (count > 0) {
            finfo.addrs = addrs;
            finfo.count = count;
            preempt_disable();
            smp_call_function_many(mm_cpumask(mm),
                                   ipi_invlpg_selective, &finfo, 1);
            ipi_invlpg_selective(&finfo);
            preempt_enable();
        }

        ce_stats_add(stats, ktime_to_ns(ktime_sub(ktime_get(), t0)));

        mmap_read_unlock(mm);
    }

    kfree(addrs);
    return NV_OK;
}

NV_STATUS uvm_test_ce_parallel_profile(UVM_TEST_CE_PARALLEL_PROFILE_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_gpu_t *gpu;
    uvm_channel_manager_t *manager;
    uvm_rm_mem_t *test_mem = NULL;
    NV_STATUS status = NV_OK;
    NvU64 gpu_va;
    unsigned i;
    NvU32 alloc_size;
    uvm_channel_pool_t *pool_c2g, *pool_g2c, *pool_int;
    const char *type_names[] = {"CPU_TO_GPU", "GPU_TO_CPU", "GPU_INTERNAL", "MEMOPS", "GPU_TO_GPU"};
    static const NvU32 sizes[] = { 64 * 1024, 128 * 1024, 256 * 1024 };
    static const char *size_labels[] = { "64KB", "128KB", "256KB" };
    ce_stats_t st;

    uvm_va_space_down_read(va_space);
    gpu = uvm_va_space_find_first_gpu(va_space);
    if (!gpu) {
        uvm_va_space_up_read(va_space);
        return NV_ERR_INVALID_STATE;
    }
    uvm_gpu_retain(gpu);
    uvm_va_space_up_read(va_space);

    manager = gpu->channel_manager;

    // ---- CE Configuration ----
    pool_c2g = manager->pool_to_use.default_for_type[UVM_CHANNEL_TYPE_CPU_TO_GPU];
    pool_g2c = manager->pool_to_use.default_for_type[UVM_CHANNEL_TYPE_GPU_TO_CPU];
    pool_int = manager->pool_to_use.default_for_type[UVM_CHANNEL_TYPE_GPU_INTERNAL];

    params->num_usable_ces = (NvU32)bitmap_weight(manager->ce_mask, UVM_COPY_ENGINE_COUNT_MAX);
    params->cpu_to_gpu_ce  = pool_c2g ? pool_c2g->engine_index : 0;
    params->gpu_to_cpu_ce  = pool_g2c ? pool_g2c->engine_index : 0;
    params->gpu_internal_ce = pool_int ? pool_int->engine_index : 0;
    params->memops_ce      = manager->pool_to_use.default_for_type[UVM_CHANNEL_TYPE_MEMOPS]
                             ? manager->pool_to_use.default_for_type[UVM_CHANNEL_TYPE_MEMOPS]->engine_index : 0;

    pr_info("UVM:CE_PROFILE: ========================================================\n");
    pr_info("UVM:CE_PROFILE: CE Configuration for GPU %s\n", uvm_gpu_name(gpu));
    pr_info("UVM:CE_PROFILE: Usable CEs: %u (mask: 0x%lx)\n",
            params->num_usable_ces, manager->ce_mask[0]);
    for (i = 0; i < manager->num_channel_pools; i++) {
        uvm_channel_pool_t *p = &manager->channel_pools[i];
        if (p->pool_type == UVM_CHANNEL_POOL_TYPE_CE)
            pr_info("UVM:CE_PROFILE:   CE pool: engine=%u channels=%u\n",
                    p->engine_index, p->num_channels);
    }
    for (i = 0; i < UVM_CHANNEL_TYPE_CE_COUNT; i++) {
        uvm_channel_pool_t *p = manager->pool_to_use.default_for_type[i];
        if (p)
            pr_info("UVM:CE_PROFILE:   %-12s -> CE %u\n", type_names[i], p->engine_index);
        else
            pr_info("UVM:CE_PROFILE:   %-12s -> NONE\n", type_names[i]);
    }

    // ---- GPU PTE validity note ----
    pr_info("UVM:CE_PROFILE: [NOTE] GPU PTE tests use ce_hal->memset via uvm_push — the\n");
    pr_info("UVM:CE_PROFILE:   same CE HAL + push mechanism as uvm_va_block GPU PTE updates.\n");
    pr_info("UVM:CE_PROFILE:   CPU_TO_GPU(CE%u) and MEMOPS(CE%u) %s the same physical CE.\n",
            params->cpu_to_gpu_ce, params->memops_ce,
            params->cpu_to_gpu_ce == params->memops_ce ? "SHARE" : "do NOT share");

    // ---- Allocate test memory (max needed: 4 blocks × 4MB = 16MB) ----
    alloc_size = CE_PIPELINE_BLOCKS * CE_MAX_ALLOC_SIZE;
    status = uvm_rm_mem_alloc_and_map_cpu(gpu, UVM_RM_MEM_TYPE_SYS, alloc_size, 0, &test_mem);
    if (status != NV_OK) {
        pr_err("UVM:CE_PROFILE: Failed to alloc %u bytes: 0x%x\n", alloc_size, status);
        goto out_release;
    }
    gpu_va = uvm_rm_mem_get_gpu_va(test_mem, gpu, false).address;

    // ================================================================
    // Part 1: Cross-CE parallelism at different data sizes
    // ================================================================
    for (i = 0; i < ARRAY_SIZE(sizes); i++) {
        NvU32 sz = sizes[i];

        pr_info("UVM:CE_PROFILE: ---- Part1: Cross-CE @ %s (%d iters) ----\n",
                size_labels[i], CE_ITERS);

        do_single_push_test(gpu, manager, UVM_CHANNEL_TYPE_CPU_TO_GPU,
                            gpu_va, sz, &st);
        ce_stats_print("A: 1x single push (CE2 baseline)", &st);

        do_dual_push_test(gpu, manager,
                          UVM_CHANNEL_TYPE_CPU_TO_GPU, UVM_CHANNEL_TYPE_CPU_TO_GPU,
                          gpu_va, sz, &st);
        ce_stats_print("B: 2x same CE (both CPU_TO_GPU=CE2)", &st);

        do_dual_push_test(gpu, manager,
                          UVM_CHANNEL_TYPE_CPU_TO_GPU, UVM_CHANNEL_TYPE_GPU_INTERNAL,
                          gpu_va, sz, &st);
        ce_stats_print("C: 2x cross-CE (CE2+CE4 GPU_INTERNAL)", &st);

        do_dual_push_test(gpu, manager,
                          UVM_CHANNEL_TYPE_CPU_TO_GPU, UVM_CHANNEL_TYPE_GPU_TO_CPU,
                          gpu_va, sz, &st);
        ce_stats_print("D: 2x cross-CE (CE2+CE3 GPU_TO_CPU)", &st);
    }

    // ================================================================
    // Part 2: Pipeline simulation (4 blocks)
    // ================================================================
    for (i = 0; i < ARRAY_SIZE(sizes); i++) {
        NvU32 sz = sizes[i];

        pr_info("UVM:CE_PROFILE: ---- Part2: Pipeline %dx%s (%d iters) ----\n",
                CE_PIPELINE_BLOCKS, size_labels[i], CE_ITERS);

        do_pipeline_serial_test(gpu, manager, UVM_CHANNEL_TYPE_CPU_TO_GPU,
                                gpu_va, sz, CE_PIPELINE_BLOCKS, &st);
        ce_stats_print("E: Serial  (push+wait each, CE2)", &st);

        do_pipeline_async_test(gpu, manager, UVM_CHANNEL_TYPE_CPU_TO_GPU,
                               gpu_va, sz, CE_PIPELINE_BLOCKS, &st);
        ce_stats_print("F: Pipeline (push all, wait once, CE2)", &st);

        do_pipeline_cross_ce_test(gpu, manager,
                                  UVM_CHANNEL_TYPE_CPU_TO_GPU,
                                  UVM_CHANNEL_TYPE_GPU_INTERNAL,
                                  gpu_va, sz, CE_PIPELINE_BLOCKS, &st);
        ce_stats_print("G: Pipeline cross-CE (CE2/CE4 round-robin)", &st);

        do_pipeline_cross_ce_test(gpu, manager,
                                  UVM_CHANNEL_TYPE_CPU_TO_GPU,
                                  UVM_CHANNEL_TYPE_GPU_TO_CPU,
                                  gpu_va, sz, CE_PIPELINE_BLOCKS, &st);
        ce_stats_print("H: Pipeline cross-CE (CE2/CE3 round-robin)", &st);
    }

    uvm_rm_mem_free(test_mem);
    test_mem = NULL;

    // ================================================================
    // Part 3: CPU Unmap / TLB Shootdown
    //
    // Uses unmap_mapping_range(va_space->mapping, ...) — UVM's native
    // CPU PTE removal path (see block_unmap_cpu in uvm_va_block.c).
    //
    // Tests per page count:
    //   Individual 1T     — N separate unmap_mapping_range(PAGE_SIZE)
    //   Batch 1T          — 1 unmap_mapping_range(N*PAGE_SIZE)
    //   Parallel 2T       — 2 kthreads, same 2MB PTE page (same ptl)
    //   Parallel 4T       — 4 kthreads, same ptl
    //   PersistKT 2T      — persistent kthreads, same ptl
    //   PersistKT 4T      — persistent kthreads, same ptl
    //   PersistKT 2T xptl — persistent kthreads, DIFFERENT ptl (2MB apart)
    // ================================================================
#define CROSS_PTL_STRIDE (2ULL * 1024 * 1024)
    if (params->managed_base != 0 && params->managed_size != 0) {
        struct address_space *mapping = va_space->mapping;
        NvU64 cursor = params->managed_base;
        NvU64 end    = params->managed_base + params->managed_size;
        static const NvU32 unmap_counts[] = { 16, 32, 64, 128 };
        int ui;
        char label[80];

        pr_info("UVM:CE_PROFILE: ---- Part3: CPU Unmap / TLB Shootdown (%d iters) ----\n",
                UNMAP_ITERS);
        pr_info("UVM:CE_PROFILE:   managed=%llu MB  online_cpus=%u\n",
                params->managed_size >> 20, num_online_cpus());

        for (ui = 0; ui < ARRAY_SIZE(unmap_counts); ui++) {
            NvU32 np = unmap_counts[ui];
            NvU64 per_test_contig = (NvU64)UNMAP_ITERS * np * PAGE_SIZE;
            NvU64 cross_iter_span = CROSS_PTL_STRIDE + (NvU64)(np / 2) * PAGE_SIZE;
            NvU64 per_test_cross  = (NvU64)UNMAP_ITERS * cross_iter_span;
            NvU64 needed = 6 * per_test_contig + per_test_cross;

            if (cursor + needed > end) {
                pr_info("UVM:CE_PROFILE:   SKIP %u pages: insufficient managed memory\n", np);
                continue;
            }

            pr_info("UVM:CE_PROFILE: --- %u pages (%lu KB) ---\n",
                    np, (unsigned long)np * PAGE_SIZE / 1024);

            snprintf(label, sizeof(label), "Individual 1T (%ux4K)", np);
            do_cpu_unmap_individual_test(mapping, cursor, np, UNMAP_ITERS, &st);
            ce_stats_print_perpg(label, &st, np);
            cursor += per_test_contig;

            snprintf(label, sizeof(label), "Batch 1T (1x%uK)", np * 4);
            do_cpu_unmap_batch_test(mapping, cursor, np, UNMAP_ITERS, &st);
            ce_stats_print_perpg(label, &st, np);
            cursor += per_test_contig;

            snprintf(label, sizeof(label), "Parallel 2T (2x%upg indiv)", np / 2);
            do_cpu_unmap_parallel_test(mapping, cursor, np, 2, UNMAP_ITERS, &st);
            ce_stats_print_perpg(label, &st, np);
            cursor += per_test_contig;

            snprintf(label, sizeof(label), "Parallel 4T (4x%upg indiv)", np / 4);
            do_cpu_unmap_parallel_test(mapping, cursor, np, 4, UNMAP_ITERS, &st);
            ce_stats_print_perpg(label, &st, np);
            cursor += per_test_contig;

            snprintf(label, sizeof(label), "PersistKT 2T same-ptl (%upg)", np / 2);
            do_cpu_unmap_persistent_test(mapping, cursor, np, 2, UNMAP_ITERS, 0, &st);
            ce_stats_print_perpg(label, &st, np);
            cursor += per_test_contig;

            snprintf(label, sizeof(label), "PersistKT 4T same-ptl (%upg)", np / 4);
            do_cpu_unmap_persistent_test(mapping, cursor, np, 4, UNMAP_ITERS, 0, &st);
            ce_stats_print_perpg(label, &st, np);
            cursor += per_test_contig;

            snprintf(label, sizeof(label), "PersistKT 2T cross-ptl (%upg)", np / 2);
            do_cpu_unmap_persistent_test(mapping, cursor, np, 2, UNMAP_ITERS,
                                         CROSS_PTL_STRIDE, &st);
            ce_stats_print_perpg(label, &st, np);
            cursor += per_test_cross;
        }
        // ============================================================
        // Part 4: Sparse unmap — non-contiguous pages (stride=2)
        //
        // Compares three strategies for unmapping N pages scattered
        // with stride-2 (every other page) across a 2N-page range:
        //   Sparse Indiv    — N × unmap_mapping_range(PAGE_SIZE)
        //   Sparse Superset — 1 × unmap_mapping_range(2N range)
        //   Sparse Selective — clear N PTEs + 1 on_each_cpu(INVLPG)
        // ============================================================
#define SPARSE_STRIDE 2
        pr_info("UVM:CE_PROFILE: ---- Part4: Sparse Unmap stride=%d (%d iters) ----\n",
                SPARSE_STRIDE, UNMAP_ITERS);

        for (ui = 0; ui < ARRAY_SIZE(unmap_counts); ui++) {
            NvU32 np = unmap_counts[ui];
            NvU32 range_pages = np * SPARSE_STRIDE;
            NvU64 per_sparse_test = (NvU64)UNMAP_ITERS * range_pages * PAGE_SIZE;
            NvU64 sparse_needed = 3 * per_sparse_test;

            if (cursor + sparse_needed > end) {
                pr_info("UVM:CE_PROFILE:   SKIP sparse %u pages: insufficient memory\n", np);
                continue;
            }

            pr_info("UVM:CE_PROFILE: --- sparse %u/%u pages (%lu KB range) ---\n",
                    np, range_pages,
                    (unsigned long)range_pages * PAGE_SIZE / 1024);

            snprintf(label, sizeof(label),
                     "Sparse Indiv (%ux4K stride%d)", np, SPARSE_STRIDE);
            do_sparse_individual_test(mapping, cursor, np,
                                      SPARSE_STRIDE, UNMAP_ITERS, &st);
            ce_stats_print_perpg(label, &st, np);
            cursor += per_sparse_test;

            snprintf(label, sizeof(label),
                     "Sparse Superset (1x%uK)", range_pages * 4);
            do_sparse_superset_test(mapping, cursor, np,
                                    SPARSE_STRIDE, UNMAP_ITERS, &st);
            ce_stats_print_perpg(label, &st, np);
            cursor += per_sparse_test;

            snprintf(label, sizeof(label),
                     "Sparse Selective (%upg+1IPI)", np);
            do_sparse_selective_test(mapping, cursor, np,
                                     SPARSE_STRIDE, UNMAP_ITERS, &st);
            ce_stats_print_perpg(label, &st, np);
            cursor += per_sparse_test;
        }
    } else {
        pr_info("UVM:CE_PROFILE: SKIP Part3/4: no managed_base provided\n");
    }

    pr_info("UVM:CE_PROFILE: ========================================================\n");
    status = NV_OK;

out_release:
    if (test_mem)
        uvm_rm_mem_free(test_mem);
    uvm_mutex_lock(&g_uvm_global.global_lock);
    uvm_gpu_release_locked(gpu);
    uvm_mutex_unlock(&g_uvm_global.global_lock);
    return status;
}
