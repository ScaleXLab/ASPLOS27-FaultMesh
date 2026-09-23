/*******************************************************************************
    Copyright (c) 2015-2024 NVIDIA Corporation

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

#include "linux/sort.h"
#include "nv_uvm_interface.h"
#include "uvm_common.h"
#include "uvm_linux.h"
#include "uvm_global.h"
#include "uvm_gpu_replayable_faults.h"
#include "uvm_hal.h"
#include "uvm_kvmalloc.h"
#include "uvm_tools.h"
#include "uvm_va_block.h"
#include "uvm_va_range.h"
#include "uvm_va_space.h"
#include "uvm_va_space_mm.h"
#include "uvm_procfs.h"
#include "uvm_perf_thrashing.h"
#include "uvm_gpu_non_replayable_faults.h"
#include "uvm_ats_faults.h"
#include "uvm_test.h"
#include "uvm_gpu_replayable_faults_parallel.h"

// The documentation at the beginning of uvm_gpu_non_replayable_faults.c
// provides some background for understanding replayable faults, non-replayable
// faults, and how UVM services each fault type.

// The HW fault buffer flush mode instructs RM on how to flush the hardware
// replayable fault buffer; it is only used in Confidential Computing.
//
// Unless HW_FAULT_BUFFER_FLUSH_MODE_MOVE is functionally required (because UVM
// needs to inspect the faults currently present in the HW fault buffer) it is
// recommended to use HW_FAULT_BUFFER_FLUSH_MODE_DISCARD for performance
// reasons.
typedef enum
{
    // Flush the HW fault buffer, discarding all the resulting faults. UVM never
    // gets to see these faults.
    HW_FAULT_BUFFER_FLUSH_MODE_DISCARD,

    // Flush the HW fault buffer, and move all the resulting faults to the SW
    // fault ("shadow") buffer.
    HW_FAULT_BUFFER_FLUSH_MODE_MOVE,
} hw_fault_buffer_flush_mode_t;

#define UVM_PERF_REENABLE_PREFETCH_FAULTS_LAPSE_MSEC_DEFAULT 1000

// Lapse of time in milliseconds after which prefetch faults can be re-enabled.
// 0 means it is never disabled
static unsigned uvm_perf_reenable_prefetch_faults_lapse_msec = UVM_PERF_REENABLE_PREFETCH_FAULTS_LAPSE_MSEC_DEFAULT;
module_param(uvm_perf_reenable_prefetch_faults_lapse_msec, uint, S_IRUGO);

#define UVM_PERF_FAULT_BATCH_COUNT_MIN 1
#define UVM_PERF_FAULT_BATCH_COUNT_DEFAULT 256

// Number of entries that are fetched from the GPU fault buffer and serviced in
// batch
static unsigned uvm_perf_fault_batch_count = UVM_PERF_FAULT_BATCH_COUNT_DEFAULT;
module_param(uvm_perf_fault_batch_count, uint, S_IRUGO);

unsigned uvm_perf_fault_fetch_adaptive_enable = 0;
module_param(uvm_perf_fault_fetch_adaptive_enable, uint, 0644);
MODULE_PARM_DESC(uvm_perf_fault_fetch_adaptive_enable,
                 "Enable adaptive fault fetch limit based on pending buffer depth (0=off, 1=on)");

unsigned uvm_perf_fault_fetch_low_count = 256;
module_param(uvm_perf_fault_fetch_low_count, uint, 0644);
MODULE_PARM_DESC(uvm_perf_fault_fetch_low_count,
                 "Adaptive fetch limit when pending faults stay below mid threshold");

unsigned uvm_perf_fault_fetch_mid_count = 512;
module_param(uvm_perf_fault_fetch_mid_count, uint, 0644);
MODULE_PARM_DESC(uvm_perf_fault_fetch_mid_count,
                 "Adaptive fetch limit when pending faults exceed mid threshold");

unsigned uvm_perf_fault_fetch_high_count = 1024;
module_param(uvm_perf_fault_fetch_high_count, uint, 0644);
MODULE_PARM_DESC(uvm_perf_fault_fetch_high_count,
                 "Adaptive fetch limit when pending faults exceed high threshold");

unsigned uvm_perf_fault_fetch_mid_pending = 512;
module_param(uvm_perf_fault_fetch_mid_pending, uint, 0644);
MODULE_PARM_DESC(uvm_perf_fault_fetch_mid_pending,
                 "Pending-fault threshold that switches adaptive fetch to mid_count");

unsigned uvm_perf_fault_fetch_high_pending = 1024;
module_param(uvm_perf_fault_fetch_high_pending, uint, 0644);
MODULE_PARM_DESC(uvm_perf_fault_fetch_high_pending,
                 "Pending-fault threshold that switches adaptive fetch to high_count");

unsigned uvm_perf_fault_fetch_predictor_boost_enable = 0;
module_param(uvm_perf_fault_fetch_predictor_boost_enable, uint, 0644);
MODULE_PARM_DESC(uvm_perf_fault_fetch_predictor_boost_enable,
                 "Boost fetch_limit when previous-batch predictor hit ratio is high (0=off, 1=on)");

unsigned uvm_perf_fault_prev_fetch_predictor_enable = 1;
module_param(uvm_perf_fault_prev_fetch_predictor_enable, uint, 0644);
MODULE_PARM_DESC(uvm_perf_fault_prev_fetch_predictor_enable,
                 "Enable previous-batch fetch predictor bookkeeping used by skip/boost experiments (0=off, 1=on)");

unsigned uvm_perf_fault_prev_fetch_mode = 0;
module_param(uvm_perf_fault_prev_fetch_mode, uint, 0644);
MODULE_PARM_DESC(uvm_perf_fault_prev_fetch_mode,
                 "Previous-fetch predictor mode: 0=legacy fetch-cache exact key, 1=coalesced address bsearch, 2=coalesced address direct-mapped hash, 3=coalesced approximate tag table");

unsigned uvm_perf_fault_fetch_predictor_boost_threshold_pct = 40;
module_param(uvm_perf_fault_fetch_predictor_boost_threshold_pct, uint, 0644);
MODULE_PARM_DESC(uvm_perf_fault_fetch_predictor_boost_threshold_pct,
                 "Minimum previous-batch predictor candidate percentage that triggers fetch boost");

unsigned uvm_perf_fault_fetch_predictor_boost_count = 1024;
module_param(uvm_perf_fault_fetch_predictor_boost_count, uint, 0644);
MODULE_PARM_DESC(uvm_perf_fault_fetch_predictor_boost_count,
                 "Fetch limit target used when predictor-based boost is triggered");

unsigned uvm_perf_fault_pred_skip_enable = 0;
module_param(uvm_perf_fault_pred_skip_enable, uint, 0644);
MODULE_PARM_DESC(uvm_perf_fault_pred_skip_enable,
                 "Enable fetch-stage predictor skip using previous coalesced fault keys (0=off, 1=on)");

unsigned uvm_perf_fault_pred_timing_enable = 0;
module_param(uvm_perf_fault_pred_timing_enable, uint, 0644);
MODULE_PARM_DESC(uvm_perf_fault_pred_timing_enable,
                 "Enable predictor build/lookup timing counters in stale profiling (0=off, 1=on)");

// P3: reduce ktime_get() call volume during profiling by only timing 1 out
// of every N lookup calls and scaling the sampled cost back up by N. The
// accumulated time_pred_fetch_lookup_ns stays an apples-to-apples estimate
// of "every call timed" (N=1, i.e. the original behavior) for dmesg/CSV
// compatibility, but pays the ~20-25ns ktime_get() tax on only a 1/N
// fraction of lookups, so the measured filter time is closer to its
// production (un-instrumented) cost.
unsigned uvm_perf_fault_pred_timing_sample_rate = 1;
module_param(uvm_perf_fault_pred_timing_sample_rate, uint, 0644);
MODULE_PARM_DESC(uvm_perf_fault_pred_timing_sample_rate,
                 "P3: when timing is enabled, ktime_get()-time only 1 out of every N lookup calls "
                 "and scale the result by N (1=time every call, i.e. unsampled/original behavior)");

// P4: additive bias correction, in ns, applied to every timed lookup sample.
//
// A ktime_get()-delimited measurement does not run from "start of body" to
// "end of body": it runs from the clock read inside the first ktime_get() to
// the clock read inside the second, so it carries roughly one whole
// ktime_get() worth of work no matter how cheap the body is. On this machine
// that bias is ~30ns, which is an order of magnitude larger than the
// lookup itself (~2-4ns, cross-checked against the build path, which times
// one ktime_get() pair per batch instead of per key). Left uncorrected it
// made the filter look like ~10% of handler time instead of ~1%.
//
// calibrate_ktime_overhead() measures exactly this bias -- two back-to-back
// clock reads with an empty body -- and publishes the result here, so the
// recommended profiling configuration is to run the calibration once at
// load time and let it populate this automatically. 0 disables the
// correction and reproduces the old, biased-high numbers.
//
// Note P3 and P4 fix different problems and are both needed: sampling
// reduces how much the measurement *perturbs* the run, while this
// correction removes the bias in each individual sample.
unsigned uvm_perf_fault_pred_timing_overhead_ns = 0;
module_param(uvm_perf_fault_pred_timing_overhead_ns, uint, 0644);
MODULE_PARM_DESC(uvm_perf_fault_pred_timing_overhead_ns,
                 "P4: ns of ktime_get() measurement bias to subtract from each timed lookup "
                 "sample (0=no correction; populated automatically by the ktime calibration)");

// O1 tunable: table size = roundup_pow_of_two(max_batch_size * this value).
// A batch inserts up to num_coalesced_faults keys (<= max_batch_size) into a
// *direct-mapped* table (no chaining: a same-batch hash collision silently
// evicts whichever key was written first). Too small a load factor measurably
// increases collision-driven key loss -- which shows up as a *lower*
// pred_filter_fast_skips count and *more* real page-fault servicing work,
// not just a slower lookup -- so this has to stay large enough that
// collisions are rare, while still being tiny/L1-L2-resident relative to
// the old max_faults*2 sizing (e.g. 256 batch size * 16 = 4096 slots =
// 64 KiB, vs the original ~768 KiB for two arrays sized off max_faults).
unsigned uvm_perf_fault_pred_exact_load_factor = 16;
module_param(uvm_perf_fault_pred_exact_load_factor, uint, 0644);
MODULE_PARM_DESC(uvm_perf_fault_pred_exact_load_factor,
                 "O1: size the mode-2 (hash_exact) predictor table to "
                 "max_batch_size * this value (rounded up to a power of "
                 "two, minimum 64). Read once at GPU fault-buffer init.");

// O6: adaptive activation. Stale/duplicate fault-buffer entries are only
// really produced when the GPU is generating faults faster than the driver
// drains them (a backlog); a batch that finishes well short of its
// fetch-limit cap drained the whole buffer and essentially never has
// anything for the filter to find. When enabled, the filter (lookup+build)
// is skipped entirely -- not just made cheaper -- for batches while no
// backlog is signaled, at the cost of one cold-start batch's worth of
// missed skips whenever a backlog begins (the predictor table's epoch
// mechanism makes "was skipped for a while" indistinguishable from "cold",
// so this is always correctness-safe, purely a missed optimization
// opportunity in the worst case). Default 0 = always run the filter,
// identical to pre-O6 behavior.
unsigned uvm_perf_fault_pred_adaptive_enable = 0;
module_param(uvm_perf_fault_pred_adaptive_enable, uint, 0644);
MODULE_PARM_DESC(uvm_perf_fault_pred_adaptive_enable,
                 "O6: skip the stale-fault filter (lookup+build) entirely for batches that don't "
                 "signal a fault-buffer backlog (fetch did not hit its per-batch fetch-limit cap). "
                 "0=always run the filter (default), 1=adaptive gating.");

// O6: hysteresis. After a batch drops below the backlog threshold, keep the
// filter active for up to this many additional consecutive batches (as long
// as it keeps finding stale faults, the cooldown resets) before powering it
// down. Guards against turning off right as a real backlog is draining, one
// batch below the fetch-limit cap at a time.
unsigned uvm_perf_fault_pred_adaptive_cooldown = 2;
module_param(uvm_perf_fault_pred_adaptive_cooldown, uint, 0644);
MODULE_PARM_DESC(uvm_perf_fault_pred_adaptive_cooldown,
                 "O6: consecutive below-threshold, zero-stale-fault batches to keep the filter "
                 "active before powering it down (hysteresis).");

// P2: one-shot calibration probe. If non-zero, the next call to
// uvm_gpu_service_replayable_faults() runs this many back-to-back
// ktime_get() pairs with no other work in between and logs
// [KTIME_CALIBRATION] min/avg/max ns, then resets itself to 0. This
// isolates pure instrumentation overhead so it can be subtracted from the
// measured stale-fault-filter time instead of being silently folded into
// "algorithmic cost".
unsigned uvm_perf_fault_pred_calibrate_ktime_iters = 0;
module_param(uvm_perf_fault_pred_calibrate_ktime_iters, uint, 0644);
MODULE_PARM_DESC(uvm_perf_fault_pred_calibrate_ktime_iters,
                 "P2: if non-zero, run this many back-to-back ktime_get() pairs on the next fault "
                 "service call and log [KTIME_CALIBRATION] overhead stats, then reset to 0");

unsigned uvm_perf_fault_prev_batch_predictor_enable = 1;
module_param(uvm_perf_fault_prev_batch_predictor_enable, uint, 0644);
MODULE_PARM_DESC(uvm_perf_fault_prev_batch_predictor_enable,
                 "Enable previous-batch stale predictor bookkeeping on coalesced faults (0=off, 1=on)");

#define UVM_PERF_FAULT_REPLAY_POLICY_DEFAULT UVM_PERF_FAULT_REPLAY_POLICY_BATCH_FLUSH

// Policy that determines when to issue fault replays
static uvm_perf_fault_replay_policy_t uvm_perf_fault_replay_policy = UVM_PERF_FAULT_REPLAY_POLICY_DEFAULT;
module_param(uvm_perf_fault_replay_policy, uint, S_IRUGO);

#define UVM_PERF_FAULT_REPLAY_UPDATE_PUT_RATIO_DEFAULT 50

// Reading fault buffer GET/PUT pointers from the CPU is expensive. However,
// updating PUT before flushing the buffer helps minimizing the number of
// duplicates in the buffer as it discards faults that were not processed
// because of the batch size limit or because they arrived during servicing.
// If PUT is not updated, the replay operation will make them show up again
// in the buffer as duplicates.
//
// We keep track of the number of duplicates in each batch and we use
// UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT for the fault buffer flush after if the
// percentage of duplicate faults in a batch is greater than the ratio defined
// in the following module parameter. UVM_GPU_BUFFER_FLUSH_MODE_CACHED_PUT is
// used, otherwise.
static unsigned uvm_perf_fault_replay_update_put_ratio = UVM_PERF_FAULT_REPLAY_UPDATE_PUT_RATIO_DEFAULT;
module_param(uvm_perf_fault_replay_update_put_ratio, uint, S_IRUGO);

// Force UPDATE_PUT in BATCH_FLUSH path for every non-cancel replay flush.
// Useful for experiments that isolate CACHED_PUT effects.
static unsigned uvm_perf_fault_replay_force_update_put = 0;
module_param(uvm_perf_fault_replay_force_update_put, uint, 0644);
MODULE_PARM_DESC(uvm_perf_fault_replay_force_update_put,
                 "Force UPDATE_PUT in replay batch-flush path (0=off, 1=on)");

unsigned uvm_perf_fault_replay_stale_update_put_enable = 1;
module_param(uvm_perf_fault_replay_stale_update_put_enable, uint, 0644);
MODULE_PARM_DESC(uvm_perf_fault_replay_stale_update_put_enable,
                 "Count stale faults in the UPDATE_PUT trigger heuristic (0=duplicate only, 1=duplicate+stale)");

#define UVM_PERF_FAULT_MAX_BATCHES_PER_SERVICE_DEFAULT 20

#define UVM_PERF_FAULT_MAX_THROTTLE_PER_SERVICE_DEFAULT 5

// Maximum number of batches to be processed per execution of the bottom-half
static unsigned uvm_perf_fault_max_batches_per_service = UVM_PERF_FAULT_MAX_BATCHES_PER_SERVICE_DEFAULT;
module_param(uvm_perf_fault_max_batches_per_service, uint, S_IRUGO);

// Maximum number of batches with thrashing pages per execution of the bottom-half
static unsigned uvm_perf_fault_max_throttle_per_service = UVM_PERF_FAULT_MAX_THROTTLE_PER_SERVICE_DEFAULT;
module_param(uvm_perf_fault_max_throttle_per_service, uint, S_IRUGO);

static unsigned uvm_perf_fault_coalesce = 1;
module_param(uvm_perf_fault_coalesce, uint, S_IRUGO);

static unsigned uvm_block_dist_count = 0;
module_param(uvm_block_dist_count, uint, S_IRUGO | S_IWUSR);

unsigned uvm_merge_dispatch = 0;
module_param(uvm_merge_dispatch, uint, 0644);
MODULE_PARM_DESC(uvm_merge_dispatch, "Enable cross-VA-block fault merging (0=disabled, 1=enabled)");

unsigned uvm_fault_track_range_enable = 0;
module_param(uvm_fault_track_range_enable, uint, 0644);
MODULE_PARM_DESC(uvm_fault_track_range_enable,
                 "Track stale/replay counts for one VA range [start,end) (0=off, 1=on)");

unsigned uvm_perf_fault_stale_detail_enable = 1;
module_param(uvm_perf_fault_stale_detail_enable, uint, 0644);
MODULE_PARM_DESC(uvm_perf_fault_stale_detail_enable,
                 "Enable detailed stale-fault classification and lag bookkeeping (0=off, 1=on)");

unsigned long long uvm_fault_track_range_start = 0;
module_param(uvm_fault_track_range_start, ullong, 0644);
MODULE_PARM_DESC(uvm_fault_track_range_start,
                 "Start VA of tracked fault range (inclusive)");

unsigned long long uvm_fault_track_range_end = 0;
module_param(uvm_fault_track_range_end, ullong, 0644);
MODULE_PARM_DESC(uvm_fault_track_range_end,
                 "End VA of tracked fault range (exclusive)");

#define MERGE_THRESHOLD 8
#define MAX_MERGE_BLOCKS 16

// This function is used for both the initial fault buffer initialization and
// the power management resume path.
static void fault_buffer_reinit_replayable_faults(uvm_parent_gpu_t *parent_gpu)
{
    uvm_replayable_fault_buffer_info_t *replayable_faults = &parent_gpu->fault_buffer_info.replayable;

    // Read the current get/put pointers, as this might not be the first time
    // we take control of the fault buffer since the GPU was initialized,
    // or since we may need to bring UVM's cached copies back in sync following
    // a sleep cycle.
    replayable_faults->cached_get = parent_gpu->fault_buffer_hal->read_get(parent_gpu);
    replayable_faults->cached_put = parent_gpu->fault_buffer_hal->read_put(parent_gpu);
    replayable_faults->prev_batch_key_count = 0;
    replayable_faults->last_batch_pred_candidate_pct = 0;
    replayable_faults->prev_fetch_key_count = 0;
    replayable_faults->last_batch_fetch_skip_pct = 0;
    replayable_faults->prev_fetch_last_valid = false;
    // O6: start active so the very first batches after a reinit are always
    // covered, matching pre-O6 (always-on) behavior until the adaptive
    // state machine has evidence to power down.
    replayable_faults->prev_fetch_adaptive_active = true;
    replayable_faults->prev_fetch_adaptive_cooldown_remaining = 0;

    // (Re-)enable fault prefetching
    if (parent_gpu->fault_buffer_info.prefetch_faults_enabled)
        parent_gpu->arch_hal->enable_prefetch_faults(parent_gpu);
    else
        parent_gpu->arch_hal->disable_prefetch_faults(parent_gpu);
}

// There is no error handling in this function. The caller is in charge of
// calling fault_buffer_deinit_replayable_faults on failure.
static NV_STATUS fault_buffer_init_replayable_faults(uvm_parent_gpu_t *parent_gpu)
{
    NV_STATUS status = NV_OK;
    uvm_replayable_fault_buffer_info_t *replayable_faults = &parent_gpu->fault_buffer_info.replayable;
    uvm_fault_service_batch_context_t *batch_context = &replayable_faults->batch_service_context;

    UVM_ASSERT(parent_gpu->fault_buffer_info.rm_info.replayable.bufferSize %
               parent_gpu->fault_buffer_hal->entry_size(parent_gpu) == 0);

    replayable_faults->max_faults = parent_gpu->fault_buffer_info.rm_info.replayable.bufferSize /
                                    parent_gpu->fault_buffer_hal->entry_size(parent_gpu);

    // Check provided module parameter value
    parent_gpu->fault_buffer_info.max_batch_size = max(uvm_perf_fault_batch_count,
                                                       (NvU32)UVM_PERF_FAULT_BATCH_COUNT_MIN);
    parent_gpu->fault_buffer_info.max_batch_size = min(parent_gpu->fault_buffer_info.max_batch_size,
                                                       replayable_faults->max_faults);

    if (parent_gpu->fault_buffer_info.max_batch_size != uvm_perf_fault_batch_count) {
        pr_info("Invalid uvm_perf_fault_batch_count value on GPU %s: %u. Valid range [%u:%u] Using %u instead\n",
                uvm_parent_gpu_name(parent_gpu),
                uvm_perf_fault_batch_count,
                UVM_PERF_FAULT_BATCH_COUNT_MIN,
                replayable_faults->max_faults,
                parent_gpu->fault_buffer_info.max_batch_size);
    }

    batch_context->fault_cache = uvm_kvmalloc_zero(replayable_faults->max_faults * sizeof(*batch_context->fault_cache));
    if (!batch_context->fault_cache)
        return NV_ERR_NO_MEMORY;

    // fault_cache is used to signal that the tracker was initialized.
    uvm_tracker_init(&replayable_faults->replay_tracker);

    batch_context->ordered_fault_cache = uvm_kvmalloc_zero(replayable_faults->max_faults *
                                                           sizeof(*batch_context->ordered_fault_cache));
    if (!batch_context->ordered_fault_cache)
        return NV_ERR_NO_MEMORY;

    replayable_faults->prev_batch_keys = uvm_kvmalloc_zero(replayable_faults->max_faults *
                                                           sizeof(*replayable_faults->prev_batch_keys));
    if (!replayable_faults->prev_batch_keys)
        return NV_ERR_NO_MEMORY;

    replayable_faults->prev_fetch_keys = uvm_kvmalloc_zero(replayable_faults->max_faults *
                                                           sizeof(*replayable_faults->prev_fetch_keys));
    if (!replayable_faults->prev_fetch_keys)
        return NV_ERR_NO_MEMORY;

    // O1: size the mode-2 (hash_exact) table off the per-batch working set
    // (max_batch_size), not the full fault-buffer capacity (max_faults).
    // A batch never inserts more than max_batch_size keys, so a generous
    // load factor keeps collision-driven key loss rare while the whole
    // table stays well within L1/L2 (e.g. 256 batch size * 16 -> 4096
    // slots -> 64 KiB, instead of the old max_faults*2 sizing, which was
    // hundreds of KiB and >99% empty).
    replayable_faults->prev_fetch_exact_size =
        max((NvU32)64,
            (NvU32)roundup_pow_of_two(parent_gpu->fault_buffer_info.max_batch_size *
                                      max((NvU32)1, uvm_perf_fault_pred_exact_load_factor)));
    replayable_faults->prev_fetch_exact_mask = replayable_faults->prev_fetch_exact_size - 1;
    replayable_faults->prev_fetch_exact_generation = 1;
    replayable_faults->prev_fetch_exact_slots = uvm_kvmalloc_zero(replayable_faults->prev_fetch_exact_size *
                                                                   sizeof(*replayable_faults->prev_fetch_exact_slots));
    if (!replayable_faults->prev_fetch_exact_slots)
        return NV_ERR_NO_MEMORY;

    replayable_faults->prev_fetch_last_valid = false;
    replayable_faults->prev_fetch_adaptive_active = true;
    replayable_faults->prev_fetch_adaptive_cooldown_remaining = 0;

    replayable_faults->prev_fetch_tag_size = max((NvU32)64, (NvU32)roundup_pow_of_two(max((NvU32)1, replayable_faults->max_faults / 4)));
    replayable_faults->prev_fetch_tag_mask = replayable_faults->prev_fetch_tag_size - 1;
    replayable_faults->prev_fetch_tag_generation = 1;
    replayable_faults->prev_fetch_tag_table = uvm_kvmalloc_zero(replayable_faults->prev_fetch_tag_size *
                                                                sizeof(*replayable_faults->prev_fetch_tag_table));
    if (!replayable_faults->prev_fetch_tag_table)
        return NV_ERR_NO_MEMORY;

    // Mode 3 (hash_approx) only; independent of the mode-2 table above.
    replayable_faults->prev_fetch_hash_epochs = uvm_kvmalloc_zero(replayable_faults->prev_fetch_tag_size *
                                                                  sizeof(*replayable_faults->prev_fetch_hash_epochs));
    if (!replayable_faults->prev_fetch_hash_epochs)
        return NV_ERR_NO_MEMORY;

    // This value must be initialized by HAL
    UVM_ASSERT(replayable_faults->utlb_count > 0);

    batch_context->utlbs = uvm_kvmalloc_zero(replayable_faults->utlb_count * sizeof(*batch_context->utlbs));
    if (!batch_context->utlbs)
        return NV_ERR_NO_MEMORY;

    batch_context->max_utlb_id = 0;

    status = uvm_rm_locked_call(nvUvmInterfaceOwnPageFaultIntr(parent_gpu->rm_device, NV_TRUE));
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to take page fault ownership from RM: %s, GPU %s\n",
                      nvstatusToString(status),
                      uvm_parent_gpu_name(parent_gpu));
        return status;
    }

    replayable_faults->replay_policy = uvm_perf_fault_replay_policy < UVM_PERF_FAULT_REPLAY_POLICY_MAX?
                                           uvm_perf_fault_replay_policy:
                                           UVM_PERF_FAULT_REPLAY_POLICY_DEFAULT;

    if (replayable_faults->replay_policy != uvm_perf_fault_replay_policy) {
        pr_info("Invalid uvm_perf_fault_replay_policy value on GPU %s: %d. Using %d instead\n",
                uvm_parent_gpu_name(parent_gpu),
                uvm_perf_fault_replay_policy,
                replayable_faults->replay_policy);
    }

    replayable_faults->replay_update_put_ratio = min(uvm_perf_fault_replay_update_put_ratio, 100u);
    if (replayable_faults->replay_update_put_ratio != uvm_perf_fault_replay_update_put_ratio) {
        pr_info("Invalid uvm_perf_fault_replay_update_put_ratio value on GPU %s: %u. Using %u instead\n",
                uvm_parent_gpu_name(parent_gpu),
                uvm_perf_fault_replay_update_put_ratio,
                replayable_faults->replay_update_put_ratio);
    }

    // Re-enable fault prefetching just in case it was disabled in a previous run
    parent_gpu->fault_buffer_info.prefetch_faults_enabled = parent_gpu->prefetch_fault_supported;

    fault_buffer_reinit_replayable_faults(parent_gpu);

    return NV_OK;
}

static void fault_buffer_deinit_replayable_faults(uvm_parent_gpu_t *parent_gpu)
{
    uvm_replayable_fault_buffer_info_t *replayable_faults = &parent_gpu->fault_buffer_info.replayable;
    uvm_fault_service_batch_context_t *batch_context = &replayable_faults->batch_service_context;

    if (batch_context->fault_cache) {
        UVM_ASSERT(uvm_tracker_is_empty(&replayable_faults->replay_tracker));
        uvm_tracker_deinit(&replayable_faults->replay_tracker);
    }

    if (parent_gpu->fault_buffer_info.rm_info.faultBufferHandle) {
        // Re-enable prefetch faults in case we disabled them
        if (parent_gpu->prefetch_fault_supported && !parent_gpu->fault_buffer_info.prefetch_faults_enabled)
            parent_gpu->arch_hal->enable_prefetch_faults(parent_gpu);
    }

    uvm_kvfree(batch_context->fault_cache);
    uvm_kvfree(batch_context->ordered_fault_cache);
    uvm_kvfree(batch_context->utlbs);
    uvm_kvfree(replayable_faults->prev_batch_keys);
    uvm_kvfree(replayable_faults->prev_fetch_keys);
    uvm_kvfree(replayable_faults->prev_fetch_exact_slots);
    uvm_kvfree(replayable_faults->prev_fetch_hash_epochs);
    uvm_kvfree(replayable_faults->prev_fetch_tag_table);
    batch_context->fault_cache         = NULL;
    batch_context->ordered_fault_cache = NULL;
    batch_context->utlbs               = NULL;
    replayable_faults->prev_batch_keys = NULL;
    replayable_faults->prev_batch_key_count = 0;
    replayable_faults->last_batch_pred_candidate_pct = 0;
    replayable_faults->prev_fetch_keys = NULL;
    replayable_faults->prev_fetch_key_count = 0;
    replayable_faults->last_batch_fetch_skip_pct = 0;
    replayable_faults->prev_fetch_exact_slots = NULL;
    replayable_faults->prev_fetch_exact_size = 0;
    replayable_faults->prev_fetch_exact_mask = 0;
    replayable_faults->prev_fetch_exact_generation = 0;
    replayable_faults->prev_fetch_last_valid = false;
    replayable_faults->prev_fetch_adaptive_active = false;
    replayable_faults->prev_fetch_adaptive_cooldown_remaining = 0;
    replayable_faults->prev_fetch_hash_epochs = NULL;
    replayable_faults->prev_fetch_tag_table = NULL;
    replayable_faults->prev_fetch_tag_size = 0;
    replayable_faults->prev_fetch_tag_mask = 0;
    replayable_faults->prev_fetch_tag_generation = 0;
}

NV_STATUS uvm_parent_gpu_fault_buffer_init(uvm_parent_gpu_t *parent_gpu)
{
    NV_STATUS status = NV_OK;

    uvm_assert_mutex_locked(&g_uvm_global.global_lock);
    UVM_ASSERT(parent_gpu->replayable_faults_supported);

    status = uvm_rm_locked_call(nvUvmInterfaceInitFaultInfo(parent_gpu->rm_device,
                                                            &parent_gpu->fault_buffer_info.rm_info));
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to init fault buffer info from RM: %s, GPU %s\n",
                      nvstatusToString(status),
                      uvm_parent_gpu_name(parent_gpu));

        // nvUvmInterfaceInitFaultInfo may leave fields in rm_info populated
        // when it returns an error. Set the buffer handle to zero as it is
        // used by the deinitialization logic to determine if it was correctly
        // initialized.
        parent_gpu->fault_buffer_info.rm_info.faultBufferHandle = 0;
        goto fail;
    }

    status = fault_buffer_init_replayable_faults(parent_gpu);
    if (status != NV_OK)
        goto fail;

    if (parent_gpu->non_replayable_faults_supported) {
        status = uvm_parent_gpu_fault_buffer_init_non_replayable_faults(parent_gpu);
        if (status != NV_OK)
            goto fail;
    }

    return NV_OK;

fail:
    uvm_parent_gpu_fault_buffer_deinit(parent_gpu);

    return status;
}

// Reinitialize state relevant to replayable fault handling after returning
// from a power management cycle.
void uvm_parent_gpu_fault_buffer_resume(uvm_parent_gpu_t *parent_gpu)
{
    UVM_ASSERT(parent_gpu->replayable_faults_supported);

    fault_buffer_reinit_replayable_faults(parent_gpu);
}

void uvm_parent_gpu_fault_buffer_deinit(uvm_parent_gpu_t *parent_gpu)
{
    NV_STATUS status = NV_OK;

    uvm_assert_mutex_locked(&g_uvm_global.global_lock);

    if (parent_gpu->non_replayable_faults_supported)
        uvm_parent_gpu_fault_buffer_deinit_non_replayable_faults(parent_gpu);

    fault_buffer_deinit_replayable_faults(parent_gpu);

    if (parent_gpu->fault_buffer_info.rm_info.faultBufferHandle) {
        status = uvm_rm_locked_call(nvUvmInterfaceOwnPageFaultIntr(parent_gpu->rm_device, NV_FALSE));
        UVM_ASSERT(status == NV_OK);

        uvm_rm_locked_call_void(nvUvmInterfaceDestroyFaultInfo(parent_gpu->rm_device,
                                                               &parent_gpu->fault_buffer_info.rm_info));

        parent_gpu->fault_buffer_info.rm_info.faultBufferHandle = 0;
    }
}

bool uvm_parent_gpu_replayable_faults_pending(uvm_parent_gpu_t *parent_gpu)
{
    uvm_replayable_fault_buffer_info_t *replayable_faults = &parent_gpu->fault_buffer_info.replayable;

    UVM_ASSERT(parent_gpu->replayable_faults_supported);

    // Fast path 1: we left some faults unserviced in the buffer in the last pass
    if (replayable_faults->cached_get != replayable_faults->cached_put)
        return true;

    // Fast path 2: read the valid bit of the fault buffer entry pointed by the
    // cached get pointer
    if (!parent_gpu->fault_buffer_hal->entry_is_valid(parent_gpu, replayable_faults->cached_get)) {
        // Slow path: read the put pointer from the GPU register via BAR0
        // over PCIe
        replayable_faults->cached_put = parent_gpu->fault_buffer_hal->read_put(parent_gpu);

        // No interrupt pending
        if (replayable_faults->cached_get == replayable_faults->cached_put)
            return false;
    }

    return true;
}

// Push a fault cancel method on the given client. Any failure during this
// operation may lead to application hang (requiring manual Ctrl+C from the
// user) or system crash (requiring reboot).
// In that case we log an error message.
//
// gpc_id and client_id aren't used if global_cancel is true.
//
// This function acquires both the given tracker and the replay tracker
static NV_STATUS push_cancel_on_gpu(uvm_gpu_t *gpu,
                                    uvm_gpu_phys_address_t instance_ptr,
                                    bool global_cancel,
                                    NvU32 gpc_id,
                                    NvU32 client_id,
                                    uvm_tracker_t *tracker)
{
    NV_STATUS status;
    uvm_push_t push;
    uvm_replayable_fault_buffer_info_t *replayable_faults = &gpu->parent->fault_buffer_info.replayable;

    if (global_cancel) {
        status = uvm_push_begin_acquire(gpu->channel_manager,
                                        UVM_CHANNEL_TYPE_MEMOPS,
                                        &replayable_faults->replay_tracker,
                                        &push,
                                        "Cancel targeting instance_ptr {0x%llx:%s}\n",
                                        instance_ptr.address,
                                        uvm_aperture_string(instance_ptr.aperture));
    }
    else {
        status = uvm_push_begin_acquire(gpu->channel_manager,
                                        UVM_CHANNEL_TYPE_MEMOPS,
                                        &replayable_faults->replay_tracker,
                                        &push,
                                        "Cancel targeting instance_ptr {0x%llx:%s} gpc %u client %u\n",
                                        instance_ptr.address,
                                        uvm_aperture_string(instance_ptr.aperture),
                                        gpc_id,
                                        client_id);
    }

    UVM_ASSERT(status == NV_OK);
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to create push and acquire replay tracker before pushing cancel: %s, GPU %s\n",
                      nvstatusToString(status),
                      uvm_gpu_name(gpu));
        return status;
    }

    uvm_push_acquire_tracker(&push, tracker);

    if (global_cancel)
        gpu->parent->host_hal->cancel_faults_global(&push, instance_ptr);
     else
        gpu->parent->host_hal->cancel_faults_targeted(&push, instance_ptr, gpc_id, client_id);

    // We don't need to put the cancel in the GPU replay tracker since we wait
    // on it immediately.
    status = uvm_push_end_and_wait(&push);

    UVM_ASSERT(status == NV_OK);
    if (status != NV_OK)
        UVM_ERR_PRINT("Failed to wait for pushed cancel: %s, GPU %s\n", nvstatusToString(status), uvm_gpu_name(gpu));

    uvm_tracker_clear(&replayable_faults->replay_tracker);

    return status;
}

static NV_STATUS push_cancel_on_gpu_targeted(uvm_gpu_t *gpu,
                                             uvm_gpu_phys_address_t instance_ptr,
                                             NvU32 gpc_id,
                                             NvU32 client_id,
                                             uvm_tracker_t *tracker)
{
    return push_cancel_on_gpu(gpu, instance_ptr, false, gpc_id, client_id, tracker);
}

static NV_STATUS push_cancel_on_gpu_global(uvm_gpu_t *gpu, uvm_gpu_phys_address_t instance_ptr, uvm_tracker_t *tracker)
{
    UVM_ASSERT(!gpu->parent->smc.enabled);

    return push_cancel_on_gpu(gpu, instance_ptr, true, 0, 0, tracker);
}

// Volta implements a targeted VA fault cancel that simplifies the fault cancel
// process. You only need to specify the address, type, and mmu_engine_id for
// the access to be cancelled. Caller must hold the VA space lock for the access
// to be cancelled.
static NV_STATUS cancel_fault_precise_va(uvm_gpu_t *gpu,
                                         uvm_fault_buffer_entry_t *fault_entry,
                                         uvm_fault_cancel_va_mode_t cancel_va_mode)
{
    NV_STATUS status;
    uvm_gpu_va_space_t *gpu_va_space;
    uvm_gpu_phys_address_t pdb;
    uvm_push_t push;
    uvm_replayable_fault_buffer_info_t *replayable_faults = &gpu->parent->fault_buffer_info.replayable;
    NvU64 offset;

    UVM_ASSERT(gpu->parent->replayable_faults_supported);
    UVM_ASSERT(fault_entry->fatal_reason != UvmEventFatalReasonInvalid);
    UVM_ASSERT(!fault_entry->filtered);

    gpu_va_space = uvm_gpu_va_space_get_by_parent_gpu(fault_entry->va_space, gpu->parent);
    UVM_ASSERT(gpu_va_space);
    pdb = uvm_page_tree_pdb(&gpu_va_space->page_tables)->addr;

    // Record fatal fault event
    uvm_tools_record_gpu_fatal_fault(gpu->id, fault_entry->va_space, fault_entry, fault_entry->fatal_reason);

    status = uvm_push_begin_acquire(gpu->channel_manager,
                                    UVM_CHANNEL_TYPE_MEMOPS,
                                    &replayable_faults->replay_tracker,
                                    &push,
                                    "Precise cancel targeting PDB {0x%llx:%s} VA 0x%llx VEID %u with access type %s",
                                    pdb.address,
                                    uvm_aperture_string(pdb.aperture),
                                    fault_entry->fault_address,
                                    fault_entry->fault_source.ve_id,
                                    uvm_fault_access_type_string(fault_entry->fault_access_type));
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to create push and acquire replay tracker before pushing cancel: %s, GPU %s\n",
                      nvstatusToString(status),
                      uvm_gpu_name(gpu));
        return status;
    }

    // UVM aligns fault addresses to PAGE_SIZE as it is the smallest mapping
    // and coherence tracking granularity. However, the cancel method requires
    // the original address (4K-aligned) reported in the packet, which is lost
    // at this point. Since the access permissions are the same for the whole
    // 64K page, we issue a cancel per 4K range to make sure that the HW sees
    // the address reported in the packet.
    for (offset = 0; offset < PAGE_SIZE; offset += UVM_PAGE_SIZE_4K) {
        gpu->parent->host_hal->cancel_faults_va(&push, pdb, fault_entry, cancel_va_mode);
        fault_entry->fault_address += UVM_PAGE_SIZE_4K;
    }
    fault_entry->fault_address = UVM_PAGE_ALIGN_DOWN(fault_entry->fault_address - 1);

    // We don't need to put the cancel in the GPU replay tracker since we wait
    // on it immediately.
    status = uvm_push_end_and_wait(&push);
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to wait for pushed VA global fault cancel: %s, GPU %s\n",
                      nvstatusToString(status), uvm_gpu_name(gpu));
    }

    uvm_tracker_clear(&replayable_faults->replay_tracker);

    return status;
}

static NV_STATUS push_replay_on_gpu(uvm_gpu_t *gpu,
                                    uvm_fault_replay_type_t type,
                                    uvm_fault_service_batch_context_t *batch_context)
{
    NV_STATUS status;
    uvm_push_t push;
    uvm_replayable_fault_buffer_info_t *replayable_faults = &gpu->parent->fault_buffer_info.replayable;
    uvm_tracker_t *tracker = NULL;

    if (batch_context)
        tracker = &batch_context->tracker;

    status = uvm_push_begin_acquire(gpu->channel_manager, UVM_CHANNEL_TYPE_MEMOPS, tracker, &push,
                                    "Replaying faults");
    if (status != NV_OK)
        return status;

    gpu->parent->host_hal->replay_faults(&push, type);

    // Do not count REPLAY_TYPE_START_ACK_ALL's toward the replay count.
    // REPLAY_TYPE_START_ACK_ALL's are issued for cancels, and the cancel
    // algorithm checks to make sure that no REPLAY_TYPE_START's have been
    // issued using batch_context->replays.
    if (batch_context && type != UVM_FAULT_REPLAY_TYPE_START_ACK_ALL) {
        uvm_tools_broadcast_replay(gpu, &push, batch_context->batch_id, UVM_FAULT_CLIENT_TYPE_GPC);
        ++batch_context->num_replays;
    }

    uvm_push_end(&push);

    // Add this push to the GPU's replay_tracker so cancel can wait on it.
    status = uvm_tracker_add_push_safe(&replayable_faults->replay_tracker, &push);

    if (uvm_procfs_is_debug_enabled()) {
        if (type == UVM_FAULT_REPLAY_TYPE_START)
            ++replayable_faults->stats.num_replays;
        else
            ++replayable_faults->stats.num_replays_ack_all;
    }

    return status;
}

static void write_get(uvm_parent_gpu_t *parent_gpu, NvU32 get)
{
    uvm_replayable_fault_buffer_info_t *replayable_faults = &parent_gpu->fault_buffer_info.replayable;

    UVM_ASSERT(uvm_sem_is_locked(&parent_gpu->isr.replayable_faults.service_lock));

    // Write get on the GPU only if it's changed.
    if (replayable_faults->cached_get == get)
        return;

    replayable_faults->cached_get = get;

    // Update get pointer on the GPU
    parent_gpu->fault_buffer_hal->write_get(parent_gpu, get);
}

// In Confidential Computing GSP-RM owns the HW replayable fault buffer.
// Flushing the fault buffer implies flushing both the HW buffer (using a RM
// API), and the SW buffer accessible by UVM ("shadow" buffer).
//
// The HW buffer needs to be flushed first. This is because, once that flush
// completes, any faults that were present in the HW buffer have been moved to
// the shadow buffer, or have been discarded by RM.
static NV_STATUS hw_fault_buffer_flush_locked(uvm_parent_gpu_t *parent_gpu, hw_fault_buffer_flush_mode_t flush_mode)
{
    NV_STATUS status;
    NvBool is_flush_mode_move;

    UVM_ASSERT(uvm_sem_is_locked(&parent_gpu->isr.replayable_faults.service_lock));
    UVM_ASSERT((flush_mode == HW_FAULT_BUFFER_FLUSH_MODE_MOVE) || (flush_mode == HW_FAULT_BUFFER_FLUSH_MODE_DISCARD));

    if (!g_uvm_global.conf_computing_enabled)
        return NV_OK;

    is_flush_mode_move = (NvBool) (flush_mode == HW_FAULT_BUFFER_FLUSH_MODE_MOVE);
    status = nvUvmInterfaceFlushReplayableFaultBuffer(&parent_gpu->fault_buffer_info.rm_info, is_flush_mode_move);

    UVM_ASSERT(status == NV_OK);

    return status;
}

static NvU32 replayable_fault_buffer_pending_entries(NvU32 get, NvU32 put, NvU32 max_faults)
{
    if (put >= get)
        return put - get;

    return (max_faults - get) + put;
}

static bool fault_addr_in_tracked_range(NvU64 fault_address)
{
    if (!uvm_fault_track_range_enable)
        return false;

    if (uvm_fault_track_range_start >= uvm_fault_track_range_end)
        return false;

    return fault_address >= uvm_fault_track_range_start &&
           fault_address < uvm_fault_track_range_end;
}

#define TRACKED_STALE_LAG_MAP_SIZE 4096
#define TRACKED_STALE_LAG_MAP_PROBES 8

typedef struct
{
    NvU64 page_address;
    NvU32 last_service_batch_id;
} tracked_stale_lag_entry_t;

// Best-effort lag map for motivation profiling.
// Collisions/evictions are acceptable for coarse lag buckets.
static tracked_stale_lag_entry_t g_tracked_stale_lag_map[TRACKED_STALE_LAG_MAP_SIZE];

static NvU32 tracked_stale_lag_hash(NvU64 page_address)
{
    return (NvU32)((page_address >> 12) & (TRACKED_STALE_LAG_MAP_SIZE - 1));
}

static NvU32 tracked_stale_prev_lag_get(NvU64 fault_address, NvU32 current_batch_id)
{
    NvU64 page_address = UVM_PAGE_ALIGN_DOWN(fault_address);
    NvU32 slot = tracked_stale_lag_hash(page_address);
    NvU32 probe;

    for (probe = 0; probe < TRACKED_STALE_LAG_MAP_PROBES; ++probe) {
        tracked_stale_lag_entry_t *entry = &g_tracked_stale_lag_map[(slot + probe) & (TRACKED_STALE_LAG_MAP_SIZE - 1)];

        if (entry->last_service_batch_id == 0)
            return 0;

        if (entry->page_address == page_address) {
            if (current_batch_id > entry->last_service_batch_id)
                return current_batch_id - entry->last_service_batch_id;

            return 0;
        }
    }

    return 0;
}

static void tracked_stale_prev_lag_record_service(NvU64 fault_address, NvU32 batch_id)
{
    NvU64 page_address = UVM_PAGE_ALIGN_DOWN(fault_address);
    NvU32 slot = tracked_stale_lag_hash(page_address);
    NvU32 probe;
    NvU32 victim_probe = 0;

    for (probe = 0; probe < TRACKED_STALE_LAG_MAP_PROBES; ++probe) {
        tracked_stale_lag_entry_t *entry = &g_tracked_stale_lag_map[(slot + probe) & (TRACKED_STALE_LAG_MAP_SIZE - 1)];

        if (entry->last_service_batch_id == 0 || entry->page_address == page_address) {
            entry->page_address = page_address;
            entry->last_service_batch_id = batch_id;
            return;
        }

        if (entry->last_service_batch_id < g_tracked_stale_lag_map[(slot + victim_probe) & (TRACKED_STALE_LAG_MAP_SIZE - 1)].last_service_batch_id)
            victim_probe = probe;
    }

    g_tracked_stale_lag_map[(slot + victim_probe) & (TRACKED_STALE_LAG_MAP_SIZE - 1)].page_address = page_address;
    g_tracked_stale_lag_map[(slot + victim_probe) & (TRACKED_STALE_LAG_MAP_SIZE - 1)].last_service_batch_id = batch_id;
}

static void fault_buffer_skip_replayable_entry(uvm_parent_gpu_t *parent_gpu, NvU32 index)
{
    UVM_ASSERT(parent_gpu->fault_buffer_hal->entry_is_valid(parent_gpu, index));

    // Flushed faults are never decrypted, but the decryption IV associated with
    // replayable faults still requires manual adjustment so it is kept in sync
    // with the encryption IV on the GSP-RM's side.
    if (g_uvm_global.conf_computing_enabled)
        uvm_conf_computing_fault_increment_decrypt_iv(parent_gpu, 1);

    parent_gpu->fault_buffer_hal->entry_clear_valid(parent_gpu, index);
}

static NV_STATUS fault_buffer_flush_locked(uvm_gpu_t *gpu,
                                           uvm_gpu_buffer_flush_mode_t flush_mode,
                                           uvm_fault_replay_type_t fault_replay,
                                           uvm_fault_service_batch_context_t *batch_context)
{
    NvU32 get;
    NvU32 put;
    uvm_spin_loop_t spin;
    uvm_parent_gpu_t *parent_gpu = gpu->parent;
    uvm_replayable_fault_buffer_info_t *replayable_faults = &parent_gpu->fault_buffer_info.replayable;
    NV_STATUS status;

    UVM_ASSERT(uvm_sem_is_locked(&parent_gpu->isr.replayable_faults.service_lock));
    UVM_ASSERT(parent_gpu->replayable_faults_supported);

    // Wait for the prior replay to flush out old fault messages
    if (flush_mode == UVM_GPU_BUFFER_FLUSH_MODE_WAIT_UPDATE_PUT) {
        status = uvm_tracker_wait(&replayable_faults->replay_tracker);
        if (status != NV_OK)
            return status;
    }

    // Read PUT pointer from the GPU if requested
    if (flush_mode == UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT || flush_mode == UVM_GPU_BUFFER_FLUSH_MODE_WAIT_UPDATE_PUT) {
        status = hw_fault_buffer_flush_locked(parent_gpu, HW_FAULT_BUFFER_FLUSH_MODE_DISCARD);
        if (status != NV_OK)
            return status;
        replayable_faults->cached_put = parent_gpu->fault_buffer_hal->read_put(parent_gpu);
    }

    get = replayable_faults->cached_get;
    put = replayable_faults->cached_put;

    if (uvm_page_rep_profile_enable &&
        batch_context != NULL &&
        fault_replay == UVM_FAULT_REPLAY_TYPE_START) {
        NvU32 latest_put_for_profile = parent_gpu->fault_buffer_hal->read_put(parent_gpu);
        NvU32 pending_before_replay = replayable_fault_buffer_pending_entries(get,
                                                                              latest_put_for_profile,
                                                                              replayable_faults->max_faults);

        pr_info("[REPLAY_BUF] batch_id=%llu pending_before_replay=%u cached=%u "
                "coalesced=%u get=%u cached_put=%u put=%u max=%u flush_mode=%u replay_type=%u\n",
                (unsigned long long)batch_context->batch_id,
                pending_before_replay,
                batch_context->num_cached_faults,
                batch_context->num_coalesced_faults,
                get,
                put,
                latest_put_for_profile,
                replayable_faults->max_faults,
                flush_mode,
                fault_replay);
    }

    while (get != put) {
        // Wait until valid bit is set
        UVM_SPIN_WHILE(!parent_gpu->fault_buffer_hal->entry_is_valid(parent_gpu, get), &spin);

        fault_buffer_skip_replayable_entry(parent_gpu, get);
        ++get;
        if (get == replayable_faults->max_faults)
            get = 0;
    }

    write_get(gpu->parent, get);

    // Issue fault replay
    return push_replay_on_gpu(gpu, fault_replay, batch_context);
}

NV_STATUS uvm_gpu_fault_buffer_flush(uvm_gpu_t *gpu)
{
    NV_STATUS status = NV_OK;

    UVM_ASSERT(gpu->parent->replayable_faults_supported);

    // Disables replayable fault interrupts and fault servicing
    uvm_parent_gpu_replayable_faults_isr_lock(gpu->parent);

    status = fault_buffer_flush_locked(gpu,
                                       UVM_GPU_BUFFER_FLUSH_MODE_WAIT_UPDATE_PUT,
                                       UVM_FAULT_REPLAY_TYPE_START,
                                       NULL);

    // This will trigger the top half to start servicing faults again, if the
    // replay brought any back in
    uvm_parent_gpu_replayable_faults_isr_unlock(gpu->parent);
    return status;
}

static inline int cmp_fault_instance_ptr(const uvm_fault_buffer_entry_t *a,
                                         const uvm_fault_buffer_entry_t *b)
{
    int result = uvm_gpu_phys_addr_cmp(a->instance_ptr, b->instance_ptr);
    // On Volta+ we need to sort by {instance_ptr + subctx_id} pair since it can
    // map to a different VA space
    if (result != 0)
        return result;
    return UVM_CMP_DEFAULT(a->fault_source.ve_id, b->fault_source.ve_id);
}

// Compare two VA spaces
static inline int cmp_va_space(const uvm_va_space_t *a, const uvm_va_space_t *b)
{
    return UVM_CMP_DEFAULT(a, b);
}

// Compare two virtual addresses
static inline int cmp_addr(NvU64 a, NvU64 b)
{
    return UVM_CMP_DEFAULT(a, b);
}

static int cmp_fault_addr_sort(const void *a, const void *b)
{
    NvU64 va = *(const NvU64 *)a;
    NvU64 vb = *(const NvU64 *)b;

    return UVM_CMP_DEFAULT(va, vb);
}

// Compare two fault access types
static inline int cmp_access_type(uvm_fault_access_type_t a, uvm_fault_access_type_t b)
{
    UVM_ASSERT(a >= 0 && a < UVM_FAULT_ACCESS_TYPE_COUNT);
    UVM_ASSERT(b >= 0 && b < UVM_FAULT_ACCESS_TYPE_COUNT);

    // Check that fault access type enum values are ordered by "intrusiveness"
    BUILD_BUG_ON(UVM_FAULT_ACCESS_TYPE_ATOMIC_STRONG <= UVM_FAULT_ACCESS_TYPE_ATOMIC_WEAK);
    BUILD_BUG_ON(UVM_FAULT_ACCESS_TYPE_ATOMIC_WEAK <= UVM_FAULT_ACCESS_TYPE_WRITE);
    BUILD_BUG_ON(UVM_FAULT_ACCESS_TYPE_WRITE <= UVM_FAULT_ACCESS_TYPE_READ);
    BUILD_BUG_ON(UVM_FAULT_ACCESS_TYPE_READ <= UVM_FAULT_ACCESS_TYPE_PREFETCH);

    return b - a;
}

static int cmp_fault_entry_addr_access_sort(const void *a, const void *b)
{
    const uvm_fault_buffer_entry_t *const *entry_a = a;
    const uvm_fault_buffer_entry_t *const *entry_b = b;
    int result;

    result = cmp_va_space((*entry_a)->va_space, (*entry_b)->va_space);
    if (result != 0)
        return result;

    result = cmp_addr((*entry_a)->fault_address, (*entry_b)->fault_address);
    if (result != 0)
        return result;

    return cmp_access_type((*entry_a)->fault_access_type, (*entry_b)->fault_access_type);
}

typedef enum
{
    // Fetch a batch of faults from the buffer. Stop at the first entry that is
    // not ready yet
    FAULT_FETCH_MODE_BATCH_READY,

    // Fetch all faults in the buffer before PUT. Wait for all faults to become
    // ready
    FAULT_FETCH_MODE_ALL,
} fault_fetch_mode_t;

static bool prev_fetch_predictor_use_coalesced_pages(void);
static bool pred_timing_enabled(void);

// O4: one Fibonacci-style multiplicative mix (1 multiply + 2 xor-shifts)
// instead of the 3-round MurmurHash3 finalizer (3 multiplies + 3
// xor-shifts). This is a private, best-effort staleness heuristic on the
// BackLib fault-handling hot path, not a hash table exposed to adversarial
// keys, and O1 already keeps the tables small (thousands of slots at
// most), so the extra mixing rounds bought little beyond CPU cycles.
static NvU64 prev_fetch_hash_mix(NvU64 fault_address)
{
    NvU64 x = fault_address >> 12;

    x ^= x >> 32;
    x *= 0x9E3779B97F4A7C15ULL;
    x ^= x >> 29;

    return x;
}

static NvU32 prev_fetch_hash_index(NvU64 fault_address, NvU32 mask)
{
    return (NvU32)(prev_fetch_hash_mix(fault_address) & mask);
}

static NvU16 prev_fetch_tag_value(NvU64 fault_address)
{
    NvU16 tag = (NvU16)((prev_fetch_hash_mix(fault_address) >> 16) & 0xffff);

    return tag == 0 ? 1 : tag;
}

static bool prev_fetch_predictor_use_hash_exact(void)
{
    return uvm_perf_fault_prev_fetch_mode == 2;
}

static bool prev_fetch_predictor_use_hash_approx(void)
{
    return uvm_perf_fault_prev_fetch_mode == 3;
}

static bool prev_fetch_predictor_use_coalesced_addr_bsearch(void)
{
    return uvm_perf_fault_prev_fetch_mode == 1;
}

static int cmp_prev_fetch_key(const uvm_fault_prev_fetch_key_t *a, const uvm_fault_prev_fetch_key_t *b)
{
    if (prev_fetch_predictor_use_coalesced_addr_bsearch())
        return cmp_addr(a->fault_address, b->fault_address);

    int result = uvm_gpu_phys_addr_cmp(a->instance_ptr, b->instance_ptr);

    if (result != 0)
        return result;

    result = UVM_CMP_DEFAULT(a->ve_id, b->ve_id);
    if (result != 0)
        return result;

    return cmp_addr(a->fault_address, b->fault_address);
}

static int cmp_prev_fetch_key_sort(const void *_a, const void *_b)
{
    const uvm_fault_prev_fetch_key_t *a = _a;
    const uvm_fault_prev_fetch_key_t *b = _b;

    return cmp_prev_fetch_key(a, b);
}

// P3: should THIS call be the one that gets ktime_get()-timed? Free-running
// counter, so samples land evenly across the whole run rather than only the
// first calls of each batch. rate<=1 times every call (original behavior).
static bool pred_timing_sample_due(uvm_fault_service_batch_context_t *batch_context)
{
    unsigned rate = uvm_perf_fault_pred_timing_sample_rate;

    if (rate <= 1)
        return true;

    return (batch_context->pred_timing_sample_counter++ % rate) == 0;
}

static void record_prev_fetch_lookup_stats(uvm_fault_service_batch_context_t *batch_context,
                                           NvU32 cmp_count,
                                           bool hit,
                                           ktime_t start_time,
                                           bool timed)
{
    // O5: these fields only ever feed the [STALE_PROF]/[FM_UVM_STAGES]
    // debug breakdowns (gated on uvm_page_rep_profile_enable /
    // uvm_fault_stage_profile_enable elsewhere), never a scheduling
    // decision. Skip the bookkeeping entirely on the production path so a
    // disabled profiler costs one branch instead of three increments plus
    // a conditional ktime_get() diff on every single lookup.
    if (!pred_timing_enabled())
        return;

    // P3 must sample the bookkeeping as well as the clock reads. Updating
    // these counters on every lookup still cost roughly 16 ns/call in the
    // sample_rate=64 experiment, inflating handler time by 10-11 ms even
    // though ktime_get() itself was sampled. These fields are profiling-only
    // estimates, so update them on sampled calls and scale by the rate.
    if (timed) {
        NvU64 delta_ns = (NvU64)ktime_to_ns(ktime_sub(ktime_get(), start_time));
        NvU32 rate = max((NvU32)1, uvm_perf_fault_pred_timing_sample_rate);
        NvU64 bias_ns = uvm_perf_fault_pred_timing_overhead_ns;

        batch_context->num_pred_fetch_lookup_calls += rate;
        batch_context->num_pred_fetch_lookup_cmps += cmp_count * rate;
        if (hit)
            batch_context->num_pred_fetch_lookup_hits += rate;

        // P4: remove the ktime_get() bias carried by this sample. The clamp
        // is a real limitation, not a formality: when the body costs less
        // than the clock read (which is the case here), individual samples
        // land at or below the bias and get truncated to zero, so the
        // accumulated total skews high. num_pred_fetch_lookup_unresolved
        // makes that visible instead of silent.
        if (delta_ns > bias_ns) {
            delta_ns -= bias_ns;
        }
        else {
            delta_ns = 0;
            batch_context->num_pred_fetch_lookup_unresolved++;
        }

        // P3: scale the sampled cost back up by the sampling rate so the
        // accumulated total remains an "as if every call were timed"
        // estimate, keeping the [FM_UVM_STAGES] field and downstream CSV
        // parsing unchanged while paying ktime_get() only on 1/rate calls.
        batch_context->time_pred_fetch_lookup_ns += delta_ns * rate;
    }
}

// O3: does `entry` match the single-slot memo left by the immediately
// preceding call in this batch? The memo key is the full mode-1 triple, so
// a match guarantees the address (and instance_ptr/ve_id, if relevant)
// are identical to the previous call; since no predictor state changes
// between two calls within the same fetch loop, reusing the cached result
// is exact, not an approximation, for all three modes.
static bool prev_fetch_last_matches(const uvm_replayable_fault_buffer_info_t *replayable_faults,
                                    const uvm_fault_buffer_entry_t *entry)
{
    const uvm_fault_prev_fetch_key_t *last = &replayable_faults->prev_fetch_last_key;

    return replayable_faults->prev_fetch_last_valid &&
          last->fault_address == entry->fault_address &&
          last->ve_id == entry->fault_source.ve_id &&
          uvm_gpu_phys_addr_cmp(last->instance_ptr, entry->instance_ptr) == 0;
}

static void prev_fetch_last_update(uvm_replayable_fault_buffer_info_t *replayable_faults,
                                   const uvm_fault_buffer_entry_t *entry,
                                   bool hit)
{
    replayable_faults->prev_fetch_last_key.instance_ptr = entry->instance_ptr;
    replayable_faults->prev_fetch_last_key.ve_id = entry->fault_source.ve_id;
    replayable_faults->prev_fetch_last_key.fault_address = entry->fault_address;
    replayable_faults->prev_fetch_last_hit = hit;
    replayable_faults->prev_fetch_last_valid = true;
}

// P0: kept non-inlined (negligible cost: a handful of cycles per fault,
// versus tens of ns of memory-access cost this function is measuring) so
// `perf record`/`perf report` can attribute cycles/cache-misses to this
// function by name instead of folding them into whichever caller GCC
// inlines it into.
static noinline bool prev_fetch_keys_contains(uvm_replayable_fault_buffer_info_t *replayable_faults,
                                     uvm_fault_service_batch_context_t *batch_context,
                                     const uvm_fault_buffer_entry_t *entry)
{
    bool hit;

    if (prev_fetch_last_matches(replayable_faults, entry)) {
        // Zero comparisons charged: this call did no table/array access at
        // all, only the memo-field compares already folded into
        // prev_fetch_last_matches(). Timing is still recorded on the
        // profiling build so the CSV shows these as (near-)zero-cost hits
        // rather than silently vanishing from the lookup-call count.
        ktime_t start_time = 0;
        bool timed = false;

        if (pred_timing_enabled() && pred_timing_sample_due(batch_context)) {
            start_time = ktime_get();
            timed = true;
        }
        hit = replayable_faults->prev_fetch_last_hit;
        record_prev_fetch_lookup_stats(batch_context, 0, hit, start_time, timed);
        return hit;
    }

    if (prev_fetch_predictor_use_hash_exact()) {
        NvU32 index;
        ktime_t start_time = 0;
        bool timed = false;

        if (pred_timing_enabled() && pred_timing_sample_due(batch_context)) {
            start_time = ktime_get();
            timed = true;
        }

        // O1b (bounded linear probing) was tried and reverted: with O1's
        // load_factor=16 sizing, measured skip-rate was statistically
        // identical with or without probing (collisions are already rare
        // enough that probing essentially never fires), but probing forces
        // every *build* to read-before-write instead of blind-write, which
        // multi-trial A/B showed as a statistically real regression in
        // filter time. Plain direct-mapped lookup/build, same as O1+O2.
        index = prev_fetch_hash_index(entry->fault_address, replayable_faults->prev_fetch_exact_mask);
        hit = replayable_faults->prev_fetch_exact_slots[index].epoch == replayable_faults->prev_fetch_exact_generation &&
              replayable_faults->prev_fetch_exact_slots[index].addr == entry->fault_address;
        record_prev_fetch_lookup_stats(batch_context, 1, hit, start_time, timed);
        prev_fetch_last_update(replayable_faults, entry, hit);
        return hit;
    }

    if (prev_fetch_predictor_use_hash_approx()) {
        NvU32 index;
        ktime_t start_time = 0;
        bool timed = false;

        if (pred_timing_enabled() && pred_timing_sample_due(batch_context)) {
            start_time = ktime_get();
            timed = true;
        }

        index = prev_fetch_hash_index(entry->fault_address, replayable_faults->prev_fetch_tag_mask);
        hit = replayable_faults->prev_fetch_hash_epochs[index] == replayable_faults->prev_fetch_tag_generation &&
              replayable_faults->prev_fetch_tag_table[index] == prev_fetch_tag_value(entry->fault_address);
        record_prev_fetch_lookup_stats(batch_context, 1, hit, start_time, timed);
        prev_fetch_last_update(replayable_faults, entry, hit);
        return hit;
    }

    {
        NvU32 left = 0;
        NvU32 right = replayable_faults->prev_fetch_key_count;
        uvm_fault_prev_fetch_key_t needle;
        NvU32 cmp_count = 0;
        ktime_t start_time = 0;
        bool timed = false;

        if (pred_timing_enabled() && pred_timing_sample_due(batch_context)) {
            start_time = ktime_get();
            timed = true;
        }

        needle.instance_ptr = entry->instance_ptr;
        needle.ve_id = entry->fault_source.ve_id;
        needle.fault_address = entry->fault_address;

        while (left < right) {
            NvU32 mid = left + ((right - left) / 2);
            int cmp = cmp_prev_fetch_key(&needle, &replayable_faults->prev_fetch_keys[mid]);
            ++cmp_count;

            if (cmp == 0) {
                record_prev_fetch_lookup_stats(batch_context, cmp_count, true, start_time, timed);
                prev_fetch_last_update(replayable_faults, entry, true);
                return true;
            }

            if (cmp < 0)
                right = mid;
            else
                left = mid + 1;
        }

        record_prev_fetch_lookup_stats(batch_context, cmp_count, false, start_time, timed);
        prev_fetch_last_update(replayable_faults, entry, false);
        return false;
    }
}

static bool prev_fetch_predictor_enabled(void)
{
    return uvm_perf_fault_prev_fetch_predictor_enable != 0;
}

// O6: should the filter (lookup this batch / build for this batch) actually
// run? Always true unless adaptive gating is on and the backlog-detection
// state machine has decided this batch shows no sign of a backlog. See the
// state-update logic at the end of fetch_fault_buffer_entries().
static bool prev_fetch_adaptive_should_run(const uvm_replayable_fault_buffer_info_t *replayable_faults)
{
    return !uvm_perf_fault_pred_adaptive_enable || replayable_faults->prev_fetch_adaptive_active;
}

static bool prev_fetch_predictor_use_coalesced_pages(void)
{
    return uvm_perf_fault_prev_fetch_mode >= 1;
}

static bool prev_batch_predictor_enabled(void)
{
    return uvm_perf_fault_prev_batch_predictor_enable != 0;
}

static bool stale_detail_enabled(void)
{
    return uvm_perf_fault_stale_detail_enable != 0;
}

static bool pred_timing_enabled(void)
{
    return uvm_perf_fault_pred_timing_enable != 0;
}

static void finalize_prev_fetch_predictor(uvm_replayable_fault_buffer_info_t *replayable_faults,
                                          uvm_fault_service_batch_context_t *batch_context,
                                          NvU32 key_count,
                                          ktime_t start_time)
{
    NvU32 effective_count = 0;

    replayable_faults->prev_fetch_key_count = key_count;
    batch_context->num_pred_fetch_build_keys = key_count;

    effective_count = (prev_fetch_predictor_use_coalesced_pages() ?
                       batch_context->num_coalesced_faults :
                       batch_context->num_cached_faults) +
                      batch_context->num_pred_fast_skip_stale;
    replayable_faults->last_batch_fetch_skip_pct =
        effective_count == 0 ? 0 :
        min((batch_context->num_pred_fast_skip_stale * 100) / effective_count, 100u);

    if (pred_timing_enabled())
        batch_context->time_pred_fetch_build_ns += ktime_to_ns(ktime_sub(ktime_get(), start_time));
}

static void advance_prev_fetch_generation(NvU32 *generation, NvU32 *epochs, NvU32 size)
{
    ++(*generation);
    if (*generation == 0) {
        memset(epochs, 0, size * sizeof(*epochs));
        *generation = 1;
    }
}

// Mode 2 (hash_exact) equivalent of advance_prev_fetch_generation() for the
// packed uvm_fault_prev_fetch_slot_t table (O1+O2). Only touches every slot
// on the (extremely rare, ~2^32 batches) generation wraparound.
static void advance_prev_fetch_exact_generation(uvm_replayable_fault_buffer_info_t *replayable_faults)
{
    ++replayable_faults->prev_fetch_exact_generation;
    if (replayable_faults->prev_fetch_exact_generation == 0) {
        NvU32 i;

        for (i = 0; i < replayable_faults->prev_fetch_exact_size; ++i)
            replayable_faults->prev_fetch_exact_slots[i].epoch = 0;
        replayable_faults->prev_fetch_exact_generation = 1;
    }
}

static void update_prev_fetch_predictor_from_fetch_cache(uvm_replayable_fault_buffer_info_t *replayable_faults,
                                                         uvm_fault_service_batch_context_t *batch_context)
{
    NvU32 i;
    NvU32 key_count = 0;
    ktime_t start_time = 0;

    if (pred_timing_enabled())
        start_time = ktime_get();

    for (i = 0; i < batch_context->num_cached_faults; ++i) {
        uvm_fault_buffer_entry_t *entry = &batch_context->fault_cache[i];

        if (entry->filtered && !entry->pred_fetch_skip)
            continue;

        replayable_faults->prev_fetch_keys[key_count].instance_ptr = entry->instance_ptr;
        replayable_faults->prev_fetch_keys[key_count].ve_id = entry->fault_source.ve_id;
        replayable_faults->prev_fetch_keys[key_count].fault_address = entry->fault_address;
        ++key_count;
    }

    if (key_count > 1)
        sort(replayable_faults->prev_fetch_keys,
             key_count,
             sizeof(*replayable_faults->prev_fetch_keys),
             cmp_prev_fetch_key_sort,
             NULL);

    if (key_count > 0) {
        NvU32 unique = 1;
        for (i = 1; i < key_count; ++i) {
            if (cmp_prev_fetch_key(&replayable_faults->prev_fetch_keys[i],
                                   &replayable_faults->prev_fetch_keys[unique - 1]) != 0) {
                replayable_faults->prev_fetch_keys[unique++] = replayable_faults->prev_fetch_keys[i];
            }
        }
        key_count = unique;
    }

    finalize_prev_fetch_predictor(replayable_faults, batch_context, key_count, start_time);
}

static void update_prev_fetch_predictor_from_coalesced_bsearch(uvm_replayable_fault_buffer_info_t *replayable_faults,
                                                               uvm_fault_service_batch_context_t *batch_context)
{
    NvU32 i;
    NvU32 key_count = 0;
    bool single_va_space = true;
    uvm_va_space_t *first_va_space = NULL;
    ktime_t start_time = 0;

    if (pred_timing_enabled())
        start_time = ktime_get();

    for (i = 0; i < batch_context->num_coalesced_faults; ++i) {
        uvm_fault_buffer_entry_t *entry = batch_context->ordered_fault_cache[i];

        if (i == 0)
            first_va_space = entry->va_space;
        else if (entry->va_space != first_va_space)
            single_va_space = false;

        if (single_va_space &&
            key_count > 0 &&
            replayable_faults->prev_fetch_keys[key_count - 1].fault_address == entry->fault_address) {
            continue;
        }

        replayable_faults->prev_fetch_keys[key_count].instance_ptr.address = 0;
        replayable_faults->prev_fetch_keys[key_count].ve_id = 0;
        replayable_faults->prev_fetch_keys[key_count].fault_address = entry->fault_address;
        ++key_count;
    }

    if (!single_va_space && key_count > 1) {
        NvU32 unique = 1;

        sort(replayable_faults->prev_fetch_keys,
             key_count,
             sizeof(*replayable_faults->prev_fetch_keys),
             cmp_prev_fetch_key_sort,
             NULL);

        for (i = 1; i < key_count; ++i) {
            if (cmp_prev_fetch_key(&replayable_faults->prev_fetch_keys[i],
                                   &replayable_faults->prev_fetch_keys[unique - 1]) != 0) {
                replayable_faults->prev_fetch_keys[unique++] = replayable_faults->prev_fetch_keys[i];
            }
        }
        key_count = unique;
    }

    finalize_prev_fetch_predictor(replayable_faults, batch_context, key_count, start_time);
}

static void update_prev_fetch_predictor_from_coalesced_hash_exact(uvm_replayable_fault_buffer_info_t *replayable_faults,
                                                                  uvm_fault_service_batch_context_t *batch_context)
{
    NvU32 i;
    NvU32 key_count = 0;
    ktime_t start_time = 0;

    if (pred_timing_enabled())
        start_time = ktime_get();

    advance_prev_fetch_exact_generation(replayable_faults);

    for (i = 0; i < batch_context->num_coalesced_faults; ++i) {
        uvm_fault_buffer_entry_t *entry = batch_context->ordered_fault_cache[i];
        NvU32 index = prev_fetch_hash_index(entry->fault_address, replayable_faults->prev_fetch_exact_mask);

        // O2: address + validity epoch share one slot/cache line. Blind
        // write, no read-before-write probe (see O1b revert note above the
        // hash_exact lookup branch): with O1's load_factor=16, probing
        // measurably slowed down every build for no measurable skip-rate
        // gain.
        replayable_faults->prev_fetch_exact_slots[index].epoch = replayable_faults->prev_fetch_exact_generation;
        replayable_faults->prev_fetch_exact_slots[index].addr = entry->fault_address;
        ++key_count;
    }

    finalize_prev_fetch_predictor(replayable_faults, batch_context, key_count, start_time);
}

static void update_prev_fetch_predictor_from_coalesced_hash_approx(uvm_replayable_fault_buffer_info_t *replayable_faults,
                                                                   uvm_fault_service_batch_context_t *batch_context)
{
    NvU32 i;
    NvU32 key_count = 0;
    ktime_t start_time = 0;

    if (pred_timing_enabled())
        start_time = ktime_get();

    advance_prev_fetch_generation(&replayable_faults->prev_fetch_tag_generation,
                                  replayable_faults->prev_fetch_hash_epochs,
                                  replayable_faults->prev_fetch_tag_size);

    for (i = 0; i < batch_context->num_coalesced_faults; ++i) {
        uvm_fault_buffer_entry_t *entry = batch_context->ordered_fault_cache[i];
        NvU32 index = prev_fetch_hash_index(entry->fault_address, replayable_faults->prev_fetch_tag_mask);

        replayable_faults->prev_fetch_hash_epochs[index] = replayable_faults->prev_fetch_tag_generation;
        replayable_faults->prev_fetch_tag_table[index] = prev_fetch_tag_value(entry->fault_address);
        ++key_count;
    }

    finalize_prev_fetch_predictor(replayable_faults, batch_context, key_count, start_time);
}

static void update_prev_fetch_predictor(uvm_replayable_fault_buffer_info_t *replayable_faults,
                                        uvm_fault_service_batch_context_t *batch_context)
{
    if (!prev_fetch_predictor_enabled()) {
        replayable_faults->prev_fetch_key_count = 0;
        replayable_faults->last_batch_fetch_skip_pct = 0;
        batch_context->num_pred_fetch_build_keys = 0;
        return;
    }

    if (prev_fetch_predictor_use_hash_exact())
        update_prev_fetch_predictor_from_coalesced_hash_exact(replayable_faults, batch_context);
    else if (prev_fetch_predictor_use_hash_approx())
        update_prev_fetch_predictor_from_coalesced_hash_approx(replayable_faults, batch_context);
    else if (prev_fetch_predictor_use_coalesced_pages())
        update_prev_fetch_predictor_from_coalesced_bsearch(replayable_faults, batch_context);
    else
        update_prev_fetch_predictor_from_fetch_cache(replayable_faults, batch_context);
}

static NvU32 replayable_fault_buffer_fetch_limit(uvm_gpu_t *gpu, fault_fetch_mode_t fetch_mode, NvU32 pending_faults)
{
    uvm_replayable_fault_buffer_info_t *replayable_faults = &gpu->parent->fault_buffer_info.replayable;
    NvU32 fetch_limit = gpu->parent->fault_buffer_info.max_batch_size;

    if (fetch_mode == FAULT_FETCH_MODE_ALL || !uvm_perf_fault_fetch_adaptive_enable)
        goto maybe_boost;

    if (pending_faults >= uvm_perf_fault_fetch_high_pending)
        fetch_limit = min(fetch_limit, uvm_perf_fault_fetch_high_count);
    else if (pending_faults >= uvm_perf_fault_fetch_mid_pending)
        fetch_limit = min(fetch_limit, uvm_perf_fault_fetch_mid_count);
    else
        fetch_limit = min(fetch_limit, uvm_perf_fault_fetch_low_count);

maybe_boost:
    if (fetch_mode != FAULT_FETCH_MODE_ALL &&
        prev_fetch_predictor_enabled() &&
        uvm_perf_fault_fetch_predictor_boost_enable &&
        replayable_faults->last_batch_fetch_skip_pct >= uvm_perf_fault_fetch_predictor_boost_threshold_pct) {
        fetch_limit = max(fetch_limit,
                          min(gpu->parent->fault_buffer_info.max_batch_size,
                              uvm_perf_fault_fetch_predictor_boost_count));
    }

    return max(fetch_limit, 1u);
}

static void fetch_fault_buffer_merge_entry(uvm_fault_buffer_entry_t *current_entry,
                                           uvm_fault_buffer_entry_t *last_entry)
{
    UVM_ASSERT(last_entry->num_instances > 0);

    ++last_entry->num_instances;
    uvm_fault_access_type_mask_set(&last_entry->access_type_mask, current_entry->fault_access_type);

    if (current_entry->fault_access_type > last_entry->fault_access_type) {
        // If the new entry has a higher access type, it becomes the
        // fault to be serviced. Add the previous one to the list of instances
        current_entry->access_type_mask = last_entry->access_type_mask;
        current_entry->num_instances = last_entry->num_instances;
        last_entry->filtered = true;

        // We only merge faults from different uTLBs if the new fault has an
        // access type with the same or lower level of intrusiveness.
        UVM_ASSERT(current_entry->fault_source.utlb_id == last_entry->fault_source.utlb_id);

        list_replace(&last_entry->merged_instances_list, &current_entry->merged_instances_list);
        list_add(&last_entry->merged_instances_list, &current_entry->merged_instances_list);
    }
    else {
        // Add the new entry to the list of instances for reporting purposes
        current_entry->filtered = true;
        list_add(&current_entry->merged_instances_list, &last_entry->merged_instances_list);
    }
}

static bool fetch_fault_buffer_try_merge_entry(uvm_fault_buffer_entry_t *current_entry,
                                               uvm_fault_service_batch_context_t *batch_context,
                                               uvm_fault_utlb_info_t *current_tlb,
                                               bool is_same_instance_ptr)
{
    uvm_fault_buffer_entry_t *last_tlb_entry = current_tlb->last_fault;
    uvm_fault_buffer_entry_t *last_global_entry = batch_context->last_fault;

    // Check the last coalesced fault and the coalesced fault that was
    // originated from this uTLB
    const bool is_last_tlb_fault = current_tlb->num_pending_faults > 0 &&
                                   cmp_fault_instance_ptr(current_entry, last_tlb_entry) == 0 &&
                                   current_entry->fault_address == last_tlb_entry->fault_address;

    // We only merge faults from different uTLBs if the new fault has an
    // access type with the same or lower level of intrusiveness. This is to
    // avoid having to update num_pending_faults on both uTLBs and recomputing
    // last_fault.
    const bool is_last_fault = is_same_instance_ptr &&
                               current_entry->fault_address == last_global_entry->fault_address &&
                               current_entry->fault_access_type <= last_global_entry->fault_access_type;

    if (is_last_tlb_fault) {
        fetch_fault_buffer_merge_entry(current_entry, last_tlb_entry);
        if (current_entry->fault_access_type > last_tlb_entry->fault_access_type)
            current_tlb->last_fault = current_entry;

        return true;
    }
    else if (is_last_fault) {
        fetch_fault_buffer_merge_entry(current_entry, last_global_entry);
        if (current_entry->fault_access_type > last_global_entry->fault_access_type)
            batch_context->last_fault = current_entry;

        return true;
    }

    return false;
}

// Fetch entries from the fault buffer, decode them and store them in the batch
// context. We implement the fetch modes described above.
//
// When possible, we coalesce duplicate entries to minimize the fault handling
// overhead. Basically, we merge faults with the same instance pointer and page
// virtual address. We keep track of the last fault per uTLB to detect
// duplicates due to local reuse and the last fault in the whole batch to
// detect reuse across CTAs.
//
// We will service the first fault entry with the most "intrusive" (atomic >
// write > read > prefetch) access type*. That fault entry is called the
// "representative". The rest of filtered faults have the "filtered" flag set
// and are added to a list in the representative fault entry for reporting
// purposes. The representative fault entry also contains a mask with all the
// access types that produced a fault on the page.
//
// *We only merge faults from different uTLBs if the new fault has an access
// type with the same or lower level of intrusiveness.
//
// This optimization cannot be performed during fault cancel on Pascal GPUs
// (fetch_mode == FAULT_FETCH_MODE_ALL) since we need accurate tracking of all
// the faults in each uTLB in order to guarantee precise fault attribution.
static NV_STATUS fetch_fault_buffer_entries(uvm_gpu_t *gpu,
                                            uvm_fault_service_batch_context_t *batch_context,
                                            fault_fetch_mode_t fetch_mode)
{
    NvU32 get;
    NvU32 put;
    NvU32 pending_faults;
    // O6 note: kept initialized to 0 (rather than left uninitialized) because
    // the "get == put, nothing pending" early exit below jumps straight to
    // done: before the real assignment further down; 0 makes the O6
    // "did we hit the fetch-limit cap" check at done: correctly read as
    // false (no backlog) for that path instead of reading garbage.
    NvU32 fetch_limit = 0;
    NvU32 fault_index;
    NvU32 num_coalesced_faults;
    NvU32 utlb_id;
    uvm_fault_buffer_entry_t *fault_cache;
    uvm_spin_loop_t spin;
    NV_STATUS status = NV_OK;
    uvm_replayable_fault_buffer_info_t *replayable_faults = &gpu->parent->fault_buffer_info.replayable;
    // O6: decided once per batch from the *previous* batch's backlog
    // signal, so it's applied consistently to every fault in this batch's
    // lookup loop below and to this batch's own build call (whichever of
    // the two call sites in update_prev_fetch_predictor() runs, depending
    // on predictor mode).
    const bool adaptive_filter_active = prev_fetch_adaptive_should_run(replayable_faults);
    const bool in_pascal_cancel_path = (!gpu->parent->fault_cancel_va_supported && fetch_mode == FAULT_FETCH_MODE_ALL);
    const bool may_filter = uvm_perf_fault_coalesce && !in_pascal_cancel_path;

    UVM_ASSERT(uvm_sem_is_locked(&gpu->parent->isr.replayable_faults.service_lock));
    UVM_ASSERT(gpu->parent->replayable_faults_supported);

    fault_cache = batch_context->fault_cache;

    get = replayable_faults->cached_get;

    // Read put pointer from GPU and cache it
    if (get == replayable_faults->cached_put)
        replayable_faults->cached_put = gpu->parent->fault_buffer_hal->read_put(gpu->parent);

    put = replayable_faults->cached_put;

    batch_context->is_single_instance_ptr = true;
    batch_context->last_fault = NULL;

    fault_index = 0;
    num_coalesced_faults = 0;

    // Clear uTLB counters
    for (utlb_id = 0; utlb_id <= batch_context->max_utlb_id; ++utlb_id) {
        batch_context->utlbs[utlb_id].num_pending_faults = 0;
        batch_context->utlbs[utlb_id].has_fatal_faults = false;
    }
    batch_context->max_utlb_id = 0;

    if (get == put)
        goto done;

    pending_faults = replayable_fault_buffer_pending_entries(get, put, replayable_faults->max_faults);
    fetch_limit = replayable_fault_buffer_fetch_limit(gpu, fetch_mode, pending_faults);

    if (uvm_page_rep_profile_enable && fetch_mode != FAULT_FETCH_MODE_ALL) {
        pr_info("[FETCH_CTRL] pending=%u fetch_limit=%u adaptive=%u low=%u mid=%u high=%u fetch_skip_prev_pct=%u boost_en=%u boost_thresh=%u boost_count=%u\n",
                pending_faults,
                fetch_limit,
                uvm_perf_fault_fetch_adaptive_enable,
                uvm_perf_fault_fetch_low_count,
                uvm_perf_fault_fetch_mid_count,
                uvm_perf_fault_fetch_high_count,
                replayable_faults->last_batch_fetch_skip_pct,
                uvm_perf_fault_fetch_predictor_boost_enable,
                uvm_perf_fault_fetch_predictor_boost_threshold_pct,
                uvm_perf_fault_fetch_predictor_boost_count);
    }

    // Parse until get != put and have enough space to cache.
    while ((get != put) &&
           (fetch_mode == FAULT_FETCH_MODE_ALL || fault_index < fetch_limit)) {
        bool is_same_instance_ptr = true;
        uvm_fault_buffer_entry_t *current_entry = &fault_cache[fault_index];
        uvm_fault_utlb_info_t *current_tlb;

        // We cannot just wait for the last entry (the one pointed by put) to
        // become valid, we have to do it individually since entries can be
        // written out of order
        UVM_SPIN_WHILE(!gpu->parent->fault_buffer_hal->entry_is_valid(gpu->parent, get), &spin) {
            // We have some entry to work on. Let's do the rest later.
            if (fetch_mode == FAULT_FETCH_MODE_BATCH_READY && fault_index > 0)
                goto done;
        }

        // Prevent later accesses being moved above the read of the valid bit
        smp_mb__after_atomic();

        // Got valid bit set. Let's cache.
        status = gpu->parent->fault_buffer_hal->parse_replayable_entry(gpu->parent, get, current_entry);
        if (status != NV_OK)
            goto done;

        // The GPU aligns the fault addresses to 4k, but all of our tracking is
        // done in PAGE_SIZE chunks which might be larger.
        current_entry->fault_address = UVM_PAGE_ALIGN_DOWN(current_entry->fault_address);

        // Make sure that all fields in the entry are properly initialized
        current_entry->is_fatal = (current_entry->fault_type >= UVM_FAULT_TYPE_FATAL);

        if (current_entry->is_fatal) {
            // Record the fatal fault event later as we need the va_space locked
            current_entry->fatal_reason = UvmEventFatalReasonInvalidFaultType;
        }
        else {
            current_entry->fatal_reason = UvmEventFatalReasonInvalid;
        }

        current_entry->va_space = NULL;
        current_entry->filtered = false;
        current_entry->pred_prev_candidate = false;
        current_entry->pred_fetch_skip = false;
        current_entry->replayable.cancel_va_mode = UVM_FAULT_CANCEL_VA_MODE_ALL;

        if (current_entry->fault_source.utlb_id > batch_context->max_utlb_id) {
            UVM_ASSERT(current_entry->fault_source.utlb_id < replayable_faults->utlb_count);
            batch_context->max_utlb_id = current_entry->fault_source.utlb_id;
        }

        current_tlb = &batch_context->utlbs[current_entry->fault_source.utlb_id];

        if (prev_fetch_predictor_enabled() &&
            uvm_perf_fault_pred_skip_enable &&
            fetch_mode != FAULT_FETCH_MODE_ALL &&
            !current_entry->is_fatal &&
            adaptive_filter_active &&
            prev_fetch_keys_contains(replayable_faults, batch_context, current_entry)) {
            ++batch_context->num_pred_fast_skip_stale;
            goto advance_get;
        }

        if (fault_index > 0) {
            UVM_ASSERT(batch_context->last_fault);
            is_same_instance_ptr = cmp_fault_instance_ptr(current_entry, batch_context->last_fault) == 0;

            // Coalesce duplicate faults when possible
            if (may_filter && !current_entry->is_fatal) {
                bool merged = fetch_fault_buffer_try_merge_entry(current_entry,
                                                                 batch_context,
                                                                 current_tlb,
                                                                 is_same_instance_ptr);
                if (merged)
                    goto next_fault;
            }
        }

        if (batch_context->is_single_instance_ptr && !is_same_instance_ptr)
            batch_context->is_single_instance_ptr = false;

        current_entry->num_instances = 1;
        current_entry->access_type_mask = uvm_fault_access_type_mask_bit(current_entry->fault_access_type);
        INIT_LIST_HEAD(&current_entry->merged_instances_list);

        ++current_tlb->num_pending_faults;
        current_tlb->last_fault = current_entry;
        batch_context->last_fault = current_entry;

        ++num_coalesced_faults;

    next_fault:
        ++fault_index;
    advance_get:
        ++get;
        if (get == replayable_faults->max_faults)
            get = 0;
    }

done:
    batch_context->pred_adaptive_active_this_batch = adaptive_filter_active;

    // O6: decide the adaptive-filter state for the *next* batch (and, since
    // this runs before either build call site below/in preprocess_fault_batch,
    // for the rest of *this* batch's own build too -- see the comment on
    // adaptive_filter_active above). fetch_limit/fault_index are both in
    // scope here regardless of which way the loop above exited.
    if (uvm_perf_fault_pred_adaptive_enable) {
        // "Hit the cap with more still pending" is the actual definition of
        // a backlog, independent of whatever fetch_limit happens to be this
        // batch (it can itself be adaptively reduced elsewhere).
        bool batch_hit_limit = (fetch_mode != FAULT_FETCH_MODE_ALL) && fetch_limit > 0 && fault_index >= fetch_limit;

        if (batch_hit_limit) {
            replayable_faults->prev_fetch_adaptive_active = true;
            replayable_faults->prev_fetch_adaptive_cooldown_remaining = max((NvU32)1, uvm_perf_fault_pred_adaptive_cooldown);
        }
        else if (replayable_faults->prev_fetch_adaptive_active) {
            if (batch_context->num_pred_fast_skip_stale > 0) {
                // Below the cap but still finding real stale faults: the
                // backlog is draining, not gone. Keep going.
                replayable_faults->prev_fetch_adaptive_cooldown_remaining = max((NvU32)1, uvm_perf_fault_pred_adaptive_cooldown);
            }
            else if (replayable_faults->prev_fetch_adaptive_cooldown_remaining > 1) {
                // Below the cap, nothing found this batch, but still inside
                // the hysteresis window -- this could just be a momentary
                // dip, keep checking a little longer.
                --replayable_faults->prev_fetch_adaptive_cooldown_remaining;
            }
            else {
                // Below the cap, nothing found, hysteresis exhausted: power
                // down until the next backlog signal.
                replayable_faults->prev_fetch_adaptive_active = false;
                replayable_faults->prev_fetch_adaptive_cooldown_remaining = 0;
            }
        }
        // else: already inactive and still below the cap -- stays inactive.
    }

    if (!prev_fetch_predictor_use_coalesced_pages() && prev_fetch_adaptive_should_run(replayable_faults))
        update_prev_fetch_predictor(replayable_faults, batch_context);

    // Verbose per-fault-address dump — gated by uvm_fault_dbg_enable (default off).
    // Use only for small targeted runs (e.g. analyze_faults.py); real benchmarks
    // generate millions of faults and will flood dmesg if this is on.
    if (uvm_fault_dbg_enable) {
        static int __uvm_fault_dbg_cnt = 0;
        if (fault_index > 0) {
            NvU32 __i;
            printk(KERN_INFO "[UVM_FAULT_DBG] BATCH %d %u %u\n",
                   __uvm_fault_dbg_cnt, fault_index, num_coalesced_faults);
            for (__i = 0; __i < fault_index && __i < 4096; __i++) {
                printk(KERN_INFO "[UVM_FAULT_DBG] F 0x%016llx %u\n",
                       (unsigned long long)fault_cache[__i].fault_address,
                       fault_cache[__i].filtered ? 1u : 0u);
            }
            if (fault_index > 4096)
                printk(KERN_INFO "[UVM_FAULT_DBG] TRUNC %u\n",
                       fault_index - 4096);
            printk(KERN_INFO "[UVM_FAULT_DBG] END\n");
            __uvm_fault_dbg_cnt++;
        }
    }

    // Lightweight per-batch repetition profiling.
    // Phase 1 (inline dedup) only checks the last fault per uTLB / globally,
    // so num_coalesced_faults may still contain duplicates that arrived
    // non-adjacently. Do a full sort+dedup here for accurate unique count.
    if (uvm_page_rep_profile_enable && fault_index > 0) {
        NvU64 *__addrs = kmalloc(num_coalesced_faults * sizeof(NvU64), GFP_KERNEL);
        uvm_fault_buffer_entry_t **__entries = NULL;

        if (uvm_page_rep_detail_enable)
            __entries = kmalloc(num_coalesced_faults * sizeof(*__entries), GFP_KERNEL);

        if (__addrs) {
            NvU32 __i, __j = 0;
            NvU32 __unique;

            for (__i = 0; __i < fault_index; __i++) {
                if (!fault_cache[__i].filtered) {
                    __addrs[__j++] = fault_cache[__i].fault_address;
                    if (__entries)
                        __entries[__j - 1] = &fault_cache[__i];
                }
            }

            sort(__addrs, __j, sizeof(NvU64), cmp_fault_addr_sort, NULL);

            __unique = (__j > 0) ? 1 : 0;
            for (__i = 1; __i < __j; __i++) {
                if (__addrs[__i] != __addrs[__i - 1])
                    __unique++;
            }

            pr_info("[PAGE_REP] raw=%u phase1=%u unique=%u\n",
                    fault_index, num_coalesced_faults, __unique);

            if (uvm_page_rep_detail_enable) {
                if (!__entries) {
                    pr_info("[PAGE_REP_DETAIL] batch=%u status=oom reps=%u\n",
                            batch_context->batch_id, __j);
                }
                else if (__unique < __j) {
                    NvU32 __dup_pages = 0;
                    NvU32 __dup_reps = 0;
                    NvU32 __group_begin = 0;

                    sort(__entries, __j, sizeof(*__entries), cmp_fault_entry_addr_access_sort, NULL);

                    while (__group_begin < __j) {
                        NvU32 __group_end = __group_begin + 1;
                        uvm_fault_buffer_entry_t *__head = __entries[__group_begin];

                        while (__group_end < __j &&
                               cmp_va_space(__entries[__group_end]->va_space, __head->va_space) == 0 &&
                               __entries[__group_end]->fault_address == __head->fault_address) {
                            ++__group_end;
                        }

                        if (__group_end - __group_begin > 1) {
                            NvU32 __k;

                            ++__dup_pages;
                            __dup_reps += __group_end - __group_begin;
                            pr_info("[PAGE_REP_DETAIL] batch=%u addr=0x%016llx reps=%u\n",
                                    batch_context->batch_id,
                                    (unsigned long long)__head->fault_address,
                                    __group_end - __group_begin);

                            for (__k = __group_begin; __k < __group_end; ++__k) {
                                uvm_fault_buffer_entry_t *__entry = __entries[__k];
                                pr_info("[PAGE_REP_DETAIL] batch=%u addr=0x%016llx access=%s utlb=%u num_instances=%u instance_ptr=0x%llx\n",
                                        batch_context->batch_id,
                                        (unsigned long long)__entry->fault_address,
                                        uvm_fault_access_type_string(__entry->fault_access_type),
                                        __entry->fault_source.utlb_id,
                                        __entry->num_instances,
                                        (unsigned long long)__entry->instance_ptr.address);
                            }
                        }

                        __group_begin = __group_end;
                    }

                    pr_info("[PAGE_REP_DETAIL] batch=%u summary dup_pages=%u dup_reps=%u\n",
                            batch_context->batch_id, __dup_pages, __dup_reps);
                }
            }

            kfree(__entries);
            kfree(__addrs);
        }
        else {
            pr_info("[PAGE_REP] raw=%u phase1=%u unique=?(oom)\n",
                    fault_index, num_coalesced_faults);
            kfree(__entries);
        }
    }

    write_get(gpu->parent, get);

    batch_context->num_cached_faults = fault_index;
    batch_context->num_coalesced_faults = num_coalesced_faults;

    return status;
}

// Sort comparator for pointers to fault buffer entries that sorts by
// instance pointer
static int cmp_sort_fault_entry_by_instance_ptr(const void *_a, const void *_b)
{
    const uvm_fault_buffer_entry_t **a = (const uvm_fault_buffer_entry_t **)_a;
    const uvm_fault_buffer_entry_t **b = (const uvm_fault_buffer_entry_t **)_b;

    return cmp_fault_instance_ptr(*a, *b);
}

// Sort comparator for pointers to fault buffer entries that sorts by va_space,
// fault address and fault access type
static int cmp_sort_fault_entry_by_va_space_address_access_type(const void *_a, const void *_b)
{
    const uvm_fault_buffer_entry_t **a = (const uvm_fault_buffer_entry_t **)_a;
    const uvm_fault_buffer_entry_t **b = (const uvm_fault_buffer_entry_t **)_b;

    int result;

    result = cmp_va_space((*a)->va_space, (*b)->va_space);
    if (result != 0)
        return result;

    result = cmp_addr((*a)->fault_address, (*b)->fault_address);
    if (result != 0)
        return result;

    return cmp_access_type((*a)->fault_access_type, (*b)->fault_access_type);
}

// Translate all instance pointers to VA spaces. Since the buffer is ordered by
// instance_ptr, we minimize the number of translations
//
// This function returns NV_WARN_MORE_PROCESSING_REQUIRED if a fault buffer
// flush occurred and executed successfully, or the error code if it failed.
// NV_OK otherwise.
static NV_STATUS translate_instance_ptrs(uvm_gpu_t *gpu,
                                         uvm_fault_service_batch_context_t *batch_context)
{
    NvU32 i;
    NV_STATUS status;

    for (i = 0; i < batch_context->num_coalesced_faults; ++i) {
        uvm_fault_buffer_entry_t *current_entry;

        current_entry = batch_context->ordered_fault_cache[i];

        // If this instance pointer matches the previous instance pointer, just
        // copy over the already-translated va_space and move on.
        if (i != 0 && cmp_fault_instance_ptr(current_entry, batch_context->ordered_fault_cache[i - 1]) == 0) {
            current_entry->va_space = batch_context->ordered_fault_cache[i - 1]->va_space;
            continue;
        }

        status = uvm_parent_gpu_fault_entry_to_va_space(gpu->parent, current_entry, &current_entry->va_space);
        if (status != NV_OK) {
            if (status == NV_ERR_PAGE_TABLE_NOT_AVAIL) {
                // The channel is valid but the subcontext is not. This can only
                // happen if the subcontext is torn down before its work is
                // complete while other subcontexts in the same TSG are still
                // executing. This is a violation of the programming model. We
                // have limited options since the VA space is gone, meaning we
                // can't target the PDB for cancel even if we wanted to. So
                // we'll just throw away precise attribution and cancel this
                // fault using the SW method, which validates that the intended
                // context (TSG) is still running so we don't cancel an innocent
                // context.
                UVM_ASSERT(!current_entry->va_space);
                UVM_ASSERT(gpu->max_subcontexts > 0);

                if (gpu->parent->smc.enabled) {
                    status = push_cancel_on_gpu_targeted(gpu,
                                                         current_entry->instance_ptr,
                                                         current_entry->fault_source.gpc_id,
                                                         current_entry->fault_source.client_id,
                                                         &batch_context->tracker);
                }
                else {
                    status = push_cancel_on_gpu_global(gpu, current_entry->instance_ptr, &batch_context->tracker);
                }

                if (status != NV_OK)
                    return status;

                // Fall through and let the flush restart fault processing
            }
            else {
                UVM_ASSERT(status == NV_ERR_INVALID_CHANNEL);
            }

            // If the channel is gone then we're looking at a stale fault entry.
            // The fault must have been resolved already (serviced or
            // cancelled), so we can just flush the fault buffer.
            //
            // No need to use UVM_GPU_BUFFER_FLUSH_MODE_WAIT_UPDATE_PUT since
            // there was a context preemption for the entries we want to flush,
            // meaning PUT must reflect them.
            status = fault_buffer_flush_locked(gpu,
                                               UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT,
                                               UVM_FAULT_REPLAY_TYPE_START,
                                               batch_context);
            if (status != NV_OK)
                 return status;

            return NV_WARN_MORE_PROCESSING_REQUIRED;
        }
        else {
            UVM_ASSERT(current_entry->va_space);
        }
    }

    return NV_OK;
}

// Fault cache preprocessing for fault coalescing
//
// This function generates an ordered view of the given fault_cache in which
// faults are sorted by VA space, fault address (aligned to 4K) and access type
// "intrusiveness". In order to minimize the number of instance_ptr to VA space
// translations we perform a first sort by instance_ptr.
//
// This function returns NV_WARN_MORE_PROCESSING_REQUIRED if a fault buffer
// flush occurred during instance_ptr translation and executed successfully, or
// the error code if it failed. NV_OK otherwise.
//
// Current scheme:
// 1) sort by instance_ptr
// 2) translate all instance_ptrs to VA spaces
// 3) sort by va_space, fault address (fault_address is page-aligned at this
//    point) and access type
static void update_prev_batch_predictor(uvm_gpu_t *gpu, uvm_fault_service_batch_context_t *batch_context);

static NV_STATUS preprocess_fault_batch(uvm_gpu_t *gpu, uvm_fault_service_batch_context_t *batch_context)
{
    NV_STATUS status;
    NvU32 i, j;
    uvm_fault_buffer_entry_t **ordered_fault_cache = batch_context->ordered_fault_cache;

    UVM_ASSERT(batch_context->num_coalesced_faults > 0);
    UVM_ASSERT(batch_context->num_cached_faults >= batch_context->num_coalesced_faults);

    // Generate an ordered view of the fault cache in ordered_fault_cache.
    // We sort the pointers, not the entries in fault_cache

    // Initialize pointers before they are sorted. We only sort one instance per
    // coalesced fault
    for (i = 0, j = 0; i < batch_context->num_cached_faults; ++i) {
        if (!batch_context->fault_cache[i].filtered)
            ordered_fault_cache[j++] = &batch_context->fault_cache[i];
    }
    UVM_ASSERT(j == batch_context->num_coalesced_faults);

    // 1) if the fault batch contains more than one, sort by instance_ptr
    if (!batch_context->is_single_instance_ptr) {
        sort(ordered_fault_cache,
             batch_context->num_coalesced_faults,
             sizeof(*ordered_fault_cache),
             cmp_sort_fault_entry_by_instance_ptr,
             NULL);
    }

    // 2) translate all instance_ptrs to VA spaces
    status = translate_instance_ptrs(gpu, batch_context);
    if (status != NV_OK)
        return status;

    // 3) sort by va_space, fault address (GPU already reports 4K-aligned
    // address) and access type
    sort(ordered_fault_cache,
         batch_context->num_coalesced_faults,
         sizeof(*ordered_fault_cache),
         cmp_sort_fault_entry_by_va_space_address_access_type,
         NULL);

    update_prev_batch_predictor(gpu, batch_context);
    if (prev_fetch_predictor_use_coalesced_pages() &&
        prev_fetch_adaptive_should_run(&gpu->parent->fault_buffer_info.replayable))
        update_prev_fetch_predictor(&gpu->parent->fault_buffer_info.replayable, batch_context);

    return NV_OK;
}

static bool check_fault_entry_duplicate(const uvm_fault_buffer_entry_t *current_entry,
                                        const uvm_fault_buffer_entry_t *previous_entry)
{
    bool is_duplicate = false;

    if (previous_entry) {
        is_duplicate = (current_entry->va_space == previous_entry->va_space) &&
                       (current_entry->fault_address == previous_entry->fault_address);
    }

    return is_duplicate;
}

static int cmp_prev_batch_key(const uvm_fault_prev_batch_key_t *a, const uvm_fault_prev_batch_key_t *b)
{
    int result = cmp_va_space(a->va_space, b->va_space);

    if (result != 0)
        return result;

    return cmp_addr(a->fault_address, b->fault_address);
}

static bool prev_batch_keys_contains(uvm_replayable_fault_buffer_info_t *replayable_faults,
                                     uvm_fault_service_batch_context_t *batch_context,
                                     uvm_va_space_t *va_space,
                                     NvU64 fault_address)
{
    NvU32 left = 0;
    NvU32 right = replayable_faults->prev_batch_key_count;
    NvU32 cmp_count = 0;
    ktime_t start_time = 0;

    if (pred_timing_enabled())
        start_time = ktime_get();

#define RECORD_PREV_BATCH_LOOKUP(_hit) do { \
        batch_context->num_pred_prev_lookup_calls++; \
        batch_context->num_pred_prev_lookup_cmps += cmp_count; \
        if ((_hit)) batch_context->num_pred_prev_lookup_hits++; \
        if (pred_timing_enabled()) \
            batch_context->time_pred_prev_lookup_ns += \
                ktime_to_ns(ktime_sub(ktime_get(), start_time)); \
    } while (0)

    while (left < right) {
        NvU32 mid = left + ((right - left) / 2);
        uvm_fault_prev_batch_key_t needle;
        int cmp;

        needle.va_space = va_space;
        needle.fault_address = fault_address;
        cmp = cmp_prev_batch_key(&needle, &replayable_faults->prev_batch_keys[mid]);
        ++cmp_count;
        if (cmp == 0) {
            RECORD_PREV_BATCH_LOOKUP(true);
            return true;
        }

        if (cmp < 0)
            right = mid;
        else
            left = mid + 1;
    }

    RECORD_PREV_BATCH_LOOKUP(false);
#undef RECORD_PREV_BATCH_LOOKUP
    return false;
}

static void update_prev_batch_predictor(uvm_gpu_t *gpu, uvm_fault_service_batch_context_t *batch_context)
{
    uvm_replayable_fault_buffer_info_t *replayable_faults = &gpu->parent->fault_buffer_info.replayable;
    uvm_fault_buffer_entry_t *previous_entry = NULL;
    NvU32 i;
    NvU32 key_count = 0;
    ktime_t start_time = 0;

    if (pred_timing_enabled())
        start_time = ktime_get();

    batch_context->num_pred_prev_candidates = 0;

    if (!prev_batch_predictor_enabled()) {
        replayable_faults->prev_batch_key_count = 0;
        replayable_faults->last_batch_pred_candidate_pct = 0;
        for (i = 0; i < batch_context->num_coalesced_faults; ++i)
            batch_context->ordered_fault_cache[i]->pred_prev_candidate = false;
        return;
    }

    for (i = 0; i < batch_context->num_coalesced_faults; ++i) {
        uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[i];
        bool is_duplicate = previous_entry && check_fault_entry_duplicate(current_entry, previous_entry);

        current_entry->pred_prev_candidate = false;
        if (is_duplicate)
            continue;

        if (prev_batch_keys_contains(replayable_faults, batch_context,
                                     current_entry->va_space, current_entry->fault_address)) {
            current_entry->pred_prev_candidate = true;
            ++batch_context->num_pred_prev_candidates;
        }

        replayable_faults->prev_batch_keys[key_count].va_space = current_entry->va_space;
        replayable_faults->prev_batch_keys[key_count].fault_address = current_entry->fault_address;
        ++key_count;
        previous_entry = current_entry;
    }

    replayable_faults->prev_batch_key_count = key_count;
    batch_context->num_pred_prev_build_keys = key_count;
    replayable_faults->last_batch_pred_candidate_pct =
        batch_context->num_coalesced_faults == 0 ? 0 :
        min((batch_context->num_pred_prev_candidates * 100) / batch_context->num_coalesced_faults, 100u);

    if (pred_timing_enabled())
        batch_context->time_pred_prev_build_ns +=
            ktime_to_ns(ktime_sub(ktime_get(), start_time));
}

static void update_batch_and_notify_fault(uvm_gpu_t *gpu,
                                          uvm_fault_service_batch_context_t *batch_context,
                                          uvm_va_block_t *va_block,
                                          uvm_processor_id_t preferred_location,
                                          uvm_fault_buffer_entry_t *current_entry,
                                          bool is_duplicate)
{
    if (is_duplicate)
        batch_context->num_duplicate_faults += current_entry->num_instances;
    else
        batch_context->num_duplicate_faults += current_entry->num_instances - 1;

    uvm_perf_event_notify_gpu_fault(&current_entry->va_space->perf_events,
                                    va_block,
                                    gpu->id,
                                    preferred_location,
                                    current_entry,
                                    batch_context->batch_id,
                                    is_duplicate);
}

static void mark_fault_invalid_prefetch(uvm_fault_service_batch_context_t *batch_context,
                                        uvm_fault_buffer_entry_t *fault_entry)
{
    fault_entry->is_invalid_prefetch = true;

    // For block faults, the following counter might be updated more than once
    // for the same fault if block_context->num_retries > 0. As a result, this
    // counter might be higher than the actual count. In order for this counter
    // to be always accurate, block_context needs to passed down the stack from
    // all callers. But since num_retries > 0 case is uncommon and imprecise
    // invalid_prefetch counter doesn't affect functionality (other than
    // disabling prefetching if the counter indicates lots of invalid prefetch
    // faults), this is ok.
    batch_context->num_invalid_prefetch_faults += fault_entry->num_instances;
}

static void mark_fault_throttled(uvm_fault_service_batch_context_t *batch_context,
                                 uvm_fault_buffer_entry_t *fault_entry)
{
    fault_entry->is_throttled = true;
    batch_context->has_throttled_faults = true;
}

static void mark_fault_fatal(uvm_fault_service_batch_context_t *batch_context,
                             uvm_fault_buffer_entry_t *fault_entry,
                             UvmEventFatalReason fatal_reason,
                             uvm_fault_cancel_va_mode_t cancel_va_mode)
{
    uvm_fault_utlb_info_t *utlb = &batch_context->utlbs[fault_entry->fault_source.utlb_id];

    fault_entry->is_fatal = true;
    fault_entry->fatal_reason = fatal_reason;
    fault_entry->replayable.cancel_va_mode = cancel_va_mode;

    utlb->has_fatal_faults = true;

    if (!batch_context->fatal_va_space) {
        UVM_ASSERT(fault_entry->va_space);
        batch_context->fatal_va_space = fault_entry->va_space;
    }
}

static void fault_entry_duplicate_flags(uvm_fault_service_batch_context_t *batch_context,
                                        uvm_fault_buffer_entry_t *current_entry,
                                        const uvm_fault_buffer_entry_t *previous_entry)
{
    UVM_ASSERT(previous_entry);
    UVM_ASSERT(check_fault_entry_duplicate(current_entry, previous_entry));

    // Propagate the is_invalid_prefetch flag across all prefetch faults
    // on the page
    if (previous_entry->is_invalid_prefetch)
        mark_fault_invalid_prefetch(batch_context, current_entry);

    // If a page is throttled, all faults on the page must be skipped
    if (previous_entry->is_throttled)
        mark_fault_throttled(batch_context, current_entry);
}

// This function computes the maximum access type that can be serviced for the
// reported fault instances given the logical permissions of the VA range. If
// none of the fault instances can be serviced UVM_FAULT_ACCESS_TYPE_COUNT is
// returned instead.
//
// In the case that there are faults that cannot be serviced, this function
// also sets the flags required for fault cancellation. Prefetch faults do not
// need to be cancelled since they disappear on replay.
//
// The UVM driver considers two scenarios for logical permissions violation:
// - All access types are invalid. For example, when faulting from a processor
// that doesn't have access to the preferred location of a range group when it
// is not migratable. In this case all accesses to the page must be cancelled.
// - Write/atomic accesses are invalid. Basically, when trying to modify a
// read-only VA range. In this case we restrict fault cancelling to those types
// of accesses.
//
// Return values:
// - service_access_type: highest access type that can be serviced.
static uvm_fault_access_type_t check_fault_access_permissions(uvm_gpu_t *gpu,
                                                              uvm_fault_service_batch_context_t *batch_context,
                                                              uvm_va_block_t *va_block,
                                                              uvm_service_block_context_t *service_block_context,
                                                              uvm_fault_buffer_entry_t *fault_entry,
                                                              bool allow_migration)
{
    NV_STATUS perm_status;
    UvmEventFatalReason fatal_reason;
    uvm_fault_cancel_va_mode_t cancel_va_mode;
    uvm_fault_access_type_t ret = UVM_FAULT_ACCESS_TYPE_COUNT;
    uvm_va_block_context_t *va_block_context = service_block_context->block_context;

    perm_status = uvm_va_block_check_logical_permissions(va_block,
                                                         va_block_context,
                                                         gpu->id,
                                                         uvm_va_block_cpu_page_index(va_block,
                                                                                     fault_entry->fault_address),
                                                         fault_entry->fault_access_type,
                                                         allow_migration);
    if (perm_status == NV_OK)
        return fault_entry->fault_access_type;

    if (fault_entry->fault_access_type == UVM_FAULT_ACCESS_TYPE_PREFETCH) {
        // Only update the count the first time since logical permissions cannot
        // change while we hold the VA space lock
        // TODO: Bug 1750144: That might not be true with HMM.
        if (service_block_context->num_retries == 0)
            mark_fault_invalid_prefetch(batch_context, fault_entry);

        return ret;
    }

    // At this point we know that some fault instances cannot be serviced
    fatal_reason = uvm_tools_status_to_fatal_fault_reason(perm_status);

    if (fault_entry->fault_access_type > UVM_FAULT_ACCESS_TYPE_READ) {
        cancel_va_mode = UVM_FAULT_CANCEL_VA_MODE_WRITE_AND_ATOMIC;

        // If there are pending read accesses on the same page, we have to
        // service them before we can cancel the write/atomic faults. So we
        // retry with read fault access type.
        if (uvm_fault_access_type_mask_test(fault_entry->access_type_mask, UVM_FAULT_ACCESS_TYPE_READ)) {
            perm_status = uvm_va_block_check_logical_permissions(va_block,
                                                                 va_block_context,
                                                                 gpu->id,
                                                                 uvm_va_block_cpu_page_index(va_block,
                                                                                             fault_entry->fault_address),
                                                                 UVM_FAULT_ACCESS_TYPE_READ,
                                                                 allow_migration);
            if (perm_status == NV_OK) {
                ret = UVM_FAULT_ACCESS_TYPE_READ;
            }
            else {
                // Read accesses didn't succeed, cancel all faults
                cancel_va_mode = UVM_FAULT_CANCEL_VA_MODE_ALL;
                fatal_reason = uvm_tools_status_to_fatal_fault_reason(perm_status);
            }
        }
    }
    else {
        cancel_va_mode = UVM_FAULT_CANCEL_VA_MODE_ALL;
    }

    mark_fault_fatal(batch_context, fault_entry, fatal_reason, cancel_va_mode);

    return ret;
}

static void account_stale_fault(uvm_fault_service_batch_context_t *batch_context,
                                uvm_fault_buffer_entry_t *current_entry,
                                uvm_page_index_t page_index,
                                uvm_page_mask_t *pages_serviced_in_batch)
{
    const bool detail_enable = stale_detail_enabled();
    const bool in_tracked_range = detail_enable && fault_addr_in_tracked_range(current_entry->fault_address);

    ++batch_context->num_stale_faults;

    if (!detail_enable)
        return;

    if (uvm_page_mask_test(pages_serviced_in_batch, page_index)) {
        ++batch_context->num_stale_same_batch_faults;
    }
    else {
        NvU32 lag = tracked_stale_prev_lag_get(current_entry->fault_address, batch_context->batch_id);
        ++batch_context->num_stale_prev_batch_faults;

        if (current_entry->pred_prev_candidate)
            ++batch_context->num_pred_prev_true_stale;
        else
            ++batch_context->num_pred_prev_missed_stale;

        if (lag == 0)
            ++batch_context->num_stale_prev_lag_unknown;
        else if (lag == 1)
            ++batch_context->num_stale_prev_lag_1;
        else if (lag <= 3)
            ++batch_context->num_stale_prev_lag_2_3;
        else if (lag <= 7)
            ++batch_context->num_stale_prev_lag_4_7;
        else
            ++batch_context->num_stale_prev_lag_8_plus;

        if (in_tracked_range) {
            if (lag == 0)
                ++batch_context->num_tracked_stale_prev_lag_unknown;
            else if (lag == 1)
                ++batch_context->num_tracked_stale_prev_lag_1;
            else if (lag <= 3)
                ++batch_context->num_tracked_stale_prev_lag_2_3;
            else if (lag <= 7)
                ++batch_context->num_tracked_stale_prev_lag_4_7;
            else
                ++batch_context->num_tracked_stale_prev_lag_8_plus;
        }
    }

    if (in_tracked_range)
        ++batch_context->num_tracked_stale_faults;
}

// We notify the fault event for all faults within the block so that the
// performance heuristics are updated. Then, all required actions for the block
// data are performed by the performance heuristics code.
//
// Fatal faults are flagged as fatal for later cancellation. Servicing is not
// interrupted on fatal faults due to insufficient permissions or invalid
// addresses.
//
// Return codes:
// - NV_OK if all faults were handled (both fatal and non-fatal)
// - NV_ERR_MORE_PROCESSING_REQUIRED if servicing needs allocation retry
// - NV_ERR_NO_MEMORY if the faults could not be serviced due to OOM
// - Any other value is a UVM-global error
static NV_STATUS service_fault_batch_block_locked(uvm_gpu_t *gpu,
                                                  uvm_va_block_t *va_block,
                                                  uvm_va_block_retry_t *va_block_retry,
                                                  uvm_fault_service_batch_context_t *batch_context,
                                                  NvU32 first_fault_index,
                                                  const bool hmm_migratable,
                                                  NvU32 *block_faults)
{
    NV_STATUS status = NV_OK;
    NvU32 i;
    uvm_page_index_t first_page_index;
    uvm_page_index_t last_page_index;
    NvU32 page_fault_count = 0;
    uvm_page_mask_t pages_serviced_in_batch;
    uvm_range_group_range_iter_t iter;
    uvm_replayable_fault_buffer_info_t *replayable_faults = &gpu->parent->fault_buffer_info.replayable;
    uvm_fault_buffer_entry_t **ordered_fault_cache = batch_context->ordered_fault_cache;
    uvm_service_block_context_t *block_context = &replayable_faults->block_service_context;
    uvm_va_space_t *va_space = uvm_va_block_get_va_space(va_block);
    const uvm_va_policy_t *policy;
    NvU64 end;

    // Check that all uvm_fault_access_type_t values can fit into an NvU8
    BUILD_BUG_ON(UVM_FAULT_ACCESS_TYPE_COUNT > (int)(NvU8)-1);

    uvm_assert_mutex_locked(&va_block->lock);

    *block_faults = 0;

    first_page_index = PAGES_PER_UVM_VA_BLOCK;
    last_page_index = 0;

    // Initialize fault service block context
    uvm_processor_mask_zero(&block_context->resident_processors);
    block_context->thrashing_pin_count = 0;
    block_context->read_duplicate_count = 0;
    uvm_page_mask_zero(&pages_serviced_in_batch);

    uvm_range_group_range_migratability_iter_first(va_space, va_block->start, va_block->end, &iter);

    // The first entry is guaranteed to fall within this block
    UVM_ASSERT(ordered_fault_cache[first_fault_index]->va_space == va_space);
    UVM_ASSERT(ordered_fault_cache[first_fault_index]->fault_address >= va_block->start);
    UVM_ASSERT(ordered_fault_cache[first_fault_index]->fault_address <= va_block->end);

    if (uvm_va_block_is_hmm(va_block)) {
        policy = uvm_hmm_find_policy_end(va_block,
                                         block_context->block_context->hmm.vma,
                                         ordered_fault_cache[first_fault_index]->fault_address,
                                         &end);
    }
    else {
        policy = uvm_va_range_get_policy(va_block->va_range);
        end = va_block->end;
    }

    // Scan the sorted array and notify the fault event for all fault entries
    // in the block
    for (i = first_fault_index;
         i < batch_context->num_coalesced_faults &&
         ordered_fault_cache[i]->va_space == va_space &&
         ordered_fault_cache[i]->fault_address <= end;
         ++i) {
        uvm_fault_buffer_entry_t *current_entry = ordered_fault_cache[i];
        const uvm_fault_buffer_entry_t *previous_entry = NULL;
        bool read_duplicate;
        uvm_processor_id_t new_residency;
        uvm_perf_thrashing_hint_t thrashing_hint;
        uvm_page_index_t page_index = uvm_va_block_cpu_page_index(va_block, current_entry->fault_address);
        bool is_duplicate = false;
        uvm_fault_access_type_t service_access_type;
        NvU32 service_access_type_mask;

        UVM_ASSERT(current_entry->fault_access_type ==
                   uvm_fault_access_type_mask_highest(current_entry->access_type_mask));

        // Unserviceable faults were already skipped by the caller. There are no
        // unserviceable fault types that could be in the same VA block as a
        // serviceable fault.
        UVM_ASSERT(!current_entry->is_fatal);
        current_entry->is_throttled        = false;
        current_entry->is_invalid_prefetch = false;

        if (stale_detail_enabled() && fault_addr_in_tracked_range(current_entry->fault_address))
            ++batch_context->num_tracked_faults;

        if (i > first_fault_index) {
            previous_entry = ordered_fault_cache[i - 1];
            is_duplicate = check_fault_entry_duplicate(current_entry, previous_entry);
        }

        // Ensure that the migratability iterator covers the current fault
        // address
        while (iter.end < current_entry->fault_address)
            uvm_range_group_range_migratability_iter_next(va_space, &iter, va_block->end);

        UVM_ASSERT(iter.start <= current_entry->fault_address && iter.end >= current_entry->fault_address);

        // Only update counters the first time since logical permissions cannot
        // change while we hold the VA space lock.
        // TODO: Bug 1750144: That might not be true with HMM.
        if (block_context->num_retries == 0) {
            update_batch_and_notify_fault(gpu,
                                          batch_context,
                                          va_block,
                                          policy->preferred_location,
                                          current_entry,
                                          is_duplicate);
        }

        // Service the most intrusive fault per page, only. Waive the rest
        if (is_duplicate) {
            fault_entry_duplicate_flags(batch_context, current_entry, previous_entry);

            // The previous fault was non-fatal so the page has been already
            // serviced
            if (!previous_entry->is_fatal)
                continue;
        }

        service_access_type = check_fault_access_permissions(gpu,
                                                             batch_context,
                                                             va_block,
                                                             block_context,
                                                             current_entry,
                                                             iter.migratable);

        // Do not exit early due to logical errors such as access permission
        // violation.
        if (service_access_type == UVM_FAULT_ACCESS_TYPE_COUNT)
            continue;

        if (service_access_type != current_entry->fault_access_type) {
            // Some of the fault instances cannot be serviced due to invalid
            // access permissions. Recompute the access type service mask to
            // service the rest.
            UVM_ASSERT(service_access_type < current_entry->fault_access_type);
            service_access_type_mask = uvm_fault_access_type_mask_bit(service_access_type);
        }
        else {
            service_access_type_mask = current_entry->access_type_mask;
        }

        // If the GPU already has the necessary access permission, the fault
        // does not need to be serviced
        if (uvm_va_block_page_is_gpu_authorized(va_block,
                                                page_index,
                                                gpu->id,
                                                uvm_fault_access_type_to_prot(service_access_type))) {
            account_stale_fault(batch_context,
                                current_entry,
                                page_index,
                                &pages_serviced_in_batch);
            continue;
        }

        thrashing_hint = uvm_perf_thrashing_get_hint(va_block,
                                                     block_context->block_context,
                                                     current_entry->fault_address,
                                                     gpu->id);
        if (thrashing_hint.type == UVM_PERF_THRASHING_HINT_TYPE_THROTTLE) {
            // Throttling is implemented by sleeping in the fault handler on
            // the CPU and by continuing to process faults on other pages on
            // the GPU
            //
            // Only update the flag the first time since logical permissions
            // cannot change while we hold the VA space lock.
            // TODO: Bug 1750144: That might not be true with HMM.
            if (block_context->num_retries == 0)
                mark_fault_throttled(batch_context, current_entry);

            continue;
        }
        else if (thrashing_hint.type == UVM_PERF_THRASHING_HINT_TYPE_PIN) {
            if (block_context->thrashing_pin_count++ == 0)
                uvm_page_mask_zero(&block_context->thrashing_pin_mask);

            uvm_page_mask_set(&block_context->thrashing_pin_mask, page_index);
        }

        // Compute new residency and update the masks
        new_residency = uvm_va_block_select_residency(va_block,
                                                      block_context->block_context,
                                                      page_index,
                                                      gpu->id,
                                                      service_access_type_mask,
                                                      policy,
                                                      &thrashing_hint,
                                                      UVM_SERVICE_OPERATION_REPLAYABLE_FAULTS,
                                                      hmm_migratable,
                                                      &read_duplicate);

        if (!uvm_processor_mask_test_and_set(&block_context->resident_processors, new_residency))
            uvm_page_mask_zero(&block_context->per_processor_masks[uvm_id_value(new_residency)].new_residency);

        uvm_page_mask_set(&block_context->per_processor_masks[uvm_id_value(new_residency)].new_residency, page_index);
        uvm_page_mask_set(&pages_serviced_in_batch, page_index);
        if (stale_detail_enabled())
            tracked_stale_prev_lag_record_service(current_entry->fault_address, batch_context->batch_id);

        if (read_duplicate) {
            if (block_context->read_duplicate_count++ == 0)
                uvm_page_mask_zero(&block_context->read_duplicate_mask);

            uvm_page_mask_set(&block_context->read_duplicate_mask, page_index);
        }

        ++page_fault_count;

        block_context->access_type[page_index] = service_access_type;

        if (page_index < first_page_index)
            first_page_index = page_index;
        if (page_index > last_page_index)
            last_page_index = page_index;
    }

    // Apply the changes computed in the fault service block context, if there
    // are pages to be serviced
    if (page_fault_count > 0) {
        block_context->region = uvm_va_block_region(first_page_index, last_page_index + 1);
        status = uvm_va_block_service_locked(gpu->id, va_block, va_block_retry, block_context);
    }

    *block_faults = i - first_fault_index;

    ++block_context->num_retries;

    if (status == NV_OK && batch_context->fatal_va_space)
        status = uvm_va_block_set_cancel(va_block, block_context->block_context, gpu);

    return status;
}

// We notify the fault event for all faults within the block so that the
// performance heuristics are updated. The VA block lock is taken for the whole
// fault servicing although it might be temporarily dropped and re-taken if
// memory eviction is required.
//
// See the comments for function service_fault_batch_block_locked for
// implementation details and error codes.
static NV_STATUS service_fault_batch_block(uvm_gpu_t *gpu,
                                           uvm_va_block_t *va_block,
                                           uvm_fault_service_batch_context_t *batch_context,
                                           NvU32 first_fault_index,
                                           const bool hmm_migratable,
                                           NvU32 *block_faults)
{
    NV_STATUS status;
    uvm_va_block_retry_t va_block_retry;
    NV_STATUS tracker_status;
    uvm_service_block_context_t *fault_block_context = &gpu->parent->fault_buffer_info.replayable.block_service_context;

    fault_block_context->operation = UVM_SERVICE_OPERATION_REPLAYABLE_FAULTS;
    fault_block_context->num_retries = 0;

    if (uvm_va_block_is_hmm(va_block))
        uvm_hmm_migrate_begin_wait(va_block);

    uvm_mutex_lock(&va_block->lock);

    status = UVM_VA_BLOCK_RETRY_LOCKED(va_block, &va_block_retry,
                                       service_fault_batch_block_locked(gpu,
                                                                        va_block,
                                                                        &va_block_retry,
                                                                        batch_context,
                                                                        first_fault_index,
                                                                        hmm_migratable,
                                                                        block_faults));

    tracker_status = uvm_tracker_add_tracker_safe(&batch_context->tracker, &va_block->tracker);

    uvm_mutex_unlock(&va_block->lock);

    if (uvm_va_block_is_hmm(va_block))
        uvm_hmm_migrate_finish(va_block);

    return status == NV_OK? tracker_status: status;
}

typedef enum
{
    // Use this mode when calling from the normal fault servicing path
    FAULT_SERVICE_MODE_REGULAR,

    // Use this mode when servicing faults from the fault cancelling algorithm.
    // In this mode no replays are issued
    FAULT_SERVICE_MODE_CANCEL,
} fault_service_mode_t;

static void service_fault_batch_fatal(uvm_gpu_t *gpu,
                                      uvm_fault_service_batch_context_t *batch_context,
                                      NvU32 first_fault_index,
                                      NV_STATUS status,
                                      uvm_fault_cancel_va_mode_t cancel_va_mode,
                                      NvU32 *block_faults)
{
    uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[first_fault_index];
    const uvm_fault_buffer_entry_t *previous_entry = first_fault_index > 0 ?
                                                       batch_context->ordered_fault_cache[first_fault_index - 1] : NULL;
    bool is_duplicate = check_fault_entry_duplicate(current_entry, previous_entry);

    if (is_duplicate)
        fault_entry_duplicate_flags(batch_context, current_entry, previous_entry);

    if (current_entry->fault_access_type == UVM_FAULT_ACCESS_TYPE_PREFETCH)
        mark_fault_invalid_prefetch(batch_context, current_entry);
    else
        mark_fault_fatal(batch_context, current_entry, uvm_tools_status_to_fatal_fault_reason(status), cancel_va_mode);

    (*block_faults)++;
}

static void service_fault_batch_fatal_notify(uvm_gpu_t *gpu,
                                             uvm_fault_service_batch_context_t *batch_context,
                                             NvU32 first_fault_index,
                                             NV_STATUS status,
                                             uvm_fault_cancel_va_mode_t cancel_va_mode,
                                             NvU32 *block_faults)
{
    uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[first_fault_index];
    const uvm_fault_buffer_entry_t *previous_entry = first_fault_index > 0 ?
                                                       batch_context->ordered_fault_cache[first_fault_index - 1] : NULL;
    bool is_duplicate = check_fault_entry_duplicate(current_entry, previous_entry);

    service_fault_batch_fatal(gpu, batch_context, first_fault_index, status, cancel_va_mode, block_faults);

    update_batch_and_notify_fault(gpu, batch_context, NULL, UVM_ID_INVALID, current_entry, is_duplicate);
}

static NV_STATUS service_fault_batch_ats_sub_vma(uvm_gpu_va_space_t *gpu_va_space,
                                                 struct vm_area_struct *vma,
                                                 NvU64 base,
                                                 uvm_fault_service_batch_context_t *batch_context,
                                                 NvU32 fault_index_start,
                                                 NvU32 fault_index_end,
                                                 NvU32 *block_faults)
{
    NvU32 i;
    NV_STATUS status = NV_OK;
    uvm_gpu_t *gpu = gpu_va_space->gpu;
    uvm_ats_fault_context_t *ats_context = &batch_context->ats_context;
    const uvm_page_mask_t *read_fault_mask = &ats_context->read_fault_mask;
    const uvm_page_mask_t *write_fault_mask = &ats_context->write_fault_mask;
    const uvm_page_mask_t *reads_serviced_mask = &ats_context->reads_serviced_mask;
    uvm_page_mask_t *faults_serviced_mask = &ats_context->faults_serviced_mask;
    uvm_page_mask_t *accessed_mask = &ats_context->accessed_mask;

    UVM_ASSERT(vma);

    ats_context->client_type = UVM_FAULT_CLIENT_TYPE_GPC;

    uvm_page_mask_or(accessed_mask, write_fault_mask, read_fault_mask);

    status = uvm_ats_service_faults(gpu_va_space, vma, base, &batch_context->ats_context);

    // Remove prefetched pages from the serviced mask since fault servicing
    // failures belonging to prefetch pages need to be ignored.
    uvm_page_mask_and(faults_serviced_mask, faults_serviced_mask, accessed_mask);

    UVM_ASSERT(uvm_page_mask_subset(faults_serviced_mask, accessed_mask));

    if ((status != NV_OK) || uvm_page_mask_equal(faults_serviced_mask, accessed_mask)) {
        (*block_faults) += (fault_index_end - fault_index_start);
        return status;
    }

    // Check faults_serviced_mask and reads_serviced_mask for precise fault
    // attribution after calling the ATS servicing routine. The
    // errors returned from ATS servicing routine should only be
    // global errors such as OOM or ECC. uvm_gpu_service_replayable_faults()
    // handles global errors by calling cancel_fault_batch(). Precise
    // attribution isn't currently supported in such cases.
    //
    // Precise fault attribution for global errors can be handled by
    // servicing one fault at a time until fault servicing encounters an
    // error.
    // TODO: Bug 3989244: Precise ATS fault attribution for global errors.
    for (i = fault_index_start; i < fault_index_end; i++) {
        uvm_page_index_t page_index;
        uvm_fault_cancel_va_mode_t cancel_va_mode;
        uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[i];
        uvm_fault_access_type_t access_type = current_entry->fault_access_type;

        page_index = (current_entry->fault_address - base) / PAGE_SIZE;

        if (uvm_page_mask_test(faults_serviced_mask, page_index)) {
            (*block_faults)++;
            continue;
        }

        if (access_type <= UVM_FAULT_ACCESS_TYPE_READ) {
            cancel_va_mode = UVM_FAULT_CANCEL_VA_MODE_ALL;
        }
	else {
            UVM_ASSERT(access_type >= UVM_FAULT_ACCESS_TYPE_WRITE);
            if (uvm_fault_access_type_mask_test(current_entry->access_type_mask, UVM_FAULT_ACCESS_TYPE_READ) &&
                !uvm_page_mask_test(reads_serviced_mask, page_index))
                cancel_va_mode = UVM_FAULT_CANCEL_VA_MODE_ALL;
            else
                cancel_va_mode = UVM_FAULT_CANCEL_VA_MODE_WRITE_AND_ATOMIC;
        }

        service_fault_batch_fatal(gpu, batch_context, i, NV_ERR_INVALID_ADDRESS, cancel_va_mode, block_faults);
    }

    return status;
}

static void start_new_sub_batch(NvU64 *sub_batch_base,
                                NvU64 address,
                                NvU32 *sub_batch_fault_index,
                                NvU32 fault_index,
                                uvm_ats_fault_context_t *ats_context)
{
    uvm_page_mask_zero(&ats_context->read_fault_mask);
    uvm_page_mask_zero(&ats_context->write_fault_mask);

    *sub_batch_fault_index = fault_index;
    *sub_batch_base = UVM_VA_BLOCK_ALIGN_DOWN(address);
}

static NV_STATUS service_fault_batch_ats_sub(uvm_gpu_va_space_t *gpu_va_space,
                                             struct vm_area_struct *vma,
                                             uvm_fault_service_batch_context_t *batch_context,
                                             NvU32 fault_index,
                                             NvU64 outer,
                                             NvU32 *block_faults)
{
    NV_STATUS status = NV_OK;
    NvU32 i = fault_index;
    NvU32 sub_batch_fault_index;
    NvU64 sub_batch_base;
    uvm_fault_buffer_entry_t *previous_entry = NULL;
    uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[i];
    uvm_ats_fault_context_t *ats_context = &batch_context->ats_context;
    uvm_page_mask_t *read_fault_mask = &ats_context->read_fault_mask;
    uvm_page_mask_t *write_fault_mask = &ats_context->write_fault_mask;
    uvm_gpu_t *gpu = gpu_va_space->gpu;
    bool replay_per_va_block =
                        (gpu->parent->fault_buffer_info.replayable.replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_BLOCK);

    UVM_ASSERT(vma);

    outer = min(outer, (NvU64) vma->vm_end);

    start_new_sub_batch(&sub_batch_base, current_entry->fault_address, &sub_batch_fault_index, i, ats_context);

    do {
        uvm_page_index_t page_index;
        NvU64 fault_address = current_entry->fault_address;
        uvm_fault_access_type_t access_type = current_entry->fault_access_type;
        bool is_duplicate = check_fault_entry_duplicate(current_entry, previous_entry);

        // ATS faults can't be unserviceable, since unserviceable faults require
        // GMMU PTEs.
        UVM_ASSERT(!current_entry->is_fatal);

        i++;

        update_batch_and_notify_fault(gpu_va_space->gpu,
                                      batch_context,
                                      NULL,
                                      UVM_ID_INVALID,
                                      current_entry,
                                      is_duplicate);

        // End of sub-batch. Service faults gathered so far.
        if (fault_address >= (sub_batch_base + UVM_VA_BLOCK_SIZE)) {
            UVM_ASSERT(!uvm_page_mask_empty(read_fault_mask) || !uvm_page_mask_empty(write_fault_mask));

            status = service_fault_batch_ats_sub_vma(gpu_va_space,
                                                     vma,
                                                     sub_batch_base,
                                                     batch_context,
                                                     sub_batch_fault_index,
                                                     i - 1,
                                                     block_faults);
            if (status != NV_OK || replay_per_va_block)
                break;

            start_new_sub_batch(&sub_batch_base, fault_address, &sub_batch_fault_index, i - 1, ats_context);
        }

        page_index = (fault_address - sub_batch_base) / PAGE_SIZE;

        if ((access_type <= UVM_FAULT_ACCESS_TYPE_READ) ||
             uvm_fault_access_type_mask_test(current_entry->access_type_mask, UVM_FAULT_ACCESS_TYPE_READ))
            uvm_page_mask_set(read_fault_mask, page_index);

        if (access_type >= UVM_FAULT_ACCESS_TYPE_WRITE)
            uvm_page_mask_set(write_fault_mask, page_index);

        previous_entry = current_entry;
        current_entry = i < batch_context->num_coalesced_faults ? batch_context->ordered_fault_cache[i] : NULL;

    } while (current_entry &&
             (current_entry->fault_address < outer) &&
             (previous_entry->va_space == current_entry->va_space));

    // Service the last sub-batch.
    if ((status == NV_OK) && (!uvm_page_mask_empty(read_fault_mask) || !uvm_page_mask_empty(write_fault_mask))) {
        status = service_fault_batch_ats_sub_vma(gpu_va_space,
                                                 vma,
                                                 sub_batch_base,
                                                 batch_context,
                                                 sub_batch_fault_index,
                                                 i,
                                                 block_faults);
    }

    return status;
}

static NV_STATUS service_fault_batch_ats(uvm_gpu_va_space_t *gpu_va_space,
                                         struct mm_struct *mm,
                                         uvm_fault_service_batch_context_t *batch_context,
                                         NvU32 first_fault_index,
                                         NvU64 outer,
                                         NvU32 *block_faults)
{
    NvU32 i;
    NV_STATUS status = NV_OK;

    for (i = first_fault_index; i < batch_context->num_coalesced_faults;) {
        uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[i];
        const uvm_fault_buffer_entry_t *previous_entry = i > first_fault_index ?
                                                                       batch_context->ordered_fault_cache[i - 1] : NULL;
        NvU64 fault_address = current_entry->fault_address;
        struct vm_area_struct *vma;
        NvU32 num_faults_before = (*block_faults);

        if (previous_entry && (previous_entry->va_space != current_entry->va_space))
            break;

        if (fault_address >= outer)
            break;

        vma = find_vma_intersection(mm, fault_address, fault_address + 1);
        if (!vma) {
            // Since a vma wasn't found, cancel all accesses on the page since
            // cancelling write and atomic accesses will not cancel pending read
            // faults and this can lead to a deadlock since read faults need to
            // be serviced first before cancelling write faults.
            service_fault_batch_fatal_notify(gpu_va_space->gpu,
                                             batch_context,
                                             i,
                                             NV_ERR_INVALID_ADDRESS,
                                             UVM_FAULT_CANCEL_VA_MODE_ALL,
                                             block_faults);

            // Do not fail due to logical errors.
            status = NV_OK;

            break;
        }

        status = service_fault_batch_ats_sub(gpu_va_space, vma, batch_context, i, outer, block_faults);
        if (status != NV_OK)
            break;

        i += ((*block_faults) - num_faults_before);
    }

    return status;
}

// Extract block-finding logic from service_fault_batch_dispatch.
// Returns NV_OK and sets *out_block on success.
static NV_STATUS find_va_block_for_fault(uvm_va_space_t *va_space,
                                         uvm_fault_service_batch_context_t *batch_context,
                                         NvU32 fault_index,
                                         uvm_va_block_t **out_block)
{
    uvm_va_range_t *va_range = NULL;
    uvm_va_range_t *va_range_next;
    NvU64 fault_address = batch_context->ordered_fault_cache[fault_index]->fault_address;

    va_range_next = uvm_va_space_iter_first(va_space, fault_address, ~0ULL);
    if (va_range_next && (fault_address >= va_range_next->node.start))
        va_range = va_range_next;

    if (!va_range)
        return NV_ERR_INVALID_ADDRESS;

    return uvm_va_block_find_create_in_range(va_space, va_range, fault_address, out_block);
}


// Adaptive merged service: opens shared CE copy + MEMOPS map pushes and
// processes blocks one by one until MERGE_THRESHOLD faults are accumulated.
// This reduces 2N pushes to 2 for N merged blocks.
//
// Synchronization: each block's map commands must execute after its copy
// commands. block_map_gpu_to handles this by calling
// uvm_push_acquire_tracker(map_push, &va_block->tracker) which inserts a
// semaphore wait in the MEMOPS stream for the CE copy to complete.
// This works even while the copy push is still open because the semaphore
// entry is determined at push_begin time.
static NV_STATUS service_fault_batch_merged(uvm_gpu_t *gpu,
                                            uvm_va_space_t *va_space,
                                            uvm_fault_service_batch_context_t *batch_context,
                                            NvU32 first_fault_index,
                                            const bool hmm_migratable,
                                            NvU32 *out_block_faults)
{
    NV_STATUS status = NV_OK;
    uvm_va_block_context_t *va_block_context =
        gpu->parent->fault_buffer_info.replayable.block_service_context.block_context;
    uvm_push_t copy_push;
    uvm_push_t map_push;
    NvU32 total_faults = 0;
    NvU32 pos = first_fault_index;
    NvU32 blocks_processed = 0;

    status = uvm_push_begin(gpu->channel_manager,
                            UVM_CHANNEL_TYPE_CPU_TO_GPU,
                            &copy_push,
                            "Merged CE copy");
    if (status != NV_OK)
        return status;

    status = uvm_push_begin(gpu->channel_manager,
                            UVM_CHANNEL_TYPE_MEMOPS,
                            &map_push,
                            "Merged MEMOPS map");
    if (status != NV_OK) {
        uvm_push_end(&copy_push);
        return status;
    }

    va_block_context->merge_ext.copy_push = &copy_push;
    va_block_context->merge_ext.map_push = &map_push;

    while (total_faults < MERGE_THRESHOLD &&
           blocks_processed < MAX_MERGE_BLOCKS &&
           pos < batch_context->num_coalesced_faults) {

        uvm_fault_buffer_entry_t *entry = batch_context->ordered_fault_cache[pos];
        uvm_va_block_t *blk;
        NV_STATUS find_st;
        NvU32 block_faults_out = 0;

        if (entry->is_fatal || entry->va_space != va_space)
            break;

        find_st = find_va_block_for_fault(va_space, batch_context, pos, &blk);
        if (find_st != NV_OK)
            break;

        status = service_fault_batch_block(gpu, blk, batch_context, pos,
                                           hmm_migratable, &block_faults_out);
        total_faults += block_faults_out;
        pos += block_faults_out;
        blocks_processed++;

        if (status != NV_OK)
            break;
    }

    va_block_context->merge_ext.copy_push = NULL;
    va_block_context->merge_ext.map_push = NULL;
    uvm_push_end(&copy_push);
    uvm_push_end(&map_push);

    *out_block_faults = total_faults;
    return status;
}

static NV_STATUS service_fault_batch_dispatch(uvm_va_space_t *va_space,
                                              uvm_gpu_va_space_t *gpu_va_space,
                                              uvm_fault_service_batch_context_t *batch_context,
                                              NvU32 fault_index,
                                              NvU32 *block_faults,
                                              bool replay_per_va_block,
                                              const bool hmm_migratable)
{
    NV_STATUS status;
    uvm_va_range_t *va_range = NULL;
    uvm_va_range_t *va_range_next = NULL;
    uvm_va_block_t *va_block;
    uvm_gpu_t *gpu = gpu_va_space->gpu;
    uvm_va_block_context_t *va_block_context =
        gpu->parent->fault_buffer_info.replayable.block_service_context.block_context;
    uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[fault_index];
    struct mm_struct *mm = va_block_context->mm;
    NvU64 fault_address = current_entry->fault_address;

    (*block_faults) = 0;

    va_range_next = uvm_va_space_iter_first(va_space, fault_address, ~0ULL);
    if (va_range_next && (fault_address >= va_range_next->node.start)) {
        UVM_ASSERT(fault_address < va_range_next->node.end);

        va_range = va_range_next;
        va_range_next = uvm_va_space_iter_next(va_range_next, ~0ULL);
    }

    if (va_range)
        status = uvm_va_block_find_create_in_range(va_space, va_range, fault_address, &va_block);
    else if (mm)
        status = uvm_hmm_va_block_find_create(va_space, fault_address, &va_block_context->hmm.vma, &va_block);
    else
        status = NV_ERR_INVALID_ADDRESS;

    if (status == NV_OK) {
        status = service_fault_batch_block(gpu, va_block, batch_context, fault_index, hmm_migratable, block_faults);
    }
    else if ((status == NV_ERR_INVALID_ADDRESS) && uvm_ats_can_service_faults(gpu_va_space, mm)) {
        NvU64 outer = ~0ULL;

         UVM_ASSERT(replay_per_va_block ==
                    (gpu->parent->fault_buffer_info.replayable.replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_BLOCK));

        // Limit outer to the minimum of next va_range.start and first
        // fault_address' next UVM_GMMU_ATS_GRANULARITY alignment so that it's
        // enough to check whether the first fault in this dispatch belongs to a
        // GMMU region.
        if (va_range_next) {
            outer = min(va_range_next->node.start,
                           UVM_ALIGN_DOWN(fault_address + UVM_GMMU_ATS_GRANULARITY, UVM_GMMU_ATS_GRANULARITY));
        }

        // ATS lookups are disabled on all addresses within the same
        // UVM_GMMU_ATS_GRANULARITY as existing GMMU mappings (see documentation
        // in uvm_mmu.h). User mode is supposed to reserve VAs as appropriate to
        // prevent any system memory allocations from falling within the NO_ATS
        // range of other GMMU mappings, so this shouldn't happen during normal
        // operation. However, since this scenario may lead to infinite fault
        // loops, we handle it by canceling the fault.
        if (uvm_ats_check_in_gmmu_region(va_space, fault_address, va_range_next)) {
            service_fault_batch_fatal_notify(gpu,
                                             batch_context,
                                             fault_index,
                                             NV_ERR_INVALID_ADDRESS,
                                             UVM_FAULT_CANCEL_VA_MODE_ALL,
                                             block_faults);

            // Do not fail due to logical errors
            status = NV_OK;
        }
        else {
            status = service_fault_batch_ats(gpu_va_space, mm, batch_context, fault_index, outer, block_faults);
        }
    }
    else {
        service_fault_batch_fatal_notify(gpu,
                                         batch_context,
                                         fault_index,
                                         status,
                                         UVM_FAULT_CANCEL_VA_MODE_ALL,
                                         block_faults);

        // Do not fail due to logical errors
        status = NV_OK;
    }

    return status;
}

// Called when a fault in the batch has been marked fatal. Flush the buffer
// under the VA and mmap locks to remove any potential stale fatal faults, then
// service all new faults for just that VA space and cancel those which are
// fatal. Faults in other VA spaces are replayed when done and will be processed
// when normal fault servicing resumes.
static NV_STATUS service_fault_batch_for_cancel(uvm_gpu_t *gpu, uvm_fault_service_batch_context_t *batch_context)
{
    NV_STATUS status = NV_OK;
    NvU32 i;
    uvm_va_space_t *va_space = batch_context->fatal_va_space;
    uvm_gpu_va_space_t *gpu_va_space = NULL;
    struct mm_struct *mm;
    uvm_replayable_fault_buffer_info_t *replayable_faults = &gpu->parent->fault_buffer_info.replayable;
    uvm_service_block_context_t *service_context = &gpu->parent->fault_buffer_info.replayable.block_service_context;
    uvm_va_block_context_t *va_block_context = service_context->block_context;

    UVM_ASSERT(gpu->parent->replayable_faults_supported);
    UVM_ASSERT(va_space);

    // Perform the flush and re-fetch while holding the mmap_lock and the
    // VA space lock. This avoids stale faults because it prevents any vma
    // modifications (mmap, munmap, mprotect) from happening between the time HW
    // takes the fault and we cancel it.
    mm = uvm_va_space_mm_retain_lock(va_space);
    uvm_va_block_context_init(va_block_context, mm);
    uvm_va_space_down_read(va_space);

    // We saw fatal faults in this VA space before. Flush while holding
    // mmap_lock to make sure those faults come back (aren't stale).
    //
    // We need to wait until all old fault messages have arrived before
    // flushing, hence UVM_GPU_BUFFER_FLUSH_MODE_WAIT_UPDATE_PUT.
    status = fault_buffer_flush_locked(gpu,
                                       UVM_GPU_BUFFER_FLUSH_MODE_WAIT_UPDATE_PUT,
                                       UVM_FAULT_REPLAY_TYPE_START,
                                       batch_context);
    if (status != NV_OK)
        goto done;

    // Wait for the flush's replay to finish to give the legitimate faults a
    // chance to show up in the buffer again.
    status = uvm_tracker_wait(&replayable_faults->replay_tracker);
    if (status != NV_OK)
        goto done;

    // We expect all replayed faults to have arrived in the buffer so we can re-
    // service them. The replay-and-wait sequence above will ensure they're all
    // in the HW buffer. When GSP owns the HW buffer, we also have to wait for
    // GSP to copy all available faults from the HW buffer into the shadow
    // buffer.
    status = hw_fault_buffer_flush_locked(gpu->parent, HW_FAULT_BUFFER_FLUSH_MODE_MOVE);
    if (status != NV_OK)
        goto done;

    // If there is no GPU VA space for the GPU, ignore all faults in the VA
    // space. This can happen if the GPU VA space has been destroyed since we
    // unlocked the VA space in service_fault_batch. That means the fatal faults
    // are stale, because unregistering the GPU VA space requires preempting the
    // context and detaching all channels in that VA space. Restart fault
    // servicing from the top.
    gpu_va_space = uvm_gpu_va_space_get_by_parent_gpu(va_space, gpu->parent);
    if (!gpu_va_space)
        goto done;

    // Re-parse the new faults
    batch_context->num_invalid_prefetch_faults = 0;
    batch_context->num_duplicate_faults        = 0;
    batch_context->num_stale_faults            = 0;
    batch_context->num_stale_same_batch_faults = 0;
    batch_context->num_stale_prev_batch_faults = 0;
    batch_context->num_stale_prev_lag_unknown = 0;
    batch_context->num_stale_prev_lag_1       = 0;
    batch_context->num_stale_prev_lag_2_3     = 0;
    batch_context->num_stale_prev_lag_4_7     = 0;
    batch_context->num_stale_prev_lag_8_plus  = 0;
    batch_context->num_pred_prev_candidates   = 0;
    batch_context->num_pred_prev_true_stale   = 0;
    batch_context->num_pred_prev_missed_stale = 0;
    batch_context->num_pred_fast_skip_stale   = 0;
    batch_context->pred_adaptive_active_this_batch = false;
    batch_context->num_pred_fetch_lookup_calls = 0;
    batch_context->num_pred_fetch_lookup_unresolved = 0;
    batch_context->num_pred_fetch_lookup_hits  = 0;
    batch_context->num_pred_fetch_lookup_cmps  = 0;
    batch_context->num_pred_fetch_build_keys   = 0;
    batch_context->time_pred_fetch_lookup_ns   = 0;
    batch_context->time_pred_fetch_build_ns    = 0;
    batch_context->num_pred_prev_lookup_calls  = 0;
    batch_context->num_pred_prev_lookup_hits   = 0;
    batch_context->num_pred_prev_lookup_cmps   = 0;
    batch_context->num_pred_prev_build_keys    = 0;
    batch_context->time_pred_prev_lookup_ns    = 0;
    batch_context->time_pred_prev_build_ns     = 0;
    batch_context->time_parallel_group_ns      = 0;
    batch_context->time_parallel_workers_ns    = 0;
    batch_context->time_parallel_replay_ns     = 0;
    batch_context->num_tracked_faults          = 0;
    batch_context->num_tracked_stale_faults    = 0;
    batch_context->num_tracked_stale_prev_lag_unknown = 0;
    batch_context->num_tracked_stale_prev_lag_1       = 0;
    batch_context->num_tracked_stale_prev_lag_2_3     = 0;
    batch_context->num_tracked_stale_prev_lag_4_7     = 0;
    batch_context->num_tracked_stale_prev_lag_8_plus  = 0;
    batch_context->num_replays                 = 0;
    batch_context->fatal_va_space              = NULL;
    batch_context->has_throttled_faults        = false;

    status = fetch_fault_buffer_entries(gpu, batch_context, FAULT_FETCH_MODE_ALL);
    if (status != NV_OK)
        goto done;

    // No more faults left. Either the previously-seen fatal entry was stale, or
    // RM killed the context underneath us.
    if (batch_context->num_cached_faults == 0)
        goto done;

    ++batch_context->batch_id;

    status = preprocess_fault_batch(gpu, batch_context);
    if (status != NV_OK) {
        if (status == NV_WARN_MORE_PROCESSING_REQUIRED) {
            // Another flush happened due to stale faults or a context-fatal
            // error. The previously-seen fatal fault might not exist anymore,
            // so restart fault servicing from the top.
            status = NV_OK;
        }

        goto done;
    }

    // Search for the target VA space
    for (i = 0; i < batch_context->num_coalesced_faults; i++) {
        uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[i];
        UVM_ASSERT(current_entry->va_space);
        if (current_entry->va_space == va_space)
            break;
    }

    while (i < batch_context->num_coalesced_faults) {
        uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[i];

        if (current_entry->va_space != va_space)
            break;

        // service_fault_batch_dispatch() doesn't expect unserviceable faults.
        // Just cancel them directly.
        if (current_entry->is_fatal) {
            status = cancel_fault_precise_va(gpu, current_entry, UVM_FAULT_CANCEL_VA_MODE_ALL);
            if (status != NV_OK)
                break;

            ++i;
        }
        else {
            uvm_ats_fault_invalidate_t *ats_invalidate = &gpu->parent->fault_buffer_info.replayable.ats_invalidate;
            NvU32 block_faults;
            const bool hmm_migratable = true;

            ats_invalidate->tlb_batch_pending = false;

            // Service all the faults that we can. We only really need to search
            // for fatal faults, but attempting to service all is the easiest
            // way to do that.
            status = service_fault_batch_dispatch(va_space, gpu_va_space, batch_context, i, &block_faults, false, hmm_migratable);
            if (status != NV_OK) {
                // TODO: Bug 3900733: clean up locking in service_fault_batch().
                // We need to drop lock and retry. That means flushing and
                // starting over.
                if (status == NV_WARN_MORE_PROCESSING_REQUIRED || status == NV_WARN_MISMATCHED_TARGET)
                    status = NV_OK;

                break;
            }

            // Invalidate TLBs before cancel to ensure that fatal faults don't
            // get stuck in HW behind non-fatal faults to the same line.
            status = uvm_ats_invalidate_tlbs(gpu_va_space, ats_invalidate, &batch_context->tracker);
            if (status != NV_OK)
                break;

            while (block_faults-- > 0) {
                current_entry = batch_context->ordered_fault_cache[i];
                if (current_entry->is_fatal) {
                    status = cancel_fault_precise_va(gpu, current_entry, current_entry->replayable.cancel_va_mode);
                    if (status != NV_OK)
                        break;
                }

                ++i;
            }
        }
    }

done:
    uvm_va_space_up_read(va_space);
    uvm_va_space_mm_release_unlock(va_space, mm);

    if (status == NV_OK) {
        // There are two reasons to flush the fault buffer here.
        //
        // 1) Functional. We need to replay both the serviced non-fatal faults
        //    and the skipped faults in other VA spaces. The former need to be
        //    restarted and the latter need to be replayed so the normal fault
        //    service mechanism can fetch and process them.
        //
        // 2) Performance. After cancelling the fatal faults, a flush removes
        //    any potential duplicated fault that may have been added while
        //    processing the faults in this batch. This flush also avoids doing
        //    unnecessary processing after the fatal faults have been cancelled,
        //    so all the rest are unlikely to remain after a replay because the
        //    context is probably in the process of dying.
        status = fault_buffer_flush_locked(gpu,
                                           UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT,
                                           UVM_FAULT_REPLAY_TYPE_START,
                                           batch_context);
    }

    return status;
}
static unsigned uvm_parallel_fault_timing_debug = 0;
module_param(uvm_parallel_fault_timing_debug, uint, S_IRUGO | S_IWUSR);

unsigned uvm_fpd_profile_enable = 0;
module_param(uvm_fpd_profile_enable, uint, 0644);
MODULE_PARM_DESC(uvm_fpd_profile_enable,
                 "Enable FPD profiling in parallel fault path (0=off, 1=on)");

// Coordinator-side timing for the actual replayable-fault service loop.
// FPD's unmap/alloc/copy fields are worker sums and may overlap; these stage
// values deliberately retain the serial fetch/preprocess/service boundaries.
unsigned uvm_fault_stage_profile_enable = 0;
module_param(uvm_fault_stage_profile_enable, uint, 0644);
MODULE_PARM_DESC(uvm_fault_stage_profile_enable,
                 "Log replayable-fault fetch/preprocess/service timing (0=off, 1=on)");

// Per-batch fault count profiling: logs "[BATCH_PROF] count=N" to dmesg for
// every processed batch. Intended for use with prefault benchmarks to measure
// how many faults (after driver-level coalescing) arrive per fetch.
unsigned uvm_batch_profile_enable = 0;
module_param(uvm_batch_profile_enable, uint, 0644);
MODULE_PARM_DESC(uvm_batch_profile_enable,
                 "Log fault count per batch to dmesg: [BATCH_PROF] count=N (0=off, 1=on)");

// Per-batch page repetition profiling.  For each batch logs:
//   [PAGE_REP] raw=N unique=M
// where raw = total faults fetched (fault_index) and unique = after
// driver-level coalescing (num_coalesced_faults).  The ratio raw/unique
// is the average number of times each unique page appears in the fault buffer.
unsigned uvm_page_rep_profile_enable = 0;
module_param(uvm_page_rep_profile_enable, uint, 0644);
MODULE_PARM_DESC(uvm_page_rep_profile_enable,
                 "Log raw/unique fault counts per batch: [PAGE_REP] raw=N unique=M (0=off, 1=on)");

unsigned uvm_page_rep_detail_enable = 0;
module_param(uvm_page_rep_detail_enable, uint, 0644);
MODULE_PARM_DESC(uvm_page_rep_detail_enable,
                 "Log residual duplicate-page details after full sort+dedup: [PAGE_REP_DETAIL] batch/addr/access/utlb (0=off, 1=on)");

unsigned uvm_pushbuffer_profile_enable = 0;
module_param(uvm_pushbuffer_profile_enable, uint, 0644);
MODULE_PARM_DESC(uvm_pushbuffer_profile_enable,
                 "Enable low-noise pushbuffer profiling logs for batch and shared push summaries (0=off, 1=on)");

// Gate for the verbose [UVM_FAULT_DBG] per-fault-address dump.  Off by
// default to avoid flooding dmesg during normal profiling runs.
unsigned uvm_fault_dbg_enable = 0;
module_param(uvm_fault_dbg_enable, uint, 0644);
MODULE_PARM_DESC(uvm_fault_dbg_enable,
                 "Enable verbose per-fault [UVM_FAULT_DBG] dump to dmesg (0=off, 1=on)");

// If num_coalesced_faults is <= this threshold, bypass the parallel kthread
// path and run the original serial batch path. This avoids worker wake/sync
// overhead on tiny batches and keeps behavior close to baseline.
static unsigned uvm_parallel_tiny_batch_fallback = 16;
module_param(uvm_parallel_tiny_batch_fallback, uint, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_parallel_tiny_batch_fallback,
                 "Bypass parallel path for tiny batches (0=disabled, default=16 faults)");

// If unique VA blocks (num_groups) after Phase 1 grouping is <= this threshold,
// process them inline on the coordinator thread instead of dispatching to kthreads.
// This avoids wake/wait/lock-cycling overhead for batches with insufficient
// inter-block parallelism (e.g. 2DCONV with ~2.7 blocks/batch).
static unsigned uvm_parallel_block_threshold = 4;
module_param(uvm_parallel_block_threshold, uint, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_parallel_block_threshold,
                 "Min unique VA blocks for kthread dispatch (0=always use kthreads, default=4)");

// Adaptive pre-scan: estimate unique blocks from fault addresses BEFORE entering
// the parallel path.  When estimated blocks <= uvm_parallel_block_threshold, bypass
// parallel entirely and use the serial path (no Phase 1 grouping overhead).
// 0 = disabled (always enter parallel), 1 = enabled (default).
static unsigned uvm_parallel_prescan = 1;
module_param(uvm_parallel_prescan, uint, S_IRUGO | S_IWUSR);
MODULE_PARM_DESC(uvm_parallel_prescan,
                 "Pre-scan fault addresses to estimate blocks before choosing path (0=off, 1=on)");

// Profiling: count serial-bypassed batches (gated by uvm_fpd_profile_enable)
atomic_t g_prescan_bypass_count = ATOMIC_INIT(0);
atomic_t g_prescan_parallel_count = ATOMIC_INIT(0);
atomic64_t g_prescan_bypass_blocks_sum = ATOMIC64_INIT(0);
atomic64_t g_prescan_parallel_blocks_sum = ATOMIC64_INIT(0);

// Blocks-per-batch histogram: bin i = count of batches with i unique blocks
// Bin 0 unused, bin PRESCAN_HIST_MAX = overflow (batches with >= PRESCAN_HIST_MAX blocks)
#define PRESCAN_HIST_MAX 64
atomic_t g_prescan_block_hist[PRESCAN_HIST_MAX + 1];
atomic_t g_prescan_hist_initialized = ATOMIC_INIT(0);

static void log_pushbuffer_batch_profile(uvm_gpu_t *gpu, uvm_fault_service_batch_context_t *batch_context)
{
    if (!uvm_pushbuffer_profile_enable)
        return;

    pr_info("[PBUF_BATCH] gpu=%s batch=%u cached=%u coalesced=%u stale=%u dup=%u invalid_prefetch=%u replays=%u throttled=%u fatal=%u\n",
            uvm_gpu_name(gpu),
            batch_context->batch_id,
            batch_context->num_cached_faults,
            batch_context->num_coalesced_faults,
            batch_context->num_stale_faults,
            batch_context->num_duplicate_faults,
            batch_context->num_invalid_prefetch_faults,
            batch_context->num_replays,
            batch_context->has_throttled_faults ? 1U : 0U,
            batch_context->fatal_va_space != NULL ? 1U : 0U);
}

static NvU32 estimate_unique_blocks(uvm_fault_service_batch_context_t *batch_context)
{
    NvU32 count = 0;
    NvU64 last_block_start = ~0ULL;
    NvU32 i;

    for (i = 0; i < batch_context->num_coalesced_faults; i++) {
        uvm_fault_buffer_entry_t *entry = batch_context->ordered_fault_cache[i];
        NvU64 block_start = entry->fault_address & ~(UVM_VA_BLOCK_SIZE - 1);
        if (block_start != last_block_start) {
            count++;
            last_block_start = block_start;
        }
    }
    return count;
}

static NV_STATUS service_fault_batch_parallel(uvm_gpu_t *gpu,
                                              fault_service_mode_t service_mode,
                                              uvm_fault_service_batch_context_t *batch_context)
{
    NV_STATUS status = NV_OK;
    NvU32 i;
    uvm_va_space_t *va_space = NULL;
    uvm_gpu_va_space_t *gpu_va_space = NULL;
    struct mm_struct *mm = NULL;
    uvm_parallel_fault_context_t *parallel_ctx = NULL;
    uvm_va_space_snapshot_t va_space_snapshot;
    uvm_block_fault_group_t *groups = NULL;
    NvU32 num_groups = 0;
    uvm_va_block_t *current_block = NULL;
    NvU32 current_block_first_fault = 0;
    NvU32 current_block_fault_count = 0;
    const bool replay_per_va_block = service_mode != FAULT_SERVICE_MODE_CANCEL &&
                                     gpu->parent->fault_buffer_info.replayable.replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_BLOCK;
    ktime_t t_start = 0, t_phase1_end = 0, t_phase2_end = 0, t_end = 0;
    (void)0; /* t_classify removed — no thin/fat sorting */
    const bool need_timing = uvm_parallel_fault_timing_debug || uvm_fpd_profile_enable
                             || uvm_merge_profile_enable || uvm_fault_stage_profile_enable;

    UVM_ASSERT(gpu->parent->replayable_faults_supported);

    if (need_timing)
        t_start = ktime_get();

    parallel_ctx = gpu->parent->fault_buffer_info.replayable.parallel_fault_ctx;
    if (!parallel_ctx)
        return NV_ERR_INVALID_STATE;

    groups = parallel_ctx->groups;

    uvm_parallel_fault_context_init(parallel_ctx);

    // Phase 1: Group faults by VA block (serial).
    // Holds va_space read lock + mmap_lock during find_block operations.
    for (i = 0; i < batch_context->num_coalesced_faults; i++) {
        uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[i];
        uvm_va_range_t *va_range = NULL;
        uvm_va_block_t *va_block = NULL;
        NvU64 fault_address = current_entry->fault_address;

        UVM_ASSERT(current_entry->va_space);

        if (current_entry->va_space != va_space) {
            if (current_block && current_block_fault_count > 0 && num_groups < UVM_PARALLEL_FAULT_MAX_BLOCKS) {
                groups[num_groups].va_block = current_block;
                groups[num_groups].first_fault_index = current_block_first_fault;
                groups[num_groups].num_faults = current_block_fault_count;
                num_groups++;
            }
            current_block = NULL;
            current_block_fault_count = 0;

            if (va_space) {
                uvm_va_space_up_read(va_space);
                uvm_va_space_mm_release_unlock(va_space, mm);
                mm = NULL;
            }

            va_space = current_entry->va_space;

            mm = uvm_va_space_mm_retain_lock(va_space);
            uvm_va_space_down_read(va_space);

            gpu_va_space = uvm_gpu_va_space_get_by_parent_gpu(va_space, gpu->parent);
            if (!gpu_va_space) {
                continue;
            }

            uvm_va_space_snapshot_init(&va_space_snapshot, va_space, gpu_va_space, mm);
        }

        if (current_entry->is_fatal) {
            if (!batch_context->fatal_va_space) {
                batch_context->fatal_va_space = va_space;
            }
            continue;
        }

        if (!gpu_va_space)
            continue;

        if (current_block &&
            fault_address >= current_block->start &&
            fault_address <= current_block->end) {
            current_block_fault_count++;
            continue;
        }

        va_range = uvm_va_space_iter_first(va_space, fault_address, ~0ULL);
        if (va_range && fault_address >= va_range->node.start && fault_address < va_range->node.end) {
            status = uvm_va_block_find_create_in_range(va_space, va_range, fault_address, &va_block);
            if (status != NV_OK) {
                continue;
            }
        }
        else {
            continue;
        }

        if (va_block != current_block) {
            if (current_block && current_block_fault_count > 0 && num_groups < UVM_PARALLEL_FAULT_MAX_BLOCKS) {
                groups[num_groups].va_block = current_block;
                groups[num_groups].first_fault_index = current_block_first_fault;
                groups[num_groups].num_faults = current_block_fault_count;
                num_groups++;
            }

            current_block = va_block;
            current_block_first_fault = i;
            current_block_fault_count = 1;
        }
        else {
            current_block_fault_count++;
        }
    }

    if (current_block && current_block_fault_count > 0 && num_groups < UVM_PARALLEL_FAULT_MAX_BLOCKS) {
        groups[num_groups].va_block = current_block;
        groups[num_groups].first_fault_index = current_block_first_fault;
        groups[num_groups].num_faults = current_block_fault_count;
        num_groups++;
    }

    if (need_timing)
        t_phase1_end = ktime_get();

    // Block-count threshold fallback: if num_groups is too small for kthread
    // parallelism to overcome dispatch overhead, process inline on the
    // coordinator thread.  Phase 1 grouping results are fully reused.
    // The coordinator already holds va_space read lock and mm from Phase 1.
    if (uvm_parallel_fault_processing == 2 && parallel_ctx->kthread_pool &&
        uvm_parallel_block_threshold > 0 &&
        num_groups > 0 && num_groups <= uvm_parallel_block_threshold) {
        uvm_kthread_pool_t *pool = parallel_ctx->kthread_pool;
        uvm_parallel_worker_context_t *wctx = &pool->worker_contexts[0];
        NvU32 fb_i;

        if (wctx->service_context && wctx->block_context) {
            wctx->block_context->mm = va_space_snapshot.mm;

            for (fb_i = 0; fb_i < num_groups; fb_i++) {
                uvm_kthread_work_item_t *item = &pool->queue[fb_i];

                item->va_block          = groups[fb_i].va_block;
                item->gpu               = gpu;
                item->snapshot          = va_space_snapshot;
                item->batch_context     = batch_context;
                item->first_fault_index = groups[fb_i].first_fault_index;
                item->num_faults        = groups[fb_i].num_faults;
                item->hmm_migratable    = true;
                item->status            = NV_OK;
                uvm_tracker_init(&item->tracker);
                item->faults_serviced   = 0;
                item->has_fatal_faults  = false;

                item->status = service_block_faults_kthread(item, wctx);

                if (item->status != NV_OK && status == NV_OK)
                    status = item->status;

                {
                    NV_STATUS ts = uvm_tracker_add_tracker_safe(
                        &batch_context->tracker, &item->tracker);
                    if (ts != NV_OK && status == NV_OK)
                        status = ts;
                }
                uvm_tracker_deinit(&item->tracker);
            }

            pr_debug("[PAR_BLOCK_FALLBACK] blocks=%u faults=%u threshold=%u\n",
                     num_groups, batch_context->num_coalesced_faults,
                     uvm_parallel_block_threshold);

            if (va_space) {
                uvm_va_space_up_read(va_space);
                if (mm)
                    uvm_va_space_mm_release_unlock(va_space, mm);
                mm = NULL;
            }

            if (need_timing)
                t_phase2_end = ktime_get();

            goto phase3;
        }
    }

    // Phase 2: Dispatch via kthread pool.
    // All blocks go through fault-count-based segmentation when merge is enabled.
    // No thin/fat sorting — segments are formed by accumulating blocks until
    // total faults >= segment fault threshold.
    if (uvm_parallel_fault_processing == 2 && parallel_ctx->kthread_pool) {
        uvm_kthread_pool_t *pool = parallel_ctx->kthread_pool;

        pool->batch_gpu        = gpu;
        pool->batch_snapshot   = va_space_snapshot;
        pool->batch_ctx        = batch_context;
        pool->batch_groups     = groups;
        pool->batch_num_groups = num_groups;

        if (va_space) {
            uvm_va_space_up_read(va_space);
            if (mm)
                uvm_up_read_mmap_lock(mm);
        }

        {
            NvU64 gns = (need_timing && ktime_to_ns(t_phase1_end) > 0)
                        ? (NvU64)ktime_to_ns(ktime_sub(t_phase1_end, t_start)) : 0;
            status = uvm_kthread_dispatch_and_wait(pool, num_groups,
                                                   gns, &batch_context->tracker);
        }

        if (va_space && mm)
            uvm_va_space_mm_release(va_space);
    }

    if (need_timing)
        t_phase2_end = ktime_get();

phase3:
    // Phase 3: Replay
    if (replay_per_va_block && !batch_context->fatal_va_space && status == NV_OK) {
        if (num_groups > 0) {
            status = push_replay_on_gpu(gpu, UVM_FAULT_REPLAY_TYPE_START, batch_context);
            if (status == NV_OK)
                ++batch_context->batch_id;
        }
    }

    if (need_timing)
        t_end = ktime_get();

    if (uvm_fpd_profile_enable && num_groups > 0 &&
        uvm_parallel_fault_processing == 2 && parallel_ctx->kthread_pool) {
#define PAR_FPD_HIST_MAX 256
#define PAR_FPD_BD_PHASES 7
        uvm_kthread_pool_t *prof_pool = parallel_ctx->kthread_pool;
        NvU64 *fpd_bd[PAR_FPD_BD_PHASES];
        NvU64 fpd_time_ns[PAR_FPD_HIST_MAX + 1];
        NvU32 fpd_cnt[PAR_FPD_HIST_MAX + 1];
        NvU32 dispatch_iterations = num_groups;
        NvU32 total_block_faults_sum = 0;
        NvU32 max_block_faults = 0;
        NvU32 min_block_faults = ~0U;
        NvU64 total_dispatch_ns = 0;
        NvU32 block_dist_count_local = 0;
        struct { NvU32 first; NvU32 count; } block_dist[128];
        bool alloc_ok = true;
        int _k;

        for (_k = 0; _k < PAR_FPD_BD_PHASES; _k++) {
            fpd_bd[_k] = kzalloc((PAR_FPD_HIST_MAX + 1) * sizeof(NvU64), GFP_KERNEL);
            if (!fpd_bd[_k]) {
                while (--_k >= 0) kfree(fpd_bd[_k]);
                alloc_ok = false;
                break;
            }
        }

        if (alloc_ok) {
            NvU64 replay_ns = ktime_to_ns(ktime_sub(t_end, t_phase2_end));

            memset(fpd_time_ns, 0, sizeof(fpd_time_ns));
            memset(fpd_cnt, 0, sizeof(fpd_cnt));

            for (i = 0; i < num_groups; i++) {
                uvm_kthread_work_item_t *item = &prof_pool->queue[i];
                NvU32 bf = item->num_faults;
                NvU32 bf_idx = (bf <= PAR_FPD_HIST_MAX) ? bf : PAR_FPD_HIST_MAX;
                NvU64 item_ns = (item->dispatch_end_ns > item->dispatch_start_ns) ?
                                (item->dispatch_end_ns - item->dispatch_start_ns) : 0;

                fpd_time_ns[bf_idx] += item_ns;
                fpd_cnt[bf_idx]++;
                total_block_faults_sum += bf;
                total_dispatch_ns += item_ns;
                if (bf > max_block_faults) max_block_faults = bf;
                if (bf < min_block_faults) min_block_faults = bf;

                fpd_bd[0][bf_idx] += item->breakdown_ns.unmap_ns;
                fpd_bd[1][bf_idx] += item->breakdown_ns.alloc_ns;
                fpd_bd[2][bf_idx] += item->breakdown_ns.copy_ns;
                fpd_bd[3][bf_idx] += item->breakdown_ns.map_ns;
                fpd_bd[5][bf_idx] += item->breakdown_ns.subregion_count;
                fpd_bd[6][bf_idx] += item->breakdown_ns.pages_migrated;

                if (block_dist_count_local < 128) {
                    block_dist[block_dist_count_local].first = item->first_fault_index;
                    block_dist[block_dist_count_local].count = bf;
                    block_dist_count_local++;
                }
            }

            if (replay_ns > 0 && num_groups > 0) {
                NvU32 last_bf = prof_pool->queue[num_groups - 1].num_faults;
                NvU32 last_idx = (last_bf <= PAR_FPD_HIST_MAX) ? last_bf : PAR_FPD_HIST_MAX;
                fpd_bd[4][last_idx] += replay_ns;
            }

            {
                NvU64 avg_ns = dispatch_iterations > 0 ? total_dispatch_ns / dispatch_iterations : 0;
                NvU32 avg_faults = dispatch_iterations > 0 ? total_block_faults_sum / dispatch_iterations : 0;
                NvU64 avg_per_coalesced_ns = batch_context->num_coalesced_faults > 0 ?
                    total_dispatch_ns / batch_context->num_coalesced_faults : 0;
                NvU64 upper_bound_milli = max_block_faults > 0 ?
                    div64_u64((NvU64)total_block_faults_sum * 1000, max_block_faults) : 0;
                NvU32 max_ratio_pct = total_block_faults_sum > 0 ?
                    (NvU32)div64_u64((NvU64)max_block_faults * 100, total_block_faults_sum) : 0;
                pr_info("[BLOCK_FAULTS] cached=%u | coalesced=%u | duplicates=%u | "
                        "dispatches=%u | total_block_faults=%u | "
                        "avg_bf/dispatch=%u | min=%u | max=%u | "
                        "upper_bound_x1000=%llu | max_over_total_pct=%u | "
                        "total_time_ns=%llu | avg_time_ns/dispatch=%llu | "
                        "avg_time_ns/coalesced=%llu\n",
                        batch_context->num_cached_faults,
                        batch_context->num_coalesced_faults,
                        batch_context->num_cached_faults - batch_context->num_coalesced_faults,
                        dispatch_iterations,
                        total_block_faults_sum,
                        avg_faults,
                        (min_block_faults == ~0U) ? 0 : min_block_faults,
                        max_block_faults,
                        (unsigned long long)upper_bound_milli,
                        max_ratio_pct,
                        (unsigned long long)total_dispatch_ns,
                        (unsigned long long)avg_ns,
                        (unsigned long long)avg_per_coalesced_ns);
            }

            if (uvm_block_dist_count > 0) {
                NvU32 bd_j;
                uvm_block_dist_count--;
                pr_info("[VA_BLOCK_DIST] total_blocks=%u total_faults=%u\n",
                        block_dist_count_local, batch_context->num_coalesced_faults);
                for (bd_j = 0; bd_j < block_dist_count_local; bd_j++) {
                    NvU32 last = block_dist[bd_j].first + block_dist[bd_j].count - 1;
                    pr_info("  block[%u]: faults[%u~%u] (%u)\n",
                            bd_j, block_dist[bd_j].first, last, block_dist[bd_j].count);
                }
            }

            {
                char fpd_buf[900];
                int fpd_pos = 0;
                NvU32 k;
                for (k = 1; k <= PAR_FPD_HIST_MAX; k++) {
                    if (fpd_cnt[k] == 0)
                        continue;
                    if (fpd_pos >= 850) {
                        pr_info("[FPD_HIST] %s\n", fpd_buf);
                        fpd_pos = 0;
                    }
                    if (fpd_pos > 0)
                        fpd_pos += scnprintf(fpd_buf + fpd_pos,
                                             sizeof(fpd_buf) - fpd_pos, "|");
                    fpd_pos += scnprintf(fpd_buf + fpd_pos,
                                         sizeof(fpd_buf) - fpd_pos,
                                         "%u:%llu:%u", k,
                                         (unsigned long long)fpd_time_ns[k],
                                         fpd_cnt[k]);
                }
                if (fpd_pos > 0)
                    pr_info("[FPD_HIST] %s\n", fpd_buf);
            }

            {
                char bd_buf[1200];
                int bd_pos = 0;
                NvU32 k;
                for (k = 1; k <= PAR_FPD_HIST_MAX; k++) {
                    if (fpd_cnt[k] == 0)
                        continue;
                    if (bd_pos >= 950) {
                        pr_info("[FPD_BREAKDOWN] %s\n", bd_buf);
                        bd_pos = 0;
                    }
                    if (bd_pos > 0)
                        bd_pos += scnprintf(bd_buf + bd_pos,
                                            sizeof(bd_buf) - bd_pos, "|");
                    bd_pos += scnprintf(bd_buf + bd_pos,
                                        sizeof(bd_buf) - bd_pos,
                                        "%u:%llu:%llu:%llu:%llu:%llu:%u:%llu:%llu",
                                        k,
                                        (unsigned long long)fpd_bd[0][k],
                                        (unsigned long long)fpd_bd[1][k],
                                        (unsigned long long)fpd_bd[2][k],
                                        (unsigned long long)fpd_bd[3][k],
                                        (unsigned long long)fpd_bd[4][k],
                                        fpd_cnt[k],
                                        (unsigned long long)fpd_bd[5][k],
                                        (unsigned long long)fpd_bd[6][k]);
                }
                if (bd_pos > 0)
                    pr_info("[FPD_BREAKDOWN] %s\n", bd_buf);
            }

            for (_k = 0; _k < PAR_FPD_BD_PHASES; _k++)
                kfree(fpd_bd[_k]);
        }
#undef PAR_FPD_HIST_MAX
#undef PAR_FPD_BD_PHASES
    }

    if (need_timing && num_groups > 0) {
        s64 phase1_us = ktime_to_us(ktime_sub(t_phase1_end, t_start));
        s64 phase2_us = ktime_to_us(ktime_sub(t_phase2_end, t_phase1_end));
        s64 phase3_us = ktime_to_us(ktime_sub(t_end, t_phase2_end));
        s64 total_us = ktime_to_us(ktime_sub(t_end, t_start));
        if (uvm_fault_stage_profile_enable) {
            batch_context->time_parallel_group_ns += (NvU64)phase1_us * 1000;
            batch_context->time_parallel_workers_ns += (NvU64)phase2_us * 1000;
            batch_context->time_parallel_replay_ns += (NvU64)phase3_us * 1000;
        }
        if (uvm_parallel_fault_timing_debug) {
            pr_info("UVM:PARALLEL:TIMING faults=%u blocks=%u P1(group)=%lldus P2(parallel)=%lldus P3(replay)=%lldus total=%lldus\n",
                    batch_context->num_coalesced_faults, num_groups, phase1_us, phase2_us, phase3_us, total_us);
        }
    }

    return status;
}

// Scan the ordered view of faults and group them by different va_blocks
// (managed faults) and service faults for each va_block, in batch.
// Service non-managed faults one at a time as they are encountered during the
// scan.
//
// Fatal faults are marked for later processing by the caller.
static NV_STATUS service_fault_batch(uvm_gpu_t *gpu,
                                     fault_service_mode_t service_mode,
                                     uvm_fault_service_batch_context_t *batch_context)
{
    NV_STATUS status = NV_OK;
    NvU32 i;
    uvm_va_space_t *va_space = NULL;
    uvm_gpu_va_space_t *gpu_va_space = NULL;
    uvm_ats_fault_invalidate_t *ats_invalidate = &gpu->parent->fault_buffer_info.replayable.ats_invalidate;
    struct mm_struct *mm = NULL;
    const bool replay_per_va_block = service_mode != FAULT_SERVICE_MODE_CANCEL &&
                                     gpu->parent->fault_buffer_info.replayable.replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_BLOCK;
    uvm_service_block_context_t *service_context =
        &gpu->parent->fault_buffer_info.replayable.block_service_context;
    uvm_va_block_context_t *va_block_context = service_context->block_context;
    bool hmm_migratable = true;
    NvU64 *fpd_bd[7];

    UVM_ASSERT(gpu->parent->replayable_faults_supported);

    if (uvm_parallel_fault_enabled() &&
        service_mode == FAULT_SERVICE_MODE_REGULAR &&
        !(uvm_parallel_tiny_batch_fallback > 0 &&
          batch_context->num_coalesced_faults <= uvm_parallel_tiny_batch_fallback)) {

        // Adaptive pre-scan: estimate unique blocks from sorted fault addresses.
        // If too few blocks for profitable parallelism, stay on serial path.
        if (uvm_parallel_prescan && uvm_parallel_block_threshold > 0) {
            NvU32 est_blocks = estimate_unique_blocks(batch_context);

            if (uvm_fpd_profile_enable) {
                NvU32 bin = (est_blocks < PRESCAN_HIST_MAX) ? est_blocks : PRESCAN_HIST_MAX;
                atomic_inc(&g_prescan_block_hist[bin]);
            }

            if (est_blocks <= uvm_parallel_block_threshold) {
                if (uvm_fpd_profile_enable) {
                    atomic_inc(&g_prescan_bypass_count);
                    atomic64_add(est_blocks, &g_prescan_bypass_blocks_sum);
                }
                goto serial_path;
            }
            if (uvm_fpd_profile_enable) {
                atomic_inc(&g_prescan_parallel_count);
                atomic64_add(est_blocks, &g_prescan_parallel_blocks_sum);
            }
        }

        return service_fault_batch_parallel(gpu, service_mode, batch_context);
    }

    // Parallel is enabled but this tiny batch is forced down the serial path.
    if (uvm_parallel_fault_enabled() &&
        service_mode == FAULT_SERVICE_MODE_REGULAR &&
        uvm_parallel_tiny_batch_fallback > 0 &&
        batch_context->num_coalesced_faults <= uvm_parallel_tiny_batch_fallback) {
        pr_debug("[PAR_TINY_FALLBACK] faults=%u threshold=%u\n",
                 batch_context->num_coalesced_faults,
                 uvm_parallel_tiny_batch_fallback);
    }

serial_path:

    ats_invalidate->tlb_batch_pending = false;

#define FPD_HIST_MAX 256
#define FPD_BD_PHASES 7
    {
        int _k;
        if (uvm_fpd_profile_enable) {
            for (_k = 0; _k < FPD_BD_PHASES; _k++) {
                fpd_bd[_k] = kzalloc((FPD_HIST_MAX + 1) * sizeof(NvU64), GFP_KERNEL);
                if (!fpd_bd[_k]) {
                    while (--_k >= 0) kfree(fpd_bd[_k]);
                    return NV_ERR_NO_MEMORY;
                }
            }
        } else {
            for (_k = 0; _k < FPD_BD_PHASES; _k++)
                fpd_bd[_k] = NULL;
        }
    }

    {
    NvU32 dispatch_iterations = 0;
    NvU32 total_block_faults_sum = 0;
    NvU32 max_block_faults = 0;
    NvU32 min_block_faults = ~0U;
    NvU64 dispatch_start_ns, dispatch_end_ns;
    NvU64 total_dispatch_ns = 0;
    struct { NvU32 first; NvU32 count; } block_dist[128];
    NvU32 block_dist_count_local = 0;
    NvU64 fpd_time_ns[FPD_HIST_MAX + 1];
    NvU32 fpd_cnt[FPD_HIST_MAX + 1];
    if (uvm_fpd_profile_enable) {
        memset(fpd_time_ns, 0, sizeof(fpd_time_ns));
        memset(fpd_cnt, 0, sizeof(fpd_cnt));
    }

    for (i = 0; i < batch_context->num_coalesced_faults;) {
        NvU32 block_faults;
        uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[i];
        uvm_fault_utlb_info_t *utlb = &batch_context->utlbs[current_entry->fault_source.utlb_id];

        UVM_ASSERT(current_entry->va_space);

        if (current_entry->va_space != va_space) {
            // Fault on a different va_space, drop the lock of the old one...
            if (va_space != NULL) {
                // TLB entries are invalidated per GPU VA space
                status = uvm_ats_invalidate_tlbs(gpu_va_space, ats_invalidate, &batch_context->tracker);
                if (status != NV_OK)
                    goto fail;

                uvm_va_space_up_read(va_space);
                uvm_va_space_mm_release_unlock(va_space, mm);
                mm = NULL;
            }

            va_space = current_entry->va_space;

            // ... and take the lock of the new one

            // If an mm is registered with the VA space, we have to retain it
            // in order to lock it before locking the VA space. It is guaranteed
            // to remain valid until we release. If no mm is registered, we
            // can only service managed faults, not ATS/HMM faults.
            mm = uvm_va_space_mm_retain_lock(va_space);
            uvm_va_block_context_init(va_block_context, mm);

            uvm_va_space_down_read(va_space);
            gpu_va_space = uvm_gpu_va_space_get_by_parent_gpu(va_space, gpu->parent);
        }

        // Some faults could be already fatal if they cannot be handled by
        // the UVM driver
        if (current_entry->is_fatal) {
            ++i;
            if (!batch_context->fatal_va_space)
                batch_context->fatal_va_space = va_space;

            utlb->has_fatal_faults = true;
            UVM_ASSERT(utlb->num_pending_faults > 0);
            continue;
        }

        if (!gpu_va_space) {
            // If there is no GPU VA space for the GPU, ignore the fault. This
            // can happen if a GPU VA space is destroyed without explicitly
            // freeing all memory ranges and there are stale entries in the
            // buffer that got fixed by the servicing in a previous batch.
            ++i;
            continue;
        }

        if (uvm_fpd_profile_enable) {
            dispatch_start_ns = NV_GETTIME();
            va_block_context->breakdown_ns.unmap_ns = 0;
            va_block_context->breakdown_ns.alloc_ns = 0;
            va_block_context->breakdown_ns.copy_ns  = 0;
            va_block_context->breakdown_ns.map_ns   = 0;
            va_block_context->breakdown_ns.subregion_count = 0;
            va_block_context->breakdown_ns.pages_migrated  = 0;
        }

        // if (uvm_merge_dispatch && !current_entry->is_fatal && gpu_va_space) {
        //     NvU32 check_pos = i + MERGE_THRESHOLD - 1;
        //     bool skip_merge = false;

        //     // Check 1: if MERGE_THRESHOLD faults are all in the same
        //     // 2MB block, it already batches well -- skip merge.
        //     if (check_pos < batch_context->num_coalesced_faults) {
        //         uvm_fault_buffer_entry_t *check_entry =
        //             batch_context->ordered_fault_cache[check_pos];
        //         if (check_entry->va_space == va_space &&
        //             !check_entry->is_fatal &&
        //             UVM_VA_BLOCK_ALIGN_DOWN(current_entry->fault_address) ==
        //             UVM_VA_BLOCK_ALIGN_DOWN(check_entry->fault_address)) {
        //             skip_merge = true;
        //         }
        //     }

        //     // Check 2: if the NEXT fault is in the same block, this block
        //     // has >=2 faults and can batch its own push efficiently.
        //     // Only merge when current block has exactly 1 fault (random
        //     // access pattern where each block has a single fault).
        //     if (!skip_merge && i + 1 < batch_context->num_coalesced_faults) {
        //         uvm_fault_buffer_entry_t *next_entry =
        //             batch_context->ordered_fault_cache[i + 1];
        //         if (next_entry->va_space == va_space &&
        //             !next_entry->is_fatal &&
        //             UVM_VA_BLOCK_ALIGN_DOWN(current_entry->fault_address) ==
        //             UVM_VA_BLOCK_ALIGN_DOWN(next_entry->fault_address)) {
        //             skip_merge = true;
        //         }
        //     }

        //     if (!skip_merge) {
        //         status = service_fault_batch_merged(gpu, va_space,
        //                                             batch_context, i,
        //                                             hmm_migratable,
        //                                             &block_faults);
        //         goto dispatch_done;
        //     }
        // }

        status = service_fault_batch_dispatch(va_space,
                                              gpu_va_space,
                                              batch_context,
                                              i,
                                              &block_faults,
                                              replay_per_va_block,
                                              hmm_migratable);

dispatch_done:
        if (uvm_fpd_profile_enable) {
            dispatch_end_ns = NV_GETTIME();
            total_dispatch_ns += (dispatch_end_ns - dispatch_start_ns);
            dispatch_iterations++;
            total_block_faults_sum += block_faults;
            if (block_faults > max_block_faults)
                max_block_faults = block_faults;
            if (block_faults < min_block_faults)
                min_block_faults = block_faults;

            {
                NvU32 bf_idx = (block_faults <= FPD_HIST_MAX) ? block_faults : FPD_HIST_MAX;
                fpd_time_ns[bf_idx] += (dispatch_end_ns - dispatch_start_ns);
                fpd_cnt[bf_idx]++;
                fpd_bd[0][bf_idx] += va_block_context->breakdown_ns.unmap_ns;
                fpd_bd[1][bf_idx] += va_block_context->breakdown_ns.alloc_ns;
                fpd_bd[2][bf_idx] += va_block_context->breakdown_ns.copy_ns;
                fpd_bd[3][bf_idx] += va_block_context->breakdown_ns.map_ns;
                fpd_bd[5][bf_idx] += va_block_context->breakdown_ns.subregion_count;
                fpd_bd[6][bf_idx] += va_block_context->breakdown_ns.pages_migrated;
            }
        }

        // TODO: Bug 3900733: clean up locking in service_fault_batch().
        if (status == NV_WARN_MORE_PROCESSING_REQUIRED || status == NV_WARN_MISMATCHED_TARGET) {
            if (status == NV_WARN_MISMATCHED_TARGET)
                hmm_migratable = false;
            uvm_va_space_up_read(va_space);
            uvm_va_space_mm_release_unlock(va_space, mm);
            mm = NULL;
            va_space = NULL;
            status = NV_OK;
            continue;
        }

        if (status != NV_OK)
            goto fail;

        if (uvm_fpd_profile_enable && block_dist_count_local < 128) {
            block_dist[block_dist_count_local].first = i;
            block_dist[block_dist_count_local].count = block_faults;
            block_dist_count_local++;
        }
        hmm_migratable = true;
        i += block_faults;

        // Don't issue replays in cancel mode
        if (replay_per_va_block && !batch_context->fatal_va_space) {
            NvU64 _replay_ts = uvm_fpd_profile_enable ? NV_GETTIME() : 0;
            status = push_replay_on_gpu(gpu, UVM_FAULT_REPLAY_TYPE_START, batch_context);
            if (uvm_fpd_profile_enable) {
                NvU32 bf_idx = (block_faults <= FPD_HIST_MAX) ? block_faults : FPD_HIST_MAX;
                fpd_bd[4][bf_idx] += NV_GETTIME() - _replay_ts;
            }
            if (status != NV_OK)
                goto fail;

            // Increment the batch id if UVM_PERF_FAULT_REPLAY_POLICY_BLOCK
            // is used, as we issue a replay after servicing each VA block
            // and we can service a number of VA blocks before returning.
            ++batch_context->batch_id;
        }
    }

    if (va_space != NULL) {
        NV_STATUS invalidate_status = uvm_ats_invalidate_tlbs(gpu_va_space, ats_invalidate, &batch_context->tracker);
        if (invalidate_status != NV_OK)
            status = invalidate_status;
    }

    if (uvm_fpd_profile_enable && dispatch_iterations > 0) {
        NvU64 avg_ns = total_dispatch_ns / dispatch_iterations;
        NvU32 avg_faults = total_block_faults_sum / dispatch_iterations;
        NvU64 avg_time_per_coalesced_ns = (batch_context->num_coalesced_faults > 0) ?
            total_dispatch_ns / batch_context->num_coalesced_faults : 0;
        NvU64 upper_bound_milli = max_block_faults > 0 ?
            div64_u64((NvU64)total_block_faults_sum * 1000, max_block_faults) : 0;
        NvU32 max_ratio_pct = total_block_faults_sum > 0 ?
            (NvU32)div64_u64((NvU64)max_block_faults * 100, total_block_faults_sum) : 0;
        pr_info("[BLOCK_FAULTS] cached=%u | coalesced=%u | duplicates=%u | "
                "dispatches=%u | total_block_faults=%u | "
                "avg_bf/dispatch=%u | min=%u | max=%u | "
                "upper_bound_x1000=%llu | max_over_total_pct=%u | "
                "total_time_ns=%llu | avg_time_ns/dispatch=%llu | "
                "avg_time_ns/coalesced=%llu\n",
                batch_context->num_cached_faults,
                batch_context->num_coalesced_faults,
                batch_context->num_cached_faults - batch_context->num_coalesced_faults,
                dispatch_iterations,
                total_block_faults_sum,
                avg_faults,
                (min_block_faults == ~0U) ? 0 : min_block_faults,
                max_block_faults,
                (unsigned long long)upper_bound_milli,
                max_ratio_pct,
                (unsigned long long)total_dispatch_ns,
                (unsigned long long)avg_ns,
                (unsigned long long)avg_time_per_coalesced_ns);
        if (uvm_block_dist_count > 0) {
            NvU32 bd_j;
            uvm_block_dist_count--;
            pr_info("[VA_BLOCK_DIST] total_blocks=%u total_faults=%u\n",
                    block_dist_count_local, batch_context->num_coalesced_faults);
            for (bd_j = 0; bd_j < block_dist_count_local; bd_j++) {
                NvU32 last = block_dist[bd_j].first + block_dist[bd_j].count - 1;
                pr_info("  block[%u]: faults[%u~%u] (%u)\n",
                        bd_j, block_dist[bd_j].first, last, block_dist[bd_j].count);
            }
        }

        {
            char fpd_buf[900];
            int fpd_pos = 0;
            NvU32 k;
            for (k = 1; k <= FPD_HIST_MAX; k++) {
                if (fpd_cnt[k] == 0)
                    continue;
                if (fpd_pos >= 850) {
                    pr_info("[FPD_HIST] %s\n", fpd_buf);
                    fpd_pos = 0;
                }
                if (fpd_pos > 0)
                    fpd_pos += scnprintf(fpd_buf + fpd_pos,
                                         sizeof(fpd_buf) - fpd_pos, "|");
                fpd_pos += scnprintf(fpd_buf + fpd_pos,
                                     sizeof(fpd_buf) - fpd_pos,
                                     "%u:%llu:%u", k,
                                     (unsigned long long)fpd_time_ns[k],
                                     fpd_cnt[k]);
            }
            if (fpd_pos > 0)
                pr_info("[FPD_HIST] %s\n", fpd_buf);
        }

        {
            char bd_buf[1200];
            int bd_pos = 0;
            NvU32 k;
            for (k = 1; k <= FPD_HIST_MAX; k++) {
                if (fpd_cnt[k] == 0)
                    continue;
                if (bd_pos >= 950) {
                    pr_info("[FPD_BREAKDOWN] %s\n", bd_buf);
                    bd_pos = 0;
                }
                if (bd_pos > 0)
                    bd_pos += scnprintf(bd_buf + bd_pos,
                                        sizeof(bd_buf) - bd_pos, "|");
                bd_pos += scnprintf(bd_buf + bd_pos,
                                    sizeof(bd_buf) - bd_pos,
                                    "%u:%llu:%llu:%llu:%llu:%llu:%u:%llu:%llu",
                                    k,
                                    (unsigned long long)fpd_bd[0][k],
                                    (unsigned long long)fpd_bd[1][k],
                                    (unsigned long long)fpd_bd[2][k],
                                    (unsigned long long)fpd_bd[3][k],
                                    (unsigned long long)fpd_bd[4][k],
                                    fpd_cnt[k],
                                    (unsigned long long)fpd_bd[5][k],
                                    (unsigned long long)fpd_bd[6][k]);
            }
            if (bd_pos > 0)
                pr_info("[FPD_BREAKDOWN] %s\n", bd_buf);
        }
    }
    } /* end of instrumentation block scope */

fail:
    {
        int _k;
        for (_k = 0; _k < FPD_BD_PHASES; _k++)
            kfree(fpd_bd[_k]);
    }
    if (va_space != NULL) {
        uvm_va_space_up_read(va_space);
        uvm_va_space_mm_release_unlock(va_space, mm);
    }

    return status;
}

// Tells if the given fault entry is the first one in its uTLB
static bool is_first_fault_in_utlb(uvm_fault_service_batch_context_t *batch_context, NvU32 fault_index)
{
    NvU32 i;
    NvU32 utlb_id = batch_context->fault_cache[fault_index].fault_source.utlb_id;

    for (i = 0; i < fault_index; ++i) {
        uvm_fault_buffer_entry_t *current_entry = &batch_context->fault_cache[i];

        // We have found a prior fault in the same uTLB
        if (current_entry->fault_source.utlb_id == utlb_id)
            return false;
    }

    return true;
}

// Compute the number of fatal and non-fatal faults for a page in the given uTLB
static void faults_for_page_in_utlb(uvm_fault_service_batch_context_t *batch_context,
                                    uvm_va_space_t *va_space,
                                    NvU64 addr,
                                    NvU32 utlb_id,
                                    NvU32 *fatal_faults,
                                    NvU32 *non_fatal_faults)
{
    NvU32 i;

    *fatal_faults = 0;
    *non_fatal_faults = 0;

    // Fault filtering is not allowed in the TLB-based fault cancel path
    UVM_ASSERT(batch_context->num_cached_faults == batch_context->num_coalesced_faults);

    for (i = 0; i < batch_context->num_cached_faults; ++i) {
        uvm_fault_buffer_entry_t *current_entry = &batch_context->fault_cache[i];

        if (current_entry->fault_source.utlb_id == utlb_id &&
            current_entry->va_space == va_space && current_entry->fault_address == addr) {
            // We have found the page
            if (current_entry->is_fatal)
                ++(*fatal_faults);
            else
                ++(*non_fatal_faults);
        }
    }
}

// Function that tells if there are addresses (reminder: they are aligned to 4K)
// with non-fatal faults only
static bool no_fatal_pages_in_utlb(uvm_fault_service_batch_context_t *batch_context,
                                   NvU32 start_index,
                                   NvU32 utlb_id)
{
    NvU32 i;

    // Fault filtering is not allowed in the TLB-based fault cancel path
    UVM_ASSERT(batch_context->num_cached_faults == batch_context->num_coalesced_faults);

    for (i = start_index; i < batch_context->num_cached_faults; ++i) {
        uvm_fault_buffer_entry_t *current_entry = &batch_context->fault_cache[i];

        if (current_entry->fault_source.utlb_id == utlb_id) {
            // We have found a fault for the uTLB
            NvU32 fatal_faults;
            NvU32 non_fatal_faults;

            faults_for_page_in_utlb(batch_context,
                                    current_entry->va_space,
                                    current_entry->fault_address,
                                    utlb_id,
                                    &fatal_faults,
                                    &non_fatal_faults);

            if (non_fatal_faults > 0 && fatal_faults == 0)
                return true;
        }
    }

    return false;
}

static void record_fatal_fault_helper(uvm_gpu_t *gpu, uvm_fault_buffer_entry_t *entry, UvmEventFatalReason reason)
{
    uvm_va_space_t *va_space;

    va_space = entry->va_space;
    UVM_ASSERT(va_space);
    uvm_va_space_down_read(va_space);
    // Record fatal fault event
    uvm_tools_record_gpu_fatal_fault(gpu->id, va_space, entry, reason);
    uvm_va_space_up_read(va_space);
}

// This function tries to find and issue a cancel for each uTLB that meets
// the requirements to guarantee precise fault attribution:
// - No new faults can arrive on the uTLB (uTLB is in lockdown)
// - The first fault in the buffer for a specific uTLB is fatal
// - There are no other addresses in the uTLB with non-fatal faults only
//
// This function and the related helpers iterate over faults as read from HW,
// not through the ordered fault view
//
// TODO: Bug 1766754
// This is very costly, although not critical for performance since we are
// cancelling.
// - Build a list with all the faults within a uTLB
// - Sort by uTLB id
static NV_STATUS try_to_cancel_utlbs(uvm_gpu_t *gpu, uvm_fault_service_batch_context_t *batch_context)
{
    NvU32 i;

    // Fault filtering is not allowed in the TLB-based fault cancel path
    UVM_ASSERT(batch_context->num_cached_faults == batch_context->num_coalesced_faults);

    for (i = 0; i < batch_context->num_cached_faults; ++i) {
        uvm_fault_buffer_entry_t *current_entry = &batch_context->fault_cache[i];
        uvm_fault_utlb_info_t *utlb = &batch_context->utlbs[current_entry->fault_source.utlb_id];
        NvU32 gpc_id = current_entry->fault_source.gpc_id;
        NvU32 utlb_id = current_entry->fault_source.utlb_id;
        NvU32 client_id = current_entry->fault_source.client_id;

        // Only fatal faults are considered
        if (!current_entry->is_fatal)
            continue;

        // Only consider uTLBs in lock-down
        if (!utlb->in_lockdown)
            continue;

        // Issue a single cancel per uTLB
        if (utlb->cancelled)
            continue;

        if (is_first_fault_in_utlb(batch_context, i) &&
            !no_fatal_pages_in_utlb(batch_context, i + 1, utlb_id)) {
            NV_STATUS status;

            record_fatal_fault_helper(gpu, current_entry, current_entry->fatal_reason);

            status = push_cancel_on_gpu_targeted(gpu,
                                                 current_entry->instance_ptr,
                                                 gpc_id,
                                                 client_id,
                                                 &batch_context->tracker);
            if (status != NV_OK)
                return status;

            utlb->cancelled = true;
        }
    }

    return NV_OK;
}

static NvU32 find_fatal_fault_in_utlb(uvm_fault_service_batch_context_t *batch_context,
                                      NvU32 utlb_id)
{
    NvU32 i;

    // Fault filtering is not allowed in the TLB-based fault cancel path
    UVM_ASSERT(batch_context->num_cached_faults == batch_context->num_coalesced_faults);

    for (i = 0; i < batch_context->num_cached_faults; ++i) {
        if (batch_context->fault_cache[i].is_fatal &&
            batch_context->fault_cache[i].fault_source.utlb_id == utlb_id)
            return i;
    }

    return i;
}

static NvU32 is_fatal_fault_in_buffer(uvm_fault_service_batch_context_t *batch_context,
                                      uvm_fault_buffer_entry_t *fault)
{
    NvU32 i;

    // Fault filtering is not allowed in the TLB-based fault cancel path
    UVM_ASSERT(batch_context->num_cached_faults == batch_context->num_coalesced_faults);

    for (i = 0; i < batch_context->num_cached_faults; ++i) {
        uvm_fault_buffer_entry_t *current_entry = &batch_context->fault_cache[i];
        if (cmp_fault_instance_ptr(current_entry, fault) == 0 &&
            current_entry->fault_address == fault->fault_address &&
            current_entry->fault_access_type == fault->fault_access_type &&
            current_entry->fault_source.utlb_id == fault->fault_source.utlb_id) {
            return true;
        }
    }

    return false;
}

// Cancel all faults in the given fault service batch context, even those not
// marked as fatal.
static NV_STATUS cancel_faults_all(uvm_gpu_t *gpu,
                                   uvm_fault_service_batch_context_t *batch_context,
                                   UvmEventFatalReason reason)
{
    NV_STATUS status = NV_OK;
    NV_STATUS fault_status;
    NvU32 i = 0;

    UVM_ASSERT(gpu->parent->fault_cancel_va_supported);
    UVM_ASSERT(reason != UvmEventFatalReasonInvalid);

    while (i < batch_context->num_coalesced_faults && status == NV_OK) {
        uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[i];
        uvm_va_space_t *va_space = current_entry->va_space;
        bool skip_va_space;

        UVM_ASSERT(va_space);

        uvm_va_space_down_read(va_space);

        // If there is no GPU VA space for the GPU, ignore all faults in
        // that VA space. This can happen if the GPU VA space has been
        // destroyed since we unlocked the VA space in service_fault_batch.
        // Ignoring the fault avoids targetting a PDB that might have been
        // reused by another process.
        skip_va_space = !uvm_gpu_va_space_get_by_parent_gpu(va_space, gpu->parent);

        for (;
             i < batch_context->num_coalesced_faults && current_entry->va_space == va_space;
             current_entry = batch_context->ordered_fault_cache[++i]) {
            uvm_fault_cancel_va_mode_t cancel_va_mode;

            if (skip_va_space)
                continue;

            if (current_entry->is_fatal) {
                UVM_ASSERT(current_entry->fatal_reason != UvmEventFatalReasonInvalid);
                cancel_va_mode = current_entry->replayable.cancel_va_mode;
            }
            else {
                current_entry->fatal_reason = reason;
                cancel_va_mode = UVM_FAULT_CANCEL_VA_MODE_ALL;
            }

            status = cancel_fault_precise_va(gpu, current_entry, cancel_va_mode);
            if (status != NV_OK)
                break;
        }

        uvm_va_space_up_read(va_space);
    }

    // Because each cancel itself triggers a replay, there may be a large number
    // of new duplicated faults in the buffer after cancelling all the known
    // ones. Flushing the buffer discards them to avoid unnecessary processing.
    fault_status = fault_buffer_flush_locked(gpu,
                                             UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT,
                                             UVM_FAULT_REPLAY_TYPE_START,
                                             batch_context);

    // We report the first encountered error.
    if (status == NV_OK)
        status = fault_status;

    return status;
}

// Function called when the system has found a global error and needs to
// trigger RC in RM.
static void cancel_fault_batch_tlb(uvm_gpu_t *gpu,
                                   uvm_fault_service_batch_context_t *batch_context,
                                   UvmEventFatalReason reason)
{
    NvU32 i;

    for (i = 0; i < batch_context->num_coalesced_faults; ++i) {
        NV_STATUS status = NV_OK;
        uvm_fault_buffer_entry_t *current_entry;
        uvm_fault_buffer_entry_t *coalesced_entry;

        current_entry = batch_context->ordered_fault_cache[i];

        // The list iteration below skips the entry used as 'head'.
        // Report the 'head' entry explicitly.
        uvm_va_space_down_read(current_entry->va_space);
        uvm_tools_record_gpu_fatal_fault(gpu->id, current_entry->va_space, current_entry, reason);

        list_for_each_entry(coalesced_entry, &current_entry->merged_instances_list, merged_instances_list)
            uvm_tools_record_gpu_fatal_fault(gpu->id, current_entry->va_space, coalesced_entry, reason);
        uvm_va_space_up_read(current_entry->va_space);

        // We need to cancel each instance pointer to correctly handle faults from multiple contexts.
        status = push_cancel_on_gpu_global(gpu, current_entry->instance_ptr, &batch_context->tracker);
        if (status != NV_OK)
            break;
    }
}

static void cancel_fault_batch(uvm_gpu_t *gpu,
                               uvm_fault_service_batch_context_t *batch_context,
                               UvmEventFatalReason reason)
{
    // Return code is ignored since we're on a global error path and wouldn't be
    // able to recover anyway.
    if (gpu->parent->fault_cancel_va_supported)
        cancel_faults_all(gpu, batch_context, reason);
    else
        cancel_fault_batch_tlb(gpu, batch_context, reason);
}


// Current fault cancel algorithm
//
// 1- Disable prefetching to avoid new requests keep coming and flooding the
// buffer.
// LOOP
//   2- Record one fatal fault per uTLB to check if it shows up after the replay
//   3- Flush fault buffer (REPLAY_TYPE_START_ACK_ALL to prevent new faults from
//      coming to TLBs with pending faults)
//   4- Wait for replay to finish
//   5- Fetch all faults from buffer
//   6- Check what uTLBs are in lockdown mode and can be cancelled
//   7- Preprocess faults (order per va_space, fault address, access type)
//   8- Service all non-fatal faults and mark all non-serviceable faults as fatal
//      6.1- If fatal faults are not found, we are done
//   9- Search for a uTLB which can be targeted for cancel, as described in
//      try_to_cancel_utlbs. If found, cancel it.
// END LOOP
// 10- Re-enable prefetching
//
// NOTE: prefetch faults MUST NOT trigger fault cancel. We make sure that no
// prefetch faults are left in the buffer by disabling prefetching and
// flushing the fault buffer afterwards (prefetch faults are not replayed and,
// therefore, will not show up again)
static NV_STATUS cancel_faults_precise_tlb(uvm_gpu_t *gpu, uvm_fault_service_batch_context_t *batch_context)
{
    NV_STATUS status;
    NV_STATUS tracker_status;
    uvm_replayable_fault_buffer_info_t *replayable_faults = &gpu->parent->fault_buffer_info.replayable;
    bool first = true;

    UVM_ASSERT(gpu->parent->replayable_faults_supported);

    // 1) Disable prefetching to avoid new requests keep coming and flooding
    //    the buffer
    if (gpu->parent->fault_buffer_info.prefetch_faults_enabled)
        gpu->parent->arch_hal->disable_prefetch_faults(gpu->parent);

    while (1) {
        NvU32 utlb_id;

        // 2) Record one fatal fault per uTLB to check if it shows up after
        // the replay. This is used to handle the case in which the uTLB is
        // being cancelled from behind our backs by RM. See the comment in
        // step 6.
        for (utlb_id = 0; utlb_id <= batch_context->max_utlb_id; ++utlb_id) {
            uvm_fault_utlb_info_t *utlb = &batch_context->utlbs[utlb_id];

            if (!first && utlb->has_fatal_faults) {
                NvU32 idx = find_fatal_fault_in_utlb(batch_context, utlb_id);
                UVM_ASSERT(idx < batch_context->num_cached_faults);

                utlb->prev_fatal_fault = batch_context->fault_cache[idx];
            }
            else {
                utlb->prev_fatal_fault.fault_address = (NvU64)-1;
            }
        }
        first = false;

        // 3) Flush fault buffer. After this call, all faults from any of the
        // faulting uTLBs are before PUT. New faults from other uTLBs can keep
        // arriving. Therefore, in each iteration we just try to cancel faults
        // from uTLBs that contained fatal faults in the previous iterations
        // and will cause the TLB to stop generating new page faults after the
        // following replay with type UVM_FAULT_REPLAY_TYPE_START_ACK_ALL.
        //
        // No need to use UVM_GPU_BUFFER_FLUSH_MODE_WAIT_UPDATE_PUT since we
        // don't care too much about old faults, just new faults from uTLBs
        // which faulted before the replay.
        status = fault_buffer_flush_locked(gpu,
                                           UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT,
                                           UVM_FAULT_REPLAY_TYPE_START_ACK_ALL,
                                           batch_context);
        if (status != NV_OK)
            break;

        // 4) Wait for replay to finish
        status = uvm_tracker_wait(&replayable_faults->replay_tracker);
        if (status != NV_OK)
            break;

        batch_context->num_invalid_prefetch_faults = 0;
        batch_context->num_duplicate_faults        = 0;
        batch_context->num_stale_faults            = 0;
        batch_context->num_stale_same_batch_faults = 0;
        batch_context->num_stale_prev_batch_faults = 0;
        batch_context->num_stale_prev_lag_unknown = 0;
        batch_context->num_stale_prev_lag_1       = 0;
        batch_context->num_stale_prev_lag_2_3     = 0;
        batch_context->num_stale_prev_lag_4_7     = 0;
        batch_context->num_stale_prev_lag_8_plus  = 0;
        batch_context->num_pred_prev_candidates   = 0;
        batch_context->num_pred_prev_true_stale   = 0;
        batch_context->num_pred_prev_missed_stale = 0;
        batch_context->num_pred_fast_skip_stale   = 0;
        batch_context->pred_adaptive_active_this_batch = false;
        batch_context->num_pred_fetch_lookup_calls = 0;
        batch_context->num_pred_fetch_lookup_unresolved = 0;
        batch_context->num_pred_fetch_lookup_hits  = 0;
        batch_context->num_pred_fetch_lookup_cmps  = 0;
        batch_context->num_pred_fetch_build_keys   = 0;
        batch_context->time_pred_fetch_lookup_ns   = 0;
        batch_context->time_pred_fetch_build_ns    = 0;
        batch_context->num_pred_prev_lookup_calls  = 0;
        batch_context->num_pred_prev_lookup_hits   = 0;
        batch_context->num_pred_prev_lookup_cmps   = 0;
        batch_context->num_pred_prev_build_keys    = 0;
        batch_context->time_pred_prev_lookup_ns    = 0;
        batch_context->time_pred_prev_build_ns     = 0;
        batch_context->time_parallel_group_ns      = 0;
        batch_context->time_parallel_workers_ns    = 0;
        batch_context->time_parallel_replay_ns     = 0;
        batch_context->num_tracked_faults          = 0;
        batch_context->num_tracked_stale_faults    = 0;
        batch_context->num_tracked_stale_prev_lag_unknown = 0;
        batch_context->num_tracked_stale_prev_lag_1       = 0;
        batch_context->num_tracked_stale_prev_lag_2_3     = 0;
        batch_context->num_tracked_stale_prev_lag_4_7     = 0;
        batch_context->num_tracked_stale_prev_lag_8_plus  = 0;
        batch_context->num_replays                 = 0;
        batch_context->fatal_va_space              = NULL;
        batch_context->has_throttled_faults        = false;

        // 5) Fetch all faults from buffer
        status = fetch_fault_buffer_entries(gpu, batch_context, FAULT_FETCH_MODE_ALL);
        if (status != NV_OK)
            break;

        ++batch_context->batch_id;

        UVM_ASSERT(batch_context->num_cached_faults == batch_context->num_coalesced_faults);

        // No more faults left, we are done
        if (batch_context->num_cached_faults == 0)
            break;

        // 6) Check what uTLBs are in lockdown mode and can be cancelled
        for (utlb_id = 0; utlb_id <= batch_context->max_utlb_id; ++utlb_id) {
            uvm_fault_utlb_info_t *utlb = &batch_context->utlbs[utlb_id];

            utlb->in_lockdown = false;
            utlb->cancelled   = false;

            if (utlb->prev_fatal_fault.fault_address != (NvU64)-1) {
                // If a previously-reported fault shows up again we can "safely"
                // assume that the uTLB that contains it is in lockdown mode
                // and no new translations will show up before cancel.
                // A fatal fault could only be removed behind our backs by RM
                // issuing a cancel, which only happens when RM is resetting the
                // engine. That means the instance pointer can't generate any
                // new faults, so we won't have an ABA problem where a new
                // fault arrives with the same state.
                if (is_fatal_fault_in_buffer(batch_context, &utlb->prev_fatal_fault))
                    utlb->in_lockdown = true;
            }
        }

        // 7) Preprocess faults
        status = preprocess_fault_batch(gpu, batch_context);
        if (status == NV_WARN_MORE_PROCESSING_REQUIRED)
            continue;
        else if (status != NV_OK)
            break;

        // 8) Service all non-fatal faults and mark all non-serviceable faults
        // as fatal
        status = service_fault_batch(gpu, FAULT_SERVICE_MODE_CANCEL, batch_context);
        UVM_ASSERT(batch_context->num_replays == 0);
        if (status == NV_ERR_NO_MEMORY)
            continue;
        else if (status != NV_OK)
            break;

        // No more fatal faults left, we are done
        if (!batch_context->fatal_va_space)
            break;

        // 9) Search for uTLBs that contain fatal faults and meet the
        // requirements to be cancelled
        try_to_cancel_utlbs(gpu, batch_context);
    }

    // 10) Re-enable prefetching
    if (gpu->parent->fault_buffer_info.prefetch_faults_enabled)
        gpu->parent->arch_hal->enable_prefetch_faults(gpu->parent);

    if (status == NV_OK)
        status = push_replay_on_gpu(gpu, UVM_FAULT_REPLAY_TYPE_START, batch_context);

    tracker_status = uvm_tracker_wait(&batch_context->tracker);

    return status == NV_OK? tracker_status: status;
}

static NV_STATUS cancel_faults_precise(uvm_gpu_t *gpu, uvm_fault_service_batch_context_t *batch_context)
{
    UVM_ASSERT(batch_context->fatal_va_space);
    if (gpu->parent->fault_cancel_va_supported)
        return service_fault_batch_for_cancel(gpu, batch_context);

    return cancel_faults_precise_tlb(gpu, batch_context);
}

static void enable_disable_prefetch_faults(uvm_parent_gpu_t *parent_gpu, uvm_fault_service_batch_context_t *batch_context)
{
    if (!parent_gpu->prefetch_fault_supported)
        return;

    // If more than 66% of faults are invalid prefetch accesses, disable
    // prefetch faults for a while.
    // num_invalid_prefetch_faults may be higher than the actual count. See the
    // comment in mark_fault_invalid_prefetch(..).
    // Some tests rely on this logic (and ratio) to correctly disable prefetch
    // fault reporting. If the logic changes, the tests will have to be changed.
    if (parent_gpu->fault_buffer_info.prefetch_faults_enabled &&
        uvm_perf_reenable_prefetch_faults_lapse_msec > 0 &&
        ((batch_context->num_invalid_prefetch_faults * 3 > parent_gpu->fault_buffer_info.max_batch_size * 2) ||
         (uvm_enable_builtin_tests &&
          parent_gpu->rm_info.isSimulated &&
          batch_context->num_invalid_prefetch_faults > 5))) {
        uvm_parent_gpu_disable_prefetch_faults(parent_gpu);
    }
    else if (!parent_gpu->fault_buffer_info.prefetch_faults_enabled) {
        NvU64 lapse = NV_GETTIME() - parent_gpu->fault_buffer_info.disable_prefetch_faults_timestamp;

        // Reenable prefetch faults after some time
        if (lapse > ((NvU64)uvm_perf_reenable_prefetch_faults_lapse_msec * (1000 * 1000)))
            uvm_parent_gpu_enable_prefetch_faults(parent_gpu);
    }
}

// P2: see uvm_perf_fault_pred_calibrate_ktime_iters above.
static void calibrate_ktime_overhead(void)
{
    unsigned iters = uvm_perf_fault_pred_calibrate_ktime_iters;
    unsigned i;
    NvU64 total_ns = 0;
    NvU64 min_ns = ~0ULL;
    NvU64 max_ns = 0;
    NvU64 avg_ns;

    if (iters == 0)
        return;

    // Self-disable first so a slow calibration run (e.g. a large iteration
    // count) can't be re-triggered by a concurrent fault-service call.
    uvm_perf_fault_pred_calibrate_ktime_iters = 0;

    for (i = 0; i < iters; ++i) {
        ktime_t t0 = ktime_get();
        ktime_t t1 = ktime_get();
        NvU64 delta_ns = (NvU64)ktime_to_ns(ktime_sub(t1, t0));

        total_ns += delta_ns;
        if (delta_ns < min_ns)
            min_ns = delta_ns;
        if (delta_ns > max_ns)
            max_ns = delta_ns;
    }

    avg_ns = total_ns / iters;

    // P4: the loop above measures a ktime_get()-delimited interval around an
    // empty body, which is precisely the additive bias every timed lookup
    // sample carries. Publish it so the correction is applied automatically
    // rather than depending on someone reading this line out of dmesg and
    // writing it back by hand. An explicit non-zero setting wins, so a
    // deliberate override isn't clobbered by a later calibration run.
    if (uvm_perf_fault_pred_timing_overhead_ns == 0)
        uvm_perf_fault_pred_timing_overhead_ns = (unsigned)avg_ns;

    pr_info("[KTIME_CALIBRATION] iters=%u avg_ns=%llu min_ns=%llu max_ns=%llu applied_bias_ns=%u\n",
            iters,
            (unsigned long long)avg_ns,
            (unsigned long long)min_ns,
            (unsigned long long)max_ns,
            uvm_perf_fault_pred_timing_overhead_ns);
}

void uvm_gpu_service_replayable_faults(uvm_gpu_t *gpu)
{
    NvU32 num_replays = 0;
    NvU32 num_batches = 0;
    NvU32 num_throttled = 0;
    NvU32 num_update_put_flushes = 0;
    NV_STATUS status = NV_OK;
    uvm_replayable_fault_buffer_info_t *replayable_faults = &gpu->parent->fault_buffer_info.replayable;
    uvm_fault_service_batch_context_t *batch_context = &replayable_faults->batch_service_context;
    NvU64 profile_fetch_ns = 0;
    NvU64 profile_preprocess_ns = 0;
    NvU64 profile_service_ns = 0;
    NvU64 profile_cached_faults = 0;
    NvU64 profile_coalesced_faults = 0;
    NvU64 profile_pred_fetch_lookup_ns = 0;
    NvU64 profile_pred_fetch_build_ns = 0;
    NvU64 profile_pred_fast_skips = 0;
    NvU64 profile_pred_prev_lookup_ns = 0;
    NvU64 profile_pred_prev_build_ns = 0;
    NvU64 profile_parallel_group_ns = 0;
    NvU64 profile_parallel_workers_ns = 0;
    NvU64 profile_parallel_replay_ns = 0;
    NvU32 profile_batches = 0;
    // O6: how many of profile_batches actually ran the filter, for
    // validating the adaptive gating decision (see
    // uvm_perf_fault_pred_adaptive_enable).
    NvU32 profile_adaptive_active_batches = 0;
    // P4: timed lookup samples the clock couldn't resolve above its own
    // overhead; see uvm_perf_fault_pred_timing_overhead_ns.
    NvU64 profile_pred_lookup_unresolved = 0;

    UVM_ASSERT(gpu->parent->replayable_faults_supported);

    calibrate_ktime_overhead();

    uvm_tracker_init(&batch_context->tracker);

    // Process all faults in the buffer
    while (1) {
        if (num_throttled >= uvm_perf_fault_max_throttle_per_service ||
            num_batches >= uvm_perf_fault_max_batches_per_service) {
            break;
        }

        batch_context->num_invalid_prefetch_faults = 0;
        batch_context->num_duplicate_faults        = 0;
        batch_context->num_stale_faults            = 0;
        batch_context->num_stale_same_batch_faults = 0;
        batch_context->num_stale_prev_batch_faults = 0;
        batch_context->num_stale_prev_lag_unknown = 0;
        batch_context->num_stale_prev_lag_1       = 0;
        batch_context->num_stale_prev_lag_2_3     = 0;
        batch_context->num_stale_prev_lag_4_7     = 0;
        batch_context->num_stale_prev_lag_8_plus  = 0;
        batch_context->num_pred_prev_candidates   = 0;
        batch_context->num_pred_prev_true_stale   = 0;
        batch_context->num_pred_prev_missed_stale = 0;
        batch_context->num_pred_fast_skip_stale   = 0;
        batch_context->pred_adaptive_active_this_batch = false;
        batch_context->num_pred_fetch_lookup_calls = 0;
        batch_context->num_pred_fetch_lookup_unresolved = 0;
        batch_context->num_pred_fetch_lookup_hits  = 0;
        batch_context->num_pred_fetch_lookup_cmps  = 0;
        batch_context->num_pred_fetch_build_keys   = 0;
        batch_context->time_pred_fetch_lookup_ns   = 0;
        batch_context->time_pred_fetch_build_ns    = 0;
        batch_context->num_pred_prev_lookup_calls  = 0;
        batch_context->num_pred_prev_lookup_hits   = 0;
        batch_context->num_pred_prev_lookup_cmps   = 0;
        batch_context->num_pred_prev_build_keys    = 0;
        batch_context->time_pred_prev_lookup_ns    = 0;
        batch_context->time_pred_prev_build_ns     = 0;
        batch_context->time_parallel_group_ns      = 0;
        batch_context->time_parallel_workers_ns    = 0;
        batch_context->time_parallel_replay_ns     = 0;
        batch_context->num_tracked_faults          = 0;
        batch_context->num_tracked_stale_faults    = 0;
        batch_context->num_tracked_stale_prev_lag_unknown = 0;
        batch_context->num_tracked_stale_prev_lag_1       = 0;
        batch_context->num_tracked_stale_prev_lag_2_3     = 0;
        batch_context->num_tracked_stale_prev_lag_4_7     = 0;
        batch_context->num_tracked_stale_prev_lag_8_plus  = 0;
        batch_context->num_replays                 = 0;
        batch_context->fatal_va_space              = NULL;
        batch_context->has_throttled_faults        = false;

        if (uvm_fault_stage_profile_enable) {
            NvU64 stage_start_ns = NV_GETTIME();
            status = fetch_fault_buffer_entries(gpu, batch_context, FAULT_FETCH_MODE_BATCH_READY);
            profile_fetch_ns += NV_GETTIME() - stage_start_ns;
        }
        else {
            status = fetch_fault_buffer_entries(gpu, batch_context, FAULT_FETCH_MODE_BATCH_READY);
        }
        if (status != NV_OK)
            break;

        if (batch_context->num_cached_faults == 0)
            break;

        ++batch_context->batch_id;

        if (uvm_fault_stage_profile_enable) {
            NvU64 stage_start_ns = NV_GETTIME();
            status = preprocess_fault_batch(gpu, batch_context);
            profile_preprocess_ns += NV_GETTIME() - stage_start_ns;
        }
        else {
            status = preprocess_fault_batch(gpu, batch_context);
        }

        // Keep the aggregate in the same low-volume stage profile record.
        // In contrast to uvm_batch_profile_enable, this does not emit one
        // kernel-log line per batch and therefore does not perturb a large
        // cold-fault run with dmesg I/O.
        if (uvm_fault_stage_profile_enable) {
            profile_cached_faults += batch_context->num_cached_faults;
            profile_coalesced_faults += batch_context->num_coalesced_faults;
            // These are intentionally kept in the aggregate stage record.
            // Per-batch logging perturbs large cold-fault experiments.
            profile_pred_fetch_lookup_ns += batch_context->time_pred_fetch_lookup_ns;
            profile_pred_fetch_build_ns += batch_context->time_pred_fetch_build_ns;
            profile_pred_fast_skips += batch_context->num_pred_fast_skip_stale;
            profile_pred_prev_lookup_ns += batch_context->time_pred_prev_lookup_ns;
            profile_pred_prev_build_ns += batch_context->time_pred_prev_build_ns;
            profile_pred_lookup_unresolved += batch_context->num_pred_fetch_lookup_unresolved;
            if (batch_context->pred_adaptive_active_this_batch)
                ++profile_adaptive_active_batches;
        }

        num_replays += batch_context->num_replays;

        if (uvm_batch_profile_enable && batch_context->num_coalesced_faults > 0)
            pr_info("[BATCH_PROF] count=%u\n", batch_context->num_coalesced_faults);

        if (status == NV_WARN_MORE_PROCESSING_REQUIRED)
            continue;
        else if (status != NV_OK)
            break;

        if (uvm_fault_stage_profile_enable) {
            NvU64 stage_start_ns = NV_GETTIME();
            status = service_fault_batch(gpu, FAULT_SERVICE_MODE_REGULAR, batch_context);
            profile_service_ns += NV_GETTIME() - stage_start_ns;
            ++profile_batches;
        }
        else {
            status = service_fault_batch(gpu, FAULT_SERVICE_MODE_REGULAR, batch_context);
        }

        if (uvm_fault_stage_profile_enable) {
            profile_parallel_group_ns += batch_context->time_parallel_group_ns;
            profile_parallel_workers_ns += batch_context->time_parallel_workers_ns;
            profile_parallel_replay_ns += batch_context->time_parallel_replay_ns;
        }

        // We may have issued replays even if status != NV_OK if
        // UVM_PERF_FAULT_REPLAY_POLICY_BLOCK is being used or the fault buffer
        // was flushed
        num_replays += batch_context->num_replays;

        enable_disable_prefetch_faults(gpu->parent, batch_context);

        if (status != NV_OK) {
            // Unconditionally cancel all faults to trigger RC. This will not
            // provide precise attribution, but this case handles global
            // errors such as OOM or ECC where it's not reasonable to
            // guarantee precise attribution. We ignore the return value of
            // the cancel operation since this path is already returning an
            // error code.
            cancel_fault_batch(gpu, batch_context, uvm_tools_status_to_fatal_fault_reason(status));
            break;
        }

        if (batch_context->fatal_va_space) {
            status = uvm_tracker_wait(&batch_context->tracker);
            if (status == NV_OK) {
                status = cancel_faults_precise(gpu, batch_context);
                if (status == NV_OK) {
                    // Cancel handling should've issued at least one replay
                    UVM_ASSERT(batch_context->num_replays > 0);
                    ++num_batches;
                    continue;
                }
            }

            break;
        }

        if (replayable_faults->replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_BATCH) {
            status = push_replay_on_gpu(gpu, UVM_FAULT_REPLAY_TYPE_START, batch_context);
            if (status != NV_OK)
                break;
            ++num_replays;
        }
        else if (replayable_faults->replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_BATCH_FLUSH) {
            uvm_gpu_buffer_flush_mode_t flush_mode = UVM_GPU_BUFFER_FLUSH_MODE_CACHED_PUT;
            NvU32 update_put_redundant_faults = batch_context->num_duplicate_faults;

            if (uvm_perf_fault_replay_force_update_put) {
                flush_mode = UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT;
                ++num_update_put_flushes;
            }
            else {
                if (uvm_perf_fault_replay_stale_update_put_enable)
                    update_put_redundant_faults += batch_context->num_stale_faults;

                if (update_put_redundant_faults * 100 >
                     batch_context->num_cached_faults *
                     replayable_faults->replay_update_put_ratio) {
                    flush_mode = UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT;
                    ++num_update_put_flushes;
                }
            }

            status = fault_buffer_flush_locked(gpu, flush_mode, UVM_FAULT_REPLAY_TYPE_START, batch_context);
            if (status != NV_OK)
                break;
            ++num_replays;
            status = uvm_tracker_wait(&replayable_faults->replay_tracker);
            if (status != NV_OK)
                break;
        }

        if (uvm_page_rep_profile_enable && batch_context->num_coalesced_faults > 0)
            pr_info("[STALE_PROF] batch_id=%llu faults=%u stale=%u stale_same_batch=%u stale_prev_batch=%u stale_prev_lag_unknown=%u stale_prev_lag_1=%u stale_prev_lag_2_3=%u stale_prev_lag_4_7=%u stale_prev_lag_8_plus=%u pred_prev_candidates=%u pred_prev_true_stale=%u pred_prev_missed_stale=%u pred_fast_skip_stale=%u pred_lookup_calls=%u pred_lookup_hits=%u pred_lookup_cmps=%u pred_build_keys=%u pred_lookup_ns=%llu pred_build_ns=%llu prev_book_lookup_calls=%u prev_book_lookup_hits=%u prev_book_lookup_cmps=%u prev_book_build_keys=%u prev_book_lookup_ns=%llu prev_book_build_ns=%llu dup=%u tracked_faults=%u tracked_stale=%u tracked_prev_lag_unknown=%u tracked_prev_lag_1=%u tracked_prev_lag_2_3=%u tracked_prev_lag_4_7=%u tracked_prev_lag_8_plus=%u replays=%u\n",
                    (unsigned long long)batch_context->batch_id,
                    batch_context->num_coalesced_faults,
                    batch_context->num_stale_faults,
                    batch_context->num_stale_same_batch_faults,
                    batch_context->num_stale_prev_batch_faults,
                    batch_context->num_stale_prev_lag_unknown,
                    batch_context->num_stale_prev_lag_1,
                    batch_context->num_stale_prev_lag_2_3,
                    batch_context->num_stale_prev_lag_4_7,
                    batch_context->num_stale_prev_lag_8_plus,
                    batch_context->num_pred_prev_candidates,
                    batch_context->num_pred_prev_true_stale,
                    batch_context->num_pred_prev_missed_stale,
                    batch_context->num_pred_fast_skip_stale,
                    batch_context->num_pred_fetch_lookup_calls,
                    batch_context->num_pred_fetch_lookup_hits,
                    batch_context->num_pred_fetch_lookup_cmps,
                    batch_context->num_pred_fetch_build_keys,
                    (unsigned long long)batch_context->time_pred_fetch_lookup_ns,
                    (unsigned long long)batch_context->time_pred_fetch_build_ns,
                    batch_context->num_pred_prev_lookup_calls,
                    batch_context->num_pred_prev_lookup_hits,
                    batch_context->num_pred_prev_lookup_cmps,
                    batch_context->num_pred_prev_build_keys,
                    (unsigned long long)batch_context->time_pred_prev_lookup_ns,
                    (unsigned long long)batch_context->time_pred_prev_build_ns,
                    batch_context->num_duplicate_faults,
                    batch_context->num_tracked_faults,
                    batch_context->num_tracked_stale_faults,
                    batch_context->num_tracked_stale_prev_lag_unknown,
                    batch_context->num_tracked_stale_prev_lag_1,
                    batch_context->num_tracked_stale_prev_lag_2_3,
                    batch_context->num_tracked_stale_prev_lag_4_7,
                    batch_context->num_tracked_stale_prev_lag_8_plus,
                    batch_context->num_replays);

        log_pushbuffer_batch_profile(gpu, batch_context);

        if (batch_context->has_throttled_faults)
            ++num_throttled;

        ++num_batches;
    }

    if (status == NV_WARN_MORE_PROCESSING_REQUIRED)
        status = NV_OK;

    // Make sure that we issue at least one replay if no replay has been
    // issued yet to avoid dropping faults that do not show up in the buffer
    if ((status == NV_OK && replayable_faults->replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_ONCE) ||
        num_replays == 0)
        status = push_replay_on_gpu(gpu, UVM_FAULT_REPLAY_TYPE_START, batch_context);

    uvm_tracker_deinit(&batch_context->tracker);

    if (uvm_page_rep_profile_enable && num_batches > 0)
        pr_info("[STALE_PROF] SUMMARY batches=%u update_put=%u\n",
                num_batches, num_update_put_flushes);

    if (uvm_fault_stage_profile_enable && profile_batches > 0) {
        pr_info("[FM_UVM_STAGES] gpu=%s batches=%u cached_faults=%llu coalesced_faults=%llu fetch_ns=%llu preprocess_ns=%llu service_ns=%llu pred_filter_lookup_ns=%llu pred_filter_build_ns=%llu pred_filter_fast_skips=%llu prev_book_lookup_ns=%llu prev_book_build_ns=%llu par_group_ns=%llu par_worker_wall_ns=%llu par_replay_ns=%llu adaptive_active_batches=%u lookup_bias_ns=%u lookup_unresolved=%llu\n",
                uvm_gpu_name(gpu), profile_batches,
                (unsigned long long)profile_cached_faults,
                (unsigned long long)profile_coalesced_faults,
                (unsigned long long)profile_fetch_ns,
                (unsigned long long)profile_preprocess_ns,
                (unsigned long long)profile_service_ns,
                (unsigned long long)profile_pred_fetch_lookup_ns,
                (unsigned long long)profile_pred_fetch_build_ns,
                (unsigned long long)profile_pred_fast_skips,
                (unsigned long long)profile_pred_prev_lookup_ns,
                (unsigned long long)profile_pred_prev_build_ns,
                (unsigned long long)profile_parallel_group_ns,
                (unsigned long long)profile_parallel_workers_ns,
                (unsigned long long)profile_parallel_replay_ns,
                profile_adaptive_active_batches,
                uvm_perf_fault_pred_timing_overhead_ns,
                (unsigned long long)profile_pred_lookup_unresolved);
    }

    if (status != NV_OK)
        UVM_DBG_PRINT("Error servicing replayable faults on GPU: %s\n", uvm_gpu_name(gpu));
}

void uvm_parent_gpu_enable_prefetch_faults(uvm_parent_gpu_t *parent_gpu)
{
    UVM_ASSERT(parent_gpu->isr.replayable_faults.handling);
    UVM_ASSERT(parent_gpu->prefetch_fault_supported);

    if (!parent_gpu->fault_buffer_info.prefetch_faults_enabled) {
        parent_gpu->arch_hal->enable_prefetch_faults(parent_gpu);
        parent_gpu->fault_buffer_info.prefetch_faults_enabled = true;
    }
}

void uvm_parent_gpu_disable_prefetch_faults(uvm_parent_gpu_t *parent_gpu)
{
    UVM_ASSERT(parent_gpu->isr.replayable_faults.handling);
    UVM_ASSERT(parent_gpu->prefetch_fault_supported);

    if (parent_gpu->fault_buffer_info.prefetch_faults_enabled) {
        parent_gpu->arch_hal->disable_prefetch_faults(parent_gpu);
        parent_gpu->fault_buffer_info.prefetch_faults_enabled = false;
        parent_gpu->fault_buffer_info.disable_prefetch_faults_timestamp = NV_GETTIME();
    }
}

const char *uvm_perf_fault_replay_policy_string(uvm_perf_fault_replay_policy_t replay_policy)
{
    BUILD_BUG_ON(UVM_PERF_FAULT_REPLAY_POLICY_MAX != 4);

    switch (replay_policy) {
        UVM_ENUM_STRING_CASE(UVM_PERF_FAULT_REPLAY_POLICY_BLOCK);
        UVM_ENUM_STRING_CASE(UVM_PERF_FAULT_REPLAY_POLICY_BATCH);
        UVM_ENUM_STRING_CASE(UVM_PERF_FAULT_REPLAY_POLICY_BATCH_FLUSH);
        UVM_ENUM_STRING_CASE(UVM_PERF_FAULT_REPLAY_POLICY_ONCE);
        UVM_ENUM_STRING_DEFAULT();
    }
}

NV_STATUS uvm_test_get_prefetch_faults_reenable_lapse(UVM_TEST_GET_PREFETCH_FAULTS_REENABLE_LAPSE_PARAMS *params,
                                                      struct file *filp)
{
    params->reenable_lapse = uvm_perf_reenable_prefetch_faults_lapse_msec;

    return NV_OK;
}

NV_STATUS uvm_test_set_prefetch_faults_reenable_lapse(UVM_TEST_SET_PREFETCH_FAULTS_REENABLE_LAPSE_PARAMS *params,
                                                      struct file *filp)
{
    uvm_perf_reenable_prefetch_faults_lapse_msec = params->reenable_lapse;

    return NV_OK;
}

NV_STATUS uvm_test_drain_replayable_faults(UVM_TEST_DRAIN_REPLAYABLE_FAULTS_PARAMS *params, struct file *filp)
{
    uvm_gpu_t *gpu;
    NV_STATUS status = NV_OK;
    uvm_spin_loop_t spin;
    bool pending = true;
    uvm_va_space_t *va_space = uvm_va_space_get(filp);

    gpu = uvm_va_space_retain_gpu_by_uuid(va_space, &params->gpu_uuid);
    if (!gpu)
        return NV_ERR_INVALID_DEVICE;

    uvm_spin_loop_init(&spin);

    do {
        uvm_parent_gpu_replayable_faults_isr_lock(gpu->parent);
        pending = uvm_parent_gpu_replayable_faults_pending(gpu->parent);
        uvm_parent_gpu_replayable_faults_isr_unlock(gpu->parent);

        if (!pending)
            break;

        if (fatal_signal_pending(current)) {
            status = NV_ERR_SIGNAL_PENDING;
            break;
        }

        UVM_SPIN_LOOP(&spin);
    } while (uvm_spin_loop_elapsed(&spin) < params->timeout_ns);

    if (pending && status == NV_OK)
        status = NV_ERR_TIMEOUT;

    uvm_gpu_release(gpu);

    return status;
}
