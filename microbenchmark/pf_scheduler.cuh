#ifndef __PF_SCHEDULER_CUH__
#define __PF_SCHEDULER_CUH__

// ============================================================================
// GPU Prefault Scheduler
//
// Persistent 128-thread block that drains a ring buffer of page-fault requests,
// radix-sorts by page number, deduplicates, and triggers HMM faults in parallel.
//
// Design mirrors io_scheduler (ring buffer + prod_seq + radix sort) but:
//   - No NVMe / BaM dependency (standalone CUDA header)
//   - No done_flags / done_seq (completion via per-page status table)
//   - Keys-only sort (no vals needed)
//   - Touch phase: volatile reads to trigger HMM page migration
// ============================================================================

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

// ============================================================================
// Constants
// ============================================================================
static constexpr uint32_t kPfSchedThreads  = 256;   // 8 warps
static constexpr uint32_t kPfMaxBatch      = 512;   // max entries per drain batch
static constexpr uint32_t kPfPageShift     = 12;    // 4KB pages
static constexpr uint32_t kPfPageSize      = 4096;
static constexpr uint32_t kPfRadixPasses   = 6;     // 6 passes x 8 bits = 48 bits
#ifndef PF_DRAIN_WAIT_MAX
#define PF_DRAIN_WAIT_MAX 8192
#endif
static constexpr uint32_t kPfDrainWaitMax  = PF_DRAIN_WAIT_MAX;  // ns, adaptive wait cap
static constexpr uint32_t kPfPagesPerVaBlock = 512;              // 2MB VA Block / 4KB page
static constexpr bool     kPfDebugBypassSortDedup = false;
static constexpr bool     kPfDebugValidateKeys    = true;

// ============================================================================
// Utility (standalone, no BaM headers)
// ============================================================================
__device__ __forceinline__ uint32_t pf_lane_id()
{
    uint32_t ret;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(ret));
    return ret;
}

__device__ __forceinline__ unsigned long long pf_globaltimer()
{
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// ============================================================================
// Page status enum
// ============================================================================
typedef enum : uint32_t {
    PF_UNKNOWN = 0,   // not on GPU, no fault requested
    PF_PENDING = 1,   // fault request enqueued to scheduler
    PF_READY   = 2    // page on GPU, safe to read
} pf_page_status_t;

// ============================================================================
// Ring buffer entry (8 bytes)
// ============================================================================
struct pf_request_t {
    uint64_t page_addr;   // page-aligned virtual address
};

// ============================================================================
// Prefault scheduler queue
// ============================================================================
struct pf_queue_t {
    // Ring buffer
    pf_request_t*      entries;       // [capacity]
    volatile uint32_t* prod_seq;      // [capacity] per-slot ready marker
    uint32_t           capacity;
    uint32_t           capacity_mask;

    // Cursors and control (use volatile + atomicXxx)
    uint32_t           prod_tail;     // next slot to claim
    uint32_t           cons_head;     // next slot to consume
    uint32_t           shutdown;      // 1 = exit after draining

    // Global page status table
    volatile pf_page_status_t* page_status;   // [n_pages]
    uint64_t           base_addr;     // page-aligned base of UVM allocation
    uint64_t           n_pages;

    // Radix sort double buffers (global memory, [capacity] each)
    uint64_t*          sort_keys;
    uint64_t*          sort_keys_alt;

    // Stats (written by consumer thread 0 on shutdown)
    uint64_t           total_requests;
    uint64_t           unique_pages;
    uint64_t           batch_count;
    uint64_t           sched_drain_ns;
    uint64_t           sched_sort_ns;
    uint64_t           sched_dedup_ns;
    uint64_t           sched_touch_ns;
    uint64_t           sched_notify_ns;
    uint64_t           sum_vblocks;     // sum of unique VA blocks across all batches
};

__device__ __forceinline__
bool pf_valid_page_id(const pf_queue_t* q, uint64_t page_addr, uint64_t* page_id)
{
    if (page_addr < q->base_addr) return false;
    uint64_t id = (page_addr - q->base_addr) >> kPfPageShift;
    if (id >= q->n_pages) return false;
    *page_id = id;
    return true;
}

// ============================================================================
// Device-side Producer API
// ============================================================================

// Scan prod_seq from 'start' forward, return count of consecutive ready entries.
__device__ __forceinline__
uint32_t pf_scan_ready(const pf_queue_t* q, uint32_t start, uint32_t max_count)
{
    uint32_t tail = *(volatile uint32_t*)&q->prod_tail;
    uint32_t pending = tail - start;
    if (pending > max_count) pending = max_count;
    for (uint32_t i = 0; i < pending; i++) {
        if (q->prod_seq[(start + i) & q->capacity_mask] != (start + i + 1u))
            return i;
    }
    return pending;
}

// Enqueue a page fault request. Returns true if this thread enqueued (won CAS).
// The atomicCAS on page_status ensures at most one enqueue per page globally.
__device__ __forceinline__
bool pf_sched_enqueue(pf_queue_t* q, uint64_t page_addr)
{
    uint64_t page_id = 0;
    if (!pf_valid_page_id(q, page_addr, &page_id)) return false;

    // Fast check: already ready?
    pf_page_status_t st = (pf_page_status_t)atomicAdd(
        (uint32_t*)&q->page_status[page_id], 0);
    if (st == PF_READY) return false;

    // Try to claim: UNKNOWN -> PENDING
    uint32_t old = atomicCAS((uint32_t*)&q->page_status[page_id],
                             (uint32_t)PF_UNKNOWN, (uint32_t)PF_PENDING);
    if (old != (uint32_t)PF_UNKNOWN) return false;  // someone else owns it

    // Won — enqueue to ring buffer
    uint32_t slot = atomicAdd(&q->prod_tail, 1u);
    uint32_t idx  = slot & q->capacity_mask;

    // Spin if queue full
    while ((slot - *(volatile uint32_t*)&q->cons_head) >= q->capacity) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(64);
#endif
    }

    // Write entry
    q->entries[idx].page_addr = page_addr;

    __threadfence();

    // Mark slot ready
    q->prod_seq[idx] = slot + 1u;

    return true;
}

// Full producer flow: enqueue if needed, then wait for page to be ready.
template <typename T>
__device__ __forceinline__
T pf_sched_load(pf_queue_t* q, const T* ptr, uint64_t i)
{
    const uintptr_t addr = reinterpret_cast<uintptr_t>(&ptr[i]);
    const uint64_t page_addr = addr & ~((uint64_t)kPfPageSize - 1);
    uint64_t page_id = 0;
    if (!pf_valid_page_id(q, page_addr, &page_id)) {
        printf("[PF_SCHED][BAD_LOAD] addr=0x%llx page_addr=0x%llx base=0x%llx i=%llu n_pages=%llu\n",
               (unsigned long long)addr,
               (unsigned long long)page_addr,
               (unsigned long long)q->base_addr,
               (unsigned long long)i,
               (unsigned long long)q->n_pages);
        asm volatile("trap;");
        return ptr[i];
    }

    // Fast path: already ready.
    pf_page_status_t st = (pf_page_status_t)atomicAdd(
        (uint32_t*)&q->page_status[page_id], 0);
    if (st == PF_READY) {
        const volatile T* vptr = (const volatile T*)ptr;
        return vptr[i];
    }

    // Warp coalesce: reduce intra-warp duplicate enqueues
    const uint32_t mask = __activemask();
    const uint32_t eq_mask = __match_any_sync(mask, (unsigned long long)page_id);
    const int master = __ffs(eq_mask) - 1;
    const uint32_t lane = pf_lane_id();

    if (lane == (uint32_t)master) {
        pf_sched_enqueue(q, page_addr);
    }

    // Spin on page_status until READY
    while (((pf_page_status_t)atomicAdd(
        (uint32_t*)&q->page_status[page_id], 0)) != PF_READY) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(100);
#endif
    }
    __syncwarp(eq_mask);

    // Acquire-like read side: after observing READY, fence and then perform
    // a volatile global load so data read is ordered after the status check.
    __threadfence();
    const volatile T* vptr = (const volatile T*)ptr;
    return vptr[i];
}

// Device-side shutdown signal
__device__ __forceinline__
void pf_sched_signal_shutdown(pf_queue_t* q)
{
    atomicExch(&q->shutdown, 1u);
}

// ============================================================================
// Consumer internals: keys-only radix sort
// ============================================================================
__device__ __noinline__
void pf_radix_sort_keys(
    uint64_t* __restrict__ keys,
    uint64_t* __restrict__ keys_alt,
    uint32_t N, uint32_t n_passes)
{
    __shared__ uint32_t s_hist[256];
    const uint32_t tid = threadIdx.x;

    for (uint32_t pass = 0; pass < n_passes; pass++) {
        const uint32_t shift = pass * 8;

        // 1. Clear histogram
        for (uint32_t i = tid; i < 256; i += kPfSchedThreads)
            s_hist[i] = 0;
        __syncthreads();

        // 2. Build histogram (128 threads)
        for (uint32_t i = tid; i < N; i += kPfSchedThreads)
            atomicAdd_block(&s_hist[(keys[i] >> shift) & 0xFF], 1);
        __syncthreads();

        // 3. Exclusive prefix sum (thread 0)
        if (tid == 0) {
            uint32_t s = 0;
            for (int d = 0; d < 256; d++) {
                uint32_t c = s_hist[d];
                s_hist[d] = s;
                s += c;
            }
        }
        __syncthreads();

        // 4. Serial scatter (thread 0, stable)
        if (tid == 0) {
            for (uint32_t i = 0; i < N; i++) {
                uint32_t d = (keys[i] >> shift) & 0xFF;
                uint32_t p = s_hist[d]++;
                keys_alt[p] = keys[i];
            }
        }
        __syncthreads();

        // 5. Swap src <-> alt
        { uint64_t* t = keys; keys = keys_alt; keys_alt = t; }
    }
    // n_passes is even -> result in original buffer
}

// ============================================================================
// Consumer internals: parallel dedup on sorted keys
// ============================================================================
__device__ __noinline__
uint32_t pf_dedup_sorted(uint64_t* keys, uint32_t N)
{
    __shared__ uint32_t sh_wmask[kPfSchedThreads / 32];
    __shared__ uint32_t sh_unique_count;
    __shared__ uint64_t sh_unique_buf[kPfMaxBatch];

    const uint32_t tid     = threadIdx.x;
    const uint32_t lane    = tid & 31;
    const uint32_t warp_id = tid >> 5;

    if (tid == 0) {
        sh_unique_count = 0;
    }
    __syncthreads();

    for (uint32_t chunk = 0; chunk < N; chunk += kPfSchedThreads) {
        uint32_t pos = chunk + tid;
        bool is_new = false;
        if (pos < N) {
            is_new = (pos == 0) || (keys[pos] != keys[pos - 1]);
        }

        uint32_t wmask = __ballot_sync(0xFFFFFFFF, is_new);
        if (lane == 0) sh_wmask[warp_id] = wmask;
        __syncthreads();

        // Thread 0: extract unique positions in order
        if (tid == 0) {
            for (int w = 0; w < (int)(kPfSchedThreads / 32); w++) {
                uint32_t m = sh_wmask[w];
                while (m != 0) {
                    uint32_t bit = __ffs(m) - 1;
                    uint32_t abs_pos = chunk + w * 32 + bit;
                    if (abs_pos < N) {
                        sh_unique_buf[sh_unique_count++] = keys[abs_pos];
                    }
                    m &= m - 1;
                }
            }
        }
        __syncthreads();
    }

    // Copy unique keys back to keys[0..n_unique-1]
    uint32_t n_unique = sh_unique_count;
    for (uint32_t i = tid; i < n_unique; i += kPfSchedThreads) {
        keys[i] = sh_unique_buf[i];
    }
    __syncthreads();

    return n_unique;
}

// ============================================================================
// Consumer persistent kernel
// ============================================================================
__global__ void pf_scheduler_kernel(pf_queue_t* q)
{
    if (blockIdx.x != 0) return;
    const uint32_t tid = threadIdx.x;
    if (tid >= kPfSchedThreads) return;

    __shared__ uint32_t sh_base_head;
    __shared__ uint32_t sh_accumulated;
    __shared__ uint32_t sh_avail;
    __shared__ uint32_t sh_wmask[kPfSchedThreads / 32];
    __shared__ uint32_t sh_drain_ns;
    __shared__ uint32_t sh_n_unique;

    uint64_t acc_drain = 0, acc_sort = 0, acc_dedup = 0;
    uint64_t acc_touch = 0, acc_notify = 0;
    uint64_t batch_count = 0;
    uint64_t total_reqs = 0, total_unique = 0;
    uint64_t sum_vblocks = 0;

    uint64_t* g_keys     = q->sort_keys;
    uint64_t* g_keys_alt = q->sort_keys_alt;

    __syncthreads();

    for (;;) {
        unsigned long long t0 = 0, t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0;
        if (tid == 0) t0 = pf_globaltimer();

        // ================================================================
        // Phase 1: DRAIN (parallel ballot scan + fused load)
        // ================================================================
        if (tid == 0) {
            sh_base_head   = *(volatile uint32_t*)&q->cons_head;
            sh_accumulated = 0;
            if (batch_count == 0) sh_drain_ns = kPfDrainWaitMax;
        }
        __syncthreads();

        for (;;) {
            // -- Parallel scan + fused drain --
            {
                const uint32_t lane_   = tid & 31;
                const uint32_t warp_id = tid >> 5;

                if (tid == 0) {
                    uint32_t pending = *(volatile uint32_t*)&q->prod_tail
                                     - (sh_base_head + sh_accumulated);
                    uint32_t room = kPfMaxBatch - sh_accumulated;
                    sh_avail = (pending < room) ? pending : room;
                    if (sh_avail > kPfSchedThreads) sh_avail = kPfSchedThreads;
                }
                __syncthreads();
                uint32_t round_max = sh_avail;

                uint32_t my_slot = sh_base_head + sh_accumulated + tid;
                uint32_t my_idx  = my_slot & q->capacity_mask;
                bool ready = (tid < round_max) &&
                             (q->prod_seq[my_idx] == my_slot + 1u);

                uint32_t wmask = __ballot_sync(0xFFFFFFFF, ready);
                if (lane_ == 0) sh_wmask[warp_id] = wmask;
                __syncthreads();

                // Thread 0: count consecutive ready entries
                if (tid == 0) {
                    uint32_t a = 0;
                    for (int w = 0; w < (int)(kPfSchedThreads / 32); w++) {
                        if (sh_wmask[w] == 0xFFFFFFFF) { a += 32; }
                        else { a += __ffs(~sh_wmask[w]) - 1; break; }
                    }
                    if (a > round_max) a = round_max;
                    sh_avail = a;
                }
                __syncthreads();

                uint32_t avail = sh_avail;

                if (avail > 0) {
                    // Fused drain: ready threads load entry -> sort key
                    if (tid < avail) {
                        pf_request_t e;
                        e.page_addr = (uint64_t)atomicAdd(
                            (unsigned long long*)&q->entries[my_idx].page_addr,
                            0ULL);
                        uint32_t acc = sh_accumulated;
                        uint64_t page_num = (e.page_addr - q->base_addr) >> kPfPageShift;
                        if (kPfDebugValidateKeys) {
                            if (e.page_addr < q->base_addr || page_num >= q->n_pages) {
                                printf("[PF_SCHED][BAD_DRAIN] slot=%u idx=%u page_addr=0x%llx base=0x%llx page_num=%llu n_pages=%llu\n",
                                       my_slot, my_idx,
                                       (unsigned long long)e.page_addr,
                                       (unsigned long long)q->base_addr,
                                       (unsigned long long)page_num,
                                       (unsigned long long)q->n_pages);
                                asm volatile("trap;");
                            }
                        }
                        g_keys[acc + tid] = page_num;
                    }
                    __syncthreads();
                    if (tid == 0) sh_accumulated += avail;
                    __syncthreads();
                    if (avail == kPfSchedThreads && sh_accumulated < kPfMaxBatch)
                        continue;  // full round, try more
                }
            }

            // -- Handle empty queue / shutdown / adaptive wait --
            {
                uint32_t avail = sh_avail;

                if (avail == 0 && sh_accumulated == 0) {
                    // Empty queue — check shutdown
                    if (*(volatile uint32_t*)&q->shutdown) {
                        if (tid == 0) {
                            sh_avail = pf_scan_ready(q, sh_base_head, q->capacity);
                        }
                        __syncthreads();
                        if (sh_avail == 0) {
                            // Save stats and exit
                            if (tid == 0) {
                                q->total_requests  = total_reqs;
                                q->unique_pages    = total_unique;
                                q->batch_count     = batch_count;
                                q->sched_drain_ns  = acc_drain;
                                q->sched_sort_ns   = acc_sort;
                                q->sched_dedup_ns  = acc_dedup;
                                q->sched_touch_ns  = acc_touch;
                                q->sched_notify_ns = acc_notify;
                                q->sum_vblocks     = sum_vblocks;
                            }
                            return;
                        }
                        // Found remaining entries, loop back to drain
                        continue;
                    }
                    // Not shutdown, idle wait
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
                    if (tid == 0) __nanosleep(64);
#endif
                    __syncthreads();
                    continue;
                }

                if (sh_accumulated == 0) continue;

                // Adaptive wait: try to accumulate more before sorting.
                // The old logic only waited when ns >= 64, which meant
                // PF_DRAIN_WAIT_MAX values below 64 effectively skipped the
                // wait path. Allow the scheduler to probe smaller wait caps too.
                bool drain_more = false;
                uint32_t ns = sh_drain_ns;
                while (ns > 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
                    if (tid == 0) __nanosleep(ns);
#endif
                    __syncthreads();
                    if (tid == 0) {
                        sh_avail = pf_scan_ready(q,
                            sh_base_head + sh_accumulated,
                            kPfMaxBatch - sh_accumulated);
                    }
                    __syncthreads();
                    if (sh_avail > 0) { drain_more = true; break; }
                    if (ns == 1) break;
                    ns /= 2;
                }
                if (drain_more) {
                    if (tid == 0 && sh_drain_ns < kPfDrainWaitMax) sh_drain_ns *= 2;
                    continue;
                }
                if (tid == 0 && sh_drain_ns > 1) sh_drain_ns /= 2;
                break;  // proceed to sort
            }
        }

        uint32_t accumulated = sh_accumulated;
        if (tid == 0) {
            t1 = pf_globaltimer(); acc_drain += (t1 - t0);
        }

        uint32_t n_unique = 0;
        if (!kPfDebugBypassSortDedup) {
            // ================================================================
            // Phase 2: SORT (radix sort, keys only)
            // ================================================================
            pf_radix_sort_keys(g_keys, g_keys_alt, accumulated, kPfRadixPasses);
            __syncthreads();
            if (tid == 0) {
                t2 = pf_globaltimer(); acc_sort += (t2 - t1);
            }

            // ================================================================
            // Phase 3: DEDUP (parallel boundary detection)
            // ================================================================
            n_unique = pf_dedup_sorted(g_keys, accumulated);
            if (tid == 0) sh_n_unique = n_unique;
            __syncthreads();
            n_unique = sh_n_unique;
            if (tid == 0) {
                t3 = pf_globaltimer(); acc_dedup += (t3 - t2);
            }
        } else {
            n_unique = accumulated;
            if (tid == 0) {
                t2 = pf_globaltimer();
                t3 = t2;
            }
        }

        // Count unique VA blocks in this batch (keys are sorted)
        if (tid == 0 && n_unique > 0) {
            uint32_t vb_count = 1;
            uint64_t prev_vb = g_keys[0] / kPfPagesPerVaBlock;
            for (uint32_t j = 1; j < n_unique; j++) {
                uint64_t cur_vb = g_keys[j] / kPfPagesPerVaBlock;
                if (cur_vb != prev_vb) { vb_count++; prev_vb = cur_vb; }
            }
            sum_vblocks += vb_count;
        }

        // ================================================================
        // Phase 4: TOUCH (128 threads parallel volatile reads)
        // ================================================================
        if (kPfDebugValidateKeys) {
            if (tid == 0) {
                for (uint32_t j = 0; j < n_unique; j++) {
                    if (g_keys[j] >= q->n_pages) {
                        printf("[PF_SCHED][BAD_KEY] j=%u key=%llu n_unique=%u accumulated=%u base=0x%llx n_pages=%llu\n",
                               j, (unsigned long long)g_keys[j], n_unique, accumulated,
                               (unsigned long long)q->base_addr,
                               (unsigned long long)q->n_pages);
                        asm volatile("trap;");
                    }
                }
            }
            __syncthreads();
        }
        for (uint32_t j = tid; j < n_unique; j += kPfSchedThreads) {
            uint64_t page_num  = g_keys[j];
            uint64_t touch_addr = q->base_addr + page_num * kPfPageSize;
            volatile uint64_t tmp = *(volatile uint64_t*)touch_addr;
            (void)tmp;
        }
        __syncthreads();
        if (tid == 0) {
            t4 = pf_globaltimer(); acc_touch += (t4 - t3);
        }

        // ================================================================
        // Phase 5: NOTIFY (page_status = READY, advance cons_head)
        // ================================================================
        __threadfence();  // ensure touches globally visible
        for (uint32_t j = tid; j < n_unique; j += kPfSchedThreads) {
            uint64_t page_num = g_keys[j];
            atomicExch((uint32_t*)&q->page_status[page_num], (uint32_t)PF_READY);
        }
        __threadfence();  // ensure READY visible to producers
        __syncthreads();

        if (tid == 0) {
            atomicAdd(&q->cons_head, accumulated);
            total_reqs   += accumulated;
            total_unique += n_unique;

            t5 = pf_globaltimer();
            acc_notify += (t5 - t4);
            batch_count++;
        }
        __syncthreads();
    }
}

// ============================================================================
// Host API: create / destroy / reset
// ============================================================================
inline pf_queue_t* pf_sched_create(uint32_t capacity,
                                    uint64_t base_addr,
                                    uint64_t n_pages,
                                    int cuda_device)
{
    if (capacity == 0 || (capacity & (capacity - 1)) != 0) {
        fprintf(stderr, "pf_sched_create: capacity must be power of 2 (got %u)\n",
                capacity);
        return nullptr;
    }

    cudaSetDevice(cuda_device);

    // Queue struct in managed memory (host + device accessible)
    pf_queue_t* q = nullptr;
    cudaMallocManaged(&q, sizeof(pf_queue_t));
    if (!q) {
        fprintf(stderr, "pf_sched_create: failed to allocate queue struct\n");
        return nullptr;
    }

    // Ring buffer entries
    pf_request_t* entries = nullptr;
    cudaMalloc(&entries, (size_t)capacity * sizeof(pf_request_t));
    cudaMemset(entries, 0, (size_t)capacity * sizeof(pf_request_t));

    // Per-slot ready markers
    volatile uint32_t* prod_seq = nullptr;
    cudaMalloc((void**)&prod_seq, (size_t)capacity * sizeof(uint32_t));
    cudaMemset((void*)prod_seq, 0, (size_t)capacity * sizeof(uint32_t));

    // Page status table
    volatile pf_page_status_t* page_status = nullptr;
    cudaMalloc((void**)&page_status, n_pages * sizeof(pf_page_status_t));
    cudaMemset((void*)page_status, 0, n_pages * sizeof(pf_page_status_t));

    // Sort buffers
    uint64_t* sort_keys = nullptr;
    uint64_t* sort_keys_alt = nullptr;
    cudaMalloc(&sort_keys,     (size_t)capacity * sizeof(uint64_t));
    cudaMalloc(&sort_keys_alt, (size_t)capacity * sizeof(uint64_t));

    q->entries       = entries;
    q->prod_seq      = prod_seq;
    q->capacity      = capacity;
    q->capacity_mask = capacity - 1;
    q->prod_tail     = 0;
    q->cons_head     = 0;
    q->shutdown      = 0;
    q->page_status   = page_status;
    q->base_addr     = base_addr;
    q->n_pages       = n_pages;
    q->sort_keys     = sort_keys;
    q->sort_keys_alt = sort_keys_alt;
    q->total_requests  = 0;
    q->unique_pages    = 0;
    q->batch_count     = 0;
    q->sched_drain_ns  = 0;
    q->sched_sort_ns   = 0;
    q->sched_dedup_ns  = 0;
    q->sched_touch_ns  = 0;
    q->sched_notify_ns = 0;
    q->sum_vblocks     = 0;

    cudaDeviceSynchronize();
    return q;
}

inline void pf_sched_destroy(pf_queue_t* q)
{
    if (!q) return;
    if (q->entries)       cudaFree(q->entries);
    if (q->prod_seq)      cudaFree((void*)q->prod_seq);
    if (q->page_status)   cudaFree((void*)q->page_status);
    if (q->sort_keys)     cudaFree(q->sort_keys);
    if (q->sort_keys_alt) cudaFree(q->sort_keys_alt);
    cudaFree(q);
}

inline void pf_sched_reset(pf_queue_t* q)
{
    if (!q) return;
    cudaMemset((void*)q->prod_seq, 0, q->capacity * sizeof(uint32_t));
    cudaMemset((void*)q->page_status, 0, q->n_pages * sizeof(pf_page_status_t));
    q->prod_tail       = 0;
    q->cons_head       = 0;
    q->shutdown        = 0;
    q->total_requests  = 0;
    q->unique_pages    = 0;
    q->batch_count     = 0;
    q->sched_drain_ns  = 0;
    q->sched_sort_ns   = 0;
    q->sched_dedup_ns  = 0;
    q->sched_touch_ns  = 0;
    q->sched_notify_ns = 0;
    q->sum_vblocks     = 0;
    cudaDeviceSynchronize();
}

inline void pf_sched_print_stats(const pf_queue_t* q)
{
    if (!q) return;
    const char* show = getenv("PF_SCHED_STATS");
    if (!show || strcmp(show, "1") != 0) return;
    printf("[PF_SCHED] Stats:\n");
    printf("  total_requests: %lu\n", (unsigned long)q->total_requests);
    printf("  unique_pages:   %lu\n", (unsigned long)q->unique_pages);
    printf("  batch_count:    %lu\n", (unsigned long)q->batch_count);
    if (q->unique_pages > 0) {
        printf("  dedup_ratio:    %.2fx (%lu -> %lu)\n",
               (double)q->total_requests / (double)q->unique_pages,
               (unsigned long)q->total_requests,
               (unsigned long)q->unique_pages);
    }
    printf("  Timing breakdown (ns):\n");
    printf("    drain:  %lu\n", (unsigned long)q->sched_drain_ns);
    printf("    sort:   %lu\n", (unsigned long)q->sched_sort_ns);
    printf("    dedup:  %lu\n", (unsigned long)q->sched_dedup_ns);
    printf("    touch:  %lu\n", (unsigned long)q->sched_touch_ns);
    printf("    notify: %lu\n", (unsigned long)q->sched_notify_ns);
}

#endif // __PF_SCHEDULER_CUH__
