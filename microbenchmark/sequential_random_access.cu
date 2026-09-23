// ============================================================================
// sequential_random_access.cu — sequential and random access at several thread counts
//
// Access patterns with duplicate pages (controlled by DUP_FACTOR):
//   Total UVM allocation : TEST_NUM_PAGES pages
//   Unique pages accessed: N_UNIQUE = TEST_NUM_PAGES / DUP_FACTOR
//   Total accesses       : TEST_NUM_PAGES
//
// Methods:
//   plain   — direct volatile reads, UVM handles faults on-demand
//   sched   — pf_scheduler (ring buffer + sort + dedup + 128-thread touch)
//   direct  — atomicCAS dedup + winner thread touches directly (no scheduler)
//   grouped — atomicCAS dedup + shared-memory VA Block bucketing + ordered touch
//   vbatch  — global per-VA-Block buffer + cross-block accumulation + sequenced flush
//
// Build:
//   nvcc -arch=sm_80 -O3 -DTEST_NUM_PAGES=32768 -DDUP_FACTOR=4 \
//        sequential_random_access.cu -o sequential_random_access
// ============================================================================

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "pf_scheduler.cuh"

// ============================================================================
// Configuration
// ============================================================================
#ifndef TEST_NUM_PAGES
#define TEST_NUM_PAGES       32768
#endif

#ifndef DUP_FACTOR
#define DUP_FACTOR           4
#endif

#define N_UNIQUE_PAGES       (TEST_NUM_PAGES / DUP_FACTOR)

#define TEST_PAGE_SIZE       4096ULL
#define TEST_ELEMS_PER_PAGE  (TEST_PAGE_SIZE / sizeof(uint64_t))  // 512
#define TEST_DATA_SIZE       ((uint64_t)TEST_NUM_PAGES * TEST_PAGE_SIZE)

// Sweep mode may use larger data (up to 65536 pages = 256 MB)
#define SWEEP_MAX_PAGES      65536u
#define SWEEP_DATA_SIZE      ((uint64_t)SWEEP_MAX_PAGES * TEST_PAGE_SIZE)
#define RING_CAPACITY        (SWEEP_MAX_PAGES * 4)

// VA Block grouping constants
#define PAGES_PER_VA_BLOCK   512u    // 2MB VA Block = 512 × 4KB pages
#define MAX_VA_BLOCKS        ((SWEEP_MAX_PAGES + PAGES_PER_VA_BLOCK - 1) / PAGES_PER_VA_BLOCK)
#define FLUSH_BATCH_SIZE     256u    // pages per flush (matches UVM driver batch)
#define VB_BUF_PAGES_MAX     PAGES_PER_VA_BLOCK  // max unique pages per VA Block

// ============================================================================
// Shutdown helper (for sched variant)
// ============================================================================
__global__ void shutdown_scheduler(pf_queue_t* q) {
    pf_sched_signal_shutdown(q);
}

// ============================================================================
// "Direct" approach: atomicCAS dedup + winner touches, no scheduler
// ============================================================================

// Minimal coordination state (no ring buffer, no sort buffers)
struct pf_direct_t {
    volatile uint32_t* page_status;  // [n_pages]: 0=UNKNOWN, 1=PENDING, 2=READY
    uint64_t base_addr;
    uint64_t n_pages;
};

// Create/destroy/reset helpers (host side)
static pf_direct_t* pf_direct_create(uint64_t base_addr, uint64_t n_pages)
{
    pf_direct_t h;
    h.base_addr = base_addr;
    h.n_pages   = n_pages;

    uint32_t* d_status = nullptr;
    cudaMalloc(&d_status, n_pages * sizeof(uint32_t));
    cudaMemset(d_status, 0, n_pages * sizeof(uint32_t));
    h.page_status = d_status;

    pf_direct_t* d_obj = nullptr;
    cudaMalloc(&d_obj, sizeof(pf_direct_t));
    cudaMemcpy(d_obj, &h, sizeof(pf_direct_t), cudaMemcpyHostToDevice);
    return d_obj;
}

static void pf_direct_destroy(pf_direct_t* d_obj)
{
    pf_direct_t h;
    cudaMemcpy(&h, d_obj, sizeof(pf_direct_t), cudaMemcpyDeviceToHost);
    if (h.page_status) cudaFree((void*)h.page_status);
    cudaFree(d_obj);
}

// ============================================================================
// "VBatch" approach: global per-VA-Block buffers + sequenced cooperative flush
//
// CAS winners from ALL thread blocks write page_ids to global per-VA-Block
// buffers.  When a buffer accumulates FLUSH_BATCH_SIZE (256) pages, one
// thread block claims the flush via atomicCAS, takes a global sequence
// ticket, waits for the previous flush to start, then 256 threads
// cooperatively touch all pages from that VA Block batch.
//
// Provides:  (1) cross-block VA Block accumulation
//            (2) 256-thread parallel touch per flush
//            (3) sequenced flushes — different VA Blocks don't interleave
// ============================================================================

struct pf_vbatch_t {
    volatile uint32_t* page_status;  // [n_pages]: 0=UNKNOWN, 1=PENDING, 2=READY
    uint64_t base_addr;
    uint64_t n_pages;
    uint32_t n_va_blocks;
    // Per-VA-Block buffers (contiguous device memory)
    uint32_t* vb_count;     // [n_va_blocks] slot counter (atomicAdd)
    uint32_t* vb_flushed;   // [n_va_blocks] pages claimed for flush (atomicCAS)
    uint32_t* vb_ready;     // [n_va_blocks * VB_BUF_PAGES_MAX] per-slot ready flag
    uint32_t* vb_pages;     // [n_va_blocks * VB_BUF_PAGES_MAX] page_ids
    // Global flush sequencing
    uint32_t* flush_seq;    // next ticket number
    uint32_t* flush_done;   // flushes that have started
};

static pf_vbatch_t* pf_vbatch_create(uint64_t base_addr, uint64_t n_pages)
{
    pf_vbatch_t h;
    h.base_addr = base_addr;
    h.n_pages = n_pages;
    h.n_va_blocks = (uint32_t)((n_pages + PAGES_PER_VA_BLOCK - 1) / PAGES_PER_VA_BLOCK);

    uint32_t nvb = h.n_va_blocks;
    size_t buf_slots = (size_t)nvb * VB_BUF_PAGES_MAX;

    cudaMalloc((void**)&h.page_status, n_pages * sizeof(uint32_t));
    cudaMemset((void*)h.page_status, 0, n_pages * sizeof(uint32_t));

    cudaMalloc(&h.vb_count,   nvb * sizeof(uint32_t));
    cudaMalloc(&h.vb_flushed, nvb * sizeof(uint32_t));
    cudaMalloc(&h.vb_ready,   buf_slots * sizeof(uint32_t));
    cudaMalloc(&h.vb_pages,   buf_slots * sizeof(uint32_t));
    cudaMemset(h.vb_count,   0, nvb * sizeof(uint32_t));
    cudaMemset(h.vb_flushed, 0, nvb * sizeof(uint32_t));
    cudaMemset(h.vb_ready,   0, buf_slots * sizeof(uint32_t));
    cudaMemset(h.vb_pages,   0, buf_slots * sizeof(uint32_t));

    cudaMalloc(&h.flush_seq,  sizeof(uint32_t));
    cudaMalloc(&h.flush_done, sizeof(uint32_t));
    cudaMemset(h.flush_seq,  0, sizeof(uint32_t));
    cudaMemset(h.flush_done, 0, sizeof(uint32_t));

    pf_vbatch_t* d_obj = nullptr;
    cudaMalloc(&d_obj, sizeof(pf_vbatch_t));
    cudaMemcpy(d_obj, &h, sizeof(pf_vbatch_t), cudaMemcpyHostToDevice);
    return d_obj;
}

static void pf_vbatch_destroy(pf_vbatch_t* d_obj)
{
    pf_vbatch_t h;
    cudaMemcpy(&h, d_obj, sizeof(pf_vbatch_t), cudaMemcpyDeviceToHost);
    if (h.page_status) cudaFree((void*)h.page_status);
    if (h.vb_count)    cudaFree(h.vb_count);
    if (h.vb_flushed)  cudaFree(h.vb_flushed);
    if (h.vb_ready)    cudaFree(h.vb_ready);
    if (h.vb_pages)    cudaFree(h.vb_pages);
    if (h.flush_seq)   cudaFree(h.flush_seq);
    if (h.flush_done)  cudaFree(h.flush_done);
    cudaFree(d_obj);
}

// Device load function: CAS dedup + winner touches directly
template <typename T>
__device__ __forceinline__
T pf_direct_load(pf_direct_t* d, const T* ptr, uint64_t i)
{
    const uintptr_t addr = reinterpret_cast<uintptr_t>(&ptr[i]);
    const uint64_t page_addr = addr & ~((uint64_t)TEST_PAGE_SIZE - 1);
    const uint64_t page_id = (page_addr - d->base_addr) >> 12;  // / 4096

    // Fast path: already ready
    uint32_t st = atomicAdd((uint32_t*)&d->page_status[page_id], 0u);
    if (st == 2u) {  // PF_READY
        const volatile T* vptr = (const volatile T*)ptr;
        return vptr[i];
    }

    // Warp coalesce: only one thread per unique page in this warp proceeds
    const uint32_t mask = __activemask();
    const uint32_t eq_mask = __match_any_sync(mask, (unsigned long long)page_id);
    const int master = __ffs(eq_mask) - 1;
    const uint32_t lane = threadIdx.x & 31;

    if (lane == (uint32_t)master) {
        // Try to claim: UNKNOWN(0) -> PENDING(1)
        uint32_t old = atomicCAS((uint32_t*)&d->page_status[page_id], 0u, 1u);
        if (old == 0u) {
            // I won — touch the page directly (triggers UVM fault + migration)
            volatile uint64_t tmp = *(volatile uint64_t*)page_addr;
            (void)tmp;
            __threadfence();
            // Mark READY(2)
            atomicExch((uint32_t*)&d->page_status[page_id], 2u);
        }
    }

    // Spin-wait until READY
    while (atomicAdd((uint32_t*)&d->page_status[page_id], 0u) != 2u) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(100);
#endif
    }
    __syncwarp(eq_mask);

    __threadfence();
    const volatile T* vptr = (const volatile T*)ptr;
    return vptr[i];
}

// ============================================================================
// Plain kernels
// ============================================================================

__global__ void kern_seq_dup_plain(const uint64_t* ptr, uint64_t n_accesses,
                                    uint64_t n_unique, uint64_t* d_sink)
{
    const uint64_t bsz   = blockDim.x;
    const uint64_t chunk  = (n_accesses + gridDim.x - 1) / gridDim.x;
    const uint64_t start  = (uint64_t)blockIdx.x * chunk;
    const uint64_t end    = (start + chunk < n_accesses) ? start + chunk : n_accesses;
    const volatile uint64_t* vptr = (const volatile uint64_t*)ptr;

    uint64_t local_xor = 0;
    for (uint64_t i = start + threadIdx.x; i < end; i += bsz) {
        uint64_t page = i % n_unique;
        local_xor ^= vptr[page * TEST_ELEMS_PER_PAGE];
    }
    atomicXor((unsigned long long*)d_sink, (unsigned long long)local_xor);
}

__global__ void kern_rand_dup_plain(const uint64_t* ptr, uint64_t n_accesses,
                                     const uint32_t* indices, uint64_t* d_sink)
{
    const uint64_t bsz   = blockDim.x;
    const uint64_t chunk  = (n_accesses + gridDim.x - 1) / gridDim.x;
    const uint64_t start  = (uint64_t)blockIdx.x * chunk;
    const uint64_t end    = (start + chunk < n_accesses) ? start + chunk : n_accesses;
    const volatile uint64_t* vptr = (const volatile uint64_t*)ptr;

    uint64_t local_xor = 0;
    for (uint64_t i = start + threadIdx.x; i < end; i += bsz) {
        local_xor ^= vptr[(uint64_t)indices[i] * TEST_ELEMS_PER_PAGE];
    }
    atomicXor((unsigned long long*)d_sink, (unsigned long long)local_xor);
}

// ============================================================================
// Scheduler-backed kernels
// ============================================================================

__global__ void kern_seq_dup_sched(pf_queue_t* q, const uint64_t* ptr,
                                    uint64_t n_accesses, uint64_t n_unique,
                                    uint64_t* d_sink)
{
    const uint64_t bsz   = blockDim.x;
    const uint64_t chunk  = (n_accesses + gridDim.x - 1) / gridDim.x;
    const uint64_t start  = (uint64_t)blockIdx.x * chunk;
    const uint64_t end    = (start + chunk < n_accesses) ? start + chunk : n_accesses;

    uint64_t local_xor = 0;
    for (uint64_t i = start + threadIdx.x; i < end; i += bsz) {
        uint64_t page = i % n_unique;
        uint64_t val = pf_sched_load<uint64_t>(q, ptr, page * TEST_ELEMS_PER_PAGE);
        local_xor ^= val;
    }
    atomicXor((unsigned long long*)d_sink, (unsigned long long)local_xor);
}

__global__ void kern_rand_dup_sched(pf_queue_t* q, const uint64_t* ptr,
                                     uint64_t n_accesses, const uint32_t* indices,
                                     uint64_t* d_sink)
{
    const uint64_t bsz   = blockDim.x;
    const uint64_t chunk  = (n_accesses + gridDim.x - 1) / gridDim.x;
    const uint64_t start  = (uint64_t)blockIdx.x * chunk;
    const uint64_t end    = (start + chunk < n_accesses) ? start + chunk : n_accesses;

    uint64_t local_xor = 0;
    for (uint64_t i = start + threadIdx.x; i < end; i += bsz) {
        uint64_t val = pf_sched_load<uint64_t>(q, ptr,
                        (uint64_t)indices[i] * TEST_ELEMS_PER_PAGE);
        local_xor ^= val;
    }
    atomicXor((unsigned long long*)d_sink, (unsigned long long)local_xor);
}

// ============================================================================
// Direct kernels: atomicCAS dedup + winner touches
// ============================================================================

__global__ void kern_seq_dup_direct(pf_direct_t* d, const uint64_t* ptr,
                                     uint64_t n_accesses, uint64_t n_unique,
                                     uint64_t* d_sink)
{
    const uint64_t bsz   = blockDim.x;
    const uint64_t chunk  = (n_accesses + gridDim.x - 1) / gridDim.x;
    const uint64_t start  = (uint64_t)blockIdx.x * chunk;
    const uint64_t end    = (start + chunk < n_accesses) ? start + chunk : n_accesses;

    uint64_t local_xor = 0;
    for (uint64_t i = start + threadIdx.x; i < end; i += bsz) {
        uint64_t page = i % n_unique;
        uint64_t val = pf_direct_load<uint64_t>(d, ptr, page * TEST_ELEMS_PER_PAGE);
        local_xor ^= val;
    }
    atomicXor((unsigned long long*)d_sink, (unsigned long long)local_xor);
}

__global__ void kern_rand_dup_direct(pf_direct_t* d, const uint64_t* ptr,
                                      uint64_t n_accesses, const uint32_t* indices,
                                      uint64_t* d_sink)
{
    const uint64_t bsz   = blockDim.x;
    const uint64_t chunk  = (n_accesses + gridDim.x - 1) / gridDim.x;
    const uint64_t start  = (uint64_t)blockIdx.x * chunk;
    const uint64_t end    = (start + chunk < n_accesses) ? start + chunk : n_accesses;

    uint64_t local_xor = 0;
    for (uint64_t i = start + threadIdx.x; i < end; i += bsz) {
        uint64_t val = pf_direct_load<uint64_t>(d, ptr,
                        (uint64_t)indices[i] * TEST_ELEMS_PER_PAGE);
        local_xor ^= val;
    }
    atomicXor((unsigned long long*)d_sink, (unsigned long long)local_xor);
}

// ============================================================================
// Grouped kernels: atomicCAS dedup + shared-memory VA Block bucketing
//
// Same CAS dedup as "direct", but CAS winners do NOT touch immediately.
// Instead, won page_ids are collected in shared memory, bucket-sorted by
// VA Block (2MB / 512 pages), then all threads cooperatively touch in
// VA Block order.  This groups temporally-close faults into the same
// VA Block so the UVM driver can batch them (up to 256 per batch).
//
// 4 __syncthreads() per loop iteration; touch parallelism = blockDim.x
// (same app threads, no separate scheduler kernel).
// ============================================================================

__global__ void kern_seq_dup_grouped(pf_direct_t* d, const uint64_t* ptr,
                                      uint64_t n_accesses, uint64_t n_unique,
                                      uint64_t* d_sink)
{
    const uint32_t ltid = threadIdx.x;
    const uint32_t bsz = blockDim.x;
    const uint64_t chunk = (n_accesses + gridDim.x - 1) / gridDim.x;
    const uint64_t start = (uint64_t)blockIdx.x * chunk;
    const uint64_t end   = (start + chunk < n_accesses) ? start + chunk : n_accesses;

    // Cache device-memory fields in registers
    const uint64_t base_addr = d->base_addr;
    volatile uint32_t* const page_status = d->page_status;

    __shared__ uint32_t sh_page_ids[256];
    __shared__ uint32_t sh_won[256];
    __shared__ uint32_t sh_vb_count[MAX_VA_BLOCKS];
    __shared__ uint32_t sh_ordered[256];
    __shared__ uint32_t sh_n_won;

    uint64_t local_xor = 0;
    const uint64_t n_iters = (chunk + bsz - 1) / bsz;

    for (uint64_t iter = 0; iter < n_iters; iter++) {
        const uint64_t i = start + iter * bsz + ltid;
        const bool active = (i < end);

        // --- Phase 1: CAS dedup (identical to "direct") ---
        uint32_t page_id = 0;
        bool i_won = false;

        if (active) {
            page_id = (uint32_t)(i % n_unique);
            uint32_t st = atomicAdd((uint32_t*)&page_status[page_id], 0u);
            if (st != 2u) {
                const uint32_t mask = __activemask();
                const uint32_t eq_mask = __match_any_sync(mask,
                                            (unsigned long long)page_id);
                const int master = __ffs(eq_mask) - 1;
                if ((ltid & 31) == (uint32_t)master) {
                    uint32_t old = atomicCAS((uint32_t*)&page_status[page_id],
                                             0u, 1u);
                    i_won = (old == 0u);
                }
            }
        }

        // Write per-thread results + clear histogram (fused before sync)
        sh_page_ids[ltid] = page_id;
        sh_won[ltid] = i_won ? 1u : 0u;
        for (uint32_t b = ltid; b < (uint32_t)MAX_VA_BLOCKS; b += bsz)
            sh_vb_count[b] = 0;
        if (ltid == 0) sh_n_won = 0;
        __syncthreads();  // sync 1: page_ids, won flags, histogram all ready

        // --- Phase 2: thread 0 — histogram + prefix sum + scatter ---
        if (ltid == 0) {
            uint32_t nw = 0;
            for (uint32_t t = 0; t < bsz; t++) {
                if (sh_won[t]) {
                    sh_vb_count[sh_page_ids[t] / PAGES_PER_VA_BLOCK]++;
                    nw++;
                }
            }
            sh_n_won = nw;
            // In-place prefix sum (sh_vb_count becomes offsets)
            uint32_t sum = 0;
            for (uint32_t b = 0; b < (uint32_t)MAX_VA_BLOCKS; b++) {
                uint32_t c = sh_vb_count[b];
                sh_vb_count[b] = sum;
                sum += c;
            }
            // Scatter won pages into sh_ordered in VA Block order
            for (uint32_t t = 0; t < bsz; t++) {
                if (sh_won[t]) {
                    uint32_t vb = sh_page_ids[t] / PAGES_PER_VA_BLOCK;
                    sh_ordered[sh_vb_count[vb]++] = sh_page_ids[t];
                }
            }
        }
        __syncthreads();  // sync 2: sh_ordered ready

        const uint32_t n_won = sh_n_won;

        // --- Phase 3: cooperative touch in VA Block order ---
        for (uint32_t j = ltid; j < n_won; j += bsz) {
            uint64_t addr = base_addr + (uint64_t)sh_ordered[j] * TEST_PAGE_SIZE;
            volatile uint64_t tmp = *(volatile uint64_t*)addr;
            (void)tmp;
        }
        if (n_won > 0) __threadfence();

        // --- Phase 4: mark READY ---
        for (uint32_t j = ltid; j < n_won; j += bsz) {
            atomicExch((uint32_t*)&page_status[sh_ordered[j]], 2u);
        }
        if (n_won > 0) __threadfence();
        __syncthreads();  // sync 3: all touches visible

        // --- Phase 5: spin-wait for own page + read data ---
        if (active) {
            while (atomicAdd((uint32_t*)&page_status[page_id], 0u) != 2u) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
                __nanosleep(100);
#endif
            }
            const volatile uint64_t* vptr = (const volatile uint64_t*)ptr;
            local_xor ^= vptr[(i % n_unique) * TEST_ELEMS_PER_PAGE];
        }
        __syncthreads();  // sync 4: safe to reuse shared memory
    }

    atomicXor((unsigned long long*)d_sink, (unsigned long long)local_xor);
}

__global__ void kern_rand_dup_grouped(pf_direct_t* d, const uint64_t* ptr,
                                       uint64_t n_accesses, const uint32_t* indices,
                                       uint64_t* d_sink)
{
    const uint32_t ltid = threadIdx.x;
    const uint32_t bsz = blockDim.x;
    const uint64_t chunk = (n_accesses + gridDim.x - 1) / gridDim.x;
    const uint64_t start = (uint64_t)blockIdx.x * chunk;
    const uint64_t end   = (start + chunk < n_accesses) ? start + chunk : n_accesses;

    const uint64_t base_addr = d->base_addr;
    volatile uint32_t* const page_status = d->page_status;

    __shared__ uint32_t sh_page_ids[256];
    __shared__ uint32_t sh_won[256];
    __shared__ uint32_t sh_vb_count[MAX_VA_BLOCKS];
    __shared__ uint32_t sh_ordered[256];
    __shared__ uint32_t sh_n_won;

    uint64_t local_xor = 0;
    const uint64_t n_iters = (chunk + bsz - 1) / bsz;

    for (uint64_t iter = 0; iter < n_iters; iter++) {
        const uint64_t i = start + iter * bsz + ltid;
        const bool active = (i < end);

        uint32_t page_id = 0;
        bool i_won = false;

        if (active) {
            page_id = indices[i];
            uint32_t st = atomicAdd((uint32_t*)&page_status[page_id], 0u);
            if (st != 2u) {
                const uint32_t mask = __activemask();
                const uint32_t eq_mask = __match_any_sync(mask,
                                            (unsigned long long)page_id);
                const int master = __ffs(eq_mask) - 1;
                if ((ltid & 31) == (uint32_t)master) {
                    uint32_t old = atomicCAS((uint32_t*)&page_status[page_id],
                                             0u, 1u);
                    i_won = (old == 0u);
                }
            }
        }

        sh_page_ids[ltid] = page_id;
        sh_won[ltid] = i_won ? 1u : 0u;
        for (uint32_t b = ltid; b < (uint32_t)MAX_VA_BLOCKS; b += bsz)
            sh_vb_count[b] = 0;
        if (ltid == 0) sh_n_won = 0;
        __syncthreads();

        if (ltid == 0) {
            uint32_t nw = 0;
            for (uint32_t t = 0; t < bsz; t++) {
                if (sh_won[t]) {
                    sh_vb_count[sh_page_ids[t] / PAGES_PER_VA_BLOCK]++;
                    nw++;
                }
            }
            sh_n_won = nw;
            uint32_t sum = 0;
            for (uint32_t b = 0; b < (uint32_t)MAX_VA_BLOCKS; b++) {
                uint32_t c = sh_vb_count[b];
                sh_vb_count[b] = sum;
                sum += c;
            }
            for (uint32_t t = 0; t < bsz; t++) {
                if (sh_won[t]) {
                    uint32_t vb = sh_page_ids[t] / PAGES_PER_VA_BLOCK;
                    sh_ordered[sh_vb_count[vb]++] = sh_page_ids[t];
                }
            }
        }
        __syncthreads();

        const uint32_t n_won = sh_n_won;

        for (uint32_t j = ltid; j < n_won; j += bsz) {
            uint64_t addr = base_addr + (uint64_t)sh_ordered[j] * TEST_PAGE_SIZE;
            volatile uint64_t tmp = *(volatile uint64_t*)addr;
            (void)tmp;
        }
        if (n_won > 0) __threadfence();

        for (uint32_t j = ltid; j < n_won; j += bsz) {
            atomicExch((uint32_t*)&page_status[sh_ordered[j]], 2u);
        }
        if (n_won > 0) __threadfence();
        __syncthreads();

        if (active) {
            while (atomicAdd((uint32_t*)&page_status[page_id], 0u) != 2u) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
                __nanosleep(100);
#endif
            }
            const volatile uint64_t* vptr = (const volatile uint64_t*)ptr;
            local_xor ^= vptr[(uint64_t)indices[i] * TEST_ELEMS_PER_PAGE];
        }
        __syncthreads();
    }

    atomicXor((unsigned long long*)d_sink, (unsigned long long)local_xor);
}

// ============================================================================
// VBatch kernels: global per-VA-Block buffer + sequenced cooperative flush
// ============================================================================

__global__ void kern_seq_dup_vbatch(pf_vbatch_t* v, const uint64_t* ptr,
                                     uint64_t n_accesses, uint64_t n_unique,
                                     uint64_t* d_sink)
{
    const uint32_t ltid = threadIdx.x;
    const uint32_t bsz = blockDim.x;
    const uint64_t chunk = (n_accesses + gridDim.x - 1) / gridDim.x;
    const uint64_t start = (uint64_t)blockIdx.x * chunk;
    const uint64_t end   = (start + chunk < n_accesses) ? start + chunk : n_accesses;

    // Cache struct fields in registers
    const uint64_t base_addr     = v->base_addr;
    volatile uint32_t* const page_status = v->page_status;
    const uint32_t n_vb          = v->n_va_blocks;
    uint32_t* const vb_count     = v->vb_count;
    uint32_t* const vb_flushed   = v->vb_flushed;
    uint32_t* const vb_ready     = v->vb_ready;
    uint32_t* const vb_pages_    = v->vb_pages;
    __shared__ uint32_t sh_flush_vb;
    __shared__ uint32_t sh_flush_base;
    __shared__ uint32_t sh_flush_count;

    uint64_t local_xor = 0;
    const uint64_t n_iters = (chunk + bsz - 1) / bsz;

    for (uint64_t iter = 0; iter < n_iters; iter++) {
        const uint64_t i = start + iter * bsz + ltid;
        const bool active = (i < end);

        // --- Phase 1: CAS dedup + write to global VA Block buffer ---
        uint32_t page_id = 0;
        if (active) {
            page_id = (uint32_t)(i % n_unique);
            uint32_t st = atomicAdd((uint32_t*)&page_status[page_id], 0u);
            if (st != 2u) {
                const uint32_t mask = __activemask();
                const uint32_t eq_mask = __match_any_sync(mask,
                                            (unsigned long long)page_id);
                const int master = __ffs(eq_mask) - 1;
                if ((ltid & 31) == (uint32_t)master) {
                    uint32_t old = atomicCAS((uint32_t*)&page_status[page_id],
                                             0u, 1u);
                    if (old == 0u) {
                        uint32_t vb = page_id / PAGES_PER_VA_BLOCK;
                        uint32_t slot = atomicAdd(&vb_count[vb], 1);
                        vb_pages_[vb * VB_BUF_PAGES_MAX + slot] = page_id;
                        __threadfence();
                        atomicExch(&vb_ready[vb * VB_BUF_PAGES_MAX + slot], 1u);
                    }
                }
            }
        }
        __syncthreads();

        // --- Phase 2: Scan + flush ready VA Block batches (one at a time) ---
        for (;;) {
            if (ltid == 0) {
                sh_flush_vb = 0xFFFFFFFF;
                for (uint32_t vb = 0; vb < n_vb; vb++) {
                    uint32_t cnt = atomicAdd(&vb_count[vb], 0);
                    uint32_t f   = atomicAdd(&vb_flushed[vb], 0);
                    if (cnt > f) {
                        uint32_t batch = (cnt - f < FLUSH_BATCH_SIZE)
                                       ? (cnt - f) : FLUSH_BATCH_SIZE;
                        if (atomicCAS(&vb_flushed[vb], f, f + batch) == f) {
                            sh_flush_vb    = vb;
                            sh_flush_base  = f;
                            sh_flush_count = batch;
                            break;
                        }
                    }
                }
            }
            __syncthreads();
            if (sh_flush_vb == 0xFFFFFFFF) break;

            uint32_t fvb    = sh_flush_vb;
            uint32_t fbase  = sh_flush_base;
            uint32_t fcount = sh_flush_count;

            // Wait for all slots in batch to be ready
            for (uint32_t j = ltid; j < fcount; j += bsz) {
                while (atomicAdd(&vb_ready[fvb * VB_BUF_PAGES_MAX + fbase + j],
                                 0) == 0)
                    __nanosleep(32);
            }
            __syncthreads();

            // Cooperative touch (256 threads, same VA Block)
            for (uint32_t j = ltid; j < fcount; j += bsz) {
                uint32_t pg = vb_pages_[fvb * VB_BUF_PAGES_MAX + fbase + j];
                uint64_t addr = base_addr + (uint64_t)pg * TEST_PAGE_SIZE;
                volatile uint64_t tmp = *(volatile uint64_t*)addr;
                (void)tmp;
            }
            __threadfence();

            // Mark READY
            for (uint32_t j = ltid; j < fcount; j += bsz) {
                uint32_t pg = vb_pages_[fvb * VB_BUF_PAGES_MAX + fbase + j];
                atomicExch((uint32_t*)&page_status[pg], 2u);
            }
            __threadfence();
            __syncthreads();
        }

        // --- Phase 3: Spin-wait (with timeout fallback) + read data ---
        if (active) {
            uint32_t spins = 0;
            while (atomicAdd((uint32_t*)&page_status[page_id], 0u) != 2u) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
                __nanosleep(100);
#endif
                if (++spins > 100000) {
                    // Timeout: direct touch for unflushed partial-batch pages
                    volatile uint64_t tmp = *(volatile uint64_t*)(
                        base_addr + (uint64_t)page_id * TEST_PAGE_SIZE);
                    (void)tmp;
                    __threadfence();
                    atomicExch((uint32_t*)&page_status[page_id], 2u);
                    break;
                }
            }
            const volatile uint64_t* vptr = (const volatile uint64_t*)ptr;
            local_xor ^= vptr[(i % n_unique) * TEST_ELEMS_PER_PAGE];
        }
        __syncthreads();
    }

    atomicXor((unsigned long long*)d_sink, (unsigned long long)local_xor);
}

__global__ void kern_rand_dup_vbatch(pf_vbatch_t* v, const uint64_t* ptr,
                                      uint64_t n_accesses, const uint32_t* indices,
                                      uint64_t* d_sink)
{
    const uint32_t ltid = threadIdx.x;
    const uint32_t bsz = blockDim.x;
    const uint64_t chunk = (n_accesses + gridDim.x - 1) / gridDim.x;
    const uint64_t start = (uint64_t)blockIdx.x * chunk;
    const uint64_t end   = (start + chunk < n_accesses) ? start + chunk : n_accesses;

    const uint64_t base_addr     = v->base_addr;
    volatile uint32_t* const page_status = v->page_status;
    // Cooperative drain shared memory (VA Block ordered touch)
    __shared__ uint32_t sh_drain_ids[256];
    __shared__ uint32_t sh_drain_vbc[MAX_VA_BLOCKS];
    __shared__ uint32_t sh_drain_ord[256];
    __shared__ uint32_t sh_n_drain;

    uint64_t local_xor = 0;
    const uint64_t n_iters = (chunk + bsz - 1) / bsz;

    for (uint64_t iter = 0; iter < n_iters; iter++) {
        const uint64_t i = start + iter * bsz + ltid;
        const bool active = (i < end);

        // --- CAS dedup only (no global buffer for random access) ---
        uint32_t page_id = 0;
        bool i_won = false;
        if (active) {
            page_id = indices[i];
            uint32_t st = atomicAdd((uint32_t*)&page_status[page_id], 0u);
            if (st != 2u) {
                const uint32_t mask = __activemask();
                const uint32_t eq_mask = __match_any_sync(mask,
                                            (unsigned long long)page_id);
                const int master = __ffs(eq_mask) - 1;
                if ((ltid & 31) == (uint32_t)master) {
                    uint32_t old = atomicCAS((uint32_t*)&page_status[page_id],
                                             0u, 1u);
                    i_won = (old == 0u);
                }
            }
        }

        // --- Cooperative drain: VA Block ordered touch ---
        {
            uint32_t my_drain = 0xFFFFFFFFu;
            if (i_won)
                my_drain = page_id;
            sh_drain_ids[ltid] = my_drain;
            for (uint32_t b = ltid; b < (uint32_t)MAX_VA_BLOCKS; b += bsz)
                sh_drain_vbc[b] = 0;
            if (ltid == 0) sh_n_drain = 0;
            __syncthreads();

            if (ltid == 0) {
                uint32_t nd = 0;
                for (uint32_t t = 0; t < bsz; t++) {
                    if (sh_drain_ids[t] != 0xFFFFFFFFu) {
                        sh_drain_vbc[sh_drain_ids[t] / PAGES_PER_VA_BLOCK]++;
                        nd++;
                    }
                }
                sh_n_drain = nd;
                uint32_t sum = 0;
                for (uint32_t b = 0; b < (uint32_t)MAX_VA_BLOCKS; b++) {
                    uint32_t c = sh_drain_vbc[b];
                    sh_drain_vbc[b] = sum;
                    sum += c;
                }
                for (uint32_t t = 0; t < bsz; t++) {
                    if (sh_drain_ids[t] != 0xFFFFFFFFu) {
                        uint32_t vb = sh_drain_ids[t] / PAGES_PER_VA_BLOCK;
                        sh_drain_ord[sh_drain_vbc[vb]++] = sh_drain_ids[t];
                    }
                }
            }
            __syncthreads();

            uint32_t nd = sh_n_drain;
            for (uint32_t j = ltid; j < nd; j += bsz) {
                uint64_t addr = base_addr + (uint64_t)sh_drain_ord[j] * TEST_PAGE_SIZE;
                volatile uint64_t tmp = *(volatile uint64_t*)addr;
                (void)tmp;
            }
            if (nd > 0) __threadfence();
            for (uint32_t j = ltid; j < nd; j += bsz) {
                atomicExch((uint32_t*)&page_status[sh_drain_ord[j]], 2u);
            }
            if (nd > 0) __threadfence();
            __syncthreads();
        }
        if (active) {
            const volatile uint64_t* vptr = (const volatile uint64_t*)ptr;
            local_xor ^= vptr[(uint64_t)indices[i] * TEST_ELEMS_PER_PAGE];
        }
        __syncthreads();
    }

    atomicXor((unsigned long long*)d_sink, (unsigned long long)local_xor);
}

// ============================================================================
// Expected XOR for verification
// ============================================================================

static uint64_t expected_xor_seq(uint64_t n_accesses, int n_blocks, int n_threads)
{
    uint64_t chunk = (n_accesses + n_blocks - 1) / n_blocks;
    uint64_t xor_acc = 0;
    for (int b = 0; b < n_blocks; b++) {
        uint64_t bstart = (uint64_t)b * chunk;
        uint64_t bend = bstart + chunk;
        if (bend > n_accesses) bend = n_accesses;
        for (int t = 0; t < n_threads; t++) {
            uint64_t local_xor = 0;
            for (uint64_t i = bstart + t; i < bend; i += n_threads) {
                uint64_t page = i % N_UNIQUE_PAGES;
                local_xor ^= page;
            }
            xor_acc ^= local_xor;
        }
    }
    return xor_acc;
}

static uint64_t expected_xor_rand(const uint32_t* indices, uint64_t n_accesses,
                                   int n_blocks, int n_threads)
{
    uint64_t chunk = (n_accesses + n_blocks - 1) / n_blocks;
    uint64_t xor_acc = 0;
    for (int b = 0; b < n_blocks; b++) {
        uint64_t bstart = (uint64_t)b * chunk;
        uint64_t bend = bstart + chunk;
        if (bend > n_accesses) bend = n_accesses;
        for (int t = 0; t < n_threads; t++) {
            uint64_t local_xor = 0;
            for (uint64_t i = bstart + t; i < bend; i += n_threads) {
                local_xor ^= (uint64_t)indices[i];
            }
            xor_acc ^= local_xor;
        }
    }
    return xor_acc;
}

static uint32_t count_unique(const uint32_t* indices, uint32_t n)
{
    bool* seen = (bool*)calloc(N_UNIQUE_PAGES, sizeof(bool));
    uint32_t count = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (!seen[indices[i]]) {
            seen[indices[i]] = true;
            count++;
        }
    }
    free(seen);
    return count;
}

// ============================================================================
// run_variant_plain
// ============================================================================
static float run_variant_plain(bool random_access, int n_blocks, int n_threads,
                                uint64_t n_accesses,
                                uint64_t* data, uint32_t* d_indices,
                                uint64_t expected)
{
    cudaMemPrefetchAsync(data, SWEEP_DATA_SIZE, cudaCpuDeviceId);
    cudaDeviceSynchronize();

    uint64_t* d_sink;
    cudaMalloc(&d_sink, sizeof(uint64_t));
    cudaMemset(d_sink, 0, sizeof(uint64_t));

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    cudaEventRecord(ev0, stream);
    if (random_access) {
        kern_rand_dup_plain<<<n_blocks, n_threads, 0, stream>>>(
            data, n_accesses, d_indices, d_sink);
    } else {
        kern_seq_dup_plain<<<n_blocks, n_threads, 0, stream>>>(
            data, n_accesses, (uint64_t)N_UNIQUE_PAGES, d_sink);
    }
    cudaEventRecord(ev1, stream);
    cudaStreamSynchronize(stream);

    cudaError_t err = cudaGetLastError();
    float ms = -1.0f;
    if (err != cudaSuccess) {
        printf("[ERROR] plain kernel: %s\n", cudaGetErrorString(err));
    } else {
        cudaEventElapsedTime(&ms, ev0, ev1);
        uint64_t h_sink;
        cudaMemcpy(&h_sink, d_sink, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        if (h_sink != expected) {
            printf("[ERROR] plain: XOR mismatch (got %llu, expected %llu)\n",
                   (unsigned long long)h_sink, (unsigned long long)expected);
            ms = -1.0f;
        }
    }

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaStreamDestroy(stream);
    cudaFree(d_sink);
    return ms;
}

// ============================================================================
// run_variant_sched
// ============================================================================
static float run_variant_sched(bool random_access, int n_blocks, int n_threads,
                                uint64_t n_accesses,
                                uint64_t* data, uint32_t* d_indices,
                                uint64_t expected)
{
    cudaMemPrefetchAsync(data, SWEEP_DATA_SIZE, cudaCpuDeviceId);
    cudaDeviceSynchronize();

    uint64_t base_raw = (uint64_t)data;
    uint64_t base_addr = base_raw & ~(TEST_PAGE_SIZE - 1ULL);
    uint64_t n_pages_q = SWEEP_MAX_PAGES + 1;
    pf_queue_t* q = pf_sched_create(RING_CAPACITY, base_addr, n_pages_q, 0);
    if (!q) return -1.0f;

    uint64_t* d_sink;
    cudaMalloc(&d_sink, sizeof(uint64_t));
    cudaMemset(d_sink, 0, sizeof(uint64_t));

    cudaStream_t sched_stream, app_stream;
    cudaStreamCreateWithFlags(&sched_stream, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&app_stream,   cudaStreamNonBlocking);

    pf_scheduler_kernel<<<1, kPfSchedThreads, 0, sched_stream>>>(q);
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("[ERROR] scheduler launch: %s\n", cudaGetErrorString(err));
            pf_sched_destroy(q); cudaFree(d_sink);
            cudaStreamDestroy(sched_stream); cudaStreamDestroy(app_stream);
            return -1.0f;
        }
    }

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    cudaEventRecord(ev0, app_stream);
    if (random_access) {
        kern_rand_dup_sched<<<n_blocks, n_threads, 0, app_stream>>>(
            q, data, n_accesses, d_indices, d_sink);
    } else {
        kern_seq_dup_sched<<<n_blocks, n_threads, 0, app_stream>>>(
            q, data, n_accesses, (uint64_t)N_UNIQUE_PAGES, d_sink);
    }
    cudaEventRecord(ev1, app_stream);
    cudaStreamSynchronize(app_stream);

    float ms = -1.0f;
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[ERROR] sched app kernel: %s\n", cudaGetErrorString(err));
    } else {
        cudaEventElapsedTime(&ms, ev0, ev1);
        uint64_t h_sink;
        cudaMemcpy(&h_sink, d_sink, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        if (h_sink != expected) {
            printf("[ERROR] sched: XOR mismatch (got %llu, expected %llu)\n",
                   (unsigned long long)h_sink, (unsigned long long)expected);
            ms = -1.0f;
        }
    }

    shutdown_scheduler<<<1, 1, 0, app_stream>>>(q);
    cudaStreamSynchronize(app_stream);
    cudaStreamSynchronize(sched_stream);

    pf_sched_print_stats(q);

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaStreamDestroy(sched_stream);
    cudaStreamDestroy(app_stream);
    cudaFree(d_sink);
    pf_sched_destroy(q);
    return ms;
}

// ============================================================================
// run_variant_direct
// ============================================================================
static float run_variant_direct(bool random_access, int n_blocks, int n_threads,
                                 uint64_t n_accesses,
                                 uint64_t* data, uint32_t* d_indices,
                                 uint64_t expected)
{
    cudaMemPrefetchAsync(data, SWEEP_DATA_SIZE, cudaCpuDeviceId);
    cudaDeviceSynchronize();

    uint64_t base_raw = (uint64_t)data;
    uint64_t base_addr = base_raw & ~(TEST_PAGE_SIZE - 1ULL);
    uint64_t n_pages_d = SWEEP_MAX_PAGES + 1;
    pf_direct_t* d = pf_direct_create(base_addr, n_pages_d);
    if (!d) return -1.0f;

    uint64_t* d_sink;
    cudaMalloc(&d_sink, sizeof(uint64_t));
    cudaMemset(d_sink, 0, sizeof(uint64_t));

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    cudaEventRecord(ev0, stream);
    if (random_access) {
        kern_rand_dup_direct<<<n_blocks, n_threads, 0, stream>>>(
            d, data, n_accesses, d_indices, d_sink);
    } else {
        kern_seq_dup_direct<<<n_blocks, n_threads, 0, stream>>>(
            d, data, n_accesses, (uint64_t)N_UNIQUE_PAGES, d_sink);
    }
    cudaEventRecord(ev1, stream);
    cudaStreamSynchronize(stream);

    cudaError_t err = cudaGetLastError();
    float ms = -1.0f;
    if (err != cudaSuccess) {
        printf("[ERROR] direct kernel: %s\n", cudaGetErrorString(err));
    } else {
        cudaEventElapsedTime(&ms, ev0, ev1);
        uint64_t h_sink;
        cudaMemcpy(&h_sink, d_sink, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        if (h_sink != expected) {
            printf("[ERROR] direct: XOR mismatch (got %llu, expected %llu)\n",
                   (unsigned long long)h_sink, (unsigned long long)expected);
            ms = -1.0f;
        }
    }

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaStreamDestroy(stream);
    cudaFree(d_sink);
    pf_direct_destroy(d);
    return ms;
}

// ============================================================================
// run_variant_grouped
// ============================================================================
static float run_variant_grouped(bool random_access, int n_blocks, int n_threads,
                                  uint64_t n_accesses,
                                  uint64_t* data, uint32_t* d_indices,
                                  uint64_t expected)
{
    cudaMemPrefetchAsync(data, SWEEP_DATA_SIZE, cudaCpuDeviceId);
    cudaDeviceSynchronize();

    uint64_t base_raw = (uint64_t)data;
    uint64_t base_addr = base_raw & ~(TEST_PAGE_SIZE - 1ULL);
    uint64_t n_pages_d = SWEEP_MAX_PAGES + 1;
    pf_direct_t* d = pf_direct_create(base_addr, n_pages_d);
    if (!d) return -1.0f;

    uint64_t* d_sink;
    cudaMalloc(&d_sink, sizeof(uint64_t));
    cudaMemset(d_sink, 0, sizeof(uint64_t));

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    cudaEventRecord(ev0, stream);
    if (random_access) {
        kern_rand_dup_grouped<<<n_blocks, n_threads, 0, stream>>>(
            d, data, n_accesses, d_indices, d_sink);
    } else {
        kern_seq_dup_grouped<<<n_blocks, n_threads, 0, stream>>>(
            d, data, n_accesses, (uint64_t)N_UNIQUE_PAGES, d_sink);
    }
    cudaEventRecord(ev1, stream);
    cudaStreamSynchronize(stream);

    cudaError_t err = cudaGetLastError();
    float ms = -1.0f;
    if (err != cudaSuccess) {
        printf("[ERROR] grouped kernel: %s\n", cudaGetErrorString(err));
    } else {
        cudaEventElapsedTime(&ms, ev0, ev1);
        uint64_t h_sink;
        cudaMemcpy(&h_sink, d_sink, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        if (h_sink != expected) {
            printf("[ERROR] grouped: XOR mismatch (got %llu, expected %llu)\n",
                   (unsigned long long)h_sink, (unsigned long long)expected);
            ms = -1.0f;
        }
    }

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaStreamDestroy(stream);
    cudaFree(d_sink);
    pf_direct_destroy(d);
    return ms;
}

// ============================================================================
// run_variant_vbatch
// ============================================================================
static float run_variant_vbatch(bool random_access, int n_blocks, int n_threads,
                                 uint64_t n_accesses,
                                 uint64_t* data, uint32_t* d_indices,
                                 uint64_t expected)
{
    cudaMemPrefetchAsync(data, SWEEP_DATA_SIZE, cudaCpuDeviceId);
    cudaDeviceSynchronize();

    uint64_t base_raw = (uint64_t)data;
    uint64_t base_addr = base_raw & ~(TEST_PAGE_SIZE - 1ULL);
    uint64_t n_pages_v = SWEEP_MAX_PAGES + 1;
    pf_vbatch_t* vb = pf_vbatch_create(base_addr, n_pages_v);
    if (!vb) return -1.0f;

    uint64_t* d_sink;
    cudaMalloc(&d_sink, sizeof(uint64_t));
    cudaMemset(d_sink, 0, sizeof(uint64_t));

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);

    cudaEventRecord(ev0, stream);
    if (random_access) {
        kern_rand_dup_vbatch<<<n_blocks, n_threads, 0, stream>>>(
            vb, data, n_accesses, d_indices, d_sink);
    } else {
        kern_seq_dup_vbatch<<<n_blocks, n_threads, 0, stream>>>(
            vb, data, n_accesses, (uint64_t)N_UNIQUE_PAGES, d_sink);
    }
    cudaEventRecord(ev1, stream);
    cudaStreamSynchronize(stream);

    cudaError_t err = cudaGetLastError();
    float ms = -1.0f;
    if (err != cudaSuccess) {
        printf("[ERROR] vbatch kernel: %s\n", cudaGetErrorString(err));
    } else {
        cudaEventElapsedTime(&ms, ev0, ev1);
        uint64_t h_sink;
        cudaMemcpy(&h_sink, d_sink, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        if (h_sink != expected) {
            printf("[ERROR] vbatch: XOR mismatch (got %llu, expected %llu)\n",
                   (unsigned long long)h_sink, (unsigned long long)expected);
            ms = -1.0f;
        }
    }

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaStreamDestroy(stream);
    cudaFree(d_sink);
    pf_vbatch_destroy(vb);
    return ms;
}

// ============================================================================
// run_test: five-way comparison
// ============================================================================
static bool run_test(const char* name, bool random_access,
                     int n_blocks, int n_threads, uint64_t n_accesses,
                     uint64_t* data, uint32_t* h_indices, uint32_t* d_indices)
{
    printf("\n---- %s ----\n", name);

    uint64_t total_threads = (uint64_t)n_blocks * n_threads;
    printf("  total_accesses=%llu, unique_pages=%u, dup_factor=%u, threads=%llu\n",
           (unsigned long long)n_accesses, (unsigned)N_UNIQUE_PAGES,
           (unsigned)DUP_FACTOR, (unsigned long long)total_threads);

    if (random_access) {
        uint32_t actual_unique = count_unique(h_indices, (uint32_t)n_accesses);
        printf("  rand indices: %u unique out of %llu (actual dup_ratio=%.2fx)\n",
               actual_unique, (unsigned long long)n_accesses,
               (double)n_accesses / actual_unique);
    }

    uint64_t expected;
    if (random_access) {
        expected = expected_xor_rand(h_indices, n_accesses, n_blocks, n_threads);
    } else {
        expected = expected_xor_seq(n_accesses, n_blocks, n_threads);
    }

    float ms_plain   = run_variant_plain(random_access, n_blocks, n_threads,
                                          n_accesses, data, d_indices, expected);
    float ms_sched   = run_variant_sched(random_access, n_blocks, n_threads,
                                          n_accesses, data, d_indices, expected);
    float ms_direct  = run_variant_direct(random_access, n_blocks, n_threads,
                                           n_accesses, data, d_indices, expected);
    float ms_grouped = run_variant_grouped(random_access, n_blocks, n_threads,
                                            n_accesses, data, d_indices, expected);
    float ms_vbatch  = run_variant_vbatch(random_access, n_blocks, n_threads,
                                           n_accesses, data, d_indices, expected);

    bool ok = (ms_plain > 0 && ms_sched > 0 && ms_direct > 0
               && ms_grouped > 0 && ms_vbatch > 0);

    if (ms_plain > 0)   printf("  plain   : %7.2f ms\n", ms_plain);
    else                 printf("  plain   : FAIL\n");

    if (ms_sched > 0)   printf("  sched   : %7.2f ms  (%.2fx vs plain)\n",
                                ms_sched, ms_plain > 0 ? ms_plain / ms_sched : 0);
    else                 printf("  sched   : FAIL\n");

    if (ms_direct > 0)  printf("  direct  : %7.2f ms  (%.2fx vs plain)\n",
                                ms_direct, ms_plain > 0 ? ms_plain / ms_direct : 0);
    else                 printf("  direct  : FAIL\n");

    if (ms_grouped > 0) printf("  grouped : %7.2f ms  (%.2fx vs plain)\n",
                                ms_grouped, ms_plain > 0 ? ms_plain / ms_grouped : 0);
    else                 printf("  grouped : FAIL\n");

    if (ms_vbatch > 0)  printf("  vbatch  : %7.2f ms  (%.2fx vs plain)\n",
                                ms_vbatch, ms_plain > 0 ? ms_plain / ms_vbatch : 0);
    else                 printf("  vbatch  : FAIL\n");

    return ok;
}

// ============================================================================
// case filter
// ============================================================================
static bool case_enabled(const char* filter, const char* token)
{
    return !filter || strcmp(filter, token) == 0;
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char** argv)
{
    setbuf(stdout, NULL);
    srand(42);

    bool smoke_only = false;
    const char* case_filter = getenv("PF_SCHED_CASE");
    const char* smoke_env = getenv("PF_SCHED_SMOKE_ONLY");
    if (case_filter && case_filter[0] == '\0')
        case_filter = NULL;
    if ((smoke_env && strcmp(smoke_env, "1") == 0) ||
        (argc >= 2 && strcmp(argv[1], "--smoke") == 0)) {
        smoke_only = true;
    }

    printf("Five-way Comparison: plain vs sched vs direct vs grouped vs vbatch\n");
    printf("  plain   = direct volatile reads (UVM on-demand faults)\n");
    printf("  sched   = pf_scheduler (ring + sort + dedup + touch)\n");
    printf("  direct  = atomicCAS dedup + winner touches (no scheduler)\n");
    printf("  grouped = atomicCAS dedup + VA Block bucketing + ordered touch\n");
    printf("  vbatch  = cross-block VA Block buffer + sequenced flush\n");
    printf("TEST_NUM_PAGES=%u, N_UNIQUE_PAGES=%u, DUP_FACTOR=%u\n",
           (unsigned)TEST_NUM_PAGES, (unsigned)N_UNIQUE_PAGES,
           (unsigned)DUP_FACTOR);
    printf("TEST_DATA_SIZE=%llu bytes (%.2f MB), unique data=%.2f MB\n",
           (unsigned long long)TEST_DATA_SIZE,
           (double)TEST_DATA_SIZE / (1024.0 * 1024.0),
           (double)N_UNIQUE_PAGES * TEST_PAGE_SIZE / (1024.0 * 1024.0));
    if (smoke_only) printf("SMOKE MODE: 4-block cases only.\n");
    if (case_filter) printf("CASE FILTER: %s\n", case_filter);

    // Allocate UVM data (SWEEP_MAX_PAGES for scaled sweep)
    uint64_t* data = nullptr;
    cudaMallocManaged(&data, SWEEP_DATA_SIZE);
    if (!data) { printf("[ERROR] cudaMallocManaged failed\n"); return 1; }

    for (uint64_t p = 0; p < (uint64_t)SWEEP_MAX_PAGES; p++)
        for (uint64_t k = 0; k < TEST_ELEMS_PER_PAGE; k++)
            data[p * TEST_ELEMS_PER_PAGE + k] = p;

    // Generate random indices: each in [0, N_UNIQUE_PAGES)
    // Allocate enough for sweep mode (max 32 blocks × 8192 chunk = 262144)
    const uint32_t MAX_ACCESSES = 262144u;
    uint32_t* h_indices = (uint32_t*)malloc(MAX_ACCESSES * sizeof(uint32_t));
    for (uint32_t i = 0; i < MAX_ACCESSES; i++)
        h_indices[i] = (uint32_t)(rand() % N_UNIQUE_PAGES);

    uint32_t* d_indices;
    cudaMalloc(&d_indices, MAX_ACCESSES * sizeof(uint32_t));
    cudaMemcpy(d_indices, h_indices, MAX_ACCESSES * sizeof(uint32_t),
               cudaMemcpyHostToDevice);

    int n_pass = 0, n_fail = 0;

    if (case_enabled(case_filter, "seq_dup4")) {
        if (run_test("SeqDup  (4 blocks x 256 threads)",
                     false, 4, 256, TEST_NUM_PAGES, data, h_indices, d_indices))
            n_pass++; else n_fail++;
    }
    if (case_enabled(case_filter, "rand_dup4")) {
        if (run_test("RandDup (4 blocks x 256 threads)",
                     true, 4, 256, TEST_NUM_PAGES, data, h_indices, d_indices))
            n_pass++; else n_fail++;
    }
    if (!smoke_only) {
        if (case_enabled(case_filter, "seq_dup16")) {
            if (run_test("SeqDup  (16 blocks x 256 threads)",
                         false, 16, 256, TEST_NUM_PAGES, data, h_indices, d_indices))
                n_pass++; else n_fail++;
        }
        if (case_enabled(case_filter, "rand_dup16")) {
            if (run_test("RandDup (16 blocks x 256 threads)",
                         true, 16, 256, TEST_NUM_PAGES, data, h_indices, d_indices))
                n_pass++; else n_fail++;
        }
    }

    printf("\nSummary: %d PASS, %d FAIL\n", n_pass, n_fail);

    // ================================================================
    // SWEEP MODE: PF_SWEEP=1 → seq_dup, 5 methods, threads 512..4096
    // ================================================================
    const char* sweep_env = getenv("PF_SWEEP");
    if (sweep_env && strcmp(sweep_env, "1") == 0) {
        // Per-block chunk stays constant at 8192 (matching 512-thread baseline)
        // so n_accesses scales linearly with n_blocks: n_acc = 8192 * nblk
        const uint64_t CHUNK_PER_BLOCK = (uint64_t)TEST_NUM_PAGES / 4;  // 8192

        fprintf(stderr, "%-12s %8s %10s %10s %8s\n",
                "pattern", "threads", "plain_ms", "time_ms", "speedup");
        fflush(stderr);
        printf("\n========== SWEEP: seq_dup, 5 methods, threads 512..4096 ==========\n");
        printf("threads,n_acc,plain_ms,sched_ms,direct_ms,grouped_ms,vbatch_ms,"
               "sched_x,direct_x,grouped_x,vbatch_x\n");

        int tcounts[] = {512, 1024, 2048, 4096};
        int nc = sizeof(tcounts) / sizeof(tcounts[0]);
        const int NRUNS = 3;

        for (int c = 0; c < nc; c++) {
            int total = tcounts[c];
            int tpb = 128;
            int nblk = (total + tpb - 1) / tpb;
            uint64_t n_acc = CHUNK_PER_BLOCK * nblk;

            uint64_t expected = expected_xor_seq(n_acc, nblk, tpb);

            float sum_p = 0, sum_s = 0, sum_d = 0, sum_g = 0, sum_v = 0;
            for (int r = 0; r < NRUNS; r++) {
                sum_p += run_variant_plain  (false, nblk, tpb, n_acc, data, d_indices, expected);
                sum_s += run_variant_sched  (false, nblk, tpb, n_acc, data, d_indices, expected);
                sum_d += run_variant_direct (false, nblk, tpb, n_acc, data, d_indices, expected);
                sum_g += run_variant_grouped(false, nblk, tpb, n_acc, data, d_indices, expected);
                sum_v += run_variant_vbatch (false, nblk, tpb, n_acc, data, d_indices, expected);
            }
            float mp = sum_p/NRUNS, ms = sum_s/NRUNS, md = sum_d/NRUNS;
            float mg = sum_g/NRUNS, mv = sum_v/NRUNS;
            printf("%d,%llu,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
                   total, (unsigned long long)n_acc, mp, ms, md, mg, mv,
                   mp/ms, mp/md, mp/mg, mp/mv);
            fprintf(stderr, "%-12s %8d %10.2f %10.2f %7.2fx\n",
                    "sequential", total, mp, mv, mp / mv);
            fflush(stderr);
        }

        // --- RandDup sweep (scaled: n_unique grows with blocks) ---
        // Each block gets chunk_per_block=8192 accesses with DUP_FACTOR=4
        // => n_unique = 2048 * nblk, so every block sees ~same # cold pages
        printf("\n========== SWEEP: rand_dup (scaled unique), 5 methods, threads 512..4096 ==========\n");
        printf("threads,n_acc,n_unique,plain_ms,sched_ms,direct_ms,grouped_ms,vbatch_ms,"
               "sched_x,direct_x,grouped_x,vbatch_x\n");

        for (int c = 0; c < nc; c++) {
            int total = tcounts[c];
            int tpb = 128;
            int nblk = (total + tpb - 1) / tpb;
            uint64_t n_acc = CHUNK_PER_BLOCK * nblk;
            uint64_t n_unique = (uint64_t)N_UNIQUE_PAGES * nblk / 4;
            if (n_unique > SWEEP_MAX_PAGES) n_unique = SWEEP_MAX_PAGES;

            // Regenerate random indices for this n_unique
            srand(42);
            for (uint32_t ii = 0; ii < (uint32_t)n_acc; ii++)
                h_indices[ii] = (uint32_t)(rand() % n_unique);
            cudaMemcpy(d_indices, h_indices, n_acc * sizeof(uint32_t),
                       cudaMemcpyHostToDevice);

            uint64_t expected = expected_xor_rand(h_indices, n_acc, nblk, tpb);

            float sum_p = 0, sum_s = 0, sum_d = 0, sum_g = 0, sum_v = 0;
            for (int r = 0; r < NRUNS; r++) {
                sum_p += run_variant_plain  (true, nblk, tpb, n_acc, data, d_indices, expected);
                sum_s += run_variant_sched  (true, nblk, tpb, n_acc, data, d_indices, expected);
                sum_d += run_variant_direct (true, nblk, tpb, n_acc, data, d_indices, expected);
                sum_g += run_variant_grouped(true, nblk, tpb, n_acc, data, d_indices, expected);
                sum_v += run_variant_vbatch (true, nblk, tpb, n_acc, data, d_indices, expected);
            }
            float mp = sum_p/NRUNS, ms = sum_s/NRUNS, md = sum_d/NRUNS;
            float mg = sum_g/NRUNS, mv = sum_v/NRUNS;
            printf("%d,%llu,%llu,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
                   total, (unsigned long long)n_acc, (unsigned long long)n_unique,
                   mp, ms, md, mg, mv,
                   mp/ms, mp/md, mp/mg, mp/mv);
            fprintf(stderr, "%-12s %8d %10.2f %10.2f %7.2fx\n",
                    "random", total, mp, mv, mp / mv);
            fflush(stderr);
        }
    }

    free(h_indices);
    cudaFree(d_indices);
    cudaFree(data);
    return n_fail > 0 ? 1 : 0;
}
