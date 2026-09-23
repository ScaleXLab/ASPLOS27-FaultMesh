#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

/* FrontLib.
 * Hit path: last_page / last_blk in registers, then one 2 MiB block counter.
 * The status table is read only when that block is not full. Once every page
 * is resident, views built afterwards skip the check.
 * Miss path: direct. One atomicCAS winner per 4 KiB page touches it. */

#define PF_LIKELY(x) (__builtin_expect(!!(x), 1))
static constexpr uint32_t PF_BLK_SHIFT = 9u;
static constexpr uint32_t PF_BLK_PAGES = 1u << PF_BLK_SHIFT;

static inline bool pf_zero_fault(void)
{
    const char* fr = getenv("FAULT_RATIO");
    return fr && strcmp(fr, "0") == 0;
}

static inline void prefault_to_gpu(void* ptr, size_t bytes)
{
    int dest = 0;
    cudaGetDevice(&dest);
    if (ptr && bytes)
        cudaMemPrefetchAsync(ptr, bytes, dest);
    cudaDeviceSynchronize();
}

static constexpr uint32_t PF_UNKNOWN = 0;
static constexpr uint32_t PF_PENDING = 1;
static constexpr uint32_t PF_READY = 2;

__device__ __forceinline__ uint32_t pf_warp_match_any_u64(uint32_t active, uint64_t value)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    return __match_any_sync(active, static_cast<unsigned long long>(value));
#else
    uint32_t eq_mask = 0;
    for (int l = 0; l < 32; ++l) {
        const uint32_t bit = (1u << l);
        if ((active & bit) == 0) continue;
        const uint64_t other = __shfl_sync(active, value, l);
        if (other == value) eq_mask |= bit;
    }
    return eq_mask;
#endif
}

__device__ __forceinline__ uint32_t pf_lane_id()
{
    // Warp lane is based on linear thread index, not threadIdx.x alone.
    // This is required for kernels that map warps along y/z dimensions
    // (e.g., blockDim.x=1, blockDim.y=32).
    const uint32_t linear_tid =
        (uint32_t) threadIdx.x +
        (uint32_t) blockDim.x * ((uint32_t) threadIdx.y + (uint32_t) blockDim.y * (uint32_t) threadIdx.z);
    return linear_tid & 31u;
}

__host__ __device__ __forceinline__ uint64_t pf_nblks(uint64_t n_pages)
{
    return (n_pages + PF_BLK_PAGES - 1) / PF_BLK_PAGES;
}

__host__ __device__ __forceinline__ uint32_t* pf_blk_cnt(uint32_t* status, uint64_t n_pages)
{
    return status + n_pages;
}

__host__ __device__ __forceinline__ uint32_t* pf_all_resident(uint32_t* status, uint64_t n_pages)
{
    return status + n_pages + pf_nblks(n_pages);
}

__host__ __device__ __forceinline__ uint32_t* pf_published(uint32_t* status, uint64_t n_pages)
{
    return pf_all_resident(status, n_pages) + 1;
}

template <typename T>
__host__ __device__ static uint64_t prefault_page_count(T* ptr, uint64_t n_elems, uint32_t page_shift)
{
    const uintptr_t raw_base = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t aligned_base = raw_base & ~((1ULL << page_shift) - 1ULL);
    const uintptr_t raw_end = raw_base + n_elems * sizeof(T);
    const uint64_t page_bytes = 1ULL << page_shift;
    return (raw_end - aligned_base + page_bytes - 1) >> page_shift;
}

template <typename T>
struct prefault_array_host_t {
    T* ptr;
    uint64_t n_elems;
    uint32_t page_shift;
    uint32_t* page_status;
    uint32_t* write_status;
    uint64_t n_pages;
};

static void pf_init_blk_counts(uint32_t* status, uint64_t n_pages)
{
    const uint64_t n_blks = pf_nblks(n_pages);
    std::vector<uint32_t> init(n_blks);
    for (uint64_t b = 0; b < n_blks; ++b) {
        uint64_t in_blk = n_pages - b * PF_BLK_PAGES;
        if (in_blk > PF_BLK_PAGES) in_blk = PF_BLK_PAGES;
        init[b] = static_cast<uint32_t>(PF_BLK_PAGES - in_blk);
    }
    cudaMemcpy(pf_blk_cnt(status, n_pages), init.data(), n_blks * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemset(pf_all_resident(status, n_pages), 0, 2 * sizeof(uint32_t));
}

template <typename T>
static void prefault_array_init(prefault_array_host_t<T>* arr,
                                T* ptr,
                                uint64_t n_elems,
                                uint32_t page_shift,
                                bool enable_write_status)
{
    arr->ptr = ptr;
    arr->n_elems = n_elems;
    arr->page_shift = page_shift;
    arr->n_pages = prefault_page_count(ptr, n_elems, page_shift);
    const uint64_t words = arr->n_pages + pf_nblks(arr->n_pages) + 2;
    cudaMalloc(&arr->page_status, words * sizeof(uint32_t));
    cudaMemset(arr->page_status, 0, arr->n_pages * sizeof(uint32_t));
    pf_init_blk_counts(arr->page_status, arr->n_pages);
    arr->write_status = nullptr;
    if (enable_write_status) {
        cudaMalloc(&arr->write_status, arr->n_pages * sizeof(uint32_t));
        cudaMemset(arr->write_status, 0, arr->n_pages * sizeof(uint32_t));
    }
}

template <typename T>
static void prefault_array_reset(prefault_array_host_t<T>* arr)
{
    cudaMemset(arr->page_status, 0, arr->n_pages * sizeof(uint32_t));
    pf_init_blk_counts(arr->page_status, arr->n_pages);
    if (arr->write_status != nullptr) {
        cudaMemset(arr->write_status, 0, arr->n_pages * sizeof(uint32_t));
    }
}

template <typename T>
static void prefault_array_destroy(prefault_array_host_t<T>* arr)
{
    if (arr->page_status != nullptr) cudaFree(arr->page_status);
    arr->page_status = nullptr;
    if (arr->write_status != nullptr) cudaFree(arr->write_status);
    arr->write_status = nullptr;
}

__device__ __forceinline__ bool pf_blk_full(volatile uint32_t* blk_cnt, uint64_t blk)
{
    unsigned int v;
    asm volatile("ld.volatile.global.u32 %0, [%1];"
                 : "=r"(v)
                 : "l"(blk_cnt + blk)
                 : "memory");
    return v >= PF_BLK_PAGES;
}

__device__ __forceinline__ unsigned int pf_ld_status(volatile uint32_t* page_status, uint64_t page_id)
{
    unsigned int v;
    asm volatile("ld.volatile.global.u32 %0, [%1];"
                 : "=r"(v)
                 : "l"(page_status + page_id)
                 : "memory");
    return v;
}

__device__ __forceinline__ void pf_publish_page(volatile uint32_t* page_status,
                                                volatile uint32_t* blk_cnt,
                                                uint32_t* published,
                                                uint32_t* all_resident,
                                                uint64_t page_id,
                                                uint64_t n_pages)
{
    atomicExch((uint32_t*)&page_status[page_id], PF_READY);
    atomicAdd((uint32_t*)&blk_cnt[page_id >> PF_BLK_SHIFT], 1u);
    const uint32_t pub = atomicAdd(published, 1u) + 1u;
    if (pub == n_pages) *all_resident = 1u;
}

__device__ __forceinline__ void pf_touch_page(uint64_t page_id,
                                              uintptr_t base_addr,
                                              uintptr_t aligned_base,
                                              uintptr_t end_addr,
                                              uint32_t page_shift)
{
    const uintptr_t page_base = aligned_base + (static_cast<uintptr_t>(page_id) << page_shift);
    const uintptr_t touch_addr = page_base < base_addr ? base_addr : page_base;
    if (touch_addr < end_addr) {
        const unsigned char tmp = *reinterpret_cast<const volatile unsigned char*>(touch_addr);
        (void)tmp;
    }
}

__device__ __forceinline__ void pf_direct_touch(volatile uint32_t* page_status,
                                                volatile uint32_t* blk_cnt,
                                                uint32_t* published,
                                                uint32_t* all_resident,
                                                unsigned long long* cas_wins,
                                                uint64_t page_id,
                                                uint64_t n_pages,
                                                uintptr_t base_addr,
                                                uintptr_t aligned_base,
                                                uintptr_t end_addr,
                                                uint32_t page_shift)
{
    const uint32_t active = __activemask();
    const uint32_t eq_mask = pf_warp_match_any_u64(active, page_id);
    const int master = __ffs(eq_mask) - 1;
    const uint32_t lane = pf_lane_id();
    if (lane == static_cast<uint32_t>(master)) {
        const uint32_t old = atomicCAS((uint32_t*)&page_status[page_id], PF_UNKNOWN, PF_PENDING);
        if (old == PF_UNKNOWN) {
            if (cas_wins != nullptr) atomicAdd(cas_wins, 1ULL);
            pf_touch_page(page_id, base_addr, aligned_base, end_addr, page_shift);
            __threadfence();
            pf_publish_page(page_status, blk_cnt, published, all_resident, page_id, n_pages);
        }
    }
    while (pf_ld_status(page_status, page_id) != PF_READY) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(64);
#endif
    }
}

/* Free function, not a member: a member call materializes the view
 * (including last_page / last_blk) in local memory. Returns true when
 * the whole 2 MiB block is already resident.
 * A translation unit may set PF_ENSURE_SLOW_ATTR before including this
 * header. Hellinger inlines it; the default keeps the slow path out of
 * line so other kernels do not spill the hit-path registers. */
#ifndef PF_ENSURE_SLOW_ATTR
#define PF_ENSURE_SLOW_ATTR __noinline__
#endif
template <typename T>
__device__ PF_ENSURE_SLOW_ATTR bool pf_ensure_slow(T* ptr,
                                            uint64_t n_elems,
                                            uint32_t page_shift,
                                            volatile uint32_t* page_status,
                                            volatile uint32_t* blk_cnt,
                                            uint32_t* published,
                                            uint32_t* all_resident,
                                            unsigned long long* cas_wins,
                                            uint64_t page_id,
                                            uint64_t blk,
                                            uint64_t n_pages)
{
    if (pf_blk_full(blk_cnt, blk)) return true;
    if (pf_ld_status(page_status, page_id) == PF_READY) return false;

    const uintptr_t base_addr = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t aligned_base = base_addr & ~((1ULL << page_shift) - 1ULL);
    const uintptr_t end_addr = base_addr + static_cast<uintptr_t>(n_elems) * sizeof(T);
    pf_direct_touch(page_status, blk_cnt, published, all_resident, cas_wins,
                    page_id, n_pages, base_addr, aligned_base, end_addr, page_shift);
    return false;
}

template <typename T>
struct uvm_prefault_array_dedup_t {
    T* ptr;
    T* ptr_aligned;
    uint64_t n_elems;
    uint32_t page_shift;
    uint64_t n_pages;
    volatile uint32_t* page_status;
    volatile uint32_t* blk_cnt;
    uint32_t* published;
    uint32_t* all_resident;
    volatile uint32_t* write_status;
    unsigned long long* cas_wins;
    /* Per-thread hints. mutable because operator[] is const. */
    mutable uint64_t last_page;
    mutable uint64_t last_blk;
    int checks;

    __host__ __device__ uvm_prefault_array_dedup_t(T* p,
                                                   uint64_t n,
                                                   uint32_t shift,
                                                   uint32_t* status,
                                                   uint32_t* write_status_ptr,
                                                   unsigned long long* wins)
        : ptr(p), n_elems(n), page_shift(shift), page_status(status),
          write_status(write_status_ptr), cas_wins(wins),
          last_page(~0ull), last_blk(~0ull), checks(1)
    {
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
        uintptr_t mask = ~((1ULL << shift) - 1ULL);
        ptr_aligned = reinterpret_cast<T*>(addr & mask);
        n_pages = prefault_page_count(ptr, n_elems, page_shift);
        blk_cnt = pf_blk_cnt(status, n_pages);
        all_resident = pf_all_resident(status, n_pages);
        published = pf_published(status, n_pages);
#if defined(__CUDA_ARCH__)
        checks = (*all_resident == 0u);
#endif
    }

    __device__ __forceinline__ uint64_t page_of(uint64_t i) const
    {
        const uintptr_t addr = reinterpret_cast<uintptr_t>(&ptr[i]);
        const uintptr_t aligned_base = reinterpret_cast<uintptr_t>(ptr_aligned);
        return (addr - aligned_base) >> page_shift;
    }

    __device__ __forceinline__ void ensure(uint64_t page_id) const
    {
        const uint64_t blk = page_id >> PF_BLK_SHIFT;
        if (PF_LIKELY((page_id == last_page) | (blk == last_blk))) {
            last_page = page_id;
            return;
        }
        if (pf_ensure_slow<T>(ptr, n_elems, page_shift, page_status, blk_cnt,
                              published, all_resident, cas_wins, page_id, blk, n_pages))
            last_blk = blk;
        last_page = page_id;
    }

    __device__ __forceinline__ T load(uint64_t i) const
    {
        if (checks) ensure(page_of(i));
        return ptr[i];
    }

    __device__ __forceinline__ void store(uint64_t i, T v) const
    {
        if (checks) ensure(page_of(i));
        ptr[i] = v;
    }

    __device__ __forceinline__ T operator[](uint64_t i) const { return load(i); }
};


static inline void prefault_to_cpu(void* ptr, size_t bytes)
{
    if (pf_zero_fault()) {
        prefault_to_gpu(ptr, bytes);
        return;
    }
    if (ptr && bytes)
        cudaMemPrefetchAsync(ptr, bytes, cudaCpuDeviceId);
    cudaDeviceSynchronize();
}

__global__ static void _pf_uvm_warmup_kernel(float* d, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] += 1.0f;
}

static inline void pf_uvm_warmup(void)
{
    const int N = 1 << 18;
    float* buf = nullptr;
    cudaMallocManaged(&buf, static_cast<size_t>(N) * sizeof(float));
    for (int i = 0; i < N; ++i) buf[i] = 0.0f;
    _pf_uvm_warmup_kernel<<<(N + 255) / 256, 256>>>(buf, N);
    cudaDeviceSynchronize();
    cudaFree(buf);
}
