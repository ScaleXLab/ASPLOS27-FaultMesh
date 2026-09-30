#pragma once

/* FaultMesh-FrontLib for llama.cpp weight tensors.
 *
 * REALAPP_PREFETCH=1      enable (BENCH_VARIANT=prefault is accepted too)
 * REALAPP_PF_DECODE=1     also prefault the single-token decode matvec
 * REALAPP_PF_COLD_PASSES  how many times one tensor is prefaulted (default 1).
 *                         When the model is larger than GPU memory, weights are
 *                         evicted between tokens, so set it large. */

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

static constexpr uint32_t REALAPP_PF_UNKNOWN = 0u;
static constexpr uint32_t REALAPP_PF_PENDING = 1u;
static constexpr uint32_t REALAPP_PF_READY   = 2u;
static constexpr uint64_t REALAPP_PF_PAGE_SIZE = 4096ull;
static constexpr uint64_t REALAPP_PF_PAGE_MASK = ~(REALAPP_PF_PAGE_SIZE - 1ull);
static constexpr uint32_t REALAPP_PF_SHMEM_QUEUE_CAP = 512u;

struct realapp_pf_region_view {
    uint32_t * page_status = nullptr;
    uint64_t base_addr = 0;
    uint64_t n_pages = 0;
    unsigned long long * cas_wins = nullptr;
};

static inline bool realapp_pf_is_enabled() {
    const char * direct = std::getenv("REALAPP_PREFETCH");
    if (direct != nullptr) {
        if (std::strcmp(direct, "1") == 0 || std::strcmp(direct, "true") == 0 ||
            std::strcmp(direct, "on") == 0 || std::strcmp(direct, "prefault") == 0) {
            return true;
        }
        if (std::strcmp(direct, "0") == 0 || std::strcmp(direct, "false") == 0 ||
            std::strcmp(direct, "off") == 0 || std::strcmp(direct, "plain") == 0) {
            return false;
        }
    }

    const char * bench_variant = std::getenv("BENCH_VARIANT");
    return bench_variant != nullptr && std::strcmp(bench_variant, "prefault") == 0;
}

static inline uint32_t realapp_pf_cold_passes() {
    const char * env = std::getenv("REALAPP_PF_COLD_PASSES");
    if (env != nullptr) {
        const long parsed = std::strtol(env, nullptr, 10);
        if (parsed >= 0) {
            return static_cast<uint32_t>(parsed);
        }
    }
    return 1u;
}

static inline bool realapp_pf_should_prefetch_cold(const void * ptr, size_t bytes) {
    if (!realapp_pf_is_enabled() || ptr == nullptr || bytes == 0) {
        return false;
    }

    const uint32_t cold_passes = realapp_pf_cold_passes();
    if (cold_passes == 0) {
        return false;
    }

    struct realapp_pf_host_key {
        uint64_t ptr;
        uint64_t bytes;

        bool operator==(const realapp_pf_host_key & other) const {
            return ptr == other.ptr && bytes == other.bytes;
        }
    };

    struct realapp_pf_host_key_hash {
        size_t operator()(const realapp_pf_host_key & key) const {
            return static_cast<size_t>((key.ptr >> 12) ^ (key.bytes << 1));
        }
    };

    static std::mutex mu;
    static std::unordered_map<realapp_pf_host_key, uint32_t, realapp_pf_host_key_hash> seen;

    const realapp_pf_host_key key {
        reinterpret_cast<uint64_t>(ptr),
        static_cast<uint64_t>(bytes),
    };

    std::lock_guard<std::mutex> lock(mu);
    uint32_t & count = seen[key];
    if (count >= cold_passes) {
        return false;
    }
    ++count;
    return true;
}

static inline uint64_t realapp_pf_aligned_base(const void * ptr) {
    return reinterpret_cast<uint64_t>(ptr) & REALAPP_PF_PAGE_MASK;
}

static inline uint64_t realapp_pf_num_pages(const void * ptr, size_t bytes) {
    if (ptr == nullptr || bytes == 0) {
        return 0;
    }
    const uint64_t begin = reinterpret_cast<uint64_t>(ptr);
    const uint64_t end = begin + bytes;
    const uint64_t aligned_begin = begin & REALAPP_PF_PAGE_MASK;
    const uint64_t aligned_end = (end + REALAPP_PF_PAGE_SIZE - 1ull) & REALAPP_PF_PAGE_MASK;
    return (aligned_end - aligned_begin) / REALAPP_PF_PAGE_SIZE;
}

static inline realapp_pf_region_view realapp_pf_make_view(uint32_t * page_status, const void * ptr, size_t bytes,
                                                          unsigned long long * cas_wins = nullptr) {
    realapp_pf_region_view view;
    view.page_status = page_status;
    view.base_addr = realapp_pf_aligned_base(ptr);
    view.n_pages = realapp_pf_num_pages(ptr, bytes);
    view.cas_wins = cas_wins;
    return view;
}

__device__ __forceinline__ static uint32_t realapp_pf_match_any(uint32_t mask, uint64_t value) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    return __match_any_sync(mask, static_cast<unsigned long long>(value));
#else
    const unsigned int value_lo = static_cast<unsigned int>(value);
    const unsigned int value_hi = static_cast<unsigned int>(value >> 32);
    uint32_t eq_mask = 0;
    for (int lane = 0; lane < 32; ++lane) {
        const unsigned int other_lo =
#if defined(__CUDA_ARCH__)
            __shfl_sync(mask, value_lo, lane);
#else
            value_lo;
#endif
        const unsigned int other_hi =
#if defined(__CUDA_ARCH__)
            __shfl_sync(mask, value_hi, lane);
#else
            value_hi;
#endif
        if (other_lo == value_lo && other_hi == value_hi) {
            eq_mask |= (1u << lane);
        }
    }
    return eq_mask & mask;
#endif
}

__device__ __forceinline__ static bool realapp_pf_try_claim_page(realapp_pf_region_view view,
                                                                 uintptr_t addr,
                                                                 uint32_t *page_id_out) {
    if (view.page_status == nullptr || view.n_pages == 0) {
        return false;
    }

    const uint64_t page_addr = static_cast<uint64_t>(addr) & REALAPP_PF_PAGE_MASK;
    if (page_addr < view.base_addr) {
        return false;
    }

    const uint64_t page_id = (page_addr - view.base_addr) >> 12;
    if (page_id >= view.n_pages) {
        return false;
    }

    if (page_id_out != nullptr) {
        *page_id_out = static_cast<uint32_t>(page_id);
    }

    const uint32_t st = atomicAdd((uint32_t *)&view.page_status[page_id], 0u);
    if (st == REALAPP_PF_READY) {
        return false;
    }

    const uint32_t mask = __activemask();
    const uint32_t eq_mask = realapp_pf_match_any(mask, page_id);
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t master = static_cast<uint32_t>(__ffs(eq_mask) - 1);

    if (lane != master) {
        return false;
    }

    const uint32_t old = atomicCAS((uint32_t *)&view.page_status[page_id],
                                   REALAPP_PF_UNKNOWN,
                                   REALAPP_PF_PENDING);
    if (old == REALAPP_PF_UNKNOWN && view.cas_wins != nullptr) {
        atomicAdd(view.cas_wins, 1ULL);
    }
    return old == REALAPP_PF_UNKNOWN;
}

__device__ __forceinline__ static void realapp_pf_touch_claimed_page(realapp_pf_region_view view,
                                                                     uint32_t page_id) {
    if (view.page_status == nullptr || view.n_pages == 0 || page_id >= view.n_pages) {
        return;
    }

    const uint64_t page_addr = view.base_addr + static_cast<uint64_t>(page_id) * REALAPP_PF_PAGE_SIZE;
    volatile uint64_t tmp = *(volatile uint64_t *)page_addr;
    (void) tmp;
    __threadfence();
    atomicExch((uint32_t *)&view.page_status[page_id], REALAPP_PF_READY);
}

__device__ __forceinline__ static void realapp_pf_touch_page_id(realapp_pf_region_view view,
                                                                uint64_t page_id) {
    if (view.page_status == nullptr || view.n_pages == 0 || page_id >= view.n_pages) {
        return;
    }

    uint32_t st = atomicAdd((uint32_t *)&view.page_status[page_id], 0u);
    if (st == REALAPP_PF_READY) {
        return;
    }

    const uint32_t old = atomicCAS((uint32_t *)&view.page_status[page_id],
                                   REALAPP_PF_UNKNOWN,
                                   REALAPP_PF_PENDING);
    if (old == REALAPP_PF_READY) {
        return;
    }

    if (old == REALAPP_PF_UNKNOWN) {
        if (view.cas_wins != nullptr) {
            atomicAdd(view.cas_wins, 1ULL);
        }
        realapp_pf_touch_claimed_page(view, static_cast<uint32_t>(page_id));
        return;
    }

    while (atomicAdd((uint32_t *)&view.page_status[page_id], 0u) != REALAPP_PF_READY) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(100);
#endif
    }
}

__global__ static void realapp_pf_prefetch_pages_kernel(realapp_pf_region_view view) {
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t page_id = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         page_id < view.n_pages; page_id += stride) {
        realapp_pf_touch_page_id(view, page_id);
    }
}

static inline int realapp_pf_launch_blocks(uint64_t n_pages, int threads) {
    int blocks = static_cast<int>((n_pages + threads - 1ull) / threads);
    if (blocks < 1) {
        blocks = 1;
    }
    if (blocks > 4096) {
        blocks = 4096;
    }
    return blocks;
}

static inline void realapp_pf_prefetch_view_pages(realapp_pf_region_view view,
                                                  cudaStream_t stream = 0) {
    if (view.page_status == nullptr || view.n_pages == 0) {
        return;
    }

    const int threads = 256;
    const int blocks = realapp_pf_launch_blocks(view.n_pages, threads);
    realapp_pf_prefetch_pages_kernel<<<blocks, threads, 0, stream>>>(view);
}
