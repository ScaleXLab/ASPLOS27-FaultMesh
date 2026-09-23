/*
 * gesummv.cu — GESUMMV motivation benchmark: frontend vs CTA dedup
 *
 * Goal:
 *   Start from the original PolyBench GESUMMV row-wise kernel and add a
 *   prefault path similar in spirit to bicg_hpca_aligned_bench.cu.
 *
 * Variants (selected by BENCH_VARIANT):
 *   baseline   : original row-wise compute kernel
 *   cta_dedup  : CTA-local short-window dedup, targets duplicate only
 *   frontend   : BICG-style global frontend, suppresses duplicate + stale
 *
 * stdout:
 *   gesummv,<variant>,<gpu_seconds>
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <sys/time.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include "../../frontLib/frontend_prefault_common.cuh"

#ifndef PROBLEM_N
#define PROBLEM_N  (256 * 91)
#endif
#define N PROBLEM_N
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

#define ALPHA 43532.0f
#define BETA  12313.0f

typedef float DATA_TYPE;

static constexpr uint32_t CTA_DEDUP_EMPTY = 0xffffffffu;
static constexpr int CTA_DEDUP_SLOTS = 128;
static constexpr int CTA_DEDUP_WINDOW = 32;
static constexpr int CTA_DEDUP_PROBES = 4;
static constexpr int CTA_DEDUP_BACKOFF = 400;

__device__ __forceinline__ uint32_t warp_match_any_u64(uint32_t active, uint64_t value)
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

static double rtclock()
{
    struct timeval Tp;
    gettimeofday(&Tp, nullptr);
    return Tp.tv_sec + Tp.tv_usec * 1.0e-6;
}

template <typename T>
__device__ __forceinline__ void cta_short_window_touch(T* ptr,
                                                       uint64_t i,
                                                       uint32_t page_shift,
                                                       volatile uint32_t* seen_pages,
                                                       unsigned long long* cas_wins)
{
    const uintptr_t base_addr = reinterpret_cast<uintptr_t>(ptr);
    const uintptr_t addr = reinterpret_cast<uintptr_t>(&ptr[i]);
    const uintptr_t page_mask = ~static_cast<uintptr_t>((1ULL << page_shift) - 1ULL);
    const uintptr_t base_page = base_addr & page_mask;
    const uintptr_t page_base = addr & page_mask;
    const uint32_t page_id = static_cast<uint32_t>((page_base - base_page) >> page_shift);

    const uint32_t mask = __activemask();
    const uint32_t eq_mask = warp_match_any_u64(mask, page_id);
    const int master = __ffs(eq_mask) - 1;
    const uint32_t lane = threadIdx.x & 31;

    bool leader_should_touch = false;
    bool leader_should_backoff = false;

    if (lane == static_cast<uint32_t>(master)) {
        uint32_t tag = page_id + 1;
        uint32_t start = tag & (CTA_DEDUP_SLOTS - 1);

        for (int probe = 0; probe < CTA_DEDUP_PROBES; ++probe) {
            uint32_t slot = (start + probe) & (CTA_DEDUP_SLOTS - 1);
            uint32_t old = atomicCAS((unsigned int*)&seen_pages[slot], CTA_DEDUP_EMPTY, tag);
            if (old == CTA_DEDUP_EMPTY) {
                leader_should_touch = true;
                break;
            }
            if (old == tag) {
                leader_should_backoff = true;
                break;
            }
        }
    }

    leader_should_touch = __shfl_sync(eq_mask, leader_should_touch ? 1 : 0, master);
    leader_should_backoff = __shfl_sync(eq_mask, leader_should_backoff ? 1 : 0, master);

    if (leader_should_touch) {
        if (lane == static_cast<uint32_t>(master)) {
            if (cas_wins != nullptr)
                atomicAdd(cas_wins, 1ULL);
            volatile T tmp = *reinterpret_cast<const T*>(page_base);
            (void)tmp;
        }
    }
    else if (leader_should_backoff) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(CTA_DEDUP_BACKOFF);
#endif
    }

    __syncwarp(eq_mask);
}

static void init_data(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* x)
{
    for (long long i = 0; i < N; i++) {
        x[i] = ((DATA_TYPE)i) / N;
        for (long long j = 0; j < N; j++) {
            A[i * (long long)N + j] = ((DATA_TYPE)i * j) / N;
            B[i * (long long)N + j] = ((DATA_TYPE)(i + 1) * (j + 1)) / N;
        }
    }
}

static double checksum_pair(const DATA_TYPE* y, const DATA_TYPE* tmp)
{
    double sum = 0.0;
    for (int i = 0; i < N; i++)
        sum += (double)y[i] + (double)tmp[i];
    return sum;
}

static void maybe_publish_ranges_and_wait(DATA_TYPE* A,
                                          DATA_TYPE* B,
                                          DATA_TYPE* x,
                                          DATA_TYPE* y,
                                          DATA_TYPE* tmp)
{
    const char* range_file = getenv("BENCH_RANGE_FILE");
    const char* ready_file = getenv("BENCH_READY_FILE");

    if (range_file && range_file[0]) {
        FILE* fh = fopen(range_file, "w");
        if (fh) {
            fprintf(fh, "A_START=0x%llx\n", (unsigned long long)(uintptr_t)A);
            fprintf(fh, "A_END=0x%llx\n", (unsigned long long)((uintptr_t)A + (size_t)N * N * sizeof(DATA_TYPE)));
            fprintf(fh, "B_START=0x%llx\n", (unsigned long long)(uintptr_t)B);
            fprintf(fh, "B_END=0x%llx\n", (unsigned long long)((uintptr_t)B + (size_t)N * N * sizeof(DATA_TYPE)));
            fprintf(fh, "X_START=0x%llx\n", (unsigned long long)(uintptr_t)x);
            fprintf(fh, "X_END=0x%llx\n", (unsigned long long)((uintptr_t)x + (size_t)N * sizeof(DATA_TYPE)));
            fprintf(fh, "Y_START=0x%llx\n", (unsigned long long)(uintptr_t)y);
            fprintf(fh, "Y_END=0x%llx\n", (unsigned long long)((uintptr_t)y + (size_t)N * sizeof(DATA_TYPE)));
            fprintf(fh, "TMP_START=0x%llx\n", (unsigned long long)(uintptr_t)tmp);
            fprintf(fh, "TMP_END=0x%llx\n", (unsigned long long)((uintptr_t)tmp + (size_t)N * sizeof(DATA_TYPE)));
            fclose(fh);
        }
    }

    if (ready_file && ready_file[0]) {
        while (access(ready_file, F_OK) != 0)
            usleep(1000);
    }
}

__global__ void gesummv_plain(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *x,
                              DATA_TYPE *y, DATA_TYPE *tmp)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        DATA_TYPE sum_a = 0.0f, sum_b = 0.0f;
        for (int j = 0; j < N; j++) {
            sum_a += A[(long long)i * N + j] * x[j];
            sum_b += B[(long long)i * N + j] * x[j];
        }
        tmp[i] = sum_a;
        y[i] = ALPHA * sum_a + BETA * sum_b;
    }
}

__global__ void gesummv_frontend(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *x,
                                 DATA_TYPE *y, DATA_TYPE *tmp,
                                 uint32_t page_shift,
                                 uint32_t* a_status,
                                 uint32_t* b_status,
                                 uint32_t* x_status,
                                 uint32_t* y_status,
                                 uint32_t* y_write_status,
                                 uint32_t* tmp_status,
                                 uint32_t* tmp_write_status,
                                 unsigned long long* cas_wins)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    const uvm_prefault_array_dedup_t<DATA_TYPE> a_pf{
        A, static_cast<uint64_t>(N) * N, page_shift, a_status, nullptr, cas_wins};
    const uvm_prefault_array_dedup_t<DATA_TYPE> b_pf{
        B, static_cast<uint64_t>(N) * N, page_shift, b_status, nullptr, cas_wins};
    const uvm_prefault_array_dedup_t<DATA_TYPE> x_pf{
        x, static_cast<uint64_t>(N), page_shift, x_status, nullptr, cas_wins};
    const uvm_prefault_array_dedup_t<DATA_TYPE> y_pf{
        y, static_cast<uint64_t>(N), page_shift, y_status, y_write_status, cas_wins};
    const uvm_prefault_array_dedup_t<DATA_TYPE> tmp_pf{
        tmp, static_cast<uint64_t>(N), page_shift, tmp_status, tmp_write_status, cas_wins};

    if (i < N) {
        DATA_TYPE sum_a = 0.0f, sum_b = 0.0f;
        for (int j = 0; j < N; j++) {
            sum_a += a_pf[(uint64_t)i * N + j] * x_pf[j];
            sum_b += b_pf[(uint64_t)i * N + j] * x_pf[j];
        }
        tmp_pf.store(i, sum_a);
        y_pf.store(i, ALPHA * sum_a + BETA * sum_b);
    }
}

__global__ void gesummv_cta_dedup(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *x,
                                  DATA_TYPE *y, DATA_TYPE *tmp,
                                  uint32_t page_shift,
                                  unsigned long long* cas_wins)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ uint32_t x_seen[CTA_DEDUP_SLOTS];

    for (int slot = threadIdx.x; slot < CTA_DEDUP_SLOTS; slot += blockDim.x)
        x_seen[slot] = CTA_DEDUP_EMPTY;
    __syncthreads();

    if (i < N) {
        DATA_TYPE sum_a = 0.0f, sum_b = 0.0f;
        for (int j = 0; j < N; j++) {
            if ((j % CTA_DEDUP_WINDOW) == 0) {
                for (int slot = threadIdx.x; slot < CTA_DEDUP_SLOTS; slot += blockDim.x)
                    x_seen[slot] = CTA_DEDUP_EMPTY;
                __syncthreads();
            }

            cta_short_window_touch(x, j, page_shift, x_seen, cas_wins);
            DATA_TYPE x_val = x[j];
            sum_a += A[(uint64_t)i * N + j] * x_val;
            sum_b += B[(uint64_t)i * N + j] * x_val;
        }
        tmp[i] = sum_a;
        y[i] = ALPHA * sum_a + BETA * sum_b;
    }
}

static void run_benchmark(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* x,
                          DATA_TYPE* y, DATA_TYPE* tmp)
{
    const char *bv = getenv("BENCH_VARIANT");
    bool run_baseline = !bv || strcmp(bv, "baseline") == 0 || strcmp(bv, "plain") == 0;
    bool run_cta_dedup = bv && strcmp(bv, "cta_dedup") == 0;
    bool run_frontend = bv && strcmp(bv, "frontend") == 0;

    const size_t size_A = (size_t)N * N * sizeof(DATA_TYPE);
    const size_t size_B = (size_t)N * N * sizeof(DATA_TYPE);
    const size_t size_x = (size_t)N * sizeof(DATA_TYPE);
    const size_t size_y = (size_t)N * sizeof(DATA_TYPE);
    const size_t size_tmp = (size_t)N * sizeof(DATA_TYPE);
    const uint32_t page_shift = 12;

    dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
    dim3 grid((unsigned int)ceil(((float)N) / ((float)block.x)), 1);

    prefault_array_host_t<DATA_TYPE> a_pf{};
    prefault_array_host_t<DATA_TYPE> b_pf{};
    prefault_array_host_t<DATA_TYPE> x_pf{};
    prefault_array_host_t<DATA_TYPE> y_pf{};
    prefault_array_host_t<DATA_TYPE> tmp_pf{};

    prefault_array_init(&a_pf, A, static_cast<uint64_t>(N) * N, page_shift, false);
    prefault_array_init(&b_pf, B, static_cast<uint64_t>(N) * N, page_shift, false);
    prefault_array_init(&x_pf, x, static_cast<uint64_t>(N), page_shift, false);
    prefault_array_init(&y_pf, y, static_cast<uint64_t>(N), page_shift, true);
    prefault_array_init(&tmp_pf, tmp, static_cast<uint64_t>(N), page_shift, true);

    unsigned long long* d_cas_wins = nullptr;
    unsigned long long h_cas_wins = 0;
    cudaMalloc(&d_cas_wins, sizeof(unsigned long long));

    double t_baseline = 0.0, t_cta_dedup = 0.0, t_frontend = 0.0;

    if (run_baseline) {
        prefault_to_cpu(A, size_A);
        prefault_to_cpu(B, size_B);
        prefault_to_cpu(x, size_x);
        prefault_to_cpu(y, size_y);
        prefault_to_cpu(tmp, size_tmp);
        cudaMemset(y, 0, size_y);
        cudaMemset(tmp, 0, size_tmp);
        cudaMemset(d_cas_wins, 0, sizeof(unsigned long long));

        double t0 = rtclock();
        gesummv_plain<<<grid, block>>>(A, B, x, y, tmp);
        cudaDeviceSynchronize();
        double t1 = rtclock();

        t_baseline = t1 - t0;
        fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_baseline);
        fprintf(stdout, "gesummv,baseline,%.6f\n", t_baseline);
#ifndef SKIP_CPU_VERIFY
        fprintf(stderr, "  baseline checksum=%.6e cas_wins=0\n", checksum_pair(y, tmp));
#endif
    }

    if (run_cta_dedup) {
        prefault_to_cpu(A, size_A);
        prefault_to_cpu(B, size_B);
        prefault_to_cpu(x, size_x);
        prefault_to_cpu(y, size_y);
        prefault_to_cpu(tmp, size_tmp);
        cudaMemset(y, 0, size_y);
        cudaMemset(tmp, 0, size_tmp);
        cudaMemset(d_cas_wins, 0, sizeof(unsigned long long));

        double t0 = rtclock();
        gesummv_cta_dedup<<<grid, block>>>(
            A, B, x, y, tmp, page_shift, d_cas_wins);
        cudaDeviceSynchronize();
        double t1 = rtclock();

        t_cta_dedup = t1 - t0;
        cudaMemcpy(&h_cas_wins, d_cas_wins, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        fprintf(stdout, "gesummv,cta_dedup,%.6f\n", t_cta_dedup);
        fprintf(stderr, "  cta_dedup checksum=%.6e cas_wins=%llu\n",
                checksum_pair(y, tmp), h_cas_wins);
    }

    if (run_frontend) {
        prefault_to_cpu(A, size_A);
        prefault_to_cpu(B, size_B);
        prefault_to_cpu(x, size_x);
        prefault_to_cpu(y, size_y);
        prefault_to_cpu(tmp, size_tmp);
        cudaMemset(y, 0, size_y);
        cudaMemset(tmp, 0, size_tmp);

        prefault_array_reset(&a_pf);
        prefault_array_reset(&b_pf);
        prefault_array_reset(&x_pf);
        prefault_array_reset(&y_pf);
        prefault_array_reset(&tmp_pf);
        cudaMemset(d_cas_wins, 0, sizeof(unsigned long long));

        double t0 = rtclock();
        gesummv_frontend<<<grid, block>>>(
            A, B, x, y, tmp, page_shift,
            a_pf.page_status, b_pf.page_status, x_pf.page_status,
            y_pf.page_status, y_pf.write_status,
            tmp_pf.page_status, tmp_pf.write_status,
            d_cas_wins);
        cudaDeviceSynchronize();
        double t1 = rtclock();

        t_frontend = t1 - t0;
        cudaMemcpy(&h_cas_wins, d_cas_wins, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        fprintf(stdout, "gesummv,frontend,%.6f\n", t_frontend);
        fprintf(stderr, "  frontend checksum=%.6e cas_wins=%llu\n",
                checksum_pair(y, tmp), h_cas_wins);
    }

    if (run_baseline && run_cta_dedup)
        fprintf(stderr, "  speedup vs baseline (cta_dedup): %.2fx\n", t_baseline / t_cta_dedup);
    if (run_baseline && run_frontend)
        fprintf(stderr, "  speedup vs baseline (frontend): %.2fx\n", t_baseline / t_frontend);

    cudaFree(d_cas_wins);
    prefault_array_destroy(&a_pf);
    prefault_array_destroy(&b_pf);
    prefault_array_destroy(&x_pf);
    prefault_array_destroy(&y_pf);
    prefault_array_destroy(&tmp_pf);
}

int main()
{
    DATA_TYPE *A, *B, *x, *y, *tmp;
    const size_t size_A = (size_t)N * N * sizeof(DATA_TYPE);
    const size_t size_B = (size_t)N * N * sizeof(DATA_TYPE);
    const size_t size_x = (size_t)N * sizeof(DATA_TYPE);
    const size_t size_y = (size_t)N * sizeof(DATA_TYPE);
    const size_t size_tmp = (size_t)N * sizeof(DATA_TYPE);

    cudaMallocManaged(&A, size_A);
    cudaMallocManaged(&B, size_B);
    cudaMallocManaged(&x, size_x);
    cudaMallocManaged(&y, size_y);
    cudaMallocManaged(&tmp, size_tmp);

#ifndef SKIP_CPU_VERIFY
    init_data(A, B, x);
#endif

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("setting device %d with name %s\n", 0, deviceProp.name);
    cudaSetDevice(0);

    maybe_publish_ranges_and_wait(A, B, x, y, tmp);

    run_benchmark(A, B, x, y, tmp);

    cudaFree(A);
    cudaFree(B);
    cudaFree(x);
    cudaFree(y);
    cudaFree(tmp);
    return 0;
}
