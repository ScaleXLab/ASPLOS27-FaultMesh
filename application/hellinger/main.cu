/*
 * hellinger_bench.cu — Tiled Hellinger distance: plain vs frontend-prefault
 *
 * Structure identical to GEMM tiling: a[M×N] × b[N×P] → c[M×P]
 * Inner loop: sum += sqrt(a[row,i] * b[i,col])
 * Tile loads go through the frontend abstraction → CAS dedup reduces UVM fault count.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "../../frontLib/frontend_prefault_common.cuh"

#define TILE 16

#define SQRT sqrtf
typedef float FP;

#ifndef M_SIZE
#define M_SIZE (512 * 37)
#endif
constexpr int M = M_SIZE;
constexpr int NN = M_SIZE;   // avoid conflict with cstdlib N
constexpr int P = M_SIZE;

static double rtclock() {
    struct timeval Tp;
    gettimeofday(&Tp, NULL);
    return Tp.tv_sec + Tp.tv_usec * 1.0e-6;
}

static void init_data(FP *a, FP *b, FP *c) {
    srand(123);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < NN; j++)
            a[i * NN + j] = (FP)1.0 / NN;
    for (int i = 0; i < NN; i++)
        for (int j = 0; j < P; j++)
            b[i * P + j] = rand() % 256;
    // normalize columns of b
    for (int j = 0; j < P; j++) {
        FP sum = 0;
        for (int i = 0; i < NN; i++) sum += b[i * P + j];
        for (int i = 0; i < NN; i++) b[i * P + j] /= sum;
    }
    for (int i = 0; i < M; i++)
        for (int j = 0; j < P; j++)
            c[i * P + j] = 0;
}

/* ── plain tiled hellinger ──────────────────────────────────── */
__global__ void hellinger_plain(FP *a, FP *b, FP *c,
                                int m, int n, int k) {
    __shared__ FP As[TILE][TILE], Bs[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int col = blockIdx.x * TILE + tx;
    int row = blockIdx.y * TILE + ty;

    FP sum = 0;
    for (int t = 0; t < n / TILE; t++) {
        As[ty][tx] = a[row * n + t * TILE + tx];
        Bs[ty][tx] = b[(t * TILE + ty) * k + col];
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < TILE; i++)
            sum += SQRT(As[ty][i] * Bs[i][tx]);
        __syncthreads();
    }
    FP value = (FP)1.0 - sum;
    FP gate = (!signbit(value));
    c[row * k + col] = SQRT(gate * value);
}

/* ── frontend tiled hellinger: compute unchanged, accesses wrapped ───────── */
__global__ void hellinger_frontend(FP *a, FP *b, FP *c,
                                   int m, int n, int k,
                                   uint32_t page_shift,
                                   uint32_t *a_status,
                                   uint32_t *b_status,
                                   uint32_t *c_status,
                                   uint32_t *c_write_status,
                                   unsigned long long *cas_wins) {
    __shared__ FP As[TILE][TILE], Bs[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int col = blockIdx.x * TILE + tx;
    int row = blockIdx.y * TILE + ty;
    const uvm_prefault_array_dedup_t<FP> a_pf{
        a, static_cast<uint64_t>(m) * n, page_shift, a_status, nullptr, cas_wins};
    const uvm_prefault_array_dedup_t<FP> b_pf{
        b, static_cast<uint64_t>(n) * k, page_shift, b_status, nullptr, cas_wins};
    const uvm_prefault_array_dedup_t<FP> c_pf{
        c, static_cast<uint64_t>(m) * k, page_shift, c_status, c_write_status, cas_wins};

    FP sum = 0;
    for (int t = 0; t < n / TILE; t++) {
        As[ty][tx] = a_pf[(uint64_t)row * n + t * TILE + tx];
        Bs[ty][tx] = b_pf[(uint64_t)(t * TILE + ty) * k + col];
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < TILE; i++)
            sum += SQRT(As[ty][i] * Bs[i][tx]);
        __syncthreads();
    }
    FP value = (FP)1.0 - sum;
    FP gate = (!signbit(value));
    c_pf.store((uint64_t)row * k + col, SQRT(gate * value));
}

/* ── naive (non-tiled) plain — for reference ────────────────── */
__global__ void hellinger_naive(FP *a, FP *b, FP *c,
                                int m, int n, int k) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    {
        FP sum = 0;
        for (int i = 0; i < n; i++)
            sum += SQRT(a[row * n + i] * b[i * k + col]);
        FP value = (FP)1.0 - sum;
        FP gate = (!signbit(value));
        c[row * k + col] = SQRT(gate * value);
    }
}

int main() {
    const char *bv = getenv("BENCH_VARIANT");
    const bool run_all = !bv;
    const bool is_baseline =
        bv && (strcmp(bv, "plain") == 0 || strcmp(bv, "baseline") == 0);
    const bool is_frontend =
        bv && (strcmp(bv, "prefault") == 0 || strcmp(bv, "frontend") == 0);
    bool run_plain = run_all || is_baseline;
    bool run_prefault = run_all || is_frontend;

    const size_t size_a = (size_t)M * NN * sizeof(FP);
    const size_t size_b = (size_t)NN * P * sizeof(FP);
    const size_t size_c = (size_t)M * P * sizeof(FP);
    uint64_t a_pages = (size_a + 4095) / 4096;
    uint64_t b_pages = (size_b + 4095) / 4096;
    uint64_t c_pages = (size_c + 4095) / 4096;
    const uint32_t page_shift = 12;

    FP *a_dev, *b_dev, *c_dev;
    cudaMallocManaged(&a_dev, size_a);
    cudaMallocManaged(&b_dev, size_b);
    cudaMallocManaged(&c_dev, size_c);
#ifndef SKIP_CPU_VERIFY
    init_data(a_dev, b_dev, c_dev);
#endif

    prefault_array_host_t<FP> a_pf{};
    prefault_array_host_t<FP> b_pf{};
    prefault_array_host_t<FP> c_pf{};
    if (run_prefault) {
        prefault_array_init(&a_pf, a_dev, static_cast<uint64_t>(M) * NN, page_shift, false);
        prefault_array_init(&b_pf, b_dev, static_cast<uint64_t>(NN) * P, page_shift, false);
        prefault_array_init(&c_pf, c_dev, static_cast<uint64_t>(M) * P, page_shift, true);
    }

    unsigned long long *d_cas_wins;
    cudaMalloc(&d_cas_wins, sizeof(unsigned long long));
    unsigned long long h_cas_wins = 0;

    dim3 block(TILE, TILE);
    dim3 grid((P + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    pf_uvm_warmup();

    fprintf(stderr, "Hellinger bench: M=%d N=%d P=%d TILE=%d\n", M, NN, P, TILE);
    fprintf(stderr, "Grid: %u×%u  Block: %u×%u\n", grid.x, grid.y, block.x, block.y);
    fprintf(stderr, "A_pages=%lu  B_pages=%lu  C_pages=%lu  total=%lu\n",
            a_pages, b_pages, c_pages, a_pages + b_pages + c_pages);

    double t0 = 0.0, t1 = 0.0;
    double t_plain = 0.0;
    double t_prefault = 0.0;

    if (run_plain) {
        /* ── tiled plain ──────────────────────────────────────── */
        prefault_to_cpu(a_dev, size_a);
        prefault_to_cpu(b_dev, size_b);
        prefault_to_cpu(c_dev, size_c);
        t0 = rtclock();
        hellinger_plain<<<grid, block>>>(a_dev, b_dev, c_dev, M, NN, P);
        cudaDeviceSynchronize();
        t1 = rtclock();
        t_plain = t1 - t0;
        fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_plain);
        fprintf(stdout, "hellinger,plain,%.6f\n", t_plain);
    }

    if (run_prefault) {
        /* ── tiled prefault ───────────────────────────────────── */
        prefault_to_cpu(a_dev, size_a);
        prefault_to_cpu(b_dev, size_b);
        prefault_to_cpu(c_dev, size_c);
        prefault_array_reset(&a_pf);
        prefault_array_reset(&b_pf);
        prefault_array_reset(&c_pf);
        cudaMemset(d_cas_wins, 0, sizeof(unsigned long long));
        t0 = rtclock();
        hellinger_frontend<<<grid, block>>>(
            a_dev, b_dev, c_dev, M, NN, P, page_shift,
            a_pf.page_status, b_pf.page_status, c_pf.page_status, c_pf.write_status, d_cas_wins);
        cudaDeviceSynchronize();
        t1 = rtclock();
        t_prefault = t1 - t0;
        cudaMemcpy(&h_cas_wins, d_cas_wins, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        fprintf(stdout, "hellinger,prefault,%.6f\n", t_prefault);

        fprintf(stderr, "\n  cas_wins=%llu  total_pages=%lu\n", h_cas_wins, a_pages + b_pages + c_pages);
    }
    if (run_plain && run_prefault) {
        fprintf(stderr, "  speedup vs tiled_plain:  %.2fx\n", t_plain / t_prefault);
    }

    cudaFree(d_cas_wins);
    if (run_prefault) {
        prefault_array_destroy(&a_pf);
        prefault_array_destroy(&b_pf);
        prefault_array_destroy(&c_pf);
    }
    cudaFree(a_dev); cudaFree(b_dev); cudaFree(c_dev);
    return 0;
}
