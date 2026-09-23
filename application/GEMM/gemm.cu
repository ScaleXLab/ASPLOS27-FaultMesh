/*
 * gemm_tiled_bench.cu — Tiled GEMM: plain vs frontend-prefault
 *
 * Shared-memory tiling separates page touch (tile load) from compute.
 *
 * Variants:
 *   1. tiled_plain           : baseline tiled kernel
 *   2. tiled_prefault        : frontend-managed A/B + read-style C warmup
 *   3. tiled_prefault_write  : frontend-managed A/B + write-style C warmup
 *   4. tiled_prefault_2step  : separate global prefault of A/B/C, then plain
 *
 * Compute loop is otherwise identical (shared memory reads, fully optimizable).
 *
 * Page duplication analysis (NI=NJ=NK=2048, TILE=32):
 *   - 1 page = 4KB = 1024 floats
 *   - B row = 2048 floats = 2 pages. bx=0..31 → page 0, bx=32..63 → page 1
 *   - Per tile: each warp accesses 1 B page (32 elements < 1024/page)
 *     32 warps → 32 unique B pages per tile
 *   - Blocks with same bx share the same B pages → cross-block duplication
 *   - frontend CAS dedup: only one thread wins per page, others use the fast path
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "../../frontLib/frontend_prefault_common.cuh"

#ifndef NI
#define multiplier 37
#define NI (512 * multiplier)
#define NJ (512 * multiplier)
#define NK (512 * multiplier)
#endif
#define TILE 32

#define ALPHA 32412.0f
#define BETA  2123.0f

typedef float DATA_TYPE;

static double rtclock() {
    struct timeval Tp;
    gettimeofday(&Tp, NULL);
    return Tp.tv_sec + Tp.tv_usec * 1.0e-6;
}

static void init_data(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C) {
    for (int i = 0; i < NI; i++)
        for (int j = 0; j < NK; j++)
            A[i * NK + j] = ((DATA_TYPE)i * j) / NI;
    for (int i = 0; i < NK; i++)
        for (int j = 0; j < NJ; j++)
            B[i * NJ + j] = ((DATA_TYPE)i * j + 1) / NJ;
    for (int i = 0; i < NI; i++)
        for (int j = 0; j < NJ; j++)
            C[i * NJ + j] = ((DATA_TYPE)i * j + 2) / NJ;
}

/* ── plain tiled GEMM ──────────────────────────────────────────── */
__global__ void gemm_plain(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C) {
    __shared__ DATA_TYPE As[TILE][TILE], Bs[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int j  = blockIdx.x * TILE + tx;
    int i  = blockIdx.y * TILE + ty;

    DATA_TYPE sum = 0.0f;
    for (int t = 0; t < NK / TILE; t++) {
        As[ty][tx] = A[i * NK + t * TILE + tx];
        Bs[ty][tx] = B[(t * TILE + ty) * NJ + j];
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; k++)
            sum += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    C[i * NJ + j] = ALPHA * sum + BETA * C[i * NJ + j];
}

/* ── prefault tiled GEMM: A/B prefault + read-style C warmup ────────────── */
__global__ void gemm_prefault(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C,
                              uint32_t page_shift,
                              uint32_t *a_status,
                              uint32_t *b_status,
                              uint32_t *c_status,
                              unsigned long long *cas_wins) {
    __shared__ DATA_TYPE As[TILE][TILE], Bs[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int j  = blockIdx.x * TILE + tx;
    int i  = blockIdx.y * TILE + ty;
    const uvm_prefault_array_dedup_t<DATA_TYPE> a_pf{
        A, static_cast<uint64_t>(NI) * NK, page_shift, a_status, nullptr, cas_wins};
    const uvm_prefault_array_dedup_t<DATA_TYPE> b_pf{
        B, static_cast<uint64_t>(NK) * NJ, page_shift, b_status, nullptr, cas_wins};
    const uvm_prefault_array_dedup_t<DATA_TYPE> c_pf{
        C, static_cast<uint64_t>(NI) * NJ, page_shift, c_status, nullptr, cas_wins};

    DATA_TYPE sum = 0.0f;
    for (int t = 0; t < NK / TILE; t++) {
        As[ty][tx] = a_pf[(uint64_t)i * NK + t * TILE + tx];
        Bs[ty][tx] = b_pf[(uint64_t)(t * TILE + ty) * NJ + j];
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; k++)
            sum += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    const uint64_t c_idx = (uint64_t)i * NJ + j;
    const DATA_TYPE c_val = c_pf[c_idx];
    c_pf.store(c_idx, ALPHA * sum + BETA * c_val);
}

/* ── prefault tiled GEMM: A/B prefault + write-style C warmup ───────────── */
__global__ void gemm_prefault_write(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C,
                                    uint32_t page_shift,
                                    uint32_t *a_status,
                                    uint32_t *b_status,
                                    uint32_t *c_status,
                                    uint32_t *c_write_status,
                                    unsigned long long *cas_wins) {
    __shared__ DATA_TYPE As[TILE][TILE], Bs[TILE][TILE];
    int tx = threadIdx.x, ty = threadIdx.y;
    int j  = blockIdx.x * TILE + tx;
    int i  = blockIdx.y * TILE + ty;
    const uvm_prefault_array_dedup_t<DATA_TYPE> a_pf{
        A, static_cast<uint64_t>(NI) * NK, page_shift, a_status, nullptr, cas_wins};
    const uvm_prefault_array_dedup_t<DATA_TYPE> b_pf{
        B, static_cast<uint64_t>(NK) * NJ, page_shift, b_status, nullptr, cas_wins};
    const uvm_prefault_array_dedup_t<DATA_TYPE> c_pf{
        C, static_cast<uint64_t>(NI) * NJ, page_shift, c_status, c_write_status, cas_wins};

    DATA_TYPE sum = 0.0f;
    for (int t = 0; t < NK / TILE; t++) {
        As[ty][tx] = a_pf[(uint64_t)i * NK + t * TILE + tx];
        Bs[ty][tx] = b_pf[(uint64_t)(t * TILE + ty) * NJ + j];
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; k++)
            sum += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    const uint64_t c_idx = (uint64_t)i * NJ + j;
    const DATA_TYPE c_val = c_pf[c_idx];
    c_pf.store(c_idx, ALPHA * sum + BETA * c_val);
}

/* ── two-step global prefault: one thread per page, then plain compute ───── */
__global__ void gemm_prefault_all_pages(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C,
                                        uint32_t page_shift,
                                        uint32_t *a_status,
                                        uint32_t *b_status,
                                        uint32_t *c_status,
                                        uint32_t *c_write_status,
                                        uint64_t n_pages_a, uint64_t n_pages_b, uint64_t n_pages_c,
                                        unsigned long long *cas_wins) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = n_pages_a + n_pages_b + n_pages_c;
    if (tid >= total) return;
    const uint64_t elems_per_page = 4096 / sizeof(DATA_TYPE);
    const uvm_prefault_array_dedup_t<DATA_TYPE> a_pf{
        A, static_cast<uint64_t>(NI) * NK, page_shift, a_status, nullptr, cas_wins};
    const uvm_prefault_array_dedup_t<DATA_TYPE> b_pf{
        B, static_cast<uint64_t>(NK) * NJ, page_shift, b_status, nullptr, cas_wins};
    const uvm_prefault_array_dedup_t<DATA_TYPE> c_pf{
        C, static_cast<uint64_t>(NI) * NJ, page_shift, c_status, c_write_status, cas_wins};

    if (tid < n_pages_a) {
        (void)a_pf[tid * elems_per_page];
    }
    else if (tid < n_pages_a + n_pages_b) {
        (void)b_pf[(tid - n_pages_a) * elems_per_page];
    }
    else {
        const uint64_t idx = (tid - n_pages_a - n_pages_b) * elems_per_page;
        C[idx] = c_pf[idx];
    }
}

/* ── prefault-only kernel: one thread per page, CAS dedup ──────────
 * Used by naive_dedup: migrate all A+B pages first, then run naive compute.
 * Isolates dedup benefit from tiling benefit.
 * ─────────────────────────────────────────────────────────────────── */
__global__ void gemm_prefault_only(DATA_TYPE *A, DATA_TYPE *B,
                                   uint32_t page_shift,
                                   uint32_t *a_status,
                                   uint32_t *b_status,
                                   uint64_t n_pages_a, uint64_t n_pages_b,
                                   unsigned long long *cas_wins) {
    uint64_t tid   = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = n_pages_a + n_pages_b;
    if (tid >= total) return;
    const uint64_t elems_per_page = 4096 / sizeof(DATA_TYPE);
    const uvm_prefault_array_dedup_t<DATA_TYPE> a_pf{
        A, static_cast<uint64_t>(NI) * NK, page_shift, a_status, nullptr, cas_wins};
    const uvm_prefault_array_dedup_t<DATA_TYPE> b_pf{
        B, static_cast<uint64_t>(NK) * NJ, page_shift, b_status, nullptr, cas_wins};
    if (tid < n_pages_a)
        (void)a_pf[tid * elems_per_page];
    else
        (void)b_pf[(tid - n_pages_a) * elems_per_page];
}

/* ── naive (non-tiled) plain GEMM — for reference ──────────────── */
__global__ void gemm_naive(DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *c) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i < NI) && (j < NJ)) {
        c[i * NJ + j] *= BETA;
        for (int k = 0; k < NK; k++)
            c[i * NJ + j] += ALPHA * a[i * NK + k] * b[k * NJ + j];
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
    bool run_prefault_write = bv && strcmp(bv, "prefault_c_write") == 0;
    bool run_prefault_2step = bv && strcmp(bv, "prefault_2step") == 0;

    const size_t size_a = (size_t)NI * NK * sizeof(DATA_TYPE);
    const size_t size_b = (size_t)NK * NJ * sizeof(DATA_TYPE);
    const size_t size_c = (size_t)NI * NJ * sizeof(DATA_TYPE);
    uint64_t a_pages = (size_a + 4095) / 4096;
    uint64_t b_pages = (size_b + 4095) / 4096;
    uint64_t c_pages = (size_c + 4095) / 4096;
    const uint32_t page_shift = 12;

    DATA_TYPE *A, *B, *C;
    cudaMallocManaged(&A, size_a);
    cudaMallocManaged(&B, size_b);
    cudaMallocManaged(&C, size_c);
#ifdef SKIP_CPU_VERIFY
    DATA_TYPE *C_init = nullptr;
#else
    DATA_TYPE *C_init = (DATA_TYPE *)malloc(size_c);
    init_data(A, B, C);
    memcpy(C_init, C, size_c);
#endif

    prefault_array_host_t<DATA_TYPE> a_pf{};
    prefault_array_host_t<DATA_TYPE> b_pf{};
    prefault_array_host_t<DATA_TYPE> c_pf{};
    if (run_prefault || run_prefault_write || run_prefault_2step) {
        prefault_array_init(&a_pf, A, static_cast<uint64_t>(NI) * NK, page_shift, false);
        prefault_array_init(&b_pf, B, static_cast<uint64_t>(NK) * NJ, page_shift, false);
        prefault_array_init(&c_pf, C, static_cast<uint64_t>(NI) * NJ, page_shift, true);
    }

    unsigned long long *d_cas_wins;
    cudaMalloc(&d_cas_wins, sizeof(unsigned long long));
    unsigned long long h_cas_wins = 0;

    dim3 block(TILE, TILE);
    dim3 grid(NJ / TILE, NI / TILE);

    pf_uvm_warmup();

    fprintf(stderr, "GEMM tiled bench: NI=%d NJ=%d NK=%d TILE=%d\n",
            NI, NJ, NK, TILE);
    fprintf(stderr, "Grid: %u×%u  Block: %u×%u  threads=%u\n",
            grid.x, grid.y, block.x, block.y, block.x * block.y);
    fprintf(stderr, "A_pages=%lu  B_pages=%lu  C_pages=%lu  total=%lu\n",
            a_pages, b_pages, c_pages, a_pages + b_pages + c_pages);

    double t0 = 0.0, t1 = 0.0;
    double t_naive = 0.0;
    double t_plain = 0.0;
    double t_prefault = 0.0;

    if (run_plain) {
        /* ── naive plain (non-tiled, for reference) ─────────────── */
        const char *skip_naive = getenv("SKIP_NAIVE_GEMM");
        if (!pf_zero_fault() && !(skip_naive && strcmp(skip_naive, "1") == 0)) {
            prefault_to_cpu(A, size_a);
            prefault_to_cpu(B, size_b);
            prefault_to_cpu(C, size_c);
            if (C_init) memcpy(C, C_init, size_c);
            t0 = rtclock();
            gemm_naive<<<grid, block>>>(A, B, C);
            cudaDeviceSynchronize();
            t1 = rtclock();
            t_naive = t1 - t0;
        }

        /* ── tiled plain ────────────────────────────────────────── */
        prefault_to_cpu(A, size_a);
        prefault_to_cpu(B, size_b);
        prefault_to_cpu(C, size_c);
        if (C_init) memcpy(C, C_init, size_c);
        if (pf_zero_fault()) {
            prefault_to_gpu(A, size_a);
            prefault_to_gpu(B, size_b);
            prefault_to_gpu(C, size_c);
        }
        t0 = rtclock();
        gemm_plain<<<grid, block>>>(A, B, C);
        cudaDeviceSynchronize();
        t1 = rtclock();
        t_plain = t1 - t0;
        fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_plain);
        fprintf(stdout, "gemm,plain,%.6f\n", t_plain);
    }

    if (run_prefault) {
        /* ── tiled prefault ─────────────────────────────────────── */
        prefault_to_cpu(A, size_a);
        prefault_to_cpu(B, size_b);
        prefault_to_cpu(C, size_c);
        if (C_init) memcpy(C, C_init, size_c);
        prefault_array_reset(&a_pf);
        prefault_array_reset(&b_pf);
        prefault_array_reset(&c_pf);
        cudaMemset(d_cas_wins, 0, sizeof(unsigned long long));
        t0 = rtclock();
        gemm_prefault<<<grid, block>>>(
            A, B, C, page_shift, a_pf.page_status, b_pf.page_status, c_pf.page_status, d_cas_wins);
        cudaDeviceSynchronize();
        t1 = rtclock();
        t_prefault = t1 - t0;
        cudaMemcpy(&h_cas_wins, d_cas_wins, sizeof(unsigned long long),
                   cudaMemcpyDeviceToHost);
        fprintf(stdout, "gemm,prefault,%.6f\n", t_prefault);

        fprintf(stderr, "\n  cas_wins=%llu  total_pages=%lu\n",
                h_cas_wins, a_pages + b_pages + c_pages);
    }
    if (run_prefault_write) {
        prefault_to_cpu(A, size_a);
        prefault_to_cpu(B, size_b);
        prefault_to_cpu(C, size_c);
        if (C_init) memcpy(C, C_init, size_c);
        prefault_array_reset(&a_pf);
        prefault_array_reset(&b_pf);
        prefault_array_reset(&c_pf);
        cudaMemset(d_cas_wins, 0, sizeof(unsigned long long));
        t0 = rtclock();
        gemm_prefault_write<<<grid, block>>>(
            A, B, C, page_shift,
            a_pf.page_status, b_pf.page_status, c_pf.page_status, c_pf.write_status, d_cas_wins);
        cudaDeviceSynchronize();
        t1 = rtclock();
        double t_prefault_write = t1 - t0;
        cudaMemcpy(&h_cas_wins, d_cas_wins, sizeof(unsigned long long),
                   cudaMemcpyDeviceToHost);
        fprintf(stdout, "gemm,tiled_prefault_write,%.6f\n", t_prefault_write);
        fprintf(stderr, "\n  prefault_write cas_wins=%llu  total_pages=%lu\n",
                h_cas_wins, a_pages + b_pages + c_pages);
    }

    if (run_prefault_2step) {
        dim3 pf_block(256);
        dim3 pf_grid((a_pages + b_pages + c_pages + 255) / 256);
        prefault_to_cpu(A, size_a);
        prefault_to_cpu(B, size_b);
        prefault_to_cpu(C, size_c);
        if (C_init) memcpy(C, C_init, size_c);
        prefault_array_reset(&a_pf);
        prefault_array_reset(&b_pf);
        prefault_array_reset(&c_pf);
        cudaMemset(d_cas_wins, 0, sizeof(unsigned long long));
        t0 = rtclock();
        gemm_prefault_all_pages<<<pf_grid, pf_block>>>(
            A, B, C, page_shift,
            a_pf.page_status, b_pf.page_status, c_pf.page_status, c_pf.write_status,
            a_pages, b_pages, c_pages, d_cas_wins);
        cudaDeviceSynchronize();
        gemm_plain<<<grid, block>>>(A, B, C);
        cudaDeviceSynchronize();
        t1 = rtclock();
        double t_prefault_2step = t1 - t0;
        cudaMemcpy(&h_cas_wins, d_cas_wins, sizeof(unsigned long long),
                   cudaMemcpyDeviceToHost);
        fprintf(stdout, "gemm,tiled_prefault_2step,%.6f\n", t_prefault_2step);
        fprintf(stderr, "\n  prefault_2step cas_wins=%llu  total_pages=%lu\n",
                h_cas_wins, a_pages + b_pages + c_pages);
    }
    if (run_plain && run_prefault) {
        fprintf(stderr, "  speedup vs tiled_plain:  %.2fx\n", t_plain / t_prefault);
        fprintf(stderr, "  speedup vs naive_plain:  %.2fx\n", t_naive / t_prefault);
    }

    cudaFree(d_cas_wins);
    if (run_prefault || run_prefault_write || run_prefault_2step) {
        prefault_array_destroy(&a_pf);
        prefault_array_destroy(&b_pf);
        prefault_array_destroy(&c_pf);
    }
    cudaFree(A); cudaFree(B); cudaFree(C);
    if (C_init) free(C_init);
    return 0;
}
