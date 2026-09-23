/*
 * conv2d_bench.cu — 2D Convolution: plain vs frontend-prefault
 *
 * 3×3 stencil on NI×NJ array. Load (TILE_Y+2)×(TILE_X+2) halo tile
 * into shared memory; compute from shared memory.
 * Tile loads go through the frontend abstraction → CAS dedup reduces UVM fault count.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "../../frontLib/frontend_prefault_common.cuh"

#ifndef NI
#define NI (256 * 91)
#endif
#ifndef NJ
#define NJ (256 * 91)
#endif

#define TILE_X 32
#define TILE_Y 8
#define HALO_X (TILE_X + 2)
#define HALO_Y (TILE_Y + 2)
#define HALO_ELEMS (HALO_X * HALO_Y)

typedef float DATA_TYPE;

static double rtclock() {
    struct timeval Tp;
    gettimeofday(&Tp, NULL);
    return Tp.tv_sec + Tp.tv_usec * 1.0e-6;
}

static void init_data(DATA_TYPE *A) {
    srand(42);
    for (long long i = 0; i < (long long)NI * NJ; i++)
        A[i] = (float)rand() / RAND_MAX;
}

/* ── plain 2D convolution (original, no shared memory) ──────── */
__global__ void conv2d_plain(DATA_TYPE *A, DATA_TYPE *B) {
    long long j = blockIdx.x * blockDim.x + threadIdx.x;
    long long i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > 0 && i < NI - 1 && j > 0 && j < NJ - 1) {
        B[i * NJ + j] =
             0.2f  * A[(i-1)*NJ + (j-1)] + 0.5f  * A[(i-1)*NJ + j] + (-0.8f) * A[(i-1)*NJ + (j+1)]
          + (-0.3f) * A[i*NJ + (j-1)]     + 0.6f  * A[i*NJ + j]     + (-0.9f) * A[i*NJ + (j+1)]
          + 0.4f  * A[(i+1)*NJ + (j-1)] + 0.7f  * A[(i+1)*NJ + j] + 0.10f * A[(i+1)*NJ + (j+1)];
    }
}

/* ── frontend 2D convolution: same tiled compute, frontend-managed accesses ── */
__global__ void conv2d_frontend(DATA_TYPE *A, DATA_TYPE *B,
                                uint32_t page_shift,
                                uint32_t *a_status,
                                uint32_t *b_status,
                                uint32_t *b_write_status,
                                unsigned long long *cas_wins) {
    __shared__ DATA_TYPE sh_tile[HALO_Y][HALO_X];
    const uvm_prefault_array_dedup_t<DATA_TYPE> a_pf{
        A, static_cast<uint64_t>(NI) * NJ, page_shift, a_status, nullptr, cas_wins};
    const uvm_prefault_array_dedup_t<DATA_TYPE> b_pf{
        B, static_cast<uint64_t>(NI) * NJ, page_shift, b_status, b_write_status, cas_wins};

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int nthreads = blockDim.x * blockDim.y;
    const int base_i = blockIdx.y * TILE_Y;
    const int base_j = blockIdx.x * TILE_X;

    for (int linear = tid; linear < HALO_ELEMS; linear += nthreads) {
        const int local_y = linear / HALO_X;
        const int local_x = linear % HALO_X;
        const int global_i = base_i + local_y - 1;
        const int global_j = base_j + local_x - 1;
        if (global_i >= 0 && global_i < NI && global_j >= 0 && global_j < NJ) {
            sh_tile[local_y][local_x] = a_pf[(uint64_t)global_i * NJ + global_j];
        } else {
            sh_tile[local_y][local_x] = 0.0f;
        }
    }
    __syncthreads();

    const int j = base_j + threadIdx.x;
    const int i = base_i + threadIdx.y;
    if (i > 0 && i < NI - 1 && j > 0 && j < NJ - 1) {
        const int ly = threadIdx.y + 1;
        const int lx = threadIdx.x + 1;
        const uint64_t out_idx = (uint64_t)i * NJ + j;
        const DATA_TYPE out_val =
             0.2f  * sh_tile[ly - 1][lx - 1] + 0.5f  * sh_tile[ly - 1][lx] + (-0.8f) * sh_tile[ly - 1][lx + 1]
          + (-0.3f) * sh_tile[ly][lx - 1]     + 0.6f  * sh_tile[ly][lx]     + (-0.9f) * sh_tile[ly][lx + 1]
          + 0.4f  * sh_tile[ly + 1][lx - 1] + 0.7f  * sh_tile[ly + 1][lx] + 0.10f * sh_tile[ly + 1][lx + 1];
        b_pf.store(out_idx, out_val);
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

    const size_t size_a = (size_t)NI * NJ * sizeof(DATA_TYPE);
    const size_t size_b = (size_t)NI * NJ * sizeof(DATA_TYPE);
    uint64_t a_pages = (size_a + 4095) / 4096;
    uint64_t b_pages = (size_b + 4095) / 4096;
    const uint32_t page_shift = 12;

    DATA_TYPE *A, *B;
    cudaMallocManaged(&A, size_a);
    cudaMallocManaged(&B, size_b);
#ifndef SKIP_CPU_VERIFY
    init_data(A);
#endif

    prefault_array_host_t<DATA_TYPE> a_pf{};
    prefault_array_host_t<DATA_TYPE> b_pf{};
    if (run_prefault) {
        prefault_array_init(&a_pf, A, static_cast<uint64_t>(NI) * NJ, page_shift, false);
        prefault_array_init(&b_pf, B, static_cast<uint64_t>(NI) * NJ, page_shift, true);
    }

    unsigned long long *d_cas_wins;
    cudaMalloc(&d_cas_wins, sizeof(unsigned long long));
    unsigned long long h_cas_wins = 0;

    dim3 block(TILE_X, TILE_Y);
    dim3 grid((NJ + TILE_X - 1) / TILE_X, (NI + TILE_Y - 1) / TILE_Y);

    pf_uvm_warmup();

    fprintf(stderr, "2DCONV bench: NI=%d NJ=%d\n", NI, NJ);
    fprintf(stderr, "Grid: %u×%u  Block: %u×%u  A_pages=%lu\n",
            grid.x, grid.y, block.x, block.y, a_pages);

    double t0 = 0.0, t1 = 0.0;
    double t_plain = 0.0;
    double t_prefault = 0.0;

    if (run_plain) {
        /* ── plain ─────────────────────────────────────────────── */
        prefault_to_cpu(A, size_a);
        prefault_to_cpu(B, size_b);
        t0 = rtclock();
        conv2d_plain<<<grid, block>>>(A, B);
        cudaDeviceSynchronize();
        t1 = rtclock();
        t_plain = t1 - t0;
        fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_plain);
        fprintf(stdout, "conv2d,plain,%.6f\n", t_plain);
    }

    if (run_prefault) {
        /* ── prefault ──────────────────────────────────────────── */
        prefault_to_cpu(A, size_a);
        prefault_to_cpu(B, size_b);
        prefault_array_reset(&a_pf);
        prefault_array_reset(&b_pf);
        cudaMemset(d_cas_wins, 0, sizeof(unsigned long long));
        t0 = rtclock();
        conv2d_frontend<<<grid, block>>>(
            A, B, page_shift, a_pf.page_status, b_pf.page_status, b_pf.write_status, d_cas_wins);
        cudaDeviceSynchronize();
        t1 = rtclock();
        t_prefault = t1 - t0;
        cudaMemcpy(&h_cas_wins, d_cas_wins, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        fprintf(stdout, "conv2d,prefault,%.6f\n", t_prefault);

        fprintf(stderr, "\n  cas_wins=%llu  total_pages=%lu\n", h_cas_wins, a_pages + b_pages);
    }
    if (run_plain && run_prefault) {
        fprintf(stderr, "  speedup vs plain: %.2fx\n", t_plain / t_prefault);
    }

    cudaFree(d_cas_wins);
    if (run_prefault) {
        prefault_array_destroy(&a_pf);
        prefault_array_destroy(&b_pf);
    }
    cudaFree(A); cudaFree(B);
    return 0;
}
