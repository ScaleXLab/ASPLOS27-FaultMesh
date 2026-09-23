/**
 * mvt.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cstdint>
#include <cstring>
#include <cuda.h>

#include "../polybenchUtilFuncts.h"
#include "../../frontLib/frontend_prefault_common.cuh"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Problem size */
#ifndef PROBLEM_N
#define PROBLEM_N (256 * 128)
#endif
#define N PROBLEM_N

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;


void init_array(DATA_TYPE* A, DATA_TYPE* x1, DATA_TYPE* x2, DATA_TYPE* y1, DATA_TYPE* y2,
		DATA_TYPE* a_gpu, DATA_TYPE* x1_gpu, DATA_TYPE* x2_gpu, DATA_TYPE* y_1_gpu, DATA_TYPE* y_2_gpu)
{
	long long int i, j;

	for (i = 0; i < N; i++)
	{
		x1[i] = ((DATA_TYPE) i) / N;
		x2[i] = ((DATA_TYPE) i + 1) / N;
		y1[i] = ((DATA_TYPE) i + 3) / N;
		y2[i] = ((DATA_TYPE) i + 4) / N;
		x1_gpu[i] = ((DATA_TYPE) i) / N;
		x2_gpu[i] = ((DATA_TYPE) i + 1) / N;
		y_1_gpu[i] = ((DATA_TYPE) i + 3) / N;
		y_2_gpu[i] = ((DATA_TYPE) i + 4) / N;
		for (j = 0; j < N; j++)
		{
			A[i*N + j] = ((DATA_TYPE) i*j) / N;
			a_gpu[i*N + j] = ((DATA_TYPE) i*j) / N;
		}
	}
}



void runMvt(DATA_TYPE* a, DATA_TYPE* x1, DATA_TYPE* x2, DATA_TYPE* y1, DATA_TYPE* y2)
{
	long long int i, j;
	
	for (i=0; i<N; i++) 
	{
		for (j=0; j<N; j++) 
		{
       			x1[i] = x1[i] + a[i*N + j] * y1[j];
        	}
    	}
	
	for (i=0; i<N; i++) 
	{
		for (j=0; j<N; j++) 
		{
 		       	x2[i] = x2[i] + a[j*N + i] * y2[j];
      		}
    	}
}


void compareResults(DATA_TYPE* x1, DATA_TYPE* x1_outputFromGpu, DATA_TYPE* x2, DATA_TYPE* x2_outputFromGpu)
{
	long long int i, fail;
	fail = 0;
	
	for (i=0; i<N; i++) 
	{
		if (percentDiff(x1[i], x1_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}

		if (percentDiff(x2[i], x2_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %lld\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
}


__global__ void mvt_kernel1(DATA_TYPE *a, DATA_TYPE *x1, DATA_TYPE *y_1)
{
	long long int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		long long int j;
		for(j=0; j < N; j++)
		{
			x1[i] += a[i * N + j] * y_1[j];
		}
	}
}


__global__ void mvt_kernel2(DATA_TYPE *a, DATA_TYPE *x2, DATA_TYPE *y_2)
{
	long long int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		long long int j;
		for(j=0; j < N; j++)
		{
			x2[i] += a[j * N + i] * y_2[j];	
		}
	}
}

__global__ void mvt_kernel1_frontend(DATA_TYPE *a, DATA_TYPE *x1, DATA_TYPE *y_1,
                                     uint32_t page_shift,
                                     uint32_t* a_status,
                                     uint32_t* x1_status,
                                     uint32_t* x1_write_status,
                                     uint32_t* y1_status,
                                     unsigned long long* cas_wins)
{
	long long int i = blockIdx.x * blockDim.x + threadIdx.x;
	const uvm_prefault_array_dedup_t<DATA_TYPE> a_pf{a, static_cast<uint64_t>(N) * N, page_shift, a_status, nullptr, cas_wins};
	const uvm_prefault_array_dedup_t<DATA_TYPE> x1_pf{x1, static_cast<uint64_t>(N), page_shift, x1_status, x1_write_status, cas_wins};
	const uvm_prefault_array_dedup_t<DATA_TYPE> y1_pf{y_1, static_cast<uint64_t>(N), page_shift, y1_status, nullptr, cas_wins};

	if (i < N)
	{
		DATA_TYPE sum = x1_pf[i];
		for (long long int j = 0; j < N; j++)
		{
			sum += a_pf[(uint64_t)i * N + j] * y1_pf[j];
		}
		x1_pf.store(i, sum);
	}
}

__global__ void mvt_kernel2_frontend(DATA_TYPE *a, DATA_TYPE *x2, DATA_TYPE *y_2,
                                     uint32_t page_shift,
                                     uint32_t* a_status,
                                     uint32_t* x2_status,
                                     uint32_t* x2_write_status,
                                     uint32_t* y2_status,
                                     unsigned long long* cas_wins)
{
	long long int i = blockIdx.x * blockDim.x + threadIdx.x;
	const uvm_prefault_array_dedup_t<DATA_TYPE> a_pf{a, static_cast<uint64_t>(N) * N, page_shift, a_status, nullptr, cas_wins};
	const uvm_prefault_array_dedup_t<DATA_TYPE> x2_pf{x2, static_cast<uint64_t>(N), page_shift, x2_status, x2_write_status, cas_wins};
	const uvm_prefault_array_dedup_t<DATA_TYPE> y2_pf{y_2, static_cast<uint64_t>(N), page_shift, y2_status, nullptr, cas_wins};

	if (i < N)
	{
		DATA_TYPE sum = x2_pf[i];
		for (long long int j = 0; j < N; j++)
		{
			sum += a_pf[(uint64_t)j * N + i] * y2_pf[j];
		}
		x2_pf.store(i, sum);
	}
}

void mvtCuda(DATA_TYPE* a_gpu, DATA_TYPE* x1_gpu, DATA_TYPE* x2_gpu, DATA_TYPE* y_1_gpu, DATA_TYPE* y_2_gpu)
{
	double t_start, t_end;
	const char* bv = getenv("BENCH_VARIANT");
	const bool use_frontend = (bv && strcmp(bv, "frontend") == 0);
	const uint32_t page_shift = 12;
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)ceil((float)N/ ((float)DIM_THREAD_BLOCK_X)), 1);

	prefault_array_host_t<DATA_TYPE> a_pf{};
	prefault_array_host_t<DATA_TYPE> x1_pf{};
	prefault_array_host_t<DATA_TYPE> x2_pf{};
	prefault_array_host_t<DATA_TYPE> y1_pf{};
	prefault_array_host_t<DATA_TYPE> y2_pf{};
	unsigned long long* d_cas_wins = nullptr;
	unsigned long long h_cas_wins = 0;

	if (use_frontend) {
		prefault_array_init(&a_pf, a_gpu, static_cast<uint64_t>(N) * N, page_shift, false);
		prefault_array_init(&x1_pf, x1_gpu, static_cast<uint64_t>(N), page_shift, true);
		prefault_array_init(&x2_pf, x2_gpu, static_cast<uint64_t>(N), page_shift, true);
		prefault_array_init(&y1_pf, y_1_gpu, static_cast<uint64_t>(N), page_shift, false);
		prefault_array_init(&y2_pf, y_2_gpu, static_cast<uint64_t>(N), page_shift, false);
		cudaMalloc(&d_cas_wins, sizeof(unsigned long long));
		cudaMemset(d_cas_wins, 0, sizeof(unsigned long long));
	}

	prefault_to_cpu(a_gpu, (size_t)N * N * sizeof(DATA_TYPE));
	prefault_to_cpu(x1_gpu, (size_t)N * sizeof(DATA_TYPE));
	prefault_to_cpu(x2_gpu, (size_t)N * sizeof(DATA_TYPE));
	prefault_to_cpu(y_1_gpu, (size_t)N * sizeof(DATA_TYPE));
	prefault_to_cpu(y_2_gpu, (size_t)N * sizeof(DATA_TYPE));

	t_start = rtclock();
	if (!use_frontend) {
		mvt_kernel1<<<grid,block>>>(a_gpu,x1_gpu,y_1_gpu);
		mvt_kernel2<<<grid,block>>>(a_gpu,x2_gpu,y_2_gpu);
	} else {
		prefault_array_reset(&a_pf);
		prefault_array_reset(&x1_pf);
		prefault_array_reset(&x2_pf);
		prefault_array_reset(&y1_pf);
		prefault_array_reset(&y2_pf);
		cudaMemset(d_cas_wins, 0, sizeof(unsigned long long));

		mvt_kernel1_frontend<<<grid,block>>>(
			a_gpu, x1_gpu, y_1_gpu, page_shift,
			a_pf.page_status,
			x1_pf.page_status, x1_pf.write_status,
			y1_pf.page_status, d_cas_wins);
		mvt_kernel2_frontend<<<grid,block>>>(
			a_gpu, x2_gpu, y_2_gpu, page_shift,
			a_pf.page_status,
			x2_pf.page_status, x2_pf.write_status,
			y2_pf.page_status, d_cas_wins);
	}
	cudaDeviceSynchronize();
	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
	fprintf(stdout, "mvt,%s,%0.6lf\n", use_frontend ? "frontend" : "baseline", t_end - t_start);
	if (use_frontend) {
		cudaMemcpy(&h_cas_wins, d_cas_wins, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
		fprintf(stderr, "  frontend cas_wins=%llu\n", h_cas_wins);
	}

	if (d_cas_wins) cudaFree(d_cas_wins);
	if (use_frontend) {
		prefault_array_destroy(&a_pf);
		prefault_array_destroy(&x1_pf);
		prefault_array_destroy(&x2_pf);
		prefault_array_destroy(&y1_pf);
		prefault_array_destroy(&y2_pf);
	}
	//fprintf(stdout, "kernel1 time: %0.6lfs\n", t_middle - t_start);
}


int main()
{
#ifdef SKIP_CPU_VERIFY
	DATA_TYPE* a_gpu;
	DATA_TYPE* x1_gpu;
	DATA_TYPE* x2_gpu;
	DATA_TYPE* y_1_gpu;
	DATA_TYPE* y_2_gpu;
	cudaMallocManaged(&a_gpu, (size_t)sizeof(DATA_TYPE) * N * N);
	cudaMallocManaged(&x1_gpu, (size_t)sizeof(DATA_TYPE) * N);
	cudaMallocManaged(&x2_gpu, (size_t)sizeof(DATA_TYPE) * N);
	cudaMallocManaged(&y_1_gpu, (size_t)sizeof(DATA_TYPE) * N);
	cudaMallocManaged(&y_2_gpu, (size_t)sizeof(DATA_TYPE) * N);
	GPU_argv_init();
	pf_uvm_warmup();
	mvtCuda(a_gpu, x1_gpu, x2_gpu, y_1_gpu, y_2_gpu);
	cudaFree(a_gpu);
	cudaFree(x1_gpu);
	cudaFree(x2_gpu);
	cudaFree(y_1_gpu);
	cudaFree(y_2_gpu);
	return 0;
#else
	double t_start, t_end;

	DATA_TYPE* a;
	DATA_TYPE* x1;
	DATA_TYPE* x2;
	DATA_TYPE* y_1;
	DATA_TYPE* y_2;
	DATA_TYPE* a_gpu;
	DATA_TYPE* x1_gpu;
	DATA_TYPE* x2_gpu;
	DATA_TYPE* y_1_gpu;
	DATA_TYPE* y_2_gpu;
	a = (DATA_TYPE*)malloc((size_t)N*N*sizeof(DATA_TYPE));
	x1 = (DATA_TYPE*)malloc((size_t)N*sizeof(DATA_TYPE));
	x2 = (DATA_TYPE*)malloc((size_t)N*sizeof(DATA_TYPE));
	y_1 = (DATA_TYPE*)malloc((size_t)N*sizeof(DATA_TYPE));
	y_2 = (DATA_TYPE*)malloc((size_t)N*sizeof(DATA_TYPE));
	cudaMallocManaged(&a_gpu, sizeof(DATA_TYPE) * N * N);
	cudaMallocManaged(&x1_gpu, sizeof(DATA_TYPE) * N);
	cudaMallocManaged(&x2_gpu, sizeof(DATA_TYPE) * N);
	cudaMallocManaged(&y_1_gpu, sizeof(DATA_TYPE) * N);
	cudaMallocManaged(&y_2_gpu, sizeof(DATA_TYPE) * N);
	init_array(a, x1, x2, y_1, y_2, a_gpu, x1_gpu, x2_gpu, y_1_gpu, y_2_gpu);
	
 
	GPU_argv_init();
	pf_uvm_warmup();

	mvtCuda(a_gpu, x1_gpu, x2_gpu, y_1_gpu, y_2_gpu);
	runMvt(a, x1, x2, y_1, y_2);
	compareResults(x1, x1_gpu, x2, x2_gpu);
	
	free(a);
	free(x1);
	free(x2);
	free(y_1);
	free(y_2);

	cudaFree(a_gpu);
	cudaFree(x1_gpu);
	cudaFree(x2_gpu);
	cudaFree(y_1_gpu);
	cudaFree(y_2_gpu);
  	return 0;
#endif
}

