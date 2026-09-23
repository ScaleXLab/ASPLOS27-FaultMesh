/**
 * bicg.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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
#include <sys/time.h>
#include <cstdint>
#include <cstring>
#include <cuda.h>

#include "../polybenchUtilFuncts.h"
#include "../../frontLib/frontend_prefault_common.cuh"

//Error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

#define GPU_DEVICE 0

/* Problem size. */
#ifndef NX
#define NX (256 * 128)
#endif
#ifndef NY
#define NY (256 * 128)
#endif

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

#ifndef M_PI
#define M_PI 3.14159
#endif

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;



void init_array(DATA_TYPE *A, DATA_TYPE *p, DATA_TYPE *r, DATA_TYPE *A_gpu, DATA_TYPE *p_gpu, DATA_TYPE *r_gpu)
{
	long long int i, j;

  	for (i = 0; i < NX; i++)
	{
			r[i] = i * M_PI;
			r_gpu[i] = i * M_PI;

    		for (j = 0; j < NY; j++)
		{
				  A[i*NY + j] = ((DATA_TYPE) i*j) / NX;
				  A_gpu[i*NY + j] = ((DATA_TYPE) i*j) / NX;
		}
 	}
	
	for (i = 0; i < NY; i++)
	{
			p[i] = i * M_PI;
			p_gpu[i] = i * M_PI;
	}
}


void compareResults(DATA_TYPE* s, DATA_TYPE* s_outputFromGpu, DATA_TYPE* q, DATA_TYPE* q_outputFromGpu)
{
	long long int i,fail;
	fail = 0;

	// Compare s with s_cuda
	for (i=0; i<NX; i++)
	{
		if (percentDiff(q[i], q_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}
	}

	for (i=0; i<NY; i++)
	{
		if (percentDiff(s[i], s_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;
		}		
	}
	
	// print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
	cudaSetDevice( GPU_DEVICE );
}


//Distributed (split) from initial loop and permuted into reverse order to allow parallelism...
__global__ void bicg_kernel1(DATA_TYPE *A, DATA_TYPE *r, DATA_TYPE *s)
{
	long long int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (j < NY)
	{
		s[j] = 0.0f;

		int i;
		for(i = 0; i < NX; i++)
		{
			s[j] += A[(long long)i * NY + j] * r[i];
		}
	}	
}


//Distributed (split) from initial loop to allow parallelism
__global__ void bicg_kernel2(DATA_TYPE *A, DATA_TYPE *p, DATA_TYPE *q)
{
	long long int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < NX)
	{
		q[i] = 0.0f;

		int j;
		for(j=0; j < NY; j++)
		{
			q[i] += A[i * (long long)NY + j] * p[j];
		}
	}
}

__global__ void bicg_kernel1_frontend(DATA_TYPE *A, DATA_TYPE *r, DATA_TYPE *s,
                                      uint32_t page_shift,
                                      uint32_t* a_status,
                                      uint32_t* r_status,
                                      uint32_t* s_status,
                                      uint32_t* s_write_status,
                                      unsigned long long* cas_wins)
{
	long long int j = blockIdx.x * blockDim.x + threadIdx.x;
	const uvm_prefault_array_dedup_t<DATA_TYPE> a_pf{A, static_cast<uint64_t>(NX) * NY, page_shift, a_status, nullptr, cas_wins};
	const uvm_prefault_array_dedup_t<DATA_TYPE> r_pf{r, static_cast<uint64_t>(NX), page_shift, r_status, nullptr, cas_wins};
	const uvm_prefault_array_dedup_t<DATA_TYPE> s_pf{s, static_cast<uint64_t>(NY), page_shift, s_status, s_write_status, cas_wins};

	if (j < NY)
	{
		DATA_TYPE sum = 0.0f;
		for (int i = 0; i < NX; i++)
		{
			sum += a_pf[(uint64_t)i * NY + j] * r_pf[i];
		}
		s_pf.store(j, sum);
	}
}

__global__ void bicg_kernel2_frontend(DATA_TYPE *A, DATA_TYPE *p, DATA_TYPE *q,
                                      uint32_t page_shift,
                                      uint32_t* a_status,
                                      uint32_t* p_status,
                                      uint32_t* q_status,
                                      uint32_t* q_write_status,
                                      unsigned long long* cas_wins)
{
	long long int i = blockIdx.x * blockDim.x + threadIdx.x;
	const uvm_prefault_array_dedup_t<DATA_TYPE> a_pf{A, static_cast<uint64_t>(NX) * NY, page_shift, a_status, nullptr, cas_wins};
	const uvm_prefault_array_dedup_t<DATA_TYPE> p_pf{p, static_cast<uint64_t>(NY), page_shift, p_status, nullptr, cas_wins};
	const uvm_prefault_array_dedup_t<DATA_TYPE> q_pf{q, static_cast<uint64_t>(NX), page_shift, q_status, q_write_status, cas_wins};

	if (i < NX)
	{
		DATA_TYPE sum = 0.0f;
		for (int j = 0; j < NY; j++)
		{
			sum += a_pf[(uint64_t)i * NY + j] * p_pf[j];
		}
		q_pf.store(i, sum);
	}
}


void bicg_cpu(DATA_TYPE* A, DATA_TYPE* r, DATA_TYPE* s, DATA_TYPE* p, DATA_TYPE* q)
{
	long long int i,j;
	
  	for (i = 0; i < NY; i++)
	{
		s[i] = 0.0;
	}

    for (i = 0; i < NX; i++)
    {
		q[i] = 0.0;
		for (j = 0; j < NY; j++)
	  	{
	    		s[j] = s[j] + r[i] * A[i*NY + j];
	    		q[i] = q[i] + A[i*NY + j] * p[j];
	  	}
	}
}


void bicgCuda(DATA_TYPE* A_gpu, DATA_TYPE* r_gpu, DATA_TYPE* s_gpu, DATA_TYPE* p_gpu, DATA_TYPE* q_gpu)
{
	double t_start, t_end;
	const char* bv = getenv("BENCH_VARIANT");
	const bool use_frontend = (bv && strcmp(bv, "frontend") == 0);
	const uint32_t page_shift = 12;

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)(ceil( ((float)NY) / ((float)block.x) )), 1);
	dim3 grid2((size_t)(ceil( ((float)NX) / ((float)block.x) )), 1);

	prefault_array_host_t<DATA_TYPE> a_pf{};
	prefault_array_host_t<DATA_TYPE> r_pf{};
	prefault_array_host_t<DATA_TYPE> s_pf{};
	prefault_array_host_t<DATA_TYPE> p_pf{};
	prefault_array_host_t<DATA_TYPE> q_pf{};
	unsigned long long* d_cas_wins = nullptr;
	unsigned long long h_cas_wins = 0;

	if (use_frontend) {
		prefault_array_init(&a_pf, A_gpu, static_cast<uint64_t>(NX) * NY, page_shift, false);
		prefault_array_init(&r_pf, r_gpu, static_cast<uint64_t>(NX), page_shift, false);
		prefault_array_init(&s_pf, s_gpu, static_cast<uint64_t>(NY), page_shift, true);
		prefault_array_init(&p_pf, p_gpu, static_cast<uint64_t>(NY), page_shift, false);
		prefault_array_init(&q_pf, q_gpu, static_cast<uint64_t>(NX), page_shift, true);
		cudaMalloc(&d_cas_wins, sizeof(unsigned long long));
		cudaMemset(d_cas_wins, 0, sizeof(unsigned long long));
	}

	prefault_to_cpu(A_gpu, (size_t)NX * NY * sizeof(DATA_TYPE));
	prefault_to_cpu(r_gpu, (size_t)NX * sizeof(DATA_TYPE));
	prefault_to_cpu(s_gpu, (size_t)NY * sizeof(DATA_TYPE));
	prefault_to_cpu(p_gpu, (size_t)NY * sizeof(DATA_TYPE));
	prefault_to_cpu(q_gpu, (size_t)NX * sizeof(DATA_TYPE));
	cudaMemset(s_gpu, 0, (size_t)NY * sizeof(DATA_TYPE));
	cudaMemset(q_gpu, 0, (size_t)NX * sizeof(DATA_TYPE));

	t_start = rtclock();
	if (!use_frontend) {
		bicg_kernel1<<< grid1, block >>>(A_gpu, r_gpu, s_gpu);
		cudaDeviceSynchronize();
		bicg_kernel2<<< grid2, block >>>(A_gpu, p_gpu, q_gpu);
	} else {
		prefault_array_reset(&a_pf);
		prefault_array_reset(&r_pf);
		prefault_array_reset(&s_pf);
		prefault_array_reset(&p_pf);
		prefault_array_reset(&q_pf);
		cudaMemset(d_cas_wins, 0, sizeof(unsigned long long));

		bicg_kernel1_frontend<<< grid1, block >>>(
			A_gpu, r_gpu, s_gpu, page_shift,
			a_pf.page_status, r_pf.page_status,
			s_pf.page_status, s_pf.write_status, d_cas_wins);
		cudaDeviceSynchronize();
		bicg_kernel2_frontend<<< grid2, block >>>(
			A_gpu, p_gpu, q_gpu, page_shift,
			a_pf.page_status, p_pf.page_status,
			q_pf.page_status, q_pf.write_status, d_cas_wins);
	}
	cudaDeviceSynchronize();
	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
	fprintf(stdout, "bicg,%s,%0.6lf\n", use_frontend ? "frontend" : "baseline", t_end - t_start);
	if (use_frontend) {
		cudaMemcpy(&h_cas_wins, d_cas_wins, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
		fprintf(stderr, "  frontend cas_wins=%llu\n", h_cas_wins);
	}

	if (d_cas_wins) cudaFree(d_cas_wins);
	if (use_frontend) {
		prefault_array_destroy(&a_pf);
		prefault_array_destroy(&r_pf);
		prefault_array_destroy(&s_pf);
		prefault_array_destroy(&p_pf);
		prefault_array_destroy(&q_pf);
	}
	
}

int main(int argc, char** argv)
{
	DATA_TYPE *A_gpu;
	DATA_TYPE *q_gpu;
	DATA_TYPE *p_gpu;
	DATA_TYPE *r_gpu;
	DATA_TYPE *s_gpu;

#ifdef SKIP_CPU_VERIFY
	cudaMallocManaged(&A_gpu, (size_t)sizeof(DATA_TYPE) * NX * NY);
	cudaMallocManaged(&r_gpu, (size_t)sizeof(DATA_TYPE) * NX);
	cudaMallocManaged(&s_gpu, (size_t)sizeof(DATA_TYPE) * NY);
	cudaMallocManaged(&p_gpu, (size_t)sizeof(DATA_TYPE) * NY);
	cudaMallocManaged(&q_gpu, (size_t)sizeof(DATA_TYPE) * NX);
	GPU_argv_init();
	pf_uvm_warmup();
	bicgCuda(A_gpu, r_gpu, s_gpu, p_gpu, q_gpu);
	cudaFree(A_gpu);
	cudaFree(r_gpu);
	cudaFree(s_gpu);
	cudaFree(p_gpu);
	cudaFree(q_gpu);
  	return 0;
#else
	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* r;
	DATA_TYPE* s;
	DATA_TYPE* p;
	DATA_TYPE* q;

	A = (DATA_TYPE*)malloc((size_t)NX*NY*sizeof(DATA_TYPE));
	r = (DATA_TYPE*)malloc((size_t)NX*sizeof(DATA_TYPE));
	s = (DATA_TYPE*)malloc((size_t)NY*sizeof(DATA_TYPE));
	p = (DATA_TYPE*)malloc((size_t)NY*sizeof(DATA_TYPE));
	q = (DATA_TYPE*)malloc((size_t)NX*sizeof(DATA_TYPE));

	cudaMallocManaged(&A_gpu, sizeof(DATA_TYPE) * NX * NY);
	cudaMallocManaged(&r_gpu, sizeof(DATA_TYPE) * NX);
	cudaMallocManaged(&s_gpu, sizeof(DATA_TYPE) * NY);
	cudaMallocManaged(&p_gpu, sizeof(DATA_TYPE) * NY);
	cudaMallocManaged(&q_gpu, sizeof(DATA_TYPE) * NX);

	init_array(A, p, r, A_gpu, p_gpu, r_gpu);
	
    free(A);
	free(r);
	free(s);
	free(p);
	free(q);

	GPU_argv_init();
	pf_uvm_warmup();

	bicgCuda(A_gpu, r_gpu, s_gpu, p_gpu, q_gpu);

	
	cudaFree(A_gpu);
	cudaFree(r_gpu);
	cudaFree(s_gpu);
	cudaFree(p_gpu);
	cudaFree(q_gpu);
  	return 0;
#endif
}

