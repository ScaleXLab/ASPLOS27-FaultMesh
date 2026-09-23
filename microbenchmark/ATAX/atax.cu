/**
 * atax.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

void init_array(DATA_TYPE *x_gpu, DATA_TYPE *A_gpu)
{
	long long int i, j;

	for (i = 0; i < NX; i++)
	{
		x_gpu[i] = i * M_PI;
		for (j = 0; j < NY; j++)
		{
			A_gpu[i*NY + j] = ((DATA_TYPE) i*(j)) / NX;
		}
	}
}


void init_array(DATA_TYPE *x, DATA_TYPE *A, DATA_TYPE *x_gpu, DATA_TYPE *A_gpu)
{
	long long int i, j;

	for (i = 0; i < NX; i++)
	{
		x[i] = i * M_PI;
		x_gpu[i] = i * M_PI;
		for (j = 0; j < NY; j++)
		{
			A[i*NY + j] = ((DATA_TYPE) i*(j)) / NX;
			A_gpu[i*NY + j] = ((DATA_TYPE) i*(j)) / NX;
		}
	}
}

void printResults(DATA_TYPE *z)
{
	long long int i;

	for (i=0; i<NY; i++)
	{
        	printf("%lf\n", z[i]);
	}
}


void compareResults(DATA_TYPE *z, DATA_TYPE *z_outputFromGpu)
{
	long long int i, fail;

	fail = 0;

	for (i=0; i<NY; i++)
	{
		if (percentDiff(z[i], z_outputFromGpu[i]) > PERCENT_DIFF_ERROR_THRESHOLD)
		{
			fail++;		           
			printf("%f, %f\n", z[i], z_outputFromGpu[i]);
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


__global__ void atax_kernel1(DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *tmp)
{
	long long int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < NX)
	{
		long long int j;
		tmp[i] = 0;
		for(j=0; j < NY; j++)
		{
			tmp[i] += A[i * NY + j] * x[j];
		}
	}
}

__global__ void atax_kernel2(DATA_TYPE *A, DATA_TYPE *y, DATA_TYPE *tmp)
{
	long long int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (j < NY)
	{
		y[j] = 0;
		long long int i;
		for(i=0; i < NX; i++)
		{
			y[j] += A[i * NY + j] * tmp[i];
		}
	}
}

__global__ void atax_kernel1_frontend(DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *tmp,
                                      uint32_t page_shift,
                                      uint32_t* a_status,
                                      uint32_t* x_status,
                                      uint32_t* tmp_status,
                                      uint32_t* tmp_write_status,
                                      unsigned long long* cas_wins)
{
	long long int i = blockIdx.x * blockDim.x + threadIdx.x;
	const uvm_prefault_array_dedup_t<DATA_TYPE> a_pf{A, static_cast<uint64_t>(NX) * NY, page_shift, a_status, nullptr, cas_wins};
	const uvm_prefault_array_dedup_t<DATA_TYPE> x_pf{x, static_cast<uint64_t>(NY), page_shift, x_status, nullptr, cas_wins};
	const uvm_prefault_array_dedup_t<DATA_TYPE> tmp_pf{tmp, static_cast<uint64_t>(NX), page_shift, tmp_status, tmp_write_status, cas_wins};

	if (i < NX)
	{
		DATA_TYPE sum = 0;
		for (long long int j = 0; j < NY; j++)
		{
			sum += a_pf[(uint64_t)i * NY + j] * x_pf[j];
		}
		tmp_pf.store(i, sum);
	}
}

__global__ void atax_kernel2_frontend(DATA_TYPE *A, DATA_TYPE *y, DATA_TYPE *tmp,
                                      uint32_t page_shift,
                                      uint32_t* a_status,
                                      uint32_t* y_status,
                                      uint32_t* y_write_status,
                                      uint32_t* tmp_status,
                                      unsigned long long* cas_wins)
{
	long long int j = blockIdx.x * blockDim.x + threadIdx.x;
	const uvm_prefault_array_dedup_t<DATA_TYPE> a_pf{A, static_cast<uint64_t>(NX) * NY, page_shift, a_status, nullptr, cas_wins};
	const uvm_prefault_array_dedup_t<DATA_TYPE> y_pf{y, static_cast<uint64_t>(NY), page_shift, y_status, y_write_status, cas_wins};
	const uvm_prefault_array_dedup_t<DATA_TYPE> tmp_pf{tmp, static_cast<uint64_t>(NX), page_shift, tmp_status, nullptr, cas_wins};

	if (j < NY)
	{
		DATA_TYPE sum = 0;
		for (long long int i = 0; i < NX; i++)
		{
			sum += a_pf[(uint64_t)i * NY + j] * tmp_pf[i];
		}
		y_pf.store(j, sum);
	}
}


void atax_cpu(DATA_TYPE* A, DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* tmp)
{
	long long int i,j;
	
	for (i= 0; i < NY; i++)
	{
    	y[i] = 0;
	}
  
	for (i = 0; i < NX; i++)
 	{
      	tmp[i] = 0;

      	for (j = 0; j < NY; j++)
		{
			tmp[i] = tmp[i] + A[i*NY + j] * x[j];
		}
		
      	for (j = 0; j < NY; j++)
		{
			y[j] = y[j] + A[i*NY + j] * tmp[i];
		}
    }
}


void ataxGpu(DATA_TYPE* A_gpu, DATA_TYPE* x_gpu, DATA_TYPE* y_gpu, DATA_TYPE* tmp_gpu)
{
	double t_start, t_end;
	const char* bv = getenv("BENCH_VARIANT");
	const bool use_frontend = (bv && strcmp(bv, "frontend") == 0);
	const uint32_t page_shift = 12;
	
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)(ceil( ((float)NX) / ((float)block.x) )), 1);
	dim3 grid2((size_t)(ceil( ((float)NY) / ((float)block.x) )), 1);

	prefault_array_host_t<DATA_TYPE> a_pf{};
	prefault_array_host_t<DATA_TYPE> x_pf{};
	prefault_array_host_t<DATA_TYPE> y_pf{};
	prefault_array_host_t<DATA_TYPE> tmp_pf{};
	unsigned long long* d_cas_wins = nullptr;
	unsigned long long h_cas_wins = 0;

	if (use_frontend) {
		prefault_array_init(&a_pf, A_gpu, static_cast<uint64_t>(NX) * NY, page_shift, false);
		prefault_array_init(&x_pf, x_gpu, static_cast<uint64_t>(NY), page_shift, false);
		prefault_array_init(&y_pf, y_gpu, static_cast<uint64_t>(NY), page_shift, true);
		prefault_array_init(&tmp_pf, tmp_gpu, static_cast<uint64_t>(NX), page_shift, true);
		cudaMalloc(&d_cas_wins, sizeof(unsigned long long));
		cudaMemset(d_cas_wins, 0, sizeof(unsigned long long));
	}

	prefault_to_cpu(A_gpu, (size_t)NX * NY * sizeof(DATA_TYPE));
	prefault_to_cpu(x_gpu, (size_t)NY * sizeof(DATA_TYPE));
	prefault_to_cpu(y_gpu, (size_t)NY * sizeof(DATA_TYPE));
	prefault_to_cpu(tmp_gpu, (size_t)NX * sizeof(DATA_TYPE));
	cudaMemset(y_gpu, 0, (size_t)NY * sizeof(DATA_TYPE));
	cudaMemset(tmp_gpu, 0, (size_t)NX * sizeof(DATA_TYPE));

	t_start = rtclock();
	if (!use_frontend) {
		atax_kernel1<<< grid1, block >>>(A_gpu,x_gpu,tmp_gpu);
		cudaDeviceSynchronize();
		atax_kernel2<<< grid2, block >>>(A_gpu,y_gpu,tmp_gpu);
	} else {
		prefault_array_reset(&a_pf);
		prefault_array_reset(&x_pf);
		prefault_array_reset(&y_pf);
		prefault_array_reset(&tmp_pf);
		cudaMemset(d_cas_wins, 0, sizeof(unsigned long long));
		atax_kernel1_frontend<<< grid1, block >>>(
			A_gpu, x_gpu, tmp_gpu, page_shift,
			a_pf.page_status, x_pf.page_status,
			tmp_pf.page_status, tmp_pf.write_status,
			d_cas_wins);
		cudaDeviceSynchronize();
		atax_kernel2_frontend<<< grid2, block >>>(
			A_gpu, y_gpu, tmp_gpu, page_shift,
			a_pf.page_status, y_pf.page_status, y_pf.write_status,
			tmp_pf.page_status, d_cas_wins);
	}
	cudaDeviceSynchronize();
	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
	fprintf(stdout, "atax,%s,%0.6lf\n", use_frontend ? "frontend" : "baseline", t_end - t_start);
	if (use_frontend) {
		cudaMemcpy(&h_cas_wins, d_cas_wins, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
		fprintf(stderr, "  frontend cas_wins=%llu\n", h_cas_wins);
	}

	if (d_cas_wins) cudaFree(d_cas_wins);
	if (use_frontend) {
		prefault_array_destroy(&a_pf);
		prefault_array_destroy(&x_pf);
		prefault_array_destroy(&y_pf);
		prefault_array_destroy(&tmp_pf);
	}
}


int main(int argc, char** argv)
{
	DATA_TYPE *A_gpu;
	DATA_TYPE *x_gpu;
	DATA_TYPE *y_gpu;
	DATA_TYPE *tmp_gpu;

#ifdef SKIP_CPU_VERIFY
	cudaMallocManaged(&A_gpu, (size_t)sizeof(DATA_TYPE) * NX * NY);
	cudaMallocManaged(&x_gpu, (size_t)sizeof(DATA_TYPE) * NY);
	cudaMallocManaged(&y_gpu, (size_t)sizeof(DATA_TYPE) * NY);
	cudaMallocManaged(&tmp_gpu, (size_t)sizeof(DATA_TYPE) * NX);
	GPU_argv_init();
	pf_uvm_warmup();
	ataxGpu(A_gpu, x_gpu, y_gpu, tmp_gpu);
	cudaFree(A_gpu);
	cudaFree(x_gpu);
	cudaFree(y_gpu);
	cudaFree(tmp_gpu);
	return 0;
#else
	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* x;
	DATA_TYPE* y;
	DATA_TYPE* tmp;

	A = (DATA_TYPE*)malloc((size_t)NX*NY*sizeof(DATA_TYPE));
	x = (DATA_TYPE*)malloc((size_t)NY*sizeof(DATA_TYPE));
	y = (DATA_TYPE*)malloc((size_t)NY*sizeof(DATA_TYPE));
	tmp = (DATA_TYPE*)malloc((size_t)NX*sizeof(DATA_TYPE));

	cudaMallocManaged(&A_gpu, (size_t)sizeof(DATA_TYPE) * NX * NY);
	cudaMallocManaged(&x_gpu, (size_t)sizeof(DATA_TYPE) * NY);
	cudaMallocManaged(&y_gpu, (size_t)sizeof(DATA_TYPE) * NY);
	cudaMallocManaged(&tmp_gpu, (size_t)sizeof(DATA_TYPE) * NX);

	init_array(x, A, x_gpu, A_gpu);

	GPU_argv_init();
	pf_uvm_warmup();
	ataxGpu(A_gpu, x_gpu, y_gpu, tmp_gpu);
	atax_cpu(A,x,y,tmp);
	compareResults(tmp, tmp_gpu);

	free(A);
	free(x);
	free(y);
	free(tmp);

	cudaFree(A_gpu);
	cudaFree(x_gpu);
	cudaFree(y_gpu);
	cudaFree(tmp_gpu);
    
  	return 0;
#endif
}

