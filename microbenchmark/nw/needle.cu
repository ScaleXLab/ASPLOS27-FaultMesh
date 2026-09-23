#define LIMIT -999
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "needle.h"
#include <cuda.h>
#include <sys/time.h>
#include <cstdint>
#include <cstring>

#include "../../frontLib/frontend_prefault_common.cuh"

// includes, kernels
#include "needle_kernel.cu"

/* UVM warmup: pre-starts kthread workers before the timed section */
__global__ static void _uvm_warmup_kernel(float *d, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] += 1.0f;
}
static void uvm_warmup(void)
{
    const int N = 1 << 18;
    float *buf = NULL;
    cudaMallocManaged(&buf, (size_t)N * sizeof(float));
    for (int i = 0; i < N; i++) buf[i] = 0.0f;
    _uvm_warmup_kernel<<<(N + 255) / 256, 256>>>(buf, N);
    cudaDeviceSynchronize();
    cudaFree(buf);
}

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);


int blosum62[24][24] = {
{ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4},
{-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4},
{-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4},
{-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4},
{-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4},
{-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4},
{-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4},
{-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4},
{-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4},
{-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4},
{-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4},
{-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4},
{-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4},
{ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4},
{ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4},
{-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4},
{-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4},
{ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4},
{-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4},
{-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4},
{ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4},
{-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1}
};

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{

	printf("WG size of kernel = %d \n", BLOCK_SIZE);

    runTest( argc, argv);

    return EXIT_SUCCESS;
}

void usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <max_rows/max_cols> <penalty> \n", argv[0]);
	fprintf(stderr, "\t<dimension>  - x and y dimensions\n");
	fprintf(stderr, "\t<penalty> - penalty(positive integer)\n");
	exit(1);
}

void runTest( int argc, char** argv) 
{
    int max_rows, max_cols, penalty;
    int *input_itemsets, *referrence_cuda;
	long long int size;
	    
    // the lengths of the two sequences should be able to divided by 16.
	// And at current stage  max_rows needs to equal max_cols
	if (argc == 3)
	{
		max_rows = atoi(argv[1]);
		max_cols = atoi(argv[1]);
		penalty = atoi(argv[2]);
		printf("%d, %d\n",max_rows, penalty);
	}
	else
	{
		usage(argc, argv);
    }
	
	if (atoi(argv[1]) % 16 != 0)
	{
		fprintf(stderr,"The dimension values must be a multiple of 16\n");
		exit(1);
	}
	

	max_rows = max_rows + 1;
	max_cols = max_cols + 1;
	// referrence = (int *)malloc( max_rows * max_cols * sizeof(int) );
    // input_itemsets = (int *)malloc( max_rows * max_cols * sizeof(int) );
    size = (long long int)max_cols * max_rows;

	// output_itemsets = (int *)malloc( size * sizeof(int) );
	
	cudaMallocManaged(&referrence_cuda, (size_t)sizeof(int)*size);
	cudaMallocManaged(&input_itemsets, (size_t)sizeof(int)*size);
	const uint32_t page_shift = 12;
	const uint64_t ref_n_elems = (uint64_t)size;
	const uint64_t matrix_n_elems = (uint64_t)size;
	const char* bv = getenv("BENCH_VARIANT");
	const bool use_frontend = (bv && strcmp(bv, "frontend") == 0);


	if (!input_itemsets)
		fprintf(stderr, "error: can not allocate memory");

#ifndef SKIP_CPU_VERIFY
    srand ( 7 );
		
	for (long long int i = 0 ; i < max_cols; i++)
	{
		for (long long int j = 0 ; j < max_rows; j++)
		{
			input_itemsets[i*max_cols+j] = 0;
		}
	}
	
    printf("1\n");

	printf("Start Needleman-Wunsch\n");
	
	for( long long int i=1; i< max_rows ; i++){    //please define your own sequence. 
       input_itemsets[i*max_cols] = rand() % 10 + 1;
	}
    for( long long int j=1; j< max_cols ; j++){    //please define your own sequence.
       input_itemsets[j] = rand() % 10 + 1;
	}

    printf("1\n");

	for (long long int i = 1 ; i < max_cols; i++){
		for (long long int j = 1 ; j < max_rows; j++){
		referrence_cuda[i*max_cols+j] = blosum62[input_itemsets[i*max_cols]][input_itemsets[j]];
		}
	}

    for( long long int i = 1; i< max_rows ; i++)
       input_itemsets[i*max_cols] = -i * penalty;
	for( long long int j = 1; j< max_cols ; j++)
       input_itemsets[j] = -j * penalty;
#else
	printf("Start Needleman-Wunsch (SKIP_CPU_VERIFY, 16GB BW run)\n");
#endif

	uvm_warmup();

	   cudaEvent_t start, stop;
	   cudaEventCreate(&start);
	   cudaEventCreate(&stop);
	   float elapsed_time;
	// cudaMalloc((void**)& referrence_cuda, sizeof(int)*size);
	// cudaMalloc((void**)& matrix_cuda, sizeof(int)*size);
	
	// cudaMemcpy(referrence_cuda, referrence, sizeof(int) * size, cudaMemcpyHostToDevice);
	// cudaMemcpy(matrix_cuda, input_itemsets, sizeof(int) * size, cudaMemcpyHostToDevice);

    dim3 dimGrid;
	dim3 dimBlock(BLOCK_SIZE, 1);
	int block_width = ( max_cols - 1 )/BLOCK_SIZE;
	prefault_array_host_t<int> ref_pf{};
	prefault_array_host_t<int> matrix_pf{};
	unsigned long long* d_cas_wins = nullptr;
	unsigned long long h_cas_wins = 0;
	if (use_frontend) {
		prefault_array_init(&ref_pf, referrence_cuda, ref_n_elems, page_shift, false);
		prefault_array_init(&matrix_pf, input_itemsets, matrix_n_elems, page_shift, true);
		cudaMalloc(&d_cas_wins, sizeof(unsigned long long));
		cudaMemset(d_cas_wins, 0, sizeof(unsigned long long));
	}

	prefault_to_cpu(referrence_cuda, (size_t)sizeof(int) * size);
	prefault_to_cpu(input_itemsets, (size_t)sizeof(int) * size);
	if (use_frontend) {
		prefault_array_reset(&ref_pf);
		prefault_array_reset(&matrix_pf);
		cudaMemset(d_cas_wins, 0, sizeof(unsigned long long));
	}
    cudaEventRecord(start, 0);
	printf("Processing top-left matrix\n");
	//process top-left matrix
	for( int i = 1 ; i <= block_width ; i++){
		dimGrid.x = i;
		dimGrid.y = 1;
		if (!use_frontend) {
			needle_cuda_shared_1<<<dimGrid, dimBlock>>>(referrence_cuda, input_itemsets
			                                      ,max_cols, penalty, i, block_width);
		} else {
			needle_cuda_shared_1_frontend<<<dimGrid, dimBlock>>>(
				referrence_cuda, input_itemsets,
				max_cols, penalty, i, block_width,
				page_shift,
				ref_pf.page_status,
				matrix_pf.page_status,
				matrix_pf.write_status,
				d_cas_wins,
				ref_n_elems,
				matrix_n_elems);
		}
	}
	printf("Processing bottom-right matrix\n");
    //process bottom-right matrix
	for( int i = block_width - 1  ; i >= 1 ; i--){
		dimGrid.x = i;
		dimGrid.y = 1;
		if (!use_frontend) {
			needle_cuda_shared_2<<<dimGrid, dimBlock>>>(referrence_cuda, input_itemsets
			                                      ,max_cols, penalty, i, block_width);
		} else {
			needle_cuda_shared_2_frontend<<<dimGrid, dimBlock>>>(
				referrence_cuda, input_itemsets,
				max_cols, penalty, i, block_width,
				page_shift,
				ref_pf.page_status,
				matrix_pf.page_status,
				matrix_pf.write_status,
				d_cas_wins,
				ref_n_elems,
				matrix_n_elems);
		}
	}


	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);

    printf("\nGPU Runtime: %lfs\n", (elapsed_time)/1000);
	printf("nw,%s,%lfs\n", use_frontend ? "frontend" : "baseline", (elapsed_time)/1000);
	if (use_frontend) {
		cudaMemcpy(&h_cas_wins, d_cas_wins, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
		fprintf(stderr, "  frontend cas_wins=%llu\n", h_cas_wins);
	}
#ifndef SKIP_CPU_VERIFY
    // cudaMemcpy(output_itemsets, matrix_cuda, sizeof(int) * size, cudaMemcpyDeviceToHost);
	
//#define TRACEBACK
	
	FILE *fpo = fopen("result.txt","w");
	fprintf(fpo, "print traceback value GPU:\n");
    
	for (long long int i = max_rows - 2, j = max_rows - 2; i>=0 && j>=0;){
		int nw, n, w, traceback;
		if ( i == max_rows - 2 && j == max_rows - 2 )
			fprintf(fpo, "%d ", input_itemsets[ i * max_cols + j]); //print the first element
		if ( i == 0 && j == 0 )
           break;
		if ( i > 0 && j > 0 ){
			nw = input_itemsets[(i - 1) * max_cols + j - 1];
		    w  = input_itemsets[ i * max_cols + j - 1 ];
            n  = input_itemsets[(i - 1) * max_cols + j];
		}
		else if ( i == 0 ){
		    nw = n = LIMIT;
		    w  = input_itemsets[ i * max_cols + j - 1 ];
		}
		else if ( j == 0 ){
		    nw = w = LIMIT;
            n  = input_itemsets[(i - 1) * max_cols + j];
		}
		else{
		}

		//traceback = maximum(nw, w, n);
		int new_nw, new_w, new_n;
		new_nw = nw + referrence_cuda[i * max_cols + j];
		new_w = w - penalty;
		new_n = n - penalty;
		
		traceback = maximum(new_nw, new_w, new_n);
		if(traceback == new_nw)
			traceback = nw;
		if(traceback == new_w)
			traceback = w;
		if(traceback == new_n)
            traceback = n;
			
		fprintf(fpo, "%d ", traceback);

		if(traceback == nw )
		{i--; j--; continue;}

        else if(traceback == w )
		{j--; continue;}

        else if(traceback == n )
		{i--; continue;}

		else
		;
	}
	
	fclose(fpo);
#endif

	cudaFree(referrence_cuda);
	cudaFree(input_itemsets);
	if (d_cas_wins) cudaFree(d_cas_wins);
	if (use_frontend) {
		prefault_array_destroy(&ref_pf);
		prefault_array_destroy(&matrix_pf);
	}

	// free(referrence);
	// free(input_itemsets);
	// free(output_itemsets);
	
}

