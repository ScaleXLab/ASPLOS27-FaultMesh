
#include "needle.h"
#include <stdio.h>
#include "../../frontLib/frontend_prefault_common.cuh"


#define SDATA( index)      CUT_BANK_CHECKER(sdata, index)

__device__ __host__ int 
maximum( int a,
		 int b,
		 int c){

int k;
if( a <= b )
k = b;
else 
k = a;

if( k <=c )
return(c);
else
return(k);

}

__global__ void
needle_cuda_shared_1(  int* referrence,
			  int* matrix_cuda, 
			  int cols,
			  int penalty,
			  int i,
			  int block_width) 
{
  int bx = blockIdx.x;
  int tx = threadIdx.x;

  int b_index_x = bx;
  int b_index_y = i - 1 - bx;

  long long int index   = (long long int)cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( cols + 1 );
  long long int index_n   = (long long int)cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( 1 );
  long long int index_w   = (long long int)cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + ( cols );
  long long int index_nw  = (long long int)cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x;

   __shared__  int temp[BLOCK_SIZE+1][BLOCK_SIZE+1];
   __shared__  int ref[BLOCK_SIZE][BLOCK_SIZE];

   //int temp[BLOCK_SIZE+1][BLOCK_SIZE+1];
   //int ref[BLOCK_SIZE][BLOCK_SIZE];

   if (tx == 0)
		  temp[tx][0] = matrix_cuda[index_nw];


  for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++)
  ref[ty][tx] = referrence[index + cols * ty];

  __syncthreads();

  temp[tx + 1][0] = matrix_cuda[index_w + cols * tx];

  __syncthreads();

  temp[0][tx + 1] = matrix_cuda[index_n];
  
  __syncthreads();
  

  for( int m = 0 ; m < BLOCK_SIZE ; m++){
   
	  if ( tx <= m ){

		  int t_index_x =  tx + 1;
		  int t_index_y =  m - tx + 1;

          temp[t_index_y][t_index_x] = maximum( temp[t_index_y-1][t_index_x-1] + ref[t_index_y-1][t_index_x-1],
		                                        temp[t_index_y][t_index_x-1]  - penalty, 
												temp[t_index_y-1][t_index_x]  - penalty);

		  
	  
	  }

	  __syncthreads();
  
    }

 for( int m = BLOCK_SIZE - 2 ; m >=0 ; m--){
   
	  if ( tx <= m){

		  int t_index_x =  tx + BLOCK_SIZE - m ;
		  int t_index_y =  BLOCK_SIZE - tx;

          temp[t_index_y][t_index_x] = maximum( temp[t_index_y-1][t_index_x-1] + ref[t_index_y-1][t_index_x-1],
		                                        temp[t_index_y][t_index_x-1]  - penalty, 
												temp[t_index_y-1][t_index_x]  - penalty);
	   
	  }

	  __syncthreads();
  }

  for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++)
  matrix_cuda[index + ty * cols] = temp[ty+1][tx+1];

}


__global__ void
needle_cuda_shared_2(  int* referrence,
			  int* matrix_cuda, 
			 
			  int cols,
			  int penalty,
			  int i,
			  int block_width) 
{

  int bx = blockIdx.x;
  int tx = threadIdx.x;

  int b_index_x = bx + block_width - i  ;
  int b_index_y = block_width - bx -1;

  long long int index   = (long long int)cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( cols + 1 );
  long long int index_n   = (long long int)cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( 1 );
  long long int index_w   = (long long int)cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + ( cols );
  long long int index_nw  = (long long int)cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x;

  __shared__  int temp[BLOCK_SIZE+1][BLOCK_SIZE+1];
  __shared__  int ref[BLOCK_SIZE][BLOCK_SIZE];
    
  //int temp[BLOCK_SIZE+1][BLOCK_SIZE+1];
  //int ref[BLOCK_SIZE][BLOCK_SIZE];


  for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++)
  ref[ty][tx] = referrence[index + cols * ty];

  __syncthreads();

   if (tx == 0)
		  temp[tx][0] = matrix_cuda[index_nw];
 
 
  temp[tx + 1][0] = matrix_cuda[index_w + cols * tx];

  __syncthreads();

  temp[0][tx + 1] = matrix_cuda[index_n];
  
  __syncthreads();
  

  for( int m = 0 ; m < BLOCK_SIZE ; m++){
   
	  if ( tx <= m ){

		  int t_index_x =  tx + 1;
		  int t_index_y =  m - tx + 1;

          temp[t_index_y][t_index_x] = maximum( temp[t_index_y-1][t_index_x-1] + ref[t_index_y-1][t_index_x-1],
		                                        temp[t_index_y][t_index_x-1]  - penalty, 
												temp[t_index_y-1][t_index_x]  - penalty);	  
	  
	  }

	  __syncthreads();
  
    }


 for( int m = BLOCK_SIZE - 2 ; m >=0 ; m--){
   
	  if ( tx <= m){

		  int t_index_x =  tx + BLOCK_SIZE - m ;
		  int t_index_y =  BLOCK_SIZE - tx;

          temp[t_index_y][t_index_x] = maximum( temp[t_index_y-1][t_index_x-1] + ref[t_index_y-1][t_index_x-1],
		                                        temp[t_index_y][t_index_x-1]  - penalty, 
												temp[t_index_y-1][t_index_x]  - penalty);


	  }

	  __syncthreads();
  }


  for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++)
  matrix_cuda[index + ty * cols] = temp[ty+1][tx+1];

}

__global__ void
needle_cuda_shared_1_frontend( int* referrence,
              int* matrix_cuda,
              int cols,
              int penalty,
              int i,
              int block_width,
              uint32_t page_shift,
              uint32_t* ref_status,
              uint32_t* matrix_status,
              uint32_t* matrix_write_status,
              unsigned long long* cas_wins,
              uint64_t ref_n_elems,
              uint64_t matrix_n_elems)
{
  int bx = blockIdx.x;
  int tx = threadIdx.x;

  int b_index_x = bx;
  int b_index_y = i - 1 - bx;

  long long int index   = (long long int)cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( cols + 1 );
  long long int index_n   = (long long int)cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( 1 );
  long long int index_w   = (long long int)cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + ( cols );
  long long int index_nw  = (long long int)cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x;

  __shared__  int temp[BLOCK_SIZE+1][BLOCK_SIZE+1];
  __shared__  int ref[BLOCK_SIZE][BLOCK_SIZE];

  const uvm_prefault_array_dedup_t<int> ref_pf{referrence, ref_n_elems, page_shift, ref_status, nullptr, cas_wins};
  const uvm_prefault_array_dedup_t<int> mat_pf{matrix_cuda, matrix_n_elems, page_shift, matrix_status, matrix_write_status, cas_wins};

  if (tx == 0)
      temp[tx][0] = mat_pf[index_nw];

  for (int ty = 0; ty < BLOCK_SIZE; ty++)
      ref[ty][tx] = ref_pf[index + cols * ty];

  __syncthreads();

  temp[tx + 1][0] = mat_pf[index_w + cols * tx];

  __syncthreads();

  temp[0][tx + 1] = mat_pf[index_n];

  __syncthreads();

  for (int m = 0; m < BLOCK_SIZE; m++) {
      if (tx <= m) {
          int t_index_x = tx + 1;
          int t_index_y = m - tx + 1;
          temp[t_index_y][t_index_x] = maximum(temp[t_index_y-1][t_index_x-1] + ref[t_index_y-1][t_index_x-1],
                                                temp[t_index_y][t_index_x-1] - penalty,
                                                temp[t_index_y-1][t_index_x] - penalty);
      }
      __syncthreads();
  }

  for (int m = BLOCK_SIZE - 2; m >= 0; m--) {
      if (tx <= m) {
          int t_index_x = tx + BLOCK_SIZE - m;
          int t_index_y = BLOCK_SIZE - tx;
          temp[t_index_y][t_index_x] = maximum(temp[t_index_y-1][t_index_x-1] + ref[t_index_y-1][t_index_x-1],
                                                temp[t_index_y][t_index_x-1] - penalty,
                                                temp[t_index_y-1][t_index_x] - penalty);
      }
      __syncthreads();
  }

  for (int ty = 0; ty < BLOCK_SIZE; ty++)
      mat_pf.store(index + ty * cols, temp[ty+1][tx+1]);
}

__global__ void
needle_cuda_shared_2_frontend( int* referrence,
              int* matrix_cuda,
              int cols,
              int penalty,
              int i,
              int block_width,
              uint32_t page_shift,
              uint32_t* ref_status,
              uint32_t* matrix_status,
              uint32_t* matrix_write_status,
              unsigned long long* cas_wins,
              uint64_t ref_n_elems,
              uint64_t matrix_n_elems)
{
  int bx = blockIdx.x;
  int tx = threadIdx.x;

  int b_index_x = bx + block_width - i;
  int b_index_y = block_width - bx - 1;

  long long int index   = (long long int)cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( cols + 1 );
  long long int index_n   = (long long int)cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( 1 );
  long long int index_w   = (long long int)cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + ( cols );
  long long int index_nw  = (long long int)cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x;

  __shared__  int temp[BLOCK_SIZE+1][BLOCK_SIZE+1];
  __shared__  int ref[BLOCK_SIZE][BLOCK_SIZE];

  const uvm_prefault_array_dedup_t<int> ref_pf{referrence, ref_n_elems, page_shift, ref_status, nullptr, cas_wins};
  const uvm_prefault_array_dedup_t<int> mat_pf{matrix_cuda, matrix_n_elems, page_shift, matrix_status, matrix_write_status, cas_wins};

  for (int ty = 0; ty < BLOCK_SIZE; ty++)
      ref[ty][tx] = ref_pf[index + cols * ty];

  __syncthreads();

  if (tx == 0)
      temp[tx][0] = mat_pf[index_nw];

  temp[tx + 1][0] = mat_pf[index_w + cols * tx];

  __syncthreads();

  temp[0][tx + 1] = mat_pf[index_n];

  __syncthreads();

  for (int m = 0; m < BLOCK_SIZE; m++) {
      if (tx <= m) {
          int t_index_x = tx + 1;
          int t_index_y = m - tx + 1;
          temp[t_index_y][t_index_x] = maximum(temp[t_index_y-1][t_index_x-1] + ref[t_index_y-1][t_index_x-1],
                                                temp[t_index_y][t_index_x-1] - penalty,
                                                temp[t_index_y-1][t_index_x] - penalty);
      }
      __syncthreads();
  }

  for (int m = BLOCK_SIZE - 2; m >= 0; m--) {
      if (tx <= m) {
          int t_index_x = tx + BLOCK_SIZE - m;
          int t_index_y = BLOCK_SIZE - tx;
          temp[t_index_y][t_index_x] = maximum(temp[t_index_y-1][t_index_x-1] + ref[t_index_y-1][t_index_x-1],
                                                temp[t_index_y][t_index_x-1] - penalty,
                                                temp[t_index_y-1][t_index_x] - penalty);
      }
      __syncthreads();
  }

  for (int ty = 0; ty < BLOCK_SIZE; ty++)
      mat_pf.store(index + ty * cols, temp[ty+1][tx+1]);
}

