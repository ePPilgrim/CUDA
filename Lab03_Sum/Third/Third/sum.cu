// This program completes parallel reduction on a data set.

// Included C libraries
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctime>
#include<cmath>
#include <float.h>
#include "cuda_runtime.h"
#include "cuda.h"
#include "sum_gold.cpp"

#define BLOCK_SIZE 1024
#define N_SIZE 65536//1048576//65536//4096//65536

__global__ void sum_kernel(float *g_odata, float *g_idata, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int upper = (1 + blockIdx.x) * blockDim.x;
	
	int k = blockDim.x >> 1;
	for (int i = 0; i < N; ++ i) {
		if ((idx + k) < upper) {
			g_idata[idx] = g_idata[idx] + g_idata[idx + k];
			upper -= k;
			k >>= 1;
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) g_odata[blockIdx.x] = g_idata[idx];
}

__global__ void sum_kernel2(float *g_odata, float *g_idata, int N)
{
	__shared__ float sum[BLOCK_SIZE];
	sum[threadIdx.x] = g_idata[blockIdx.x * blockDim.x + threadIdx.x];
	__syncthreads();

	int k = blockDim.x / 2;
	for (int i = 0; i < N; ++i) {
		if ((threadIdx.x + k) < (k<<1)) {
			sum[threadIdx.x] += sum[threadIdx.x + k];
			k >>= 1;
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) g_odata[blockIdx.x] = sum[0];
}

__global__ void sum_kernel3(float *g_odata, float *g_idata, int N)
{
	__shared__ float sum[BLOCK_SIZE];
	sum[threadIdx.x] = g_idata[blockIdx.x * blockDim.x + threadIdx.x];
	__syncthreads();

	int k = 1;
	for (int i = 0; i < N; ++i) {
		if ((threadIdx.x % (k << 1)) == 0) {
			sum[threadIdx.x] += sum[threadIdx.x + k];
			k <<=1;
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) g_odata[blockIdx.x] = sum[0];
}

__global__ void sum_kernel4(float *g_odata, float *g_idata, int N)
{
	__shared__ float sum[BLOCK_SIZE];
	sum[threadIdx.x] = g_idata[blockIdx.x * blockDim.x + threadIdx.x];
	__syncthreads();

	int k = blockDim.x >> 1;
	for (int i = 0; i < N; ++i) {
		if (threadIdx.x < k) {
			sum[threadIdx.x] += sum[(k << 1) - threadIdx.x - 1];
			k >>= 1;
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) g_odata[blockIdx.x] = sum[0];
}

__global__ void sum_kernel5(float *g_odata, float *g_idata, int N)
{

	// YOUR TASKS:
	// - Optimize as much as possible

}

int main( int argc, char* argv[]) 
{
  // Screen output
  printf("sum\n");
  printf("Parallel sum reduction.\n");
  printf("  ./sum <N:default=64> <THREADS_PR_BLOCK:default=MaxOnDevice>\n\n");

  // Check limitation of available device
  int dev = 0; // assumed only one device
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  printf("Device 0: Maximum number of threads per block is %d.\n", deviceProp.maxThreadsPerBlock);
  int MAX_THREADS_PR_BLOCK = deviceProp.maxThreadsPerBlock;

  int N, THREADS_PR_BLOCK;
  if (argc>1 ? N = atoi(argv[1]) : N = N_SIZE);
  if (argc>2 ? THREADS_PR_BLOCK = atoi(argv[2]) : THREADS_PR_BLOCK = MAX_THREADS_PR_BLOCK);
  if (THREADS_PR_BLOCK > MAX_THREADS_PR_BLOCK ? THREADS_PR_BLOCK = MAX_THREADS_PR_BLOCK : 0.0);
  if (THREADS_PR_BLOCK > N ? THREADS_PR_BLOCK = N : 0.0);
  printf("N: %d\n", N); 
  if (N % 32 > 0) { 
     printf("N is not a integer multiple of warpsize (32). %d \n",N%32);
     exit(1);
  }
  printf("Threads per block = %d. \n",THREADS_PR_BLOCK);

  int BLOCKS = (N + THREADS_PR_BLOCK)/THREADS_PR_BLOCK;
  if ((BLOCKS-1)*THREADS_PR_BLOCK >= N ? BLOCKS = BLOCKS-1 : 0.0 );
  printf("Blocks allocated = %d\n\n",BLOCKS);

/**************************************************
 * Create timers                                  *
 **************************************************/
	cudaEvent_t start, stop;
	float time, cpu_time, gpu_time1, gpu_time2, gpu_time3, gpu_time4, gpu_time5;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

/****************************
 * Initialization of memory *
 ****************************/

  // Pointers to CPU (host) data
  float* DataSet_h = new float[N]; 
  float* partialsums_h = new float[BLOCKS]; 

  // initialize the Data Set
  for (int i = 0; i < N; ++i) {
     DataSet_h[i] = 1.5f;
  }

  // Pointers for GPU (device) data
  float* DataSet_d; 
  float* partialsums_d; 

  // Safely allocate memory for data on device
  cudaMalloc( (void**)&DataSet_d, N * sizeof(float) );   
  cudaMalloc( (void**)&partialsums_d, BLOCKS*sizeof(float) ); 
  int logN = (int)std::ceilf(std::log2f((float)THREADS_PR_BLOCK));

/***************************
 * GPU execution (naive)   *
 ***************************/

  // Split problem into threads
  dim3 blockGrid( BLOCKS ); 
  dim3 threadBlock( THREADS_PR_BLOCK ); 

  float* sum_h = new float[1];
  cudaEventRecord(start, 0);
  for (int iter = 0; iter < 100; ++iter) 
  {
    cudaMemcpy( DataSet_d, DataSet_h, N * sizeof(float), cudaMemcpyHostToDevice);
    sum_kernel<<< blockGrid, threadBlock, THREADS_PR_BLOCK*sizeof(float) >>>(partialsums_d, DataSet_d, logN); 
    cudaThreadSynchronize();	
    cudaMemcpy( partialsums_h, partialsums_d, BLOCKS*sizeof(float), cudaMemcpyDeviceToHost) ;
    cudaThreadSynchronize();	
    sum_h[0] = 0.;  
    sum_gold(sum_h,partialsums_h,BLOCKS);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&gpu_time1, start, stop);

/***************************
 * GPU execution (v2)      *
 ***************************/

  float* sum2_h = new float[1];
  cudaEventRecord(start, 0);
  for (int iter = 0; iter < 100; ++iter) 
  {
    cudaMemcpy( DataSet_d, DataSet_h, N * sizeof(float), cudaMemcpyHostToDevice);
    sum_kernel2<<< blockGrid, threadBlock, THREADS_PR_BLOCK*sizeof(float) >>>(partialsums_d, DataSet_d, logN); 
    cudaThreadSynchronize();
    cudaMemcpy( partialsums_h, partialsums_d, BLOCKS*sizeof(float), cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();	
    sum2_h[0] = 0.;  
    sum_gold(sum2_h,partialsums_h,BLOCKS);
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&gpu_time2, start, stop);

/***************************
 * GPU execution (v3)      *
 ***************************/

  float* sum3_h = new float[1];
  cudaEventRecord(start, 0);
  for (int iter = 0; iter < 100; ++iter) 
  {
    cudaMemcpy( DataSet_d, DataSet_h, N * sizeof(float), cudaMemcpyHostToDevice);
    sum_kernel3<<< blockGrid, threadBlock, THREADS_PR_BLOCK*sizeof(float) >>>(partialsums_d, DataSet_d, logN); 
    cudaThreadSynchronize();	
    cudaMemcpy( partialsums_h, partialsums_d, BLOCKS*sizeof(float), cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();	
    sum3_h[0] = 0.;  
    sum_gold(sum3_h,partialsums_h,BLOCKS);
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&gpu_time3, start, stop);

/***************************
 * GPU execution (v4)      *
 ***************************/

  float* sum4_h = new float[1];
  cudaEventRecord(start, 0);
  for (int iter = 0; iter < 100; ++iter) 
  {
    cudaMemcpy( DataSet_d, DataSet_h, N * sizeof(float), cudaMemcpyHostToDevice);
    sum_kernel4<<< blockGrid, threadBlock, THREADS_PR_BLOCK*sizeof(float) >>>(partialsums_d, DataSet_d, logN); 
	cudaThreadSynchronize();	
    cudaMemcpy( partialsums_h, partialsums_d, BLOCKS*sizeof(float), cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();	
    sum4_h[0] = 0.;  
    sum_gold(sum4_h,partialsums_h,BLOCKS);
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&gpu_time4, start, stop);

/***************************
 * GPU execution (v5)      *
 ***************************/

  float* sum5_h = new float[1];
  cudaEventRecord(start, 0);
  for (int iter = 0; iter < 100; ++iter) 
  {
	cudaMemcpy( DataSet_d, DataSet_h, N * sizeof(float), cudaMemcpyHostToDevice) ;
    sum_kernel5<<< blockGrid, threadBlock, THREADS_PR_BLOCK*sizeof(float) >>>(partialsums_d, DataSet_d, N); 
	cudaThreadSynchronize();	
	cudaMemcpy( partialsums_h, partialsums_d, BLOCKS*sizeof(float), cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();	
    sum5_h[0] = 0.;  
    sum_gold(sum5_h,partialsums_h,BLOCKS);
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&gpu_time5, start, stop);

  /***************************
 * CPU execution           *
 ***************************/

  float* sumGold_h = new float[1];
  sumGold_h[0] = 0.0f;
  std::clock_t start_cpu;
  start_cpu = std::clock();

  for (int iter = 0; iter < 100; ++iter) {
	  sum_gold(sumGold_h, DataSet_h, N);
  }

  cpu_time = (std::clock() - start_cpu) / (double)CLOCKS_PER_SEC;
  cpu_time *= 1000.0;
  //cudaEventRecord(stop, 0);
  //cudaEventSynchronize(stop);
  //cudaEventElapsedTime(&cpu_time, start, stop);
/*********************************
 * Output timings & verification *
 *********************************/

    printf("  CPU time           : %.4f (ms)\n\n",cpu_time);
    printf("  GPU time (naive)   : %.4f (ms) , speedup %.2fx\n",gpu_time1,cpu_time/gpu_time1);
    if (abs(sum_h[0] - sumGold_h[0])<1e-4 ? printf("  PASSED\n\n") : printf("  FAILED \n")  )
    printf("  GPU time (v2)      : %.4f (ms) , speedup %.2fx\n",gpu_time2,cpu_time/gpu_time2);
    if (abs(sum2_h[0] - sumGold_h[0])<1e-4 ? printf("  PASSED\n\n") : printf("  FAILED \n")  )
		printf("  GPU time (v3)      : %.4f (ms) , speedup %.2fx\n", gpu_time3, cpu_time / gpu_time3);
    if (abs(sum3_h[0] - sumGold_h[0])<1e-4 ? printf("  PASSED\n\n") : printf("  FAILED \n")  )
		printf("  GPU time (v4)      : %.4f (ms) , speedup %.2fx\n", gpu_time4, cpu_time / gpu_time4);
    if (abs(sum4_h[0] - sumGold_h[0])<1e-4 ? printf("  PASSED\n\n") : printf("  FAILED \n")  )
		printf("  GPU time (v5)      : %.4f (ms) , speedup %.2fx\n", gpu_time5, cpu_time / gpu_time5);
    if (abs(sum5_h[0] - sumGold_h[0])<1e-4 ? printf("  PASSED\n\n") : printf("  FAILED \n")  )

/***************************
 * Verification            *
 ***************************/

  printf("sumCPU         = %.2f\n",sumGold_h[0]);
  printf("sumGPU (naive) = %.2f\n",sum_h[0]);
  printf("sumGPU (v2)    = %.2f\n",sum2_h[0]);
  printf("sumGPU (v3)    = %.2f\n",sum3_h[0]);
  printf("sumGPU (v4)    = %.2f\n",sum4_h[0]);
  printf("sumGPU (v5)    = %.2f\n",sum5_h[0]);

/***************************
 * Cleaning memory         *
 ***************************/

  // cleanup device memory
  cudaFree(partialsums_d); 
  cudaFree(DataSet_d); 

  delete[] partialsums_h; 
  delete[] DataSet_h; 

}
