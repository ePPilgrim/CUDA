#include <stdio.h>
#include "VecAdd_kernel.cu"
#include "cuda_runtime.h"

__global__ void VecAdd_kernel(const float* A, const float* B, float* C, int N)
/* Naive kernel */
{
	// Uncomment line below and define global index form block and thread indexes
	 int i = blockIdx.x * blockDim.x + threadIdx.x;

	 if (i < N) {
		 C[i] = A[i] + B[i];
	 }
}

int main(int argc, char *argv[])
{

 int N = 100;

 unsigned int size;
 float *d_A, *d_B, *d_C;
 float *h_A, *h_B, *h_C;

/****************************
 * Initialization of memory *
 ****************************/

 size = N * sizeof(float);
 h_A = (float *) malloc(size);
 h_B = (float *) malloc(size);
 h_C = (float *) malloc(size);
 for (unsigned i=0; i<N; i++) {
   h_A[i] = 1.0f;
   h_B[i] = 2.0f;
   h_C[i] = 0.0f;
 }

 // YOUR TASKS:
 // - Allocate below device arrays d_A, d_B and d_C
 // - Transfer array data from host to device arrays
 // Insert code below this line.

 cudaMalloc(&d_A, size);
 cudaMalloc(&d_B, size);
 cudaMalloc(&d_C, size);

 cudaMemcpy((void*)d_A, (const void*)h_A, size, ::cudaMemcpyHostToDevice);
 cudaMemcpy((void*)d_B, (const void*)h_B, size, ::cudaMemcpyHostToDevice);

/****************************
 * GPU execution            *
 ****************************/

 // YOUR TASK:
 // - Define below the number of threads per block and blocks per grid
 // Update the two lines below this line.

 int threadsPerBlock = 2; 
 int blocksPerGrid = 50; 

 VecAdd_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A,d_B,d_C,N);
 cudaThreadSynchronize();

 // YOUR TASK:
 // - Transfer data results stored in d_C to host array
 // Insert code below this line.
 cudaMemcpy((void*)h_C, (const void*)d_C, size, ::cudaMemcpyDeviceToHost);


/****************************
 * Verification             *
 ****************************/

 float sum = 0.0f;
 for (unsigned i=0; i<N; i++) {
    sum += h_C[i];
 }
 printf("Vector addition\n");
 if (abs(sum-3.0f*(float) N)<=1e-10) 
 {
    printf("PASSED!\n");
 }
 else
 {
    printf("FAILED!\n");
 }

/****************************
 * Cleaning memory          *
 ****************************/

 // YOUR TASK:
 // - Free device memory for the allocated d_A, d_B and d_C arrays
 // Insert code below this line.

 free(h_A);
 free(h_B);
 free(h_C);

 cudaFree(d_A);
 cudaFree(d_B);
 cudaFree(d_C);

 return 0;

}