//
// kernel routine
//
#include "cuda_runtime.h"

__global__ void my_first_kernel(float *x)
{
	// Uncomment line below and define integer "tid" as global index to vector "x"
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Uncomment line below and define x[tid] to be equal to the thread index
	x[tid] = (float)threadIdx.x;
}

