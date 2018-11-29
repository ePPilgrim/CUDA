#include <ctime>
#include<cmath>
#include<iostream>
#include<fstream>
#include<random>
#include<vector>
#include "cuda_runtime.h"
#include "cuda.h"

float* genVects(int col, int row);
float* findMeans(float* vects, int col, int row);
float* findCovVals(float* vects, int col, int row);

const int BlckSzX = 16;
const int BlckSzY = 16;
const int BlckSz = 16;
const int LogN = 4;

__global__ void mean_kern(float* vects, int row, int col,float *sum_odata, int slotcnt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	__shared__ float sum[BlckSzX][BlckSzX];
	sum[threadIdx.y][threadIdx.x] = vects[y * col + x];
	__syncthreads();

	int k = blockDim.x >> 1;
	for (int i = 0; i < LogN; ++i) {
		if (threadIdx.x < k) {
			sum[threadIdx.y][threadIdx.x] += sum[threadIdx.y][(k << 1) - threadIdx.x - 1];
			k >>= 1;
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) sum_odata[blockIdx.y * slotcnt + blockIdx.x] = sum[threadIdx.y][0];
}

__global__ void var_kern(float* vects, int row, int col, float *sum_odata, int slotcnt)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (x == 0) {
		float sum = 0.0f;
		float* subsum = &sum_odata[y * slotcnt];
		for (int i = 0; i < slotcnt; ++i) {
			sum += subsum[i];
		}
		sum_odata[y*slotcnt] = sum / (float)col;
	}
	__syncthreads();
	//vects[y * col + x] -= sum_odata[y*slotcnt];	
}

__global__ void varcor(const float* vects, int row, int col, float* out,const int slotsize) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	__shared__ float outmat[BlckSz][BlckSz];
	__shared__ float subvects[BlckSz][BlckSz];

	for (int i = blockIdx.y; i < gridDim.y; ++i) {
		int shift = blockDim.y*(i - blockIdx.y);
		subvects[threadIdx.y][threadIdx.x] = vects[(shift + y) * col + x];
		__syncthreads();
		for (int j = 0; j < BlckSz; ++j) {
			outmat[threadIdx.y][threadIdx.x] = 0.0f;
			if ((threadIdx.y + j) < BlckSz) {
				outmat[threadIdx.y][threadIdx.x] = vects[y * col + x]*subvects[threadIdx.y + j][threadIdx.x];
			}
			__syncthreads();
			shift = shift + j;
			shift = y + ((shift * (row + row + 1 - shift)) /2);
			if (threadIdx.x == 0){
				if( (shift < slotsize) && (shift >= 0)) {
					out[shift + (blockIdx.x * slotsize)] = outmat[threadIdx.y][0] + outmat[threadIdx.y][1] + outmat[threadIdx.y][2] +
						outmat[threadIdx.y][3] + outmat[threadIdx.y][4] + outmat[threadIdx.y][5] + outmat[threadIdx.y][6] +
						outmat[threadIdx.y][7] + outmat[threadIdx.y][8] + outmat[threadIdx.y][9] + outmat[threadIdx.y][10] +
						outmat[threadIdx.y][11] + outmat[threadIdx.y][12] + outmat[threadIdx.y][13] + outmat[threadIdx.y][14] +
						outmat[threadIdx.y][15];
				}
			}	
		}
	}
}

int main(int argc, char* argv[])
{
	//std::fstream input;
	//input.open("input", std::fstream::out);
	int Rows = 1024;
	int Cols = 1024;
	int BlocksX = Cols / BlckSzX;
	int BlocksY = Rows / BlckSzY;
	int slotSize = (Rows * (Rows + 1)) / 2;
	bool flag = true;

	float* vectors_h = genVects(Cols, Rows);
	float* out_vectors_h = new float[BlocksX * slotSize];
	float* means_h = new float[Rows];

		// Pointers for GPU (device) data
	float* vectors_d;
	float* means_d;
	float* out_vectors_d;

	// Safely allocate memory for data on device
	cudaMalloc((void**)&vectors_d, Rows * Cols * sizeof(float));
	cudaMalloc((void**)&means_d, Rows * BlocksX * sizeof(float));
	cudaMalloc((void**)&out_vectors_d, sizeof(float) * slotSize * BlocksX );

	cudaMemcpy(vectors_d, vectors_h, Cols * Rows * sizeof(float), cudaMemcpyHostToDevice);

	// Split problem into threads
	dim3 blockGrid1(Cols/BlckSzX, Rows/BlckSzX,1);
	dim3 threadBlock1(BlckSzX,BlckSzX,1);
	mean_kern<<<blockGrid1, threadBlock1, BlckSzX * BlckSzX * sizeof(float)>>>(vectors_d, Rows, Cols, means_d, BlocksX);
	cudaThreadSynchronize();

	dim3 blockGrid2(Cols / BlckSzX, Rows / BlckSzX, 1);
	dim3 threadBlock2(BlckSzX, BlckSzX, 1);
	var_kern<<< blockGrid2, threadBlock2, BlckSzX * BlckSzX * sizeof(float)>>>(vectors_d, Rows, Cols, means_d, BlocksX);
	cudaThreadSynchronize();

	//cudaMemcpy(means_h, means_d, Rows * sizeof(float), cudaMemcpyDeviceToHost);
	//float* master_mean = findMeans(vectors_h, Cols, Rows);
	//for (int i = 0; i < Rows; ++i) flag = flag && (master_mean[i] == means_h[i]);

	dim3 blockGrid3(Cols / BlckSzX, Rows / BlckSzY, 1);
	dim3 threadBlock3(BlckSzX, BlckSzY, 1);
	varcor<<<blockGrid3, threadBlock3>>>(vectors_d, Rows, Cols, out_vectors_d, slotSize);
	
	cudaThreadSynchronize();
	cudaMemcpy(out_vectors_h, out_vectors_d, BlocksX * slotSize * sizeof(float), cudaMemcpyDeviceToHost);
	std::string str(cudaGetErrorString(cudaGetLastError()));

	for (int i = 1; i < BlocksX; ++i) {
		for (int j = 0; j < slotSize; ++j) {
			out_vectors_h[j] += out_vectors_h[j + i * slotSize];
		}
	}
	for (int i = 0; i < slotSize; ++i) {
		out_vectors_h[i] /= (float)Rows;
	}

	float* out_vectors_h_master =  findCovVals(vectors_h, Cols, Rows);

	float minn = 0.0f;
	for (int i = 0; i < slotSize; ++i) {
		float ttt = std::abs(out_vectors_h[i] - out_vectors_h_master[i]);
		if (ttt > std::abs(minn)) minn = out_vectors_h[i] - out_vectors_h_master[i];
	}//*/

	std::cout << "Min diff = " << minn << std::endl;

	float ssum = 0.0;
	for (int i = 0; i < slotSize; ++i) {
		ssum += out_vectors_h_master[i];// out_vectors_h[i];
	}//*/

	cudaFree(vectors_d);
	cudaFree(means_d);
	cudaFree(out_vectors_d);
}

float* genVects(int col, int row) {
	float* out = new float[col * row];
	for (int i = 0; i < col * row; ++i) {
		out[i] = 1.0;// static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	}
	return out;
}

float* findMeans(float* vects, int col, int row) {
	float* means = new float[row];
	for (int i = 0; i < row; ++i) {
		float sum = 0.0f;
		float* subvects = &vects[col * i];
		for (int j = 0; j < col; ++j) {
			sum += subvects[j];
		}
		means[i] = 0;// sum / (float)col;
	}
	return means;
}

float* findCovVals(float* vects, int col, int row) {

	float* means = findMeans(vects, col, row);
	int sltsz = (row * (row + 1)) / 2;
	float* varmat = new float[sltsz];
	
	for (int i = 0; i < row; ++i) {
		int k = (i * (row + row + 1 - i)) / 2;
		for (int j = 0; (i + j) < row; ++j) {
			float sum = 0;
			for (int q = 0; q < col; ++q) {
				sum += (vects[j * col + q] - means[j]) * (vects[(j + i) * col + q] - means[j + i]);
			}
			varmat[k + j] = sum / (float)col;
		}
	}
	return varmat;
}