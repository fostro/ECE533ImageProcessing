#include "FostroInvertFilterGPU.h"

__global__ void applyInvertFilter_cu(int c, FostroType* d_red_input, FostroType* d_green_input, FostroType* d_blue_input, FostroType* d_red_output, FostroType* d_green_output, FostroType* d_blue_output, int nPixels, int nRows, int nCols);

void applyInvert(FostroGPU* gpuStruct, int c) {
	//dim3 gridSize(ceil(gpuStruct->nRows/32)+1, ceil(gpuStruct->nCols/32)+1);
	//dim3 blockSize(32,32);
	dim3 gridSize(ceil(gpuStruct->nRows/16)+1, ceil(gpuStruct->nCols/16)+1);
	dim3 blockSize(16,16);

	applyInvertFilter_cu<<<gridSize, blockSize>>>(c, gpuStruct->d_ir, gpuStruct->d_ig, gpuStruct->d_ib, gpuStruct->d_or, gpuStruct->d_og, gpuStruct->d_ob, gpuStruct->numPixels, gpuStruct->nRows, gpuStruct->nCols);

	cudaError_t error;

	if ((error = cudaThreadSynchronize()) != cudaSuccess) {
		printf("ApplyFilter Error: %s\n", cudaGetErrorString(error));
		exit(1);
        return;
	}
}

__global__ void applyInvertFilter_cu(int c, FostroType* d_red_input, FostroType* d_green_input, FostroType* d_blue_input, FostroType* d_red_output, FostroType* d_green_output, FostroType* d_blue_output, int nPixels, int nRows, int nCols) {

	int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	int idxY = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = idxX * nCols + idxY;

	if (idxX >= nRows || idxY >= nCols || idx < 0) {
		return;
	}

	switch (c) {
	case RED:
		d_red_output[idx] = MAX_PIXEL_VAL - d_red_input[idx];
		break;
	case GREEN:
		d_green_output[idx] = MAX_PIXEL_VAL - d_green_input[idx];
		break;
	case BLUE:
		d_blue_output[idx] = MAX_PIXEL_VAL - d_blue_input[idx];
		break;
	case GRAY:
		d_red_output[idx] = MAX_PIXEL_VAL - d_red_input[idx];
		d_green_output[idx] = MAX_PIXEL_VAL - d_green_input[idx];
		d_blue_output[idx] = MAX_PIXEL_VAL - d_blue_input[idx];
//		printf("%f\t%f\t%f\n", d_red_output[idx], d_green_output[idx], d_blue_output[idx]);
		break;
	}

}
