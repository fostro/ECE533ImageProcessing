#include "FostroInvertFilterGPU.h"

__global__ void applyInvertFilter_cu(int c, FostroType* d_red_input, FostroType* d_green_input, FostroType* d_blue_input, FostroType* d_red_output, FostroType* d_green_output, FostroType* d_blue_output, int nPixels, int nRows, int nCols);

void applyInvert(FostroGPU* gpuStruct, int c) {
	dim3 gridSize(ceil(gpuStruct->nCols/256)+1);
	dim3 blockSize(256);

	applyInvertFilter_cu<<<gridSize, blockSize>>>(c, gpuStruct->d_ir, gpuStruct->d_ig, gpuStruct->d_ib, gpuStruct->d_or, gpuStruct->d_og, gpuStruct->d_ob, gpuStruct->numPixels, gpuStruct->nRows, gpuStruct->nCols);

	cudaError_t error;

	if ((error = cudaThreadSynchronize()) != cudaSuccess) {
		printf("ApplyFilter Error: %s\n", cudaGetErrorString(error));
		exit(1);
        return;
	}
}

__global__ void applyInvertFilter_cu(int c, FostroType* d_red_input, FostroType* d_green_input, FostroType* d_blue_input, FostroType* d_red_output, FostroType* d_green_output, FostroType* d_blue_output, int nPixels, int nRows, int nCols) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= nCols || idx < 0) {
		return;
	}

	for (int i = 0; i < nRows; ++i) {
		switch (c) {
		case RED:
			d_red_output[idx*nRows+i] = MAX_PIXEL_VAL - d_red_input[idx*nRows+i];
			break;
		case GREEN:
			d_green_output[idx*nRows+i] = MAX_PIXEL_VAL - d_green_input[idx*nRows+i];
			break;
		case BLUE:
			d_blue_output[idx*nRows+i] = MAX_PIXEL_VAL - d_blue_input[idx*nRows+i];
			break;
		case GRAY:
			d_red_output[idx*nRows+i] = MAX_PIXEL_VAL - d_red_input[idx*nRows+i];
			d_green_output[idx*nRows+i] = MAX_PIXEL_VAL - d_green_input[idx*nRows+i];
			d_blue_output[idx*nRows+i] = MAX_PIXEL_VAL - d_blue_input[idx*nRows+i];
			break;
		}
	}
}
