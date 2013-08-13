#include "FostroSmoothFilterGPU.h"

__global__ void applySmoothFilter_cu(int c, FostroType* d_red_input, FostroType* d_green_input, FostroType* d_blue_input, FostroType* d_red_output, FostroType* d_green_output, FostroType* d_blue_output, int nRows, int nCols, long nPixels, FostroType* filter, int filterWidth);

void applySmooth(FostroGPU* gpuStruct, int c) {
//	dim3 gridSize(ceil(gpuStruct->nRows/32)+1, ceil(gpuStruct->nCols/32)+1);
//	dim3 blockSize(32,32);
	dim3 gridSize(ceil(gpuStruct->nRows/16)+1, ceil(gpuStruct->nCols/16)+1);
	dim3 blockSize(16,16);

	applySmoothFilter_cu<<<gridSize, blockSize>>>(c, gpuStruct->d_ir, gpuStruct->d_ig, gpuStruct->d_ib, gpuStruct->d_or, gpuStruct->d_og, gpuStruct->d_ob, gpuStruct->nRows, gpuStruct->nCols, gpuStruct->numPixels, gpuStruct->filter, gpuStruct->filterWidth);

	cudaError_t error;

	if ((error = cudaThreadSynchronize()) != cudaSuccess) {
		printf("ApplyFilter Error: %s\n", cudaGetErrorString(error));
		exit(1);
        return;
	}
}

__global__ void applySmoothFilter_cu(int c, FostroType* d_red_input, FostroType* d_green_input, FostroType* d_blue_input, FostroType* d_red_output, FostroType* d_green_output, FostroType* d_blue_output, int nRows, int nCols, long nPixels, FostroType* filter, int filterWidth) {

	int idxX = blockIdx.x * blockDim.x + threadIdx.x;
	int idxY = blockIdx.y * blockDim.y + threadIdx.y;

	int idx = idxX * nCols + idxY;

	if (idxX >= nRows || idxY >= nCols || idx < 0) {
		return;
	}

	FostroType tmp_red = 0;
	FostroType tmp_green = 0;
	FostroType tmp_blue = 0;

	for (int filter_r = -filterWidth/2; filter_r <= filterWidth/2; ++filter_r) {
		for (int filter_c = -filterWidth/2; filter_c <= filterWidth/2; ++filter_c) {
			int image_r = min(max(idxX + filter_r, 0), nRows - 1);
			int image_c = min(max(idxY + filter_c, 0), nCols - 1);
            
			FostroType image_val_red = d_red_input[image_r * nCols + image_c];
			FostroType image_val_green = d_green_input[image_r * nCols + image_c];
			FostroType image_val_blue = d_blue_input[image_r * nCols + image_c];

			FostroType filter_val = filter[(filter_r + filterWidth/2) * filterWidth + filter_c + filterWidth/2];
           
			tmp_red += image_val_red * filter_val;
			tmp_green += image_val_green * filter_val;
			tmp_blue += image_val_blue * filter_val;
		}	
	}

	d_red_output[idx] = tmp_red;
	d_green_output[idx] = tmp_green;
	d_blue_output[idx] = tmp_blue;
}
