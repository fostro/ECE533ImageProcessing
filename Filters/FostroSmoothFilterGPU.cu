#include "FostroSmoothFilterGPU.h"

__global__ void applySmoothFilter_cu(int c, FostroType* d_red_input, FostroType* d_green_input, FostroType* d_blue_input, FostroType* d_red_output, FostroType* d_green_output, FostroType* d_blue_output, int nRows, int nCols, long nPixels, FostroType* filter, int filterWidth);

void applySmooth(FostroGPU* gpuStruct, int c) {
	dim3 gridSize(ceil(gpuStruct->nCols/256)+1);
	dim3 blockSize(256);

	applySmoothFilter_cu<<<gridSize, blockSize>>>(c, gpuStruct->d_ir, gpuStruct->d_ig, gpuStruct->d_ib, gpuStruct->d_or, gpuStruct->d_og, gpuStruct->d_ob, gpuStruct->nRows, gpuStruct->nCols, gpuStruct->numPixels, gpuStruct->filter, gpuStruct->filterWidth);

	cudaError_t error;

	if ((error = cudaThreadSynchronize()) != cudaSuccess) {
		printf("ApplyFilter Error: %s\n", cudaGetErrorString(error));
		exit(1);
        return; }
}

__global__ void applySmoothFilter_cu(int c, FostroType* d_red_input, FostroType* d_green_input, FostroType* d_blue_input, FostroType* d_red_output, FostroType* d_green_output, FostroType* d_blue_output, int nRows, int nCols, long nPixels, FostroType* filter, int filterWidth) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= nCols || idx < 0) {
		return;
	}

	for (int i = 0; i < nRows; ++i) {
		FostroType tmp_red = 0;
		FostroType tmp_green = 0;
		FostroType tmp_blue = 0;
	
		for (int filter_r = -filterWidth/2; filter_r <= filterWidth/2; ++filter_r) {
			for (int filter_c = -filterWidth/2; filter_c <= filterWidth/2; ++filter_c) {
				int image_r = min(max(i + filter_r, 0), nRows - 1);
				int image_c = min(max(idx + filter_c, 0), nCols - 1);
	            
				FostroType image_val_red = d_red_input[image_r * nCols + image_c];
				FostroType image_val_green = d_green_input[image_r * nCols + image_c];
				FostroType image_val_blue = d_blue_input[image_r * nCols + image_c];
	
				FostroType filter_val = filter[(filter_r + filterWidth/2) * filterWidth + filter_c + filterWidth/2];
	           
				tmp_red += image_val_red * filter_val;
				tmp_green += image_val_green * filter_val;
				tmp_blue += image_val_blue * filter_val;
			}	
		}
	
		d_red_output[i*nCols+idx] = tmp_red;
		d_green_output[i*nCols+idx] = tmp_green;
		d_blue_output[i*nCols+idx] = tmp_blue;
	}
}
