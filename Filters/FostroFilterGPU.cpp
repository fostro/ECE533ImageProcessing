#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "../Utils/FostroImage.h"
#include "../Utils/FostroPixel.h"
#include "../Filters/FostroFilterGPU.h"

FostroFilterGPU::FostroFilterGPU() {
	gpuStruct = new FostroGPU;
}

FostroFilterGPU::~FostroFilterGPU() {
	if (h_red != NULL) {
		delete[] h_red;
	}

	if (h_green != NULL) {
		delete[] h_green;
	}

	if (h_blue != NULL) {
		delete[] h_blue;
	}

	if (gpuStruct != NULL) {
		delete gpuStruct;
	}
}

void FostroFilterGPU::setupFilter(FostroImage* image, int filterWidth) {
	int width = image->getWidth();
	int height = image->getHeight();

	gpuStruct->numPixels = width*height;
	gpuStruct->nCols = width;
	gpuStruct->nRows = height;

	int fullSize = filterWidth*filterWidth;
	gpuStruct->filterWidth = filterWidth;
	FostroType* filter = new FostroType[fullSize];
	for (int i = 0; i < fullSize; ++i) {
		filter[i] = 1./fullSize;
	}

	cudaSafe(cudaMalloc(&(gpuStruct->filter), width*height*sizeof(FostroType)), "Setup Malloc Filter"); 
	cudaSafe(cudaMemcpy(gpuStruct->filter, filter, fullSize*sizeof(FostroType), cudaMemcpyHostToDevice), "Setup Memcpy Filter");

	delete[] filter;

	h_red = new FostroType[width*height];
	h_green = new FostroType[width*height];
	h_blue = new FostroType[width*height];

	int x, y;
	FostroPixel* pixel;

	for (x = 0; x < height; ++x) {
		for (y = 0; y < width; ++y) {
			pixel = image->getPixel(x,y);
			h_red[x*width+y] = pixel->getRedVal();
			h_green[x*width+y] = pixel->getGreenVal();
			h_blue[x*width+y] = pixel->getBlueVal();
		}
	}

	cudaSafe(cudaMalloc(&(gpuStruct->d_ir), width*height*sizeof(FostroType)), "Setup Malloc Red Input"); 
	cudaSafe(cudaMalloc(&(gpuStruct->d_ig), width*height*sizeof(FostroType)), "Setup Malloc Green Input"); 
	cudaSafe(cudaMalloc(&(gpuStruct->d_ib), width*height*sizeof(FostroType)), "Setup Malloc Blue Input"); 

	cudaSafe(cudaMalloc(&(gpuStruct->d_or), width*height*sizeof(FostroType)), "Setup Malloc Red Output"); 
	cudaSafe(cudaMalloc(&(gpuStruct->d_og), width*height*sizeof(FostroType)), "Setup Malloc Green Output"); 
	cudaSafe(cudaMalloc(&(gpuStruct->d_ob), width*height*sizeof(FostroType)), "Setup Malloc Blue Output"); 

	cudaSafe(cudaMemcpy(gpuStruct->d_ir, h_red, width*height*sizeof(FostroType), cudaMemcpyHostToDevice), "Setup Memcpy Red");
	cudaSafe(cudaMemcpy(gpuStruct->d_ig, h_green, width*height*sizeof(FostroType), cudaMemcpyHostToDevice), "Setup Memcpy Green");
	cudaSafe(cudaMemcpy(gpuStruct->d_ib, h_blue, width*height*sizeof(FostroType), cudaMemcpyHostToDevice), "Setup Memcpy Blue");
}

void FostroFilterGPU::cleanupFilter(FostroImage* image) {
	int width = image->getWidth();
	int height = image->getHeight();

	cudaSafe(cudaMemcpy(h_red, gpuStruct->d_or, width*height*sizeof(FostroType), cudaMemcpyDeviceToHost), "Cleanup Memcpy Red");
	cudaSafe(cudaMemcpy(h_green, gpuStruct->d_og, width*height*sizeof(FostroType), cudaMemcpyDeviceToHost), "Cleanup Memcpy Green");
	cudaSafe(cudaMemcpy(h_blue, gpuStruct->d_ob, width*height*sizeof(FostroType), cudaMemcpyDeviceToHost), "Cleanup Memcpy Blue");

	int x, y;
	FostroPixel* pixel;

	for (x = 0; x < height; ++x) {
		for (y = 0; y < width; ++y) {
			pixel = image->getPixel(x,y);
			pixel->setRedVal(h_red[x*width+y]);
			pixel->setGreenVal(h_green[x*width+y]);
			pixel->setBlueVal(h_blue[x*width+y]);
		}
	}

	cudaSafe(cudaFree(gpuStruct->filter), "Cleanup Free Filter");

	cudaSafe(cudaFree(gpuStruct->d_ir), "Cleanup Free Red Input");
	cudaSafe(cudaFree(gpuStruct->d_ig), "Cleanup Free Green Input");
	cudaSafe(cudaFree(gpuStruct->d_ib), "Cleanup Free Blue Input");

	cudaSafe(cudaFree(gpuStruct->d_or), "Cleanup Free Red Output");
	cudaSafe(cudaFree(gpuStruct->d_og), "Cleanup Free Green Output");
	cudaSafe(cudaFree(gpuStruct->d_ob), "Cleanup Free Blue Output");

	delete[] h_red;
	delete[] h_green;
	delete[] h_blue;

	delete gpuStruct;
}
