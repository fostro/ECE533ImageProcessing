#pragma once

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "../Utils/FostroImage.h"
#include "../Utils/FostroPixel.h"

class FostroFilterGPU {
	public:
		FostroType* h_red;
		FostroType* h_green;
		FostroType* h_blue;

		FostroGPU* gpuStruct;

		cudaError_t error;
		FostroFilterGPU();
		~FostroFilterGPU();
		void setupFilter(FostroImage* image, int filterWidth);
		void cleanupFilter(FostroImage* image);
};
