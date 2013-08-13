#pragma once

#include "FostroFilterGPU.h"

class FostroSmoothFilterGPU : public FostroFilterGPU {
    public:
		FostroPixel* d_pixels;
		FostroSmoothFilterGPU();
		~FostroSmoothFilterGPU();
		void applyFilter(int width, int height, int c);
};
