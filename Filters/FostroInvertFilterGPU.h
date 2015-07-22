#pragma once

#include "FostroFilterGPU.h"

class FostroInvertFilterGPU : public FostroFilterGPU {
    public:
		FostroPixel* d_pixels;
		FostroInvertFilterGPU();
		~FostroInvertFilterGPU();
		void applyFilter(int width, int height, int c);
};
