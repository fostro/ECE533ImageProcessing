#pragma once

#include "FostroFilterGPU.h"

class FostroInvertFilterGPU : public FostroFilterGPU {
    public:
		FostroPixel* d_pixels;
		FostroInvertFilterGPU();
		~FostroInvertFilterGPU();
		//void applyFilter(int width, int height, int c, FostroType* d_red, FostroType* d_green, FostroType* d_blue);
		void applyFilter(int width, int height, int c);
};
