#pragma once

#include "FostroFilter.h"

class FostroErosionFilter : public FostroFilter {
    public:
		float threshold;

		FostroErosionFilter(int width, int height);
		~FostroErosionFilter();

		FostroImage* applyFilter(FostroImage* image, int c);
		float getVal(int x, int y, FostroImage* image);
};
