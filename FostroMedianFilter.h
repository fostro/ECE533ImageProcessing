#pragma once

#include "FostroFilter.h"

class FostroMedianFilter : public FostroFilter {
    public:
		FostroMedianFilter(int width, int height);
		~FostroMedianFilter();

		FostroImage* applyFilter(FostroImage* image, int c);
		int getMedianVal(unsigned long xImage, unsigned long yImage, FostroImage* image, int c);
};
