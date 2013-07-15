#pragma once

#include "FostroFilter.h"

class FostroSmoothingFilter : public FostroFilter {
    public:
		FostroSmoothingFilter(int width, int height);
		~FostroSmoothingFilter();

		FostroImage* applyFilter(FostroImage* image, int c);
};
