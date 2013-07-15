#pragma once

#include "FostroFilter.h"

class FostroDilationFilter : public FostroFilter {
    public:
		float threshold;

		FostroDilationFilter(int width, int height);
		~FostroDilationFilter();

		FostroImage* applyFilter(FostroImage* image, int c);
		float getVal(int x, int y, FostroImage* image);
};
