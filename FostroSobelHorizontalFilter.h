#pragma once

#include "FostroFilter.h"

class FostroSobelHorizontalFilter : public FostroFilter {
    public:
		FostroSobelHorizontalFilter(int width, int height);
		~FostroSobelHorizontalFilter();

		FostroImage* applyFilter(FostroImage* image, int c);
};
