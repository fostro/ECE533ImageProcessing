#pragma once

#include "FostroFilter.h"

class FostroSobelVerticalFilter : public FostroFilter {
    public:
		FostroSobelVerticalFilter(int width, int height);
		~FostroSobelVerticalFilter();

		FostroImage* applyFilter(FostroImage* image, int c);
};
