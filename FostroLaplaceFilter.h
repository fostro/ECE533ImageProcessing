#pragma once

#include "FostroFilter.h"

class FostroLaplaceFilter : public FostroFilter {
    public:
		FostroLaplaceFilter(int width, int height);
		~FostroLaplaceFilter();

		FostroImage* applyFilter(FostroImage* image, int c);
};
