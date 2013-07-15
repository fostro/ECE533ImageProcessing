#pragma once

#include "FostroFilter.h"

class FostroInvertFilter : public FostroFilter {
    public:
		FostroInvertFilter(int width, int height);
		~FostroInvertFilter();

		FostroImage* applyFilter(FostroImage* image, int c);
};
