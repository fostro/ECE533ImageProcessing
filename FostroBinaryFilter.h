#pragma once

#include "FostroFilter.h"

class FostroBinaryFilter : public FostroFilter {
    public:
		float threshold;

		FostroBinaryFilter(int width, int height);
		~FostroBinaryFilter();

		FostroImage* applyFilter(FostroImage* image, int c);
		void setThreshold(float val);
};
