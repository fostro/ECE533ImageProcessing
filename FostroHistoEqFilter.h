#pragma once

#include "FostroFilter.h"

class FostroHistoEqFilter : public FostroFilter {
    public:
		int* histo;
		float* counts;
		float min_cdf;
		float scale;

		FostroHistoEqFilter(int width, int height);
		~FostroHistoEqFilter();

		FostroImage* applyFilter(FostroImage* image, int c);
		void createHisto(FostroImage* image, int c);
		void calcCounts();
};
