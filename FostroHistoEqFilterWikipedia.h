#pragma once

#include "FostroFilter.h"

class FostroHistoEqFilterWikipedia : public FostroFilter {
    public:
		int* histo;
		float* cdf;
		float min_cdf;

		FostroHistoEqFilterWikipedia(int width, int height);
		~FostroHistoEqFilterWikipedia();

		FostroImage* applyFilter(FostroImage* image, int c);
		void createHisto(FostroImage* image, int c);
		void calcCDF(unsigned long imageSize);
		float round(float x);
};
