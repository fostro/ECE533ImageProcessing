#pragma once

#include "FostroImage.h"
#include "FostroMask.h"
#include "FostroPixel.h"

class FostroFilter {
    public:
        FostroMask* mask;
		
		FostroFilter(int width, int height);
		FostroFilter(const FostroFilter& other);
		FostroFilter& operator=(const FostroFilter& other);
		virtual ~FostroFilter();
        
		virtual FostroImage* applyFilter(FostroImage* image, int c);
		void setAllMaskValsSame(float val);
		void setAllMaskVals(float* vals);
};


