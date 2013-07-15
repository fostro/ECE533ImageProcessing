#pragma once

#include "FostroImage.h"

class FostroMask {
	public:
		int width;
		int height;
		float* mask;
        bool allSame;
		
		FostroMask(int w, int h);
		FostroMask(const FostroMask& other);
		FostroMask& operator=(const FostroMask& other);
		~FostroMask();
		
		int getWidth() const;
        int getHeight() const;
		float getMaskVal(int, int) const;

		void setMaskVal(int, int, float);
		void setAllMaskVals(float);
		bool allSameVals();
        void checkAllSameVals();
 
		int calcWidthOffset(int xImage, int xMask);
		int calcHeightOffset(int xImage, int xMask);
		float getImageValAtMaskLoc(int xMask, int yMask, int xImage, int yImage, FostroImage* image, int c);
};
