#pragma once

#include "FostroImage.h"

class FostroMask {
	public:
		int width;
		int height;
		FostroType* mask;
        bool allSame;
		
		FostroMask(int w, int h);
		FostroMask(const FostroMask& other);
		FostroMask& operator=(const FostroMask& other);
		~FostroMask();
		
		int getWidth() const;
        int getHeight() const;
		FostroType getMaskVal(int, int) const;

		void setMaskVal(int, int, FostroType);
		void setAllMaskVals(FostroType);
		bool allSameVals();
        void checkAllSameVals();
 
		int calcWidthOffset(int xImage, int xMask);
		int calcHeightOffset(int xImage, int xMask);
		FostroType getImageValAtMaskLoc(int xMask, int yMask, int xImage, int yImage, FostroImage* image, int c);
};
