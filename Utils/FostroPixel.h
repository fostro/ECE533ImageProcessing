#pragma once

#include "FostroDefines.h"

class FostroPixel {
	public:
		FostroType red;
		FostroType blue;
		FostroType green;
		FostroType alpha;
		FostroType grayscaleIntensity;

		FostroType getRedVal() const;
		FostroType getGreenVal() const;
		FostroType getBlueVal() const;
		FostroType getAlphaVal() const;
		FostroType getPixelVal(int c) const;
		
		void setRedVal(FostroType x);
		void setGreenVal(FostroType x);
		void setBlueVal(FostroType x);
		void setAlphaVal(FostroType x);
		void setPixelVal(FostroType val, int c);
		
		void invertRed();
		void invertGreen();
		void invertBlue();
		void invertAlpha();

		FostroType getGrayscaleIntensity();
		void calcGrayscaleIntensity();
};

