#include <cmath>
#include "FostroSobelHorizontalFilter.h"

FostroSobelHorizontalFilter::FostroSobelHorizontalFilter(int width, int height) : FostroFilter(width, height) {
	// this expects a 3x3 and needs to be changed to allow for different sized matrices
	float horizVals[width*height];
	horizVals[0] = -1;
	horizVals[1] = -2;
	horizVals[2] = -1;
	horizVals[3] = 0;
	horizVals[4] = 0;
	horizVals[5] = 0;
	horizVals[6] = 1;
	horizVals[7] = 2;
	horizVals[8] = 1;
	
	setAllMaskVals(horizVals);
}

FostroSobelHorizontalFilter::~FostroSobelHorizontalFilter() {
}

FostroImage* FostroSobelHorizontalFilter::applyFilter(FostroImage* image, int c) {
	FostroPixel* pixel;
	float val = 0;

	FostroImage* newImage = new FostroImage(*image, "tmpSobelH.png");

	for (unsigned long x = 0; x < image->getHeight(); x++) {
		for (unsigned long y = 0; y < image->getWidth(); y++) {
	        val = 0;
			for (int i = 0; i < mask->getHeight(); i++) {
				for (int j = 0; j < mask->getWidth(); j++) {
					//val += pixel->getGrayscaleIntensity() * mask->getMaskVal(i,j);
					val += mask->getImageValAtMaskLoc(i,j,(int)x,(int)y,image,c);
				}
			}

			val = abs(val);

            pixel = newImage->getPixel(x,y);

			switch (c) {
			case RED:
				pixel->setRedVal(val);
				break;
			case GREEN:
				pixel->setGreenVal(val);
				break;
			case BLUE:
				pixel->setBlueVal(val);
				break;
			case ALPHA:
				pixel->setAlphaVal(val);
				break;
			case GRAY:
				pixel->setRedVal(val);
				pixel->setGreenVal(val);
				pixel->setBlueVal(val);
				pixel->calcGrayscaleIntensity();
				break;
			}
		}
	}

	return newImage;
}
