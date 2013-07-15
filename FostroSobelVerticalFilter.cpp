#include <cmath>
#include "FostroSobelVerticalFilter.h"

#define abs(x) (x>0?x:-x)

FostroSobelVerticalFilter::FostroSobelVerticalFilter(int width, int height) : FostroFilter(width, height) {

	// this expects a 3x3 and needs to be changed to allow for different sized matrices
	float vertVals[width*height];
	vertVals[0] = -1;
	vertVals[1] = 0;
	vertVals[2] = 1;
	vertVals[3] = -2;
	vertVals[4] = 0;
	vertVals[5] = 2;
	vertVals[6] = -1;
	vertVals[7] = 0;
	vertVals[8] = 1;
	
	setAllMaskVals(vertVals);
}

FostroSobelVerticalFilter::~FostroSobelVerticalFilter() {
}

FostroImage* FostroSobelVerticalFilter::applyFilter(FostroImage* image, int c) {
	FostroPixel* pixel;
	float val = 0;

	FostroImage* newImage = new FostroImage(*image, "tmpSobelV.png");

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

