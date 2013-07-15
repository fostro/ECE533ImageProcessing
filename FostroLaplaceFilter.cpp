#include "FostroLaplaceFilter.h"

FostroLaplaceFilter::FostroLaplaceFilter(int width, int height) : FostroFilter(width, height) {

	// this expects a 3x3 and needs to be changed to allow for different sized matrices
	float vertVals[width*height];
	vertVals[0] = 0;
	vertVals[1] = 1;
	vertVals[2] = 0;
	vertVals[3] = 1;
	vertVals[4] = -4;
	vertVals[5] = 1;
	vertVals[6] = 0;
	vertVals[7] = 1;
	vertVals[8] = 0;
	
	setAllMaskVals(vertVals);
}

FostroLaplaceFilter::~FostroLaplaceFilter() {
}

FostroImage* FostroLaplaceFilter::applyFilter(FostroImage* image, int c) {
	FostroPixel* pixel;
	float val = 0;

	FostroImage* newImage = new FostroImage(*image, "tmpLaplace.png");

	for (unsigned long x = 0; x < image->getHeight(); x++) {
		for (unsigned long y = 0; y < image->getWidth(); y++) {
	        val = 0;
			for (int i = 0; i < mask->getHeight(); i++) {
				for (int j = 0; j < mask->getWidth(); j++) {
					val += mask->getImageValAtMaskLoc(i,j,(int)x,(int)y,image,c);
				}
			}

			val = 2*abs(val);

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
