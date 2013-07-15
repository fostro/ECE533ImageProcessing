#include "FostroSmoothingFilter.h"

FostroSmoothingFilter::FostroSmoothingFilter(int width, int height) : FostroFilter(width, height) {
	
}

FostroSmoothingFilter::~FostroSmoothingFilter() {
}

FostroImage* FostroSmoothingFilter::applyFilter(FostroImage* image, int c) {
	FostroPixel* pixel;
	FostroImage* newImage = new FostroImage(*image, "tmpSmooth.png");

	for (unsigned long x = 0; x < image->getHeight(); x++) {
		for (unsigned long y = 0; y < image->getWidth(); y++) {
	        unsigned long val = 0;
			for (int i = 0; i < mask->getHeight(); i++) {
				for (int j = 0; j < mask->getWidth(); j++) {
					val += mask->getImageValAtMaskLoc(i,j,(int)x,(int)y,image,c);
				}
			}
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
};
