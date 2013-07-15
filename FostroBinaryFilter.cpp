#include "FostroBinaryFilter.h"

FostroBinaryFilter::FostroBinaryFilter(int width, int height) : FostroFilter(width, height) {

}

FostroBinaryFilter::~FostroBinaryFilter() {
}

FostroImage* FostroBinaryFilter::applyFilter(FostroImage* image, int c) {
	FostroPixel* pixel;
	float val = 0;
	float checkVal = 0;

	FostroImage* newImage = new FostroImage(*image, "tmpThreshold.png");

	for (unsigned long x = 0; x < image->getHeight(); x++) {
		for (unsigned long y = 0; y < image->getWidth(); y++) {
			
            pixel = newImage->getPixel(x,y);

			switch (c) {
			case RED:
				checkVal = pixel->getRedVal();
				break;
			case GREEN:
				checkVal = pixel->getGreenVal();
				break;
			case BLUE:
				checkVal = pixel->getBlueVal();
				break;
			case GRAY:
				checkVal = pixel->getGrayscaleIntensity();
				break;
			}

			if (checkVal >= threshold) {
				val = MAX_PIXEL_VAL;
			} else {
				val = 0;
			}

			pixel->setRedVal(val);
			pixel->setGreenVal(val);
			pixel->setBlueVal(val);
			pixel->calcGrayscaleIntensity();
		}
	}

	return newImage;
}

void FostroBinaryFilter::setThreshold(float val) {
	threshold = val;
}
