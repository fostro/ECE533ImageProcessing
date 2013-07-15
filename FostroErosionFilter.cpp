#include "FostroErosionFilter.h"

FostroErosionFilter::FostroErosionFilter(int width, int height) : FostroFilter(width, height) {
}

FostroErosionFilter::~FostroErosionFilter() {
}

FostroImage* FostroErosionFilter::applyFilter(FostroImage* image, int c) {
	FostroPixel* pixel;
	float val = 0;

	FostroImage* newImage = new FostroImage(*image, "tmpSobelV.png");

	for (unsigned long x = 0; x < image->getHeight(); x++) {
		for (unsigned long y = 0; y < image->getWidth(); y++) {
			
			val = getVal(x, y, image);

            pixel = newImage->getPixel(x,y);

			pixel->setRedVal(val);
			pixel->setGreenVal(val);
			pixel->setBlueVal(val);
			pixel->calcGrayscaleIntensity();
		}
	}

	return newImage;
}

float FostroErosionFilter::getVal(int x, int y, FostroImage* image) {
	FostroPixel* pixel;

	for (int i = 0; i < mask->getHeight(); i++) {
		for (int j = 0; j < mask->getWidth(); j++) {
			pixel = image->getPixel(mask->calcHeightOffset(x, i), mask->calcWidthOffset(y, j));
			if (pixel->getGrayscaleIntensity() < MAX_PIXEL_VAL/2) {
				return 0;
			}
		}
	}

	return MAX_PIXEL_VAL;
}
