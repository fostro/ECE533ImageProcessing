#include "FostroInvertFilter.h"

FostroInvertFilter::FostroInvertFilter(int width, int height) : FostroFilter(width, height) {
	
}

FostroInvertFilter::~FostroInvertFilter() {
}

FostroImage* FostroInvertFilter::applyFilter(FostroImage* image, int c) {
	FostroPixel* pixel;
	FostroImage* newImage = new FostroImage(*image, "tmpInvert.png");

	for (unsigned long x = 0; x < image->getHeight(); x++) {
		for (unsigned long y = 0; y < image->getWidth(); y++) {
			pixel = newImage->getPixel(x,y);
			pixel->invertRed();
			pixel->invertGreen();
			pixel->invertBlue();
		}
	}
	return newImage;
};
