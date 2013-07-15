#include "FostroMedianFilter.h"
#include <algorithm>
#include <vector>

FostroMedianFilter::FostroMedianFilter(int width, int height) : FostroFilter(width, height) {
	
}

FostroMedianFilter::~FostroMedianFilter() {
}

FostroImage* FostroMedianFilter::applyFilter(FostroImage* image, int c) {
	FostroPixel* pixel;
	FostroImage* newImage = new FostroImage(*image, "tmpMedian.png");

	for (unsigned long x = 0; x < image->getHeight(); x++) {
		for (unsigned long y = 0; y < image->getWidth(); y++) {
			float val = 0;
			val = getMedianVal(x,y,image,c);
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

int FostroMedianFilter::getMedianVal(unsigned long xImage, unsigned long yImage, FostroImage* image, int c) {

	int mHeight = mask->getHeight();
	int mWidth = mask->getWidth();

	std::vector<int> vals;
	vals.reserve(mHeight*mWidth);

	for (int x = 0; x < mask->getHeight(); x++) {
		for (int y = 0; y < mask->getHeight(); y++) {
			vals.push_back(mask->getImageValAtMaskLoc(x,y,xImage,yImage,image,c));
		}
	}	

	std::sort(vals.begin(), vals.end());

	return vals[mHeight*mWidth/2];

}
