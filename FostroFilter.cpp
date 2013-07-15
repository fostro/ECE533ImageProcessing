#include "FostroImage.h"
#include "FostroMask.h"
#include "FostroPixel.h"
#include "FostroFilter.h"

FostroFilter::FostroFilter(int width, int height) {
    mask = new FostroMask(width, height);
}

FostroFilter::FostroFilter(const FostroFilter& other) {
	mask = new FostroMask(*other.mask);
}

FostroFilter& FostroFilter::operator=(const FostroFilter& other) {
	delete mask;
	mask = new FostroMask(*other.mask);

	return *this;
}

FostroFilter::~FostroFilter() {
	delete mask;
} 

FostroImage* FostroFilter::applyFilter(FostroImage* image, int c) {
	std::cout << "This function should never be called, something went wrong in applyFilter." << std::endl;
	return NULL;
}

void FostroFilter::setAllMaskValsSame(float val) {
	mask->setAllMaskVals(val);
}

void FostroFilter::setAllMaskVals(float* vals) {
	int mWidth = mask->getWidth();
	int mHeight = mask->getHeight();

	for (int i = 0; i < mHeight; i++) {
		for (int j = 0; j < mWidth; j++) {
			mask->setMaskVal(i, j, vals[i*mWidth+j]);
		}
	}
}
