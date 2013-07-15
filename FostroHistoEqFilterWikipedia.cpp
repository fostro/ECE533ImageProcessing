#include <cmath>
#include "FostroHistoEqFilterWikipedia.h"

FostroHistoEqFilterWikipedia::FostroHistoEqFilterWikipedia(int width, int height) : FostroFilter(width, height) {
	
}

FostroHistoEqFilterWikipedia::~FostroHistoEqFilterWikipedia() {
	delete[] histo;
	delete[] cdf;
}

FostroImage* FostroHistoEqFilterWikipedia::applyFilter(FostroImage* image, int c) {
	FostroPixel* pixel;
	FostroImage* newImage = new FostroImage(*image, "tmpHistoW.png");
	unsigned long lookupVal;
	float val;
	unsigned long iHeight = image->getHeight();
	unsigned long iWidth = image->getWidth();

	std::cout << "In applyFilter for histoEq" << std::endl;
	std::cout << "iHeight=" << iHeight << std::endl;
	std::cout << "iWidth=" << iWidth << std::endl;

	createHisto(image, c);
	calcCDF(iHeight*iWidth);

	for (unsigned long x = 0; x < iHeight; x++) {
		for (unsigned long y = 0; y < iWidth; y++) {
			pixel = image->getPixel(x,y);

			switch (c) {
			case RED:
				lookupVal = pixel->getRedVal();			// intentional truncation
				break;
			case GREEN:
				lookupVal = pixel->getGreenVal();		// intentional truncation
				break;
			case BLUE:
				lookupVal = pixel->getBlueVal();		// intentional truncation
				break;
			case ALPHA:
				lookupVal = pixel->getAlphaVal();		// intentional truncation
				break;
			case GRAY:
				lookupVal = pixel->getRedVal();			// intentional truncation
				lookupVal = pixel->getGreenVal();		// intentional truncation
				lookupVal = pixel->getBlueVal();		// intentional truncation
				pixel->calcGrayscaleIntensity();
				break;
			}

			val = round((cdf[(int)lookupVal] - min_cdf) / (iHeight*iWidth - min_cdf) * (MAX_PIXEL_VAL - 1));
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

void FostroHistoEqFilterWikipedia::createHisto(FostroImage* image, int c) {

	float val;
	FostroPixel* pixel;
	histo = new int[MAX_PIXEL_VAL];

	for (unsigned long i = 0; i < MAX_PIXEL_VAL; i++) {
		histo[i] = 0;
	}

	for (unsigned long x = 0; x < image->getHeight(); x++) {
		for (unsigned long y = 0; y < image->getHeight(); y++) {
			pixel = image->getPixel(x,y);
			switch (c) {
			case RED:
				val = pixel->getRedVal();
				break;
			case GREEN:
				val = pixel->getGreenVal();
				break;
			case BLUE:
				val = pixel->getBlueVal();
				break;
			case ALPHA:
				val = pixel->getAlphaVal();
				break;
			case GRAY:
				val = pixel->getGrayscaleIntensity();
				break;
			}

			histo[(int)val] += 1;
		}
	}
}

void FostroHistoEqFilterWikipedia::calcCDF(unsigned long imageSize) {
	cdf = new float[MAX_PIXEL_VAL];

	for (unsigned long i = 0; i < MAX_PIXEL_VAL; i++) {
		cdf[i] = 0;
	}

	min_cdf = MAX_PIXEL_VAL;

	for (unsigned long i = 0; i < MAX_PIXEL_VAL; i++) {
		for (unsigned long j = 0; j < i; j++) {
			cdf[i] += (float)histo[j];
		}
		if (cdf[i] < min_cdf && cdf[i] != 0) {
			min_cdf = cdf[i];
		}
	}
}

float FostroHistoEqFilterWikipedia::round(float x) {
	int y = x - (int)x;
	if (y >= 0.5f) {
		return (int)x+1;
	} else
		return (int)x;
}
