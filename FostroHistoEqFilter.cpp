#include <cmath>
#include "FostroHistoEqFilter.h"

FostroHistoEqFilter::FostroHistoEqFilter(int width, int height) : FostroFilter(width, height) {
	
}

FostroHistoEqFilter::~FostroHistoEqFilter() {
	delete[] histo;
	delete[] counts;
}

FostroImage* FostroHistoEqFilter::applyFilter(FostroImage* image, int c) {
	FostroPixel* pixel;
	FostroImage* newImage = new FostroImage(*image, "tmpHisto.png");
	unsigned long lookupVal;
	float val;
	unsigned long iHeight = image->getHeight();
	unsigned long iWidth = image->getWidth();
	scale = MAX_PIXEL_VAL/(float)(iHeight*iWidth);

	createHisto(image, c);
	calcCounts();

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

			val = counts[lookupVal];
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

void FostroHistoEqFilter::createHisto(FostroImage* image, int c) {

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

void FostroHistoEqFilter::calcCounts() {
	counts = new float[MAX_PIXEL_VAL];

	counts[0] = histo[0];

	for (unsigned long i = 1; i < MAX_PIXEL_VAL; i++) {
		counts[i] = histo[i] + counts[i-1];
		counts[i-1] *= scale;
	}
}

