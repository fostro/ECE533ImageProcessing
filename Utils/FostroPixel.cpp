#include "FostroPixel.h"
#include "FostroDefines.h"

FostroType FostroPixel::getRedVal() const {
	return red;
}

FostroType FostroPixel::getGreenVal() const {
	return green;
}

FostroType FostroPixel::getBlueVal() const {
	return blue;
}

FostroType FostroPixel::getAlphaVal() const {
	return alpha;
}

FostroType FostroPixel::getPixelVal(int c) const {
	FostroType val;
	switch (c) {
	case RED:
		val = red;;
		break;
	case GREEN:
		val = green;
		break;
	case BLUE:
		val = blue;
		break;
	case ALPHA:
		val = alpha;
		break;
	case GRAY:
		val = grayscaleIntensity;
		break;
	}
	return val;
}

void FostroPixel::setRedVal(FostroType x) {
	red = x;
}

void FostroPixel::setGreenVal(FostroType x) {
	green = x;
}

void FostroPixel::setBlueVal(FostroType x) {
	blue = x;
}

void FostroPixel::setAlphaVal(FostroType x) {
	alpha = x;
}

void FostroPixel::setPixelVal(FostroType val, int c) {
	switch (c) {
	case RED:
		red = val;
		break;
	case GREEN:
		green = val;
		break;
	case BLUE:
		blue = val;
		break;
	case ALPHA:
		alpha = val;
		break;
	case GRAY:
		grayscaleIntensity = val;
		break;
	}
}

void FostroPixel::invertRed() {
	red = MAX_PIXEL_VAL - red;
}

void FostroPixel::invertGreen() {
	green = MAX_PIXEL_VAL - green;
}

void FostroPixel::invertBlue() {
	blue = MAX_PIXEL_VAL - blue;
}

void FostroPixel::invertAlpha() {
	alpha = MAX_PIXEL_VAL - alpha;
}

void FostroPixel::calcGrayscaleIntensity() {
	grayscaleIntensity = (red+blue+green)/3;
}

FostroType FostroPixel::getGrayscaleIntensity() {
	return grayscaleIntensity;
}
