#include "FostroPixel.h"
#include "FostroDefines.h"

float FostroPixel::getRedVal() const {
	return red;
}

float FostroPixel::getGreenVal() const {
	return green;
}

float FostroPixel::getBlueVal() const {
	return blue;
}

float FostroPixel::getAlphaVal() const {
	return alpha;
}

float FostroPixel::getPixelVal(int c) const {
	float val;
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

void FostroPixel::setRedVal(float x) {
	red = x;
}

void FostroPixel::setGreenVal(float x) {
	green = x;
}

void FostroPixel::setBlueVal(float x) {
	blue = x;
}

void FostroPixel::setAlphaVal(float x) {
	alpha = x;
}

void FostroPixel::setPixelVal(float val, int c) {
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

float FostroPixel::getGrayscaleIntensity() {
	return grayscaleIntensity;
}
