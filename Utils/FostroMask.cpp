#include "FostroMask.h"
#include "FostroPixel.h"
#include "FostroImage.h"
#include "FostroDefines.h"

FostroMask::FostroMask(int w, int h) {
	width = w;
    height = h;
	mask = new FostroType[width*height];
}

FostroMask::FostroMask(const FostroMask& other) {
	width = other.getWidth();
	height = other.getHeight();
	
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < height; j++) {
			setMaskVal(i, j, other.getMaskVal(i,j));
		}
	}
}

FostroMask& FostroMask::operator=(const FostroMask& other) {
	delete[] mask;
	
	width = other.getWidth();
	height = other.getHeight();
	
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < height; j++) {
			setMaskVal(i, j, other.getMaskVal(i,j));
		}
	}

	return *this;
}

FostroMask::~FostroMask() {
	delete[] mask;
}

int FostroMask::getWidth() const {
	return width;
}

int FostroMask::getHeight() const {
	return height;
}

FostroType FostroMask::getMaskVal(int x, int y) const {
	return mask[x*width+y];
}

void FostroMask::setMaskVal(int x, int y, FostroType val) {
	mask[x*width+y] = val;
}

void FostroMask::setAllMaskVals(FostroType val) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			setMaskVal(i,j,val);
		}
	}
}

bool FostroMask::allSameVals() {
    return allSame;
}

void FostroMask::checkAllSameVals() {
    FostroType val = mask[0];

    allSame = true;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (mask[i*width+j] != val) {
                allSame = false;
                break;
            }
        }
        if (!allSame) {
            break;
        }
    }
}

int FostroMask::calcWidthOffset(int xImage, int xMask) {
    int halfWidth = width/2;   // intentional truncation
    return (xImage - halfWidth + xMask);
}

int FostroMask::calcHeightOffset(int xImage, int xMask) {
    int halfHeight = height/2;   // intentional truncation
    return (xImage - halfHeight + xMask);
}


// returns mask value at mask location multiplied by value in
// image at mask location relatvie to image location
FostroType FostroMask::getImageValAtMaskLoc(int xMask, int yMask, int xImage, int yImage, FostroImage* image, int c) {
    FostroPixel* pixel;
    FostroType val = 0;

	int x = calcHeightOffset(xImage, xMask);
	int y = calcWidthOffset(yImage, yMask);

    //pixel = image->getPixel(calcWidthOffset(xImage, xMask), calcHeightOffset(yImage, yMask));
    pixel = image->getPixel(x, y);
    
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
	default:
		val = -1;
    }

	if (val < 0 || val > MAX_PIXEL_VAL) {
		std::cout << "Val is " << val << " something is wrong at pixel location " << x << "," << y << " around loc " << xImage << "," << yImage << std::endl;
	
		if (c == GRAY) {
			std::cout << "Gray Value is " << pixel->getGrayscaleIntensity() << " at " << x << "," << y << std::endl;
			std::cout << "Red Value is " << pixel->getRedVal() << " at " << x << "," << y << std::endl;
		}
		if (c == RED) {
			std::cout << "Red Value Selected" << std::endl;
		}
		if (c == GREEN) {
			std::cout << "Green Value Selected" << std::endl;
		}
		if (c == BLUE) {
			std::cout << "Blue Value Selected" << std::endl;
		}
	}

	if (val == -1) {
		std::cout << "Val is -1, something went wrong in the switch case" << std::endl;
	}

//	std::cout << "val in getImageValAtMaskLoc = " << val << " from location [" << x << "][" << y << "]" << std::endl;

    return val*getMaskVal(xMask, yMask);
} 
