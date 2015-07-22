#include <iostream>
#include <wand/MagickWand.h>
#include <vector>
#include <string>
#include <algorithm>
#include "FostroImage.h"

FostroImage::FostroImage(std::string imagePath, std::string outputPath) {
	PixelWand** tpixels;
	MagickWand* image_wand;
	PixelIterator* iterator;
	MagickBooleanType status;
	MagickPixelPacket pixel;
	register long x;
	unsigned long y;

	// Read in image
	MagickWandGenesis();
	image_wand=NewMagickWand();
	status=MagickReadImage(image_wand, imagePath.c_str());

    filePath = (char*)outputPath.c_str();
	
	if (status == MagickFalse) {
		ThrowWandException(image_wand);
	}

	// loop over pixels and fill in the array
	iterator=NewPixelIterator(image_wand);

	height = (unsigned long) MagickGetImageHeight(image_wand);
	width = (unsigned long) MagickGetImageWidth(image_wand);

	pixels = new FostroPixel[width*height];

	if (iterator == (PixelIterator *) NULL)
		ThrowWandException(image_wand);
	for (y=0; y < (unsigned long) MagickGetImageHeight(image_wand); y++)
	{
		tpixels=PixelGetNextIteratorRow(iterator,&width);
		if (tpixels == (PixelWand **) NULL)
			break;
		for (x=0; x < (long) width; x++)
		{
			pixels[y*width+x].setRedVal((float)PixelGetRed(tpixels[x]));
			pixels[y*width+x].setGreenVal((float)PixelGetGreen(tpixels[x]));
			pixels[y*width+x].setBlueVal((float)PixelGetBlue(tpixels[x]));
			pixels[y*width+x].calcGrayscaleIntensity();
		}
	}
	if (y < (unsigned long) MagickGetImageHeight(image_wand))
		ThrowWandException(image_wand);

	iterator=DestroyPixelIterator(iterator);
	image_wand=DestroyMagickWand(image_wand);
	MagickWandTerminus();
}

FostroImage::FostroImage(const FostroImage& other, std::string f) {
	height = other.getHeight();
	width = other.getWidth();

	filePath = (char*)f.c_str();
    
	pixels = new FostroPixel[width*height];

    for (unsigned long x = 0; x < height; x++) {
        for (unsigned long y = 0; y < width; y++) {
            pixels[x*width+y].setRedVal(other.pixels[x*width+y].red);
            pixels[x*width+y].setGreenVal(other.pixels[x*width+y].green);
            pixels[x*width+y].setBlueVal(other.pixels[x*width+y].blue);
            pixels[x*width+y].calcGrayscaleIntensity();
        }
	}
}

FostroImage& FostroImage::operator=(const FostroImage& other) {
	if (this != NULL) {
		delete[] pixels;
	}

	height = other.getHeight();
	width = other.getWidth();
    
	pixels = new FostroPixel[width*height];

    for (unsigned long x = 0; x < height; x++) {
        for (unsigned long y = 0; y < width; y++) {
            pixels[x*width+y].setRedVal(other.pixels[x*width+y].red);
            pixels[x*width+y].setGreenVal(other.pixels[x*width+y].green);
            pixels[x*width+y].setBlueVal(other.pixels[x*width+y].blue);
            pixels[x*width+y].calcGrayscaleIntensity();
        }
	}
	
	return *this;
}

FostroImage FostroImage::operator+(const FostroImage& image1) {
	bool hflag = false;
	bool wflag = false;

	if (image1.getHeight() != this->getHeight()) {
		std::cout << "Cannot Add Images, different heights" << std::endl;
		hflag = true;
	}

	if (image1.getWidth() != this->getWidth()) {
		std::cout << "Cannot Add Images, different widths" << std::endl;
		wflag = true;
	}

	if (hflag == true || wflag == true) {
		// this should never happen
		exit(-1);
	}

	FostroImage newImage(image1);
	unsigned long width = image1.getWidth();
	unsigned long height = image1.getHeight();
	FostroPixel* newPixel; 
	const FostroPixel* pixel1;
	const FostroPixel* pixel2;

    for (unsigned long x = 0; x < height; x++) {
        for (unsigned long y = 0; y < width; y++) {
			newPixel = newImage.getPixel(x,y);
			pixel1 = image1.getConstPixel(x,y);
			pixel2 = this->getConstPixel(x,y);

			newPixel->setRedVal(pixel1->getRedVal() + pixel2->getRedVal()); 
			newPixel->setGreenVal(pixel1->getGreenVal() + pixel2->getGreenVal()); 
			newPixel->setBlueVal(pixel1->getBlueVal() + pixel2->getBlueVal()); 
			newPixel->calcGrayscaleIntensity(); 
        }
	}

	return newImage;
}

FostroImage FostroImage::operator-(const FostroImage& image1) {
	bool hflag = false;
	bool wflag = false;

	if (image1.getHeight() != this->getHeight()) {
		std::cout << "Cannot Add Images, different heights" << std::endl;
		hflag = true;
	}

	if (image1.getWidth() != this->getWidth()) {
		std::cout << "Cannot Add Images, different widths" << std::endl;
		wflag = true;
	}

	if (hflag == true || wflag == true) {
		// this should never happen
		exit(-1);
	}

	FostroImage newImage(image1);
	unsigned long width = image1.getWidth();
	unsigned long height = image1.getHeight();
	FostroPixel* newPixel; 
	const FostroPixel* pixel1;
	const FostroPixel* pixel2;

    for (unsigned long x = 0; x < height; x++) {
        for (unsigned long y = 0; y < width; y++) {
			newPixel = newImage.getPixel(x,y);
			pixel1 = image1.getConstPixel(x,y);
			pixel2 = this->getConstPixel(x,y);

			newPixel->setRedVal(pixel1->getRedVal() - pixel2->getRedVal()); 
			newPixel->setGreenVal(pixel1->getGreenVal() - pixel2->getGreenVal()); 
			newPixel->setBlueVal(pixel1->getBlueVal() - pixel2->getBlueVal()); 
			newPixel->calcGrayscaleIntensity(); 
        }
	}

	return newImage;
}


// Don't Call This
FostroImage::FostroImage() {
	std::cout << "Never call a FostroImage constructor without arguments, bad things will happen..." << std::endl;
	exit(-1);
}

FostroImage::~FostroImage() {
	delete[] pixels; 
}

void FostroImage::saveImage() {
	MagickPixelPacket pixel;
	MagickWand* image_wand;
	PixelIterator* iterator;
	PixelWand** tpixels;
	PixelWand* pwand;
	MagickBooleanType status;
	long y;
	register long x;

	MagickWandGenesis();

	pwand = NewPixelWand();
	PixelSetColor(pwand, "white");
	image_wand = NewMagickWand();
	if (image_wand == (MagickWand *) NULL) {
		ThrowWandException(image_wand);
	}

    status = MagickNewImage(image_wand, width, height, pwand);

	iterator=NewPixelIterator(image_wand);
	if (iterator == (PixelIterator *) NULL) { 
		ThrowWandException(image_wand);
	}
	for (y=0; y < (long) MagickGetImageHeight(image_wand); y++)
	{
		tpixels=PixelGetNextIteratorRow(iterator,&width);
		if (tpixels == (PixelWand **) NULL) {
			break;
		}
		for (x=0; x < (long) width; x++)
		{
			PixelSetRed(tpixels[x], (double)pixels[y*width+x].getRedVal());
			PixelSetGreen(tpixels[x], (double)pixels[y*width+x].getGreenVal());
			PixelSetBlue(tpixels[x], (double)pixels[y*width+x].getBlueVal());
		}
		(void) PixelSyncIterator(iterator);
	}
	if (y < (long) MagickGetImageHeight(image_wand)) {
		ThrowWandException(image_wand);
	}
	iterator=DestroyPixelIterator(iterator);

	status=MagickWriteImage(image_wand,filePath);

	if (status == MagickFalse) {
		ThrowWandException(image_wand);
	}
	image_wand=DestroyMagickWand(image_wand);
	MagickWandTerminus();
}

FostroPixel* FostroImage::getPixel(unsigned long x, unsigned long y) {
	if (x >= height)
		x = height-1;
	else if (x < 0)
		x = 0;
	if (y >= width)
		y = width-1;
	else if (y < 0)
		y = 0;

	return pixels + x*width+y;
}

FostroPixel const* FostroImage::getConstPixel(unsigned long x, unsigned long y) const {
	if (x >= height)
		x = height-1;
	else if (x < 0)
		x = 0;
	if (y >= width)
		y = width-1;
	else if (y < 0)
		y = 0;

	return pixels + x*width+y;
}

unsigned long FostroImage::getWidth() const {
	return width;
}

unsigned long FostroImage::getHeight() const {
	return height;
}

char* FostroImage::getFilePath() {
    return filePath;
}

void FostroImage::setFilePath(char* f) {
    filePath = f;
}

