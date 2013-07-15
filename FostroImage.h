#pragma once

#include <iostream>
#include <wand/MagickWand.h>
#include <string>
#include "FostroPixel.h"
#include "FostroImage.h"
#include "FostroDefines.h"

class FostroImage {
	public:
	
		unsigned long  width;
		unsigned long  height;
        
        char* filePath;

		FostroPixel* pixels;

		FostroImage(std::string imagePath, std::string filePath);
        FostroImage();
		FostroImage(const FostroImage& other, std::string filePath);
		FostroImage& operator=(const FostroImage& other);
		FostroImage operator+(const FostroImage& image1);
		FostroImage operator-(const FostroImage& image1);
		~FostroImage();
		void saveImage();

		FostroPixel* getPixel(unsigned long x, unsigned long y);
		FostroPixel const* getConstPixel(unsigned long x, unsigned long y) const;

		unsigned long getWidth() const;
		unsigned long getHeight() const;
		char* getFilePath();
		void setFilePath(char* f);
};
