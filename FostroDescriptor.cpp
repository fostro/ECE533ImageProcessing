#include "FostroImage.h"
#include "FostroPixel.h"
#include "FostroErosionFilter.h"
#include "FostroDescriptor.h"

FostroDescriptor::FostroDescriptor() {
}

FostroDescriptor::FostroDescriptor(const FostroDescriptor& other) {
}

FostroDescriptor& FostroDescriptor::operator=(const FostroDescriptor& other) {
	return *this;
}

FostroDescriptor::~FostroDescriptor() {
} 
 
void FostroDescriptor::calcDescriptors(FostroImage* image, int c){
	area = calcArea(image);
	perimeter = calcPerimeter(image);
	calcPerimeter2OverArea(image);
	calcMomentInvariants(image);
}

float FostroDescriptor::calcArea(FostroImage* image){
	float area = 0;
	FostroPixel* pixel;

	for (unsigned long x = 0; x < image->getHeight(); ++x) {
		for (unsigned long y = 0; y < image->getWidth(); ++y) {
			pixel = image->getPixel(x,y);	
			if (pixel->getGrayscaleIntensity() > MAX_PIXEL_VAL/2) {
				++area;
			}
		}
	}

	return area;
}

float FostroDescriptor::calcPerimeter(FostroImage* image){
	FostroErosionFilter erosionFilter(3,3);
	FostroImage* tmpImage = erosionFilter.applyFilter(image, GRAY);
	tmpImage->setFilePath("erosionPerim.png");
	tmpImage->saveImage();

	float erodedArea = calcArea(tmpImage);

	return (area - erodedArea);
}

void FostroDescriptor::calcPerimeter2OverArea(FostroImage* image){
	perim2OverArea = (perimeter*perimeter)/area;
}

void FostroDescriptor::calcMomentInvariants(FostroImage* image) {
	
	float m10 = 0;
	float m01 = 0;
	float m00 = 0;
	float tmp;
	FostroPixel* pixel;

	for (unsigned long x = 0; x < image->getHeight(); ++x) {
		for (unsigned long y = 0; y < image->getWidth(); ++y) {
			pixel = image->getPixel(x,y);	
			tmp = pixel->getGrayscaleIntensity();
			if (tmp > MAX_PIXEL_VAL/2) {
				m00 += tmp;
				m01 += y*tmp;
				m10 += x*tmp;
			}
		}
	}

	float xcenter = m10/m00;
	float ycenter = m01/m00;

	float mu00 = m00;
	float mu02 = 0;
	float mu03 = 0;
	float mu11 = 0;
	float mu12 = 0;
	float mu20 = 0;
	float mu21 = 0;
	float mu30 = 0;

	float ydiff;
	float xdiff;

	for (unsigned long x = 0; x < image->getHeight(); ++x) {
		for (unsigned long y = 0; y < image->getWidth(); ++y) {
			pixel = image->getPixel(x,y);	
			tmp = pixel->getGrayscaleIntensity();
			if (tmp > MAX_PIXEL_VAL/2) {
				xdiff = x - xcenter;
				ydiff = y - ycenter;

				mu02 += ydiff*ydiff*tmp;
				mu03 += ydiff*ydiff*ydiff*tmp;
				mu11 += xdiff*ydiff*tmp;
				mu12 += xdiff*ydiff*ydiff*tmp;
				mu20 += xdiff*xdiff*tmp;
				mu21 += xdiff*xdiff*ydiff*tmp;
				mu30 += xdiff*xdiff*xdiff*tmp;
			}
		}
	}

	float mu00pow2 = mu00*mu00;
	float mu00pow2point5 = pow(mu00,2.5f);

	float n02 = mu02/mu00pow2;
	float n03 = mu03/mu00pow2point5;
	float n11 = mu11/mu00pow2;
	float n12 = mu12/mu00pow2point5;
	float n20 = mu02/mu00pow2;
	float n21 = mu21/mu00pow2point5;;
	float n30 = mu30/mu00pow2point5;

	momentInvariants[0] = n20 + n02;
	momentInvariants[1] = (n20 - n02)*(n20 - n02) + 4*n11*n11;
	momentInvariants[2] = (n30 - 3*n12)*(n30 - 3*n12) + (3*n21 - n03)*(3*n21 - n03);
	momentInvariants[3] = (n30 + n12)*(n30 + n12) + (n21 + n03)*(n21 + n03);
	momentInvariants[4] = (n30 - 3*n12)*(n30 + n12)*((n30 + n12)*(n30 + n12) - 3*(n21 + n03)*(n21 + n03)) + (3*n21 - n03)*(n21 + n03)*(3*(n30 + n12)*(n30 + n12) - (n21 + n03)*(n21 + n03)); 
	momentInvariants[5] = (n20 - n02)*((n30 + n12)*(n30 + n12) - (n21 + n03)*(n21 + n03)) + 4*n11*(n30 + n12)*(n21 + n03);
	momentInvariants[6] = (3*n21 - n03)*(n30 + n12)*((n30 + n12)*(n30 + n12) - 3*(n21 + n03)*(n21 + n03)) - (n30 - 3*n12)*(n21 + n03)*(3*(n30 + n12)*(n30 + n12) - (n21 + n03)*(n21 + n03));


	float matrix[2][2] = {{0}};
	
	for (unsigned long x = 0; x < image->getHeight(); ++x) {
		for (unsigned long y = 0; y < image->getWidth(); ++y) {
			pixel = image->getPixel(x,y);	
			tmp = pixel->getGrayscaleIntensity();
			if (tmp > MAX_PIXEL_VAL/2) {
				xdiff = x - xcenter;
				ydiff = y - ycenter;

				matrix[0][0] += xdiff*xdiff;
				matrix[0][1] += xdiff*ydiff;
				matrix[1][1] += ydiff*ydiff;
			}
		}
	}

	matrix[1][0] = matrix[0][1];

	float T = matrix[0][0] + matrix[1][1];
	float D = matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0];
	eigenVal1 = T/2 + sqrt(T*T/4 - D);
	eigenVal2 = T/2 - sqrt(T*T/4 - D);

	if (matrix[1][0] != 0) {
		eigenVect1[0] = eigenVal1 - matrix[1][1];
		eigenVect1[1] = matrix[1][0];
		eigenVect2[0] = eigenVal2 - matrix[1][1];
		eigenVect2[1] = matrix[1][0];
	} else {
		eigenVect1[0] = 1;
		eigenVect1[1] = 0;
		eigenVect2[0] = 0;
		eigenVect2[1] = 1;
	}

	if (eigenVal1 > eigenVal2) {
		majorAxis[0] = eigenVect1[0];
		majorAxis[1] = eigenVect1[1];
		minorAxis[0] = eigenVect2[0];
		minorAxis[1] = eigenVect2[1];
	} else {
		majorAxis[0] = eigenVect2[0];
		majorAxis[1] = eigenVect2[1];
		minorAxis[0] = eigenVect1[0];
		minorAxis[1] = eigenVect1[1];
	}

}

float FostroDescriptor::getArea() {
	return area;
}

float FostroDescriptor::getPerimeter() {
	return perimeter;
}

float FostroDescriptor::getPerimeter2OverArea() {
	return perim2OverArea;
}

float FostroDescriptor::getMomentInvariant(int index) {
	return momentInvariants[index];
}

float FostroDescriptor::getMajorAxis(int index) {
	return majorAxis[index];
}

float FostroDescriptor::getMinorAxis(int index) {
	return minorAxis[index];
}

float FostroDescriptor::getEigenVector(int vectIndex, int index) {
	if (vectIndex == 0) {
		return eigenVect1[index];
	} else {
		return eigenVect2[index];
	}
}

float FostroDescriptor::getEigenValue(int index) {
	if (index == 0) {
		return eigenVal1;
	} else {
		return eigenVal2;
	}
}
