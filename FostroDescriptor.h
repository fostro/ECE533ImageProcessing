#pragma once

#include "FostroImage.h"

class FostroDescriptor {
    public:
		float area;
		float perimeter;
		float perim2OverArea;

		float momentInvariants[7];

		float majorAxis[2];
		float minorAxis[2];

		float eigenVal1;
		float eigenVal2;

		float eigenVect1[2];
		float eigenVect2[2];

		FostroDescriptor();
		FostroDescriptor(const FostroDescriptor& other);
		FostroDescriptor& operator=(const FostroDescriptor& other);
		~FostroDescriptor();
 
		void calcDescriptors(FostroImage* image, int c);
		float calcArea(FostroImage* image);
		float calcPerimeter(FostroImage* image);
		void calcPerimeter2OverArea(FostroImage* image);
		float calcMajorAxis(FostroImage* image);
		float calcMinorAxis(FostroImage* image);
		void calcMomentInvariants(FostroImage* image);

		float getArea();
		float getPerimeter();
		float getPerimeter2OverArea();
		float getMomentInvariant(int index);
		float getMajorAxis(int index);
		float getMinorAxis(int index);
		float getEigenVector(int vectIndex, int index);
		float getEigenValue(int index);
};
