#pragma once

#include "FostroImage.h"

class FostroDescriptor {
    public:
		FostroType area;
		FostroType perimeter;
		FostroType perim2OverArea;

		FostroType momentInvariants[7];

		FostroType majorAxis[2];
		FostroType minorAxis[2];

		FostroType eigenVal1;
		FostroType eigenVal2;

		FostroType eigenVect1[2];
		FostroType eigenVect2[2];

		FostroDescriptor();
		FostroDescriptor(const FostroDescriptor& other);
		FostroDescriptor& operator=(const FostroDescriptor& other);
		~FostroDescriptor();
 
		void calcDescriptors(FostroImage* image, int c);
		FostroType calcArea(FostroImage* image);
		FostroType calcPerimeter(FostroImage* image);
		void calcPerimeter2OverArea(FostroImage* image);
		FostroType calcMajorAxis(FostroImage* image);
		FostroType calcMinorAxis(FostroImage* image);
		void calcMomentInvariants(FostroImage* image);

		FostroType getArea();
		FostroType getPerimeter();
		FostroType getPerimeter2OverArea();
		FostroType getMomentInvariant(int index);
		FostroType getMajorAxis(int index);
		FostroType getMinorAxis(int index);
		FostroType getEigenVector(int vectIndex, int index);
		FostroType getEigenValue(int index);
};
