#include <iostream>
#include <cstdlib>
#include <wand/MagickWand.h>
#include <getopt.h>
#include "FostroImage.h"
#include "FostroDefines.h"
#include "FostroSmoothingFilter.h"
#include "FostroInvertFilter.h"
#include "FostroMedianFilter.h"
#include "FostroHistoEqFilter.h"
#include "FostroHistoEqFilterWikipedia.h"
#include "FostroSobelVerticalFilter.h"
#include "FostroSobelHorizontalFilter.h"
#include "FostroLaplaceFilter.h"
#include "FostroBinaryFilter.h"
#include "FostroErosionFilter.h"
#include "FostroDilationFilter.h"
#include "FostroDescriptor.h"

using namespace std;

const char* optString = "hsmno:i:glb:eda:c";

string defaultOutput = "output.png";

static const struct option longOpts[] = {
	{ "histo", no_argument, NULL, 'h' },
	{ "smooth", no_argument, NULL, 's' },
	{ "median", no_argument, NULL, 'm' },
	{ "invert", no_argument, NULL, 'n' },
	{ "output", required_argument, NULL, 'o' },
	{ "input", required_argument, NULL, 'i' },
	{ "sobel", no_argument, NULL, 'g' },
	{ "laplace", no_argument, NULL, 'l' },
	{ "binary", required_argument, NULL, 'b' },
	{ "erosion", no_argument, NULL, 'e' },
	{ "dilation", no_argument, NULL, 'd' },
	{ "segmentation", required_argument, NULL, 'a' },
	{ "descriptor", no_argument, NULL, 'c' },
};

int main(int argc, char* argv[]) {

	bool histoflag = false;
	bool smoothflag = false;
	bool mediainvertflag = false;
	bool invertflag = false;
	bool inputflag = false;
	bool histowikiflag = false;
	bool sobelflag = false;
	bool laplaceflag = false;
	bool binaryflag = false;
	bool erosionflag = false;
	bool dilationflag = false;
	bool segmentationflag = false;
	bool descriptorflag = false;

	string outputFileName;
	string inputFileName;

	int threshold = MAX_PIXEL_VAL/2;
	int segLayer = GRAY;

	if (argc < 3) {
		cout << "Usage: " << argv[0] << " inputImage outputImage" << endl;
		return -1;
	}

	int longIndex;
	int opt = getopt_long(argc, argv, optString, longOpts, &longIndex);

	outputFileName = defaultOutput;

	while (opt != -1) {
		switch (opt) {
		case 'h':
			histoflag = true;
			break;
		case 's':
			smoothflag = true;
			break;
		case 'm':
			mediainvertflag = true;
			break;
		case 'n':
			invertflag = true;
			break;
		case 'i':
			inputflag = true;
			inputFileName = optarg;
			break;
		case 'o':
			outputFileName = optarg;
			break;
		case 'g':
			sobelflag = true;
			break;
		case 'l':
			laplaceflag = true;
			break;
		case 'b':
			binaryflag = true;
			threshold = atoi(optarg);
			break;
		case 'e':
			erosionflag = true;
			binaryflag = true;
			break;
		case 'd':
			dilationflag = true;
			binaryflag = true;
			break;
		case 'a':
			segmentationflag = true;
			switch (optarg[0]) {
			case 'r':
				segLayer = RED;
				break;
			case 'g':
				segLayer = GREEN;
				break;
			case 'b':
				segLayer = BLUE;
				break;
			default:
				segLayer = GRAY;
			}
			threshold=atoi(optarg+2);
		case 'c':
			descriptorflag = true;
			break;
		}

		opt = getopt_long(argc, argv, optString, longOpts, &longIndex);
	}

	if (!inputflag) {
		cout << "No Input File Specified" << endl;
		exit(-1);
	}

	cout << "Creating Image" << endl;

	FostroImage image(inputFileName, outputFileName);

	FostroImage *newImage, *binaryImage;
	
	newImage = binaryImage = NULL;

	if (smoothflag) {
		cout << "Smoothing Image" << endl;
		FostroSmoothingFilter smoothFilter(3,3);
		smoothFilter.setAllMaskValsSame(1.0/9.0);
		newImage = smoothFilter.applyFilter(&image, GRAY);
	}

	if (invertflag) {
		cout << "Inverting Image" << endl;
		FostroInvertFilter invertFilter(3,3);
		invertFilter.setAllMaskValsSame(1);
		newImage = invertFilter.applyFilter(&image, GRAY);
	}

	if (mediainvertflag) {
		cout << "Median Filtering Image" << endl;
		FostroMedianFilter medianFilter(3,3);
		medianFilter.setAllMaskValsSame(1);
		newImage = medianFilter.applyFilter(&image, GRAY);
	}

	if (histoflag) {
		cout << "Histogram Equalization Filtering Image" << endl;
		FostroHistoEqFilter histoEqFilter(3,3);
		histoEqFilter.setAllMaskValsSame(1);
		newImage = histoEqFilter.applyFilter(&image, GRAY);
	}

	if (histowikiflag) {
		cout << "Alternate Histogram Equalization Filtering Image" << endl;
		FostroHistoEqFilterWikipedia histoEqFilter(3,3);
		histoEqFilter.setAllMaskValsSame(1);
		newImage = histoEqFilter.applyFilter(&image, GRAY);
	}

	if (sobelflag) {
		cout << "Sobel Filtering Image" << endl; 
		FostroSobelVerticalFilter sobelVertFilter(3,3);
		FostroSobelHorizontalFilter sobelHorizFilter(3,3);
		FostroImage* sobelVImage = sobelVertFilter.applyFilter(&image, GRAY);
		cout << "Saving Vertical Sobel" << endl;
		sobelVImage->saveImage();
		FostroImage* sobelHImage = sobelHorizFilter.applyFilter(&image, GRAY);
		cout << "Saving Horizontal Sobel" << endl;
		sobelHImage->saveImage();
		newImage = new FostroImage(*sobelVImage + *sobelHImage, outputFileName);
	}
	
	if (laplaceflag) {
		cout << "Laplace Filtering Image" << endl;
		FostroLaplaceFilter laplaceFilter(3,3);
		newImage = laplaceFilter.applyFilter(&image, GRAY);
	}
	
	if (binaryflag) {
		cout << "Applying Threshold to Create Binary Image" << endl;
		FostroBinaryFilter binaryFilter(3,3);
		cout << "Using threshold value of " << threshold << endl;
		binaryFilter.setThreshold(threshold);
		newImage = binaryFilter.applyFilter(&image, GRAY);
		binaryImage = new FostroImage(*newImage);
	}
	
	if (erosionflag) {
		cout << "Eroding Image" << endl;
		FostroErosionFilter erosionFilter(3,3);
		newImage = erosionFilter.applyFilter(binaryImage, GRAY);
	}
	
	if (dilationflag) {
		cout << "Dilating Image" << endl;
		FostroDilationFilter dilationFilter(3,3);
		newImage = dilationFilter.applyFilter(binaryImage, GRAY);
	}
	
	if (segmentationflag) {
		cout << "Segmenting Image" << endl;
		FostroBinaryFilter segFilter(3,3);
		segFilter.setThreshold(threshold);
		newImage = segFilter.applyFilter(&image, segLayer);
	}

	if (descriptorflag) {
		cout << "Calculating Image Descriptors" << endl;
		FostroDescriptor* descriptor = new FostroDescriptor();
		descriptor->calcDescriptors(&image, GRAY);

		cout << "Area = " << descriptor->getArea() << endl;
		cout << "Perimeter = " << descriptor->getPerimeter() << endl;
		cout << "Perimeter^2/Area = " << descriptor->getPerimeter2OverArea() << endl;
		for (int i = 0; i < 7; ++i) {
			cout << "Moment Invariant " << i << " = " << descriptor->getMomentInvariant(i) << endl;
		}
		cout << "Eigen Values are " << descriptor->getEigenValue(0) << " and " << descriptor->getEigenValue(1) << endl;
		cout << "Eigen Vectors are " << descriptor->getEigenVector(0, 0) << "," << descriptor->getEigenVector(0, 1)
			<< " and " << descriptor->getEigenVector(1, 0) << "," << descriptor->getEigenVector(1, 1) << endl;
		cout << "Major Axis is " << descriptor->getMajorAxis(0) << "," << descriptor->getMajorAxis(1) << endl;
		cout << "Minor Axis is " << descriptor->getMinorAxis(0) << "," << descriptor->getMinorAxis(1) << endl;
	}

	if (!descriptorflag) {
		newImage->setFilePath((char*)outputFileName.c_str());

		cout << "Saving Image to " << newImage->getFilePath() << endl;
		newImage->saveImage();
	}

	return 0;
}

