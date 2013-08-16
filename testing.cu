#include <iostream>
#include <cstdlib>
#include <wand/MagickWand.h>
#include <getopt.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "Utils/FostroImage.h"
#include "Utils/FostroDefines.h"
#include "Utils/FostroCudaDevice.h"
#include "Filters/FostroInvertFilterGPU.h"
#include "Filters/FostroSmoothFilterGPU.h"
#include "Utils/MonkTimer.h"

using namespace std;

const char* optString = "o:i:ns::";

string defaultOutput = "output.png";

static const struct option longOpts[] = {
	{ "output", required_argument, NULL, 'o' },
	{ "input", required_argument, NULL, 'i' },
	{ "invert", no_argument, NULL, 'n' },
	{ "smooth", optional_argument, NULL, 's' },
};

int main(int argc, char* argv[]) {

	bool inputFlag = false;
	bool invertFlag = false;
	bool smoothFlag = false;

	string outputFileName;
	string inputFileName;

	if (argc < 3) {
		cout << "Usage: " << argv[0] << " -i inputImage -o outputImage [options]" << endl;
		return -1;
	}

	int longIndex;
	int opt = getopt_long(argc, argv, optString, longOpts, &longIndex);
	int filterWidth = 3;

	outputFileName = defaultOutput;

	while (opt != -1) {
		switch (opt) {
		case 'i':
			inputFlag = true;
			inputFileName = optarg;
			break;
		case 'o':
			outputFileName = optarg;
			break;
		case 'n':
			invertFlag = true;
			break;
		case 's':
			smoothFlag = true;
			if (optarg) {
				filterWidth = atoi(optarg);
			}
			break;
		}

		opt = getopt_long(argc, argv, optString, longOpts, &longIndex);
	}

	if (!inputFlag) {
		cout << "No Input File Specified" << endl;
		exit(-1);
	}

	FostroCudaDevice device;
	cout << "Device count is " << device.getDeviceCount() << endl;
	cout << "Original Device is ";
	device.getCurrentDeviceName();
	device.setDevice(GTX580_1);
	cout << "New Device is ";
	device.getCurrentDeviceName();

	cout << "Creating Image" << endl;

	MonkTimer open("Open");
	MonkTimer setup("Setup");
	MonkTimer gpu("GPU");
	MonkTimer total("Total");

	total.start();

	open.start();

	FostroImage image(inputFileName, outputFileName);

	open.stop();

	FostroImage* newImage(&image);

	setup.start();

	if (invertFlag) {
		cout << "Inverting Image" << endl;

		cout << "Creating Invert Filter" << endl;
		FostroInvertFilterGPU* invertFilter = new FostroInvertFilterGPU();
	
		cout << "Setting Up Invert Filter" << endl;
		invertFilter->setupFilter(newImage, filterWidth);

		gpu.start();
	
		cout << "Applying Invert Filter" << endl;
		invertFilter->applyFilter(newImage->getWidth(), newImage->getHeight(), GRAY);

		gpu.stop();
	
		cout << "Cleaning Up Invert Filter" << endl;
		invertFilter->cleanupFilter(newImage);

	} else if (smoothFlag) {
		cout << "Smoothing Image" << endl;

		cout << "Creating Smooth Filter" << endl;
		FostroSmoothFilterGPU* smoothFilter = new FostroSmoothFilterGPU();
	
		cout << "Setting Up Smooth Filter" << endl;
		smoothFilter->setupFilter(newImage, filterWidth);

		gpu.start();
	
		cout << "Applying Smooth Filter" << endl;
		smoothFilter->applyFilter(newImage->getWidth(), newImage->getHeight(), GRAY);

		gpu.stop();
	
		cout << "Cleaning Up Smooth Filter" << endl;
		smoothFilter->cleanupFilter(newImage);
	}

	setup.stop();

	MonkTimer save("Save");
	save.start();

	cout << "Setting Output Name" << endl;
	newImage->setFilePath((char*)outputFileName.c_str());

	cout << "Saving Image to " << newImage->getFilePath() << endl;
	newImage->saveImage();

	save.stop();
	total.stop();

	return 0;
}

