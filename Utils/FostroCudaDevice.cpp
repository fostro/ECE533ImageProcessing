#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <iostream>
#include "FostroCudaDevice.h"
#include "FostroDefines.h"

FostroCudaDevice::FostroCudaDevice() {

}

FostroCudaDevice::~FostroCudaDevice() {

}

int FostroCudaDevice::getDeviceCount() {
	cudaError_t error;

	cudaSafe(cudaGetDeviceCount(&numDevices), "Get Device Count");
	return numDevices;
}

int FostroCudaDevice::getCurrentDevice() {
	cudaError_t error;

	cudaSafe(cudaGetDevice(&currentDevice), "Get Device");
	return currentDevice;
}

void FostroCudaDevice::getCurrentDeviceName() {
	cudaError_t error;

	cudaDeviceProp* prop = new cudaDeviceProp;

	cudaSafe(cudaGetDevice(&currentDevice), "Get Device");
	cudaSafe(cudaGetDeviceProperties(prop, currentDevice), "Get Device Properties");

	std::cout << "Device " << currentDevice << ":  " << prop->name << std::endl;

	delete prop;
}

void FostroCudaDevice::setDevice(int device) {
	cudaError_t error;

	cudaSafe(cudaSetDevice(device), "Set Device");
}
