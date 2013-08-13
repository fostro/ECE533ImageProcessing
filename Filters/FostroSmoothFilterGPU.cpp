#include "FostroSmoothFilterGPU.h"

void applySmooth(FostroGPU* gpuStruct, int c);

FostroSmoothFilterGPU::FostroSmoothFilterGPU() : FostroFilterGPU() {
	
}

FostroSmoothFilterGPU::~FostroSmoothFilterGPU() {

}

void FostroSmoothFilterGPU::applyFilter(int width, int height, int c) {
	applySmooth(gpuStruct, c);
}
