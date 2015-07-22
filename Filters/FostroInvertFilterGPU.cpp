#include "FostroInvertFilterGPU.h"

void applyInvert(FostroGPU* gpuStruct, int c);

FostroInvertFilterGPU::FostroInvertFilterGPU() : FostroFilterGPU() {
	
}

FostroInvertFilterGPU::~FostroInvertFilterGPU() {

}

void FostroInvertFilterGPU::applyFilter(int width, int height, int c) {
	applyInvert(gpuStruct, c);
}
