#include "FostroInvertFilterGPU.h"

//void apply(int width, int height, int c, FostroType* d_red_input, FostroType* d_green_input, FostroType* d_blue_input, FostroType* d_red_output, FostroType* d_green_output, FostroType* d_blue_output);
void applyInvert(FostroGPU* gpuStruct, int c);

FostroInvertFilterGPU::FostroInvertFilterGPU() : FostroFilterGPU() {
	
}

FostroInvertFilterGPU::~FostroInvertFilterGPU() {

}

void FostroInvertFilterGPU::applyFilter(int width, int height, int c) {
	//apply(width, height, c, d_red_input, d_green_input, d_blue_input, d_red_output, d_green_output, d_blue_output);
	applyInvert(gpuStruct, c);
}
