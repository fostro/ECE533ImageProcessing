#pragma once

#include <stdio.h>
#include <stdlib.h>

#define GTX680		0
#define GTX580_1	1
#define GTX580_2	2

#define ThrowWandException(wand) \
{ \
  char \
    *description; \
 \
  ExceptionType \
    severity; \
 \
  description=MagickGetException(wand,&severity); \
  (void) fprintf(stderr,"%s %s %lu %s\n",GetMagickModule(),description); \
  description=(char *) MagickRelinquishMemory(description); \
  exit(-1); \
}

//#define MAX_PIXEL_VAL	65535
#define MAX_PIXEL_VAL	1.0

#define cudaSafe(x, y) if ((error = x) != cudaSuccess) {\
        printf("%s Error: %s\n", y, cudaGetErrorString(error));\
        exit(1);\
        }
//#define FostroType	float
#define FostroType	double

struct FostroGPU {
	unsigned long numPixels;
	int nCols;
	int nRows;
	FostroType* d_ir;
	FostroType* d_ig;
	FostroType* d_ib;
	FostroType* d_or;
	FostroType* d_og;
	FostroType* d_ob;
	FostroType* filter;
	int filterWidth;
};

enum FOSTRO_COLOR{ RED=0, GREEN, BLUE, ALPHA, GRAY };
