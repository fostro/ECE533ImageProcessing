#!/bin/bash

nvcc -G -g testing.cu Filters/FostroFilterGPU.cpp Filters/FostroInvertFilterGPU.cpp Filters/FostroInvertFilterGPU.cu Utils/FostroImage.cpp Utils/FostroPixel.cpp Filters/FostroSmoothFilterGPU.cpp Filters/FostroSmoothFilterGPU.cu Utils/FostroCudaDevice.cpp -o wand -DMAGICKCORE_HDRI_ENABLE=0 -DMAGICKCORE_QUANTUM_DEPTH=16 -DMAGICKCORE_HDRI_ENABLE=0 -DMAGICKCORE_QUANTUM_DEPTH=16 -I/usr/include/ImageMagick-6 -lMagickWand-6.Q16 -lMagickCore-6.Q16 -arch sm_20
