#pragma once

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "FostroCudaDevice.h"
#include "FostroDefines.h"

class FostroCudaDevice {
	public:
		int currentDevice;
		int numDevices;

		FostroCudaDevice();
		~FostroCudaDevice();
		int getDeviceCount();
		int getCurrentDevice();
		void getCurrentDeviceName();
		void setDevice(int device);
};
