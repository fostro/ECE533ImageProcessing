#pragma once

class FostroPixel {
	public:
		float red;
		float blue;
		float green;
		float alpha;
		float grayscaleIntensity;

		float getRedVal() const;
		float getGreenVal() const;
		float getBlueVal() const;
		float getAlphaVal() const;
		float getPixelVal(int c) const;
		
		void setRedVal(float x);
		void setGreenVal(float x);
		void setBlueVal(float x);
		void setAlphaVal(float x);
		void setPixelVal(float val, int c);
		
		void invertRed();
		void invertGreen();
		void invertBlue();
		void invertAlpha();

		float getGrayscaleIntensity();
		void calcGrayscaleIntensity();
};

