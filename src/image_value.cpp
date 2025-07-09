
#ifndef F_image_value
#define F_image_value

#include <algorithm>
#include <cassert>

namespace ML::image {

	/*
		a simple iterator for iterating on an image
		with a variable number of channels.

		NOTE: image is assumed to be in row-major form
		with interleaved colour channels.
	*/
	struct value_image_iterator {
		// image dimensions.
		int X,Y,C;
		// current position.
		int x,y,c;
		// image bounds - area to iterate through.
		int x0,x1;
		int y0,y1;
		int c0,c1;
		// index in image data.
		int i;
		// iterator steps remaining.
		int irem;

		value_image_iterator() = default;
		value_image_iterator(
			int X, int Y, int C,
			int x0, int x1,
			int y0, int y1,
			int c0, int c1
		) {
			this->X = X;
			this->Y = Y;
			this->C = C;
			this->x0 = x0;
			this->x1 = x1;
			this->y0 = y0;
			this->y1 = y1;
			this->c0 = c0;
			this->c1 = c1;
			this->x = x0;
			this->y = y0;
			this->c = c0;
			this->i = ((y * X) + x) * C + c;
			this->irem = length();
			// assertions.
			assert(x0 >= 0);
			assert(y0 >= 0);
			assert(c0 >= 0);
			assert(x1 <= X);
			assert(y1 <= Y);
			assert(c1 <= C);
		}

		int length() {
			return std::max(x1-x0, 0) * std::max(y1-y0, 0) * std::max(c1-c0, 0);
		}
		bool has_next() {
			return irem > 0;
		}
		int next() {
			irem--;
			c++;
			if(c >= c1) { c=c0; x++; }
			if(x >= x1) { x=x0; y++; }
			i = ((y*X) + x)*C + c;
			return i;
		}
	};

	// TODO - implement value_image and sampling stuff for autoencoder.
}

#endif
