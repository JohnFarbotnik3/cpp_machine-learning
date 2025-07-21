
#include <vector>
#include "simd.cpp"

namespace ML::models::autoencoder_subimage {
	using std::vector;

	struct simd_image_8f_dimensions {
		int X;// image width.
		int Y;// image height.
		int C;// number of colour channels.

		simd_image_8f_dimensions() = default;
		simd_image_8f_dimensions(int X, int Y, int C) {
			this->X = X;
			this->Y = Y;
			this->C = C;
		}

		int get_offset(int x, int y, int c) const { return ((y*X) + x)*C + c; }

		int length() const { return Y*X*C; }
		int row_length() const { return X*C; }
		int pixel_length() const { return C; }
	};
	struct simd_image_8f {
		vector<vec8f> data;
		simd_image_8f_dimensions dim;

		simd_image_8f() = default;
		simd_image_8f(const simd_image_8f_dimensions dim) : dim(dim), data(dim.length(), simd_value(0)) {}
		simd_image_8f(const int X, const int Y, const int C) : dim(X,Y,C), data(X*Y*C, simd_value(0)) {}

		void clear() { data.assign(data.size(), simd_value(0)); }

		// pack multiple images into simd image.
		void pack() {}// TODO

		// unpack multiple images from simd image.
		void unpack() {}// TODO
	};
};
