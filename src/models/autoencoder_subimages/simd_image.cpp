
#ifndef F_simd_image
#define F_simd_image

#include <vector>
#include "simd.cpp"
#include "src/image/value_image.cpp"

namespace ML::models::autoencoder_subimage {
	using std::vector;
	using ML::image::value_image::value_image;

	struct simd_image_8f_dimensions {
		int X;// image width.
		int Y;// image height.
		int C;// number of colour channels.
		int STRIDE_X;
		int STRIDE_Y;

		simd_image_8f_dimensions() = default;
		simd_image_8f_dimensions(int X, int Y, int C) {
			this->X = X;
			this->Y = Y;
			this->C = C;
			this->STRIDE_X = C;
			this->STRIDE_Y = C*X;
		}

		int get_offset(int x, int y, int c) const { return y*STRIDE_Y + x*STRIDE_X + c; }

		int length() const { return Y*X*C; }

		bool equals(const simd_image_8f_dimensions& other) const {
			return (
				(other.X == X) &
				(other.Y == Y) &
				(other.C == C)
			);
		}
	};
	struct simd_image_8f {
		vector<vec8f> data;
		simd_image_8f_dimensions dim;

		simd_image_8f() = default;
		simd_image_8f(const simd_image_8f_dimensions dim) : dim(dim), data(dim.length(), simd_value(0)) {}
		simd_image_8f(const int X, const int Y, const int C) : dim(X,Y,C), data(X*Y*C, simd_value(0)) {}

		void clear() { data.assign(data.size(), simd_value(0)); }

		// pack multiple images into simd image.
		void pack(const value_image<float>* images, const int N) {
			assert(0 <= N && N <= 8);
			float values[8];
			for(int n=0;n<8;n++) values[n] = 0;
			for(int x=0;x<dim.length();x++) {
				for(int n=0;n<N;n++) values[n] = images[n].data[x];
				data[x] = _mm256_loadu_ps(values);
			}
		}

		// unpack multiple images from simd image.
		void unpack(value_image<float>* images, const int N) {
			assert(0 <= N && N <= 8);
			float values[8];
			for(int x=0;x<dim.length();x++) {
				_mm256_storeu_ps(values, data[x]);
				for(int n=0;n<N;n++) images[n].data[x] = values[n];
			}
		}
	};
};

#endif
