
#ifndef F_value_image_padded
#define F_value_image_padded

#include <cassert>
#include <vector>
#include "file_image.cpp"

namespace ML::image::extra_image_types {
	using std::vector;

	struct value_image_padded_dimensions {
	private:
		int X = 0;// width of image.
		int Y = 0;// height of image.
	public:
		int C = 0;// number of channels per pixel.
		// number of padding pixels along image perimeter.
		int pad = 0;

		value_image_padded_dimensions() = default;
		value_image_padded_dimensions(int X, int Y, int C, int pad) {
			this->X = X + pad*2;
			this->Y = Y + pad*2;
			this->C = C;
			this->pad = pad;
		}

		int outerX() const { return X; }
		int outerY() const { return Y; }
		int innerX() const { return X - pad*2; }
		int innerY() const { return Y - pad*2; }
		int outer_length() const { return outerX() * outerY() * C; }
		int inner_length() const { return innerX() * innerY() * C; }

		int get_offset_padded(const int x, const int y, const int c) const {
			return (((y+pad)*X) + (x+pad))*C + c;
		}

		bool has_padding() const {
			return pad > 0;
		}

		bool equals(const value_image_padded_dimensions& other) const {
			return (
				(other.X == X) &
				(other.Y == Y) &
				(other.C == C) &
				(other.pad == pad)
			);
		}

		bool is_within_inner_bounds(const int x, const int y) const {
			return (
				(x >= 0) & (x < innerX()) &
				(y >= 0) & (y < innerY())
			);
		}

		string toString() const {
			char buf[256];
			int len = snprintf(buf, 256, "X=%i, Y=%i, C=%i, pad=%i", X, Y, C, pad);
			return string(buf, len);
		}
	};
	template<class T>
	struct value_image_padded {
		vector<T> data;
		value_image_padded_dimensions dim;

		value_image_padded() = default;
		value_image_padded(const value_image_padded_dimensions dim) : dim(dim), data(dim.outer_length(), 0) {}

		void clear() {
			data.assign(data.size(), 0);
		}
	};

	struct interleaved_image_dimensions {
		int X;// image width.
		int Y;// image height.
		int C;// number of colour channels.
		int D;// number of interleaved images.

		interleaved_image_dimensions() = default;
		interleaved_image_dimensions(int X, int Y, int C, int D) {
			this->X = X;
			this->Y = Y;
			this->C = C;
			this->D = D;
		}

		int get_offset(int x, int y, int c) const { return (((y*X) + x)*C + c)*D; }
		int get_offset(int x, int y, int c, int d) const { return (((y*X) + x)*C + c)*D + d; }

		int length() const { return Y*X*C*D; }
		int row_length() const { return X*C*D; }
		int pixel_length() const { return C*D; }
		int channel_length() const { return D; }
	};
	template<class T>
	struct interleaved_image {
		vector<T> data;
		interleaved_image_dimensions dim;

		interleaved_image() = default;
		interleaved_image(const interleaved_image_dimensions dim) : dim(dim), data(dim.length(), 0) {}
		interleaved_image(const int X, const int Y, const int C, const int D) : dim(X,Y,C,D), data(X*Y*C*D, 0) {}

		void clear() { data.assign(data.size(), 0); }
	};

}

#endif
