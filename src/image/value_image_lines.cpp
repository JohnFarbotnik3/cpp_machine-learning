
#ifndef F_value_image_lines
#define F_value_image_lines

#include <algorithm>
#include <cassert>
#include <vector>
#include "file_image.cpp"

namespace ML::image::value_image_lines {
	using std::vector;

	struct value_image_lines_dimensions {
		int X = 0;// width of image.
		int Y = 0;// height of image.
		int C = 0;// number of channels per pixel.

		int length() const {
			return X * Y * C;
		}

		int get_offset(const int x, const int y, const int c) const {
			return ((y*X) + x)*C + c;
		}

		bool is_within_bounds(const int x, const int y) const {
			return (
				(x >= 0) & (x < X) &
				(y >= 0) & (y < Y)
			);
		}

		bool equals(const value_image_lines_dimensions& other) const {
			return (
				(other.X == X) &
				(other.Y == Y) &
				(other.C == C)
			);
		}

		string toString() const {
			char buf[256];
			int len = snprintf(buf, 256, "X=%i, Y=%i, C=%i", X, Y, C);
			return string(buf, len);
		}
	};

	/* iterator for scanline images. */
	struct value_image_lines_iterator {
		// image dimensions.
		value_image_lines_dimensions dim;
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

		value_image_lines_iterator() = default;
		value_image_lines_iterator(
			value_image_lines_dimensions dim,
			int x0, int x1,
			int y0, int y1,
			int c0, int c1
		) {
			this->dim = dim;
			this->x0 = x0;
			this->x1 = x1;
			this->y0 = y0;
			this->y1 = y1;
			this->c0 = c0;
			this->c1 = c1;
			this->x = x0;
			this->y = y0;
			this->c = c0;
			this->i = dim.get_offset(x, y, c);
			this->irem = length();
			// assertions.
			assert(x0 >= 0);
			assert(y0 >= 0);
			assert(c0 >= 0);
			assert(x1 <= dim.X);
			assert(y1 <= dim.Y);
			assert(c1 <= dim.C);
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
			i = dim.get_offset(x, y, c);
			return i;
		}
	};

	template<class T>
	struct value_image_lines {
		vector<T> data;
		value_image_lines_dimensions dim;
		// sample bounds.
		int x0,x1;
		int y0,y1;

		value_image_lines() = default;
		value_image_lines(const value_image_lines_dimensions dim) : dim(dim), data(dim.length(), 0) {}

		void clear() {
			for(int x=0;x<data.size();x++) data[x] = 0;
		}

		value_image_lines_iterator get_iterator(
			int x0, int x1,
			int y0, int y1,
			int c0, int c1
		) const {
			return value_image_lines_iterator(dim, x0, x1, y0, y1, c0, c1);
		}
	};

	void sample_area_copy(value_image_lines<float>& sample, const file_image& image) {
		// assert that image-area and sample-area match.
		assert((sample.x1 - sample.x0) == image.X);
		assert((sample.y1 - sample.y0) == image.Y);
		assert(sample.dim.C == image.C);
		sample.clear();

		value_image_lines_iterator sample_iter = sample.get_iterator(
			sample.x0, sample.x1,
			sample.y0, sample.y1,
			0, image.C
		);

		const float m = 1.0f / 255.0f;
		while(sample_iter.has_next()) {
			int ofs = image.get_offset(
				sample_iter.x - sample.x0,
				sample_iter.y - sample.y0,
				sample_iter.c
			);
			sample.data[sample_iter.i] = m * image.data[ofs];
			sample_iter.next();
		}
	}

	void sample_area_nearest_neighbour(value_image_lines<float>& sample, const file_image& image) {
		sample.clear();

		value_image_lines_iterator sample_iter = sample.get_iterator(
			sample.x0, sample.x1,
			sample.y0, sample.y1,
			0, image.C
		);

		const float m = 1.0f / 255.0f;
		const float x_inv_scale_factor = float(image.X) / float(sample.x1 - sample.x0);
		const float y_inv_scale_factor = float(image.Y) / float(sample.y1 - sample.y0);
		while(sample_iter.has_next()) {
			int ix = std::clamp(int(float(sample_iter.x - sample.x0) * x_inv_scale_factor), 0, image.X);
			int iy = std::clamp(int(float(sample_iter.y - sample.y0) * y_inv_scale_factor), 0, image.Y);
			int ofs = image.get_offset(ix, iy, sample_iter.c);
			sample.data[sample_iter.i] = m * image.data[ofs];
			sample_iter.next();
		}
	}

	void sample_area_linear(value_image_lines<float>& sample, const file_image& image) {
		sample.clear();

		// compute float conversions between sample coordinates to image coordinates.
		vector<int> floor_mx(sample.dim.X+2);
		vector<int> floor_my(sample.dim.Y+2);
		vector<int> ceil__mx(sample.dim.X+2);
		vector<int> ceil__my(sample.dim.Y+2);
		vector<float> image_to_sample_x(image.X+2);
		vector<float> image_to_sample_y(image.Y+2);
		int max_int = std::max(image.X+2, image.Y+2);
		vector<float> int_to_float(max_int);
		{
			float mx = float(image.X) / float(sample.x1 - sample.x0);
			float my = float(image.Y) / float(sample.y1 - sample.y0);
			for(int p=0;p<floor_mx.size();p++) floor_mx[p] = std::floor(float(p) * mx);
			for(int p=0;p<floor_my.size();p++) floor_my[p] = std::floor(float(p) * my);
			for(int p=0;p<ceil__mx.size();p++) ceil__mx[p] = std::ceil(float(p) * mx);
			for(int p=0;p<ceil__my.size();p++) ceil__my[p] = std::ceil(float(p) * my);
			for(int p=0;p<image_to_sample_x.size();p++) image_to_sample_x[p] = float(p) / mx;
			for(int p=0;p<image_to_sample_y.size();p++) image_to_sample_y[p] = float(p) / my;
			for(int p=0;p<int_to_float.size();p++) int_to_float[p] = float(p);
		}

		value_image_lines_iterator sample_iter = sample.get_iterator(
			sample.x0, sample.x1,
			sample.y0, sample.y1,
			0, image.C
		);

		const float m = 1.0f / 255.0f;
		while(sample_iter.has_next()) {
			int sx = sample_iter.x - sample_iter.x0;
			int sy = sample_iter.y - sample_iter.y0;
			int ix0 = floor_mx[sx];
			int ix1 = ceil__mx[sx + 1];
			int iy0 = floor_my[sy];
			int iy1 = ceil__my[sy + 1];

			// pre-compute intersection lengths independently for each axis.
			const int ixlen = ix1 - ix0;
			const int iylen = iy1 - iy0;
			float intersect_lengths_x[ixlen];
			float intersect_lengths_y[iylen];
			for(int x=0;x<ixlen;x++) {
				const int ix = x + ix0;
				const float lx0 = std::max(image_to_sample_x[ix  ], int_to_float[sx  ]);
				const float lx1 = std::min(image_to_sample_x[ix+1], int_to_float[sx+1]);
				intersect_lengths_x[x] = lx1 - lx0;
				assert(intersect_lengths_x[x] >= 0.0f);
			}
			for(int y=0;y<iylen;y++) {
				const int iy = y + iy0;
				const float ly0 = std::max(image_to_sample_y[iy  ], int_to_float[sy  ]);
				const float ly1 = std::min(image_to_sample_y[iy+1], int_to_float[sy+1]);
				intersect_lengths_y[y] = ly1 - ly0;
				assert(intersect_lengths_y[y] >= 0.0f);
			}

			// gather weighted sum of image-pixel values according to
			// area of intersection between sample-pixel and image-pixel.
			float value_sum = 0;
			for(int y=0;y<iylen;y++) {
			for(int x=0;x<ixlen;x++) {
				const int ix = x + ix0;
				const int iy = y + iy0;
				// compute area of intersection between scaled image-pixel and sample-pixel,
				// normalized such that sample-pixel area=1.
				const float area = intersect_lengths_x[x] * intersect_lengths_y[y];
				value_sum += image.data[image.get_offset(ix, iy, sample_iter.c)] * area * m;
			}}
			if(value_sum <   0.0f) printf("value_sum <   0.0f: x=%i, y=%i, c=%i, i=%i, v=%f\n", sx, sy, sample_iter.c, sample_iter.i, value_sum);
			if(value_sum > 255.0f) printf("value_sum < 255.0f: x=%i, y=%i, c=%i, i=%i, v=%f\n", sx, sy, sample_iter.c, sample_iter.i, value_sum);
			assert(value_sum >= 0.0f);
			assert(value_sum <= 255.0f);
			sample.data[sample_iter.i] = value_sum;
			sample_iter.next();
		}
	}

	/*
		generate input values by sampling loaded training images:
			- if the original image is smaller than the input area,
			then the image is just copied to input and centered.
			- if the original image is larger than the input area,
			then it is scaled to fit in input area.

		values outside the sample area are set to 0 and are ignored
		when computing error.
	*/
	void generate_sample_image(value_image_lines<float>& sample, const file_image& image) {
		const int sample_X = sample.dim.X;
		const int sample_Y = sample.dim.Y;

		// if loaded image is smaller than sample area, then copy.
		if(image.X <= sample_X && image.Y <= sample_Y) {
			int remaining_w = sample_X - image.X;
			int remaining_h = sample_Y - image.Y;
			sample.x0 = remaining_w/2;
			sample.y0 = remaining_h/2;
			sample.x1 = sample.x0 + image.X;
			sample.y1 = sample.y0 + image.Y;
			sample_area_copy(sample, image);
		}

		// if loaded image is larger than sample area, then scale down.
		else {
			// determine which dimension to scale against.
			float iw_over_sw = float(image.X) / float(sample_X);
			float ih_over_sh = float(image.Y) / float(sample_Y);
			float scale_factor = 1;
			if(iw_over_sw >= ih_over_sh) {
				// scale to sample width.
				int scaled_h = std::min(((image.Y * sample_X) / image.X), sample_Y);
				sample.x0 = 0;
				sample.x1 = sample_X;
				int remaining_h = sample_Y  - scaled_h;
				sample.y0 = remaining_h / 2;
				sample.y1 = (sample.y0 + scaled_h);
			} else {
				// scale to sample height.
				int scaled_w = std::min(((image.X * sample_Y) / image.Y), sample_X);
				sample.y0 = 0;
				sample.y1 = sample_Y;
				int remaining_w = sample_X  - scaled_w;
				sample.x0 = remaining_w / 2;
				sample.x1 = (sample.x0 + scaled_w);
			}

			// sample image data.
			sample_area_linear(sample, image);
		}
	};

	file_image to_file_image(value_image_lines<float>& sample, bool clamp_to_sample_area) {
		file_image image;
		image.X = sample.dim.X;
		image.Y = sample.dim.Y;
		image.C = sample.dim.C;
		image.data.resize(image.X * image.Y * image.C);

		value_image_lines_iterator sample_iter = clamp_to_sample_area ? sample.get_iterator(
			sample.x0, sample.x1,
			sample.y0, sample.y1,
			0, image.C
		) : sample.get_iterator(
			0, image.X,
			0, image.Y,
			0, image.C
		);

		const float m = 255.0f / 1.0f;
		while(sample_iter.has_next()) {
			int ofs = image.get_offset(sample_iter.x, sample_iter.y, sample_iter.c);
			/*
			NOTE: clamping output values fixed the apparent "dead pixels" problem.
			- it was likely pixel value underflow/overflow that I was seeing,
			which would explain why error rate was low even with very wrong pixels in output files.
			- however using unclamped conversion can reveal bugs in sampling algorithm.
			*/
			//image.data[ofs] = sample.data[sample_iter.i] * m;// TODO TEST
			image.data[ofs] = std::clamp(sample.data[sample_iter.i], 0.0f, 1.0f) * m;
			sample_iter.next();
		}
		return image;
	}
}

#endif
