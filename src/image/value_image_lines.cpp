
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

	struct sample_bounds {
		int x0, x1;
		int y0, y1;
	};

	template<class T>
	struct value_image_lines {
		vector<T> data;
		value_image_lines_dimensions dim;

		value_image_lines() = default;
		value_image_lines(const value_image_lines_dimensions dim) : dim(dim), data(dim.length(), 0) {}

		void clear() {
			data.assign(data.size(), 0);
		}
	};

	void sample_area_copy(value_image_lines<float>& sample, const file_image& image, const sample_bounds bounds) {
		assert((bounds.x1 - bounds.x0) == image.X);
		assert((bounds.y1 - bounds.y0) == image.Y);
		assert(sample.dim.C == image.C);
		sample.clear();

		const float m = 1.0f / 255.0f;
		for(int iy=0;iy<image.Y;iy++) {
		for(int ix=0;ix<image.X;ix++) {
		for(int ic=0;ic<image.C;ic++) {
			const int iofs = image.get_offset(ix, iy, ic);
			const int sofs = sample.dim.get_offset(ix+bounds.x0, iy+bounds.y0, ic);
			sample.data[sofs] = image.data[iofs] * m;
		}}}
	}

	void sample_area_nearest_neighbour(value_image_lines<float>& sample, const file_image& image, const sample_bounds bounds) {
		sample.clear();

		const float m = 1.0f / 255.0f;
		const float x_inv_scale_factor = float(image.X) / float(bounds.x1 - bounds.x0);
		const float y_inv_scale_factor = float(image.Y) / float(bounds.y1 - bounds.y0);
		for(int sy=bounds.y0;sy<bounds.y1;sy++) {
		for(int sx=bounds.x0;sx<bounds.x1;sx++) {
		for(int sc=0;sc<sample.dim.C;sc++) {
			const int ix = std::clamp(int(float(sx - bounds.x0) * x_inv_scale_factor), 0, image.X);
			const int iy = std::clamp(int(float(sy - bounds.y0) * y_inv_scale_factor), 0, image.Y);
			const int ic = sc;
			const int iofs = image.get_offset(ix, iy, ic);
			const int sofs = sample.dim.get_offset(sx, sy, sc);
			sample.data[sofs] = image.data[iofs] * m;
		}}}
	}

	void sample_area_linear(value_image_lines<float>& sample, const file_image& image, const sample_bounds bounds) {
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
			float mx = float(image.X) / float(bounds.x1 - bounds.x0);
			float my = float(image.Y) / float(bounds.y1 - bounds.y0);
			for(int p=0;p<floor_mx.size();p++) floor_mx[p] = std::floor(float(p) * mx);
			for(int p=0;p<floor_my.size();p++) floor_my[p] = std::floor(float(p) * my);
			for(int p=0;p<ceil__mx.size();p++) ceil__mx[p] = std::ceil(float(p) * mx);
			for(int p=0;p<ceil__my.size();p++) ceil__my[p] = std::ceil(float(p) * my);
			for(int p=0;p<image_to_sample_x.size();p++) image_to_sample_x[p] = float(p) / mx;
			for(int p=0;p<image_to_sample_y.size();p++) image_to_sample_y[p] = float(p) / my;
			for(int p=0;p<int_to_float.size();p++) int_to_float[p] = float(p);
		}

		const float m = 1.0f / 255.0f;
		for(int sy=bounds.y0;sy<bounds.y1;sy++) {
		for(int sx=bounds.x0;sx<bounds.x1;sx++) {
			int bx = sx - bounds.x0;
			int by = sy - bounds.y0;
			int ix0 = floor_mx[bx];
			int ix1 = ceil__mx[bx + 1];
			int iy0 = floor_my[by];
			int iy1 = ceil__my[by + 1];

			// pre-compute intersection lengths independently for each axis.
			const int ixlen = ix1 - ix0;
			const int iylen = iy1 - iy0;
			float intersect_lengths_x[ixlen];
			float intersect_lengths_y[iylen];
			for(int x=0;x<ixlen;x++) {
				const int ix = x + ix0;
				const float lx0 = std::max(image_to_sample_x[ix  ], int_to_float[bx  ]);
				const float lx1 = std::min(image_to_sample_x[ix+1], int_to_float[bx+1]);
				intersect_lengths_x[x] = lx1 - lx0;
				assert(intersect_lengths_x[x] >= 0.0f);
			}
			for(int y=0;y<iylen;y++) {
				const int iy = y + iy0;
				const float ly0 = std::max(image_to_sample_y[iy  ], int_to_float[by  ]);
				const float ly1 = std::min(image_to_sample_y[iy+1], int_to_float[by+1]);
				intersect_lengths_y[y] = ly1 - ly0;
				assert(intersect_lengths_y[y] >= 0.0f);
			}

			// gather weighted sum of image-pixel values according to
			// area of intersection between sample-pixel and image-pixel.
			float value_sums[sample.dim.C];
			for(int sc=0;sc<sample.dim.C;sc++) value_sums[sc] = 0.0f;
			for(int y=0;y<iylen;y++) {
			for(int x=0;x<ixlen;x++) {
				const int ix = x + ix0;
				const int iy = y + iy0;
				// compute area of intersection between scaled image-pixel and sample-pixel,
				// normalized such that sample-pixel area=1.
				const float area = intersect_lengths_x[x] * intersect_lengths_y[y];
				for(int sc=0;sc<sample.dim.C;sc++) value_sums[sc] += image.data[image.get_offset(ix, iy, sc)] * area;
			}}
			for(int sc=0;sc<sample.dim.C;sc++) {
				if(value_sums[sc] <   0.00f) printf("value_sum <   0.0f: x=%i, y=%i, c=%i, v=%f\n", bx, by, sc, value_sums[sc]);
				if(value_sums[sc] > 255.01f) printf("value_sum < 255.0f: x=%i, y=%i, c=%i, v=%f\n", bx, by, sc, value_sums[sc]);
				assert(value_sums[sc] >=   0.00f);
				assert(value_sums[sc] <= 255.01f);
				const int sofs = sample.dim.get_offset(sx, sy, sc);
				sample.data[sofs] = value_sums[sc] * m;
			}
		}}
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
	sample_bounds generate_sample_image(value_image_lines<float>& sample, const file_image& image) {
		const int sample_X = sample.dim.X;
		const int sample_Y = sample.dim.Y;
		sample_bounds bounds;

		// if loaded image is smaller than sample area, then copy.
		if(image.X <= sample_X && image.Y <= sample_Y) {
			int remaining_w = sample_X - image.X;
			int remaining_h = sample_Y - image.Y;
			bounds.x0 = remaining_w/2;
			bounds.y0 = remaining_h/2;
			bounds.x1 = bounds.x0 + image.X;
			bounds.y1 = bounds.y0 + image.Y;
			sample_area_copy(sample, image, bounds);
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
				bounds.x0 = 0;
				bounds.x1 = sample_X;
				int remaining_h = sample_Y  - scaled_h;
				bounds.y0 = remaining_h / 2;
				bounds.y1 = (bounds.y0 + scaled_h);
			} else {
				// scale to sample height.
				int scaled_w = std::min(((image.X * sample_Y) / image.Y), sample_X);
				bounds.y0 = 0;
				bounds.y1 = sample_Y;
				int remaining_w = sample_X  - scaled_w;
				bounds.x0 = remaining_w / 2;
				bounds.x1 = (bounds.x0 + scaled_w);
			}
			// sample image data.
			sample_area_linear(sample, image, bounds);
		}

		return bounds;
	};

	/*
		NOTE: clamping output values fixed the apparent "dead pixels" problem.
		- it was likely pixel value underflow/overflow that I was seeing,
		which would explain why error rate was low even with very wrong pixels in output files.
		- however using unclamped conversion can reveal bugs in sampling algorithm.
	*/
	file_image to_file_image(value_image_lines<float>& sample, const sample_bounds bounds, bool crop_to_sample_area) {
		const float m = 255.0f / 1.0f;
		if(crop_to_sample_area) {
			file_image image;
			image.X = bounds.x1 - bounds.x0;
			image.Y = bounds.y1 - bounds.y0;
			image.C = sample.dim.C;
			image.data.resize(image.X * image.Y * image.C);
			for(int sy=bounds.y0;sy<bounds.y1;sy++) {
			for(int sx=bounds.x0;sx<bounds.x1;sx++) {
			for(int sc=0;sc<sample.dim.C;sc++) {
				const int iofs = image.get_offset(sx-bounds.x0, sy-bounds.y0, sc);
				const int sofs = sample.dim.get_offset(sx, sy, sc);
				image.data[iofs] = std::clamp(sample.data[sofs], 0.0f, 1.0f) * m;
			}}}
			return image;
		} else {
			file_image image;
			image.X = sample.dim.X;
			image.Y = sample.dim.Y;
			image.C = sample.dim.C;
			image.data.resize(image.X * image.Y * image.C);
			for(int sy=0;sy<image.Y;sy++) {
			for(int sx=0;sx<image.X;sx++) {
			for(int sc=0;sc<image.C;sc++) {
				const int iofs = image.get_offset(sx, sy, sc);
				const int sofs = sample.dim.get_offset(sx, sy, sc);
				image.data[iofs] = std::clamp(sample.data[sofs], 0.0f, 1.0f) * m;
			}}}
			return image;
		}
	}
}

#endif
