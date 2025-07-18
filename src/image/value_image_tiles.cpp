
#ifndef F_image
#define F_image

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>
#include "./file_image.cpp"
#include "./value_image_lines.cpp"

/*
	NOTE: this namespace assumes that image-data is in row-major form
	starting from bottom-left corner, and that RGBA values are interleaved.
*/
namespace ML::image::value_image_tiles {
	using std::vector;
	using ML::image::value_image_lines::value_image_lines_iterator;
	using ML::image::value_image_lines::value_image_lines_dimensions;

	// TODO - remove dependence on "value_image_iterator".
	/*
		iterator for accessing images with a tiled memory layout.

		effectively, the super-image is made of a grid of sub-images,
		which are themselves made of a grid of pixels.

		NOTE: this is an act of desperation to try an fix what
		seems to be a memory bandwidth issue. hopefully it will improve
		caching performance considerably.
	*/
	struct value_image_tiles_iterator {
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
		// tile dimensions.
		int TX,TY,TC;
		// length of tile in image data.
		int TILE_LENGTH;
		// iterates through tiles in the image.
		value_image_lines_iterator inner_iterator;
		// iterates through pixels in the current tile.
		value_image_lines_iterator outer_iterator;

		value_image_tiles_iterator(
			int TX, int TY, int TC,
			int X, int Y, int C,
			int x0, int x1,
			int y0, int y1,
			int c0, int c1
		) {
			this->X = X;
			this->Y = Y;
			this->C = C;
			this->TX = TX;
			this->TY = TY;
			this->TC = TC;
			TILE_LENGTH = TX * TY * TC;
			this->x0 = x0;
			this->x1 = x1;
			this->y0 = y0;
			this->y1 = y1;
			this->c0 = c0;
			this->c1 = c1;
			this->x = x0;
			this->y = y0;
			this->c = c0;
			setup_outer_iterator();
			setup_inner_iterator(outer_iterator);
			this->i = outer_iterator.i * TILE_LENGTH + inner_iterator.i;
			this->irem = length();
			// assertions.
			assert(x0 >= 0);
			assert(y0 >= 0);
			assert(c0 >= 0);
			assert(x1 <= X);
			assert(y1 <= Y);
			assert(c1 <= C);
			assert(X % TX == 0);
			assert(Y % TY == 0);
			assert(C % TC == 0);
		}

		int length() {
			return std::max(x1-x0, 0) * std::max(y1-y0, 0) * std::max(c1-c0, 0);
		}
		bool has_next() {
			return irem > 0;
		}
		int next() {
			irem--;
			//printf("next: x=%i, y=%i, c=%i, i=%i\n", x, y, c, i);
			inner_iterator.next();
			if(!inner_iterator.has_next()) {
				outer_iterator.next();
				setup_inner_iterator(outer_iterator);
			}
			x = outer_iterator.x * TX + inner_iterator.x;
			y = outer_iterator.y * TY + inner_iterator.y;
			c = outer_iterator.c * TC + inner_iterator.c;
			i = outer_iterator.i * TILE_LENGTH + inner_iterator.i;
			return i;
		}

	private:
		/*
			create iterator that iterates through all tiles
			required to cover image bounds.
		*/
		void setup_outer_iterator() {
			outer_iterator = value_image_lines_iterator(
				value_image_lines_dimensions(X/TX, Y/TY, C/TC),
				x0/TX, x1/TX + (x1%TX != 0 ? 1 : 0),
				y0/TY, y1/TY + (y1%TY != 0 ? 1 : 0),
				c0/TC, c1/TC + (c1%TC != 0 ? 1 : 0)
			);
		}
		/*
			create iterator that iterates through pixels in tile,
			clamped to image bounds.

			NOTE: this depends on state of outer_iterator.
		*/
		void setup_inner_iterator(const value_image_lines_iterator& outer) {
			int tx = outer.x * TX;
			int ty = outer.y * TY;
			int tc = outer.c * TC;
			inner_iterator = value_image_lines_iterator(
				value_image_lines_dimensions(TX, TY, TC),
				std::max(x0-tx, 0), std::min(x1-tx, TX),
				std::max(y0-ty, 0), std::min(y1-ty, TY),
				std::max(c0-tc, 0), std::min(c1-tc, TC)
			);
		}
	};

	struct value_image_tiles_dimensions {
		int X = 0;
		int Y = 0;
		int C = 0;
		int TX = 0;
		int TY = 0;
		int TC = 0;

		int length() const {
			return X * Y * C;
		}

		value_image_tiles_iterator get_iterator(int x0, int x1, int y0, int y1, int c0, int c1) const {
			return value_image_tiles_iterator(TX, TY, TC, X, Y, C, x0, x1, y0, y1, c0, c1);
		}
		value_image_tiles_iterator get_iterator() const {
			return value_image_tiles_iterator(TX, TY, TC, X, Y, C, 0, X, 0, Y, 0, C);
		}

		vector<int> generate_target_indices(int x0, int x1, int y0, int y1, int ch=-1) const {
			int c0 = (ch == -1) ? 0 : ch;
			int c1 = (ch == -1) ? C : ch+1;
			value_image_tiles_iterator iter = get_iterator(x0, x1, y0, y1, c0, c1);
			vector<int> list;
			while(iter.has_next()) {
				list.push_back(iter.i);
				iter.next();
			}
			return list;
		}
	};

	/*
		an image with a variable number of channels.

		NOTE: this is intended to be populated externally,
		either by hand or with the help of image iterators.
	*/
	template<class T>
	struct value_image_tiles {
		vector<T> data;
		int X = 0;// width.
		int Y = 0;// height.
		int C = 0;// number of channels.
		int TX = 0;// tile width.
		int TY = 0;// tile height.
		int TC = 0;// tile channels.
		// sample bounds.
		int x0,x1;
		int y0,y1;

		value_image_tiles(
			int X, int Y, int C,
			int TX, int TY, int TC
		) {
			assert(X % TX == 0);
			assert(Y % TY == 0);
			assert(C % TC == 0);

			this->X = X;
			this->Y = Y;
			this->C = C;
			this->TX = TX;
			this->TY = TY;
			this->TC = TC;
			this->data.resize(X * Y * C);
			clear();
		}

		void clear() {
			for(int x=0;x<data.size();x++) data[x] = 0;
		}

		value_image_tiles_iterator get_iterator(
			int x0, int x1,
			int y0, int y1,
			int c0, int c1
		) const {
			return value_image_tiles_iterator(TX, TY, TC, X, Y, C, x0, x1, y0, y1, c0, c1);
		}
	};

	void sample_area_copy(value_image_tiles<float>& sample, const file_image& image) {
		// assert that image-area and sample-area match.
		assert((sample.x1 - sample.x0) == image.X);
		assert((sample.y1 - sample.y0) == image.Y);
		assert(sample.C == image.C);

		value_image_tiles_iterator sample_iter = sample.get_iterator(
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

	void sample_area_nearest_neighbour(value_image_tiles<float>& sample, const file_image& image) {
		value_image_tiles_iterator sample_iter = sample.get_iterator(
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

	void sample_area_linear(value_image_tiles<float>& sample, const file_image& image) {
		// compute float conversions between sample coordinates to image coordinates.
		vector<int> floor_mx(sample.X+2);
		vector<int> floor_my(sample.Y+2);
		vector<int> ceil__mx(sample.X+2);
		vector<int> ceil__my(sample.Y+2);
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

		value_image_tiles_iterator sample_iter = sample.get_iterator(
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
	void generate_sample_image(value_image_tiles<float>& sample, const file_image& image) {
		sample.clear();

		// if loaded image is smaller than sample area, then copy.
		if(image.X <= sample.X && image.Y <= sample.Y) {
			int remaining_w = sample.X - image.X;
			int remaining_h = sample.Y - image.Y;
			sample.x0 = remaining_w/2;
			sample.y0 = remaining_h/2;
			sample.x1 = sample.x0 + image.X;
			sample.y1 = sample.y0 + image.Y;
			sample_area_copy(sample, image);
		}

		// if loaded image is larger than sample area, then scale down.
		else {
			// determine which dimension to scale against.
			float iw_over_sw = float(image.X) / float(sample.X);
			float ih_over_sh = float(image.Y) / float(sample.Y);
			float scale_factor = 1;
			if(iw_over_sw >= ih_over_sh) {
				// scale to sample width.
				int scaled_h = std::min(((image.Y * sample.X) / image.X), sample.Y);
				sample.x0 = 0;
				sample.x1 = sample.X;
				int remaining_h = sample.Y  - scaled_h;
				sample.y0 = remaining_h / 2;
				sample.y1 = (sample.y0 + scaled_h);
			} else {
				// scale to sample height.
				int scaled_w = std::min(((image.X * sample.Y) / image.Y), sample.X);
				sample.y0 = 0;
				sample.y1 = sample.Y;
				int remaining_w = sample.X  - scaled_w;
				sample.x0 = remaining_w / 2;
				sample.x1 = (sample.x0 + scaled_w);
			}

			// sample image data.
			sample_area_linear(sample, image);
		}
	};

	file_image to_file_image(value_image_tiles<float>& sample, bool clamp_to_sample_area) {
		file_image image;
		image.X = sample.X;
		image.Y = sample.Y;
		image.C = sample.C;
		image.data.resize(image.X * image.Y * image.C);

		value_image_tiles_iterator sample_iter = clamp_to_sample_area ? sample.get_iterator(
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










