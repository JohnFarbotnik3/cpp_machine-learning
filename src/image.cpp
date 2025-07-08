
#ifndef F_image
#define F_image

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>
#include "./utils/file_io.cpp"

#define STB_IMAGE_IMPLEMENTATION
#include "./stb-master/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./stb-master/stb_image_write.h"

/*
	NOTE: this namespace assumes that image-data is in row-major form
	starting from bottom-left corner, and that RGBA values are interleaved.
*/
namespace ML::image {
	using std::vector;
	using std::string;
	namespace fs = std::filesystem;
	using byte = unsigned char;

	template<class T>
	struct vec2 {
		T x,y;

		static vec2 linear_combination(vec2<T> a, T a_mult, vec2<T> b, T b_mult) {
			T x = (a.x * a_mult) + (b.x * b_mult);
			T y = (a.y * a_mult) + (b.y * b_mult);
			return vec2<T>{ x,y };
		}
	};

	/*
		image containing RGBA byte data loaded from image files.
		these are typically all loaded before training.

		due to memory constraints, these are stored as bytes
		and then converted to floats during sampling.
	*/
	struct file_image {
		string path = "";
		vector<byte> data = vector<byte>(0);
		int X = 0;
		int Y = 0;
		int C = 0;
		int C_orig = 0;

		void print() const {
			printf("<loaded_image>\n");
			printf("path   : %s\n", path.c_str());
			printf("area   : %i x %i (%i -> %i channels)\n", X, Y, C_orig, C);
			printf("memsize: %.3f MiB\n", float(data.size()*sizeof(byte))/(1024*1024));
		}

		int get_offset(int x, int y, int c) const {
			return (y*X + x) * C + c;
		}

		// flip image data in the y-axis.
		static vector<byte> flip_data_y(const vector<byte>& imgdata, const int imgw, const int imgh, const int imgch) {
			vector<byte> temp(imgdata.size());
			int row_length = imgw * imgch;
			for(int y=0;y<imgh;y++) {
				int i_fl = row_length * y;
				int i_in = row_length * (imgh-1-y);
				memcpy(temp.data() + i_fl, imgdata.data() + i_in, row_length * sizeof(byte));
			}
			return temp;
		}

		static file_image load(string filepath, int channels) {
			file_image image;
			image.C = channels;
			byte* imgdata = stbi_load(filepath.c_str(), &image.X, &image.Y, &image.C_orig, image.C);
			if(imgdata == NULL) {
				image.X=0;
				image.Y=0;
			} else {
				int size = image.X * image.Y * image.C;
				vector<byte> imgdata_loaded(imgdata, imgdata+size);
				image.data = flip_data_y(imgdata_loaded, image.X, image.Y, image.C);
				image.path = filepath;
				free(imgdata);
			}
			return image;
		}

		static bool save(const file_image& image, string filepath, const int num_channels) {
			printf("save(): X=%i, Y=%i, C=%i, ch=%i, size=%lu, path=%s\n", image.X, image.Y, image.C, num_channels, image.data.size(), filepath.c_str());
			// create parent directories as needed.
			std::error_code ec;
			fs::path parent_dir = fs::path(filepath).parent_path();
			fs::create_directories(parent_dir, ec);
			if(ec) printf("error: %s\n", ec.message().c_str());
			// write file.
			bool allowed = utils::file_io::can_write_file(filepath);
			if(!allowed) return false;
			vector<byte> out_data = flip_data_y(image.data, image.X, image.Y, image.C);
			int row_length = image.X * image.C;
			int success = stbi_write_png(filepath.c_str(), image.X, image.Y, num_channels, out_data.data(), row_length * sizeof(byte));
			if(success == 0) {
				printf("failed to save image: %s\n", filepath.c_str());
				printf("errno: %s\n", strerror(errno));
			}
			return success != 0;
		}
	};

	vector<fs::directory_entry> get_image_entries_in_directory(string dir) {
		vector<string> extensions = { ".png", ".jpg", ".jpeg" };
		vector<fs::directory_entry> entries;
		fs::directory_iterator iter(dir);
		for(const fs::directory_entry entry : iter) {
			bool match = false;
			string f_path = entry.path().string();
			string f_ext  = entry.path().extension().string();
			for(const string& ext : extensions) if(ext.compare(f_ext) == 0) match=true;
			if(match) entries.push_back(entry);
		}
		return entries;
	}

	/*
		a simple iterator for iterating on an image
		with a variable number of channels.

		NOTE: image is assumed to be in row-major form
		with interleaved colour channels.
	*/
	struct variable_image_iterator {
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

		variable_image_iterator() = default;
		variable_image_iterator(
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

	/*
		iterator for accessing images with a tiled memory layout.

		effectively, the super-image is made of a grid of sub-images,
		which are themselves made of a grid of pixels.

		NOTE: this is an act of desperation to try an fix what
		seems to be a memory bandwidth issue. hopefully it will improve
		caching performance considerably.
	*/
	struct variable_image_tile_iterator {
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
		variable_image_iterator inner_iterator;
		// iterates through pixels in the current tile.
		variable_image_iterator outer_iterator;

		variable_image_tile_iterator(
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
			outer_iterator = variable_image_iterator(
				X/TX, Y/TY, C/TC,
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
		void setup_inner_iterator(const variable_image_iterator& outer) {
			int tx = outer.x * TX;
			int ty = outer.y * TY;
			int tc = outer.c * TC;
			inner_iterator = variable_image_iterator(
				TX, TY, TC,
				std::max(x0-tx, 0), std::min(x1-tx, TX),
				std::max(y0-ty, 0), std::min(y1-ty, TY),
				std::max(c0-tc, 0), std::min(c1-tc, TC)
			);
		}
	};

	/*
		an image with a variable number of channels.

		NOTE: this is intended to be populated externally,
		either by hand or with the help of image iterators.
	*/
	template<class T>
	struct variable_image_tiled {
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

		variable_image_tiled(
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

		variable_image_tile_iterator get_iterator(
			int x0, int x1,
			int y0, int y1,
			int c0, int c1
		) const {
			return variable_image_tile_iterator(TX, TY, TC, X, Y, C, x0, x1, y0, y1, c0, c1);
		}
	};

	void sample_area_copy(variable_image_tiled<float>& sample, const file_image& image) {
		// assert that image-area and sample-area match.
		assert((sample.x1 - sample.x0) == image.X);
		assert((sample.y1 - sample.y0) == image.Y);
		assert(sample.C == image.C);

		variable_image_tile_iterator sample_iter = sample.get_iterator(
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
	void sample_area_nearest_neighbour(variable_image_tiled<float>& sample, const file_image& image) {
		variable_image_tile_iterator sample_iter = sample.get_iterator(
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
	void sample_area_linear(variable_image_tiled<float>& sample, const file_image& image) {
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

		variable_image_tile_iterator sample_iter = sample.get_iterator(
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
		generate input values by sampling loaded training images.

		- if the original image is smaller than the input area,
		then the image is just copied to input and centered.

		- if the original image is larger than the input area,
		then it is scaled to fit in input area.

		values outside the sample area are set to 0 and are ignored
		when computing error.
	*/
	void generate_sample_image(variable_image_tiled<float>& sample, const file_image& image) {
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

	file_image to_byte_image(variable_image_tiled<float>& sample, bool clamp_to_sample_area) {
		file_image image;
		image.X = sample.X;
		image.Y = sample.Y;
		image.C = sample.C;
		image.data.resize(image.X * image.Y * image.C);

		variable_image_tile_iterator sample_iter = clamp_to_sample_area ? sample.get_iterator(
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










