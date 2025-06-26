
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <memory>
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
	using std::unique_ptr;// TODO - check if there is a performance difference between unique_ptr and vector.
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

	struct image_dimensions {
		int w;// layer width.
		int h;// layer height.
		image_dimensions() = default;
		image_dimensions(int w, int h) {
			this->w = w;
			this->h = h;
		}
		int  get_area() const { return w*h; }
		int  get_index(int x, int y) const { return x*h + y; }
		bool in_bounds(int x, int y) const { return (0 <= x) & (x < w) & (0 <= y) & (y < h); }
	};

	vector<int> generate_image_data_indices(int image_w, int image_h, int x, int y, int w, int h, int channel=-1) {
		// clip to image area.
		int x0 = std::max(x, 0);
		int y0 = std::max(y, 0);
		int x1 = std::min(x+w, image_w);
		int y1 = std::min(y+h, image_h);
		// generate indices.
		vector<int> list;
		for(int py=y0;py<y1;py++) {
			for(int px=x0;px<x1;px++) {
				if(channel == -1) {
					int i = ((py*image_w) + px) * 4;
					list.push_back(i+0);
					list.push_back(i+1);
					list.push_back(i+2);
					list.push_back(i+3);
				} else {
					int i = ((py*image_w) + px) * 4 + channel;
					list.push_back(i);
				}
			}}
			return list;
	}

	/*
		image containing RGBA byte data loaded from image files.
		these are typically all loaded before training.

		due to memory constraints, these are stored as bytes
		and then converted to floats during sampling.
	*/
	struct file_image {
		string path = "";
		vector<byte> data = vector<byte>(0);
		int w = 0;
		int h = 0;
		int ch_orig = 0;

		void print() const {
			printf("<loaded_image>\n");
			printf("path   : %s\n", path.c_str());
			printf("area   : %i x %i (%i channels)\n", w, h, ch_orig);
			printf("memsize: %.3f MiB\n", float(data.size()*sizeof(byte))/(1024*1024));
		}

		int get_offset(int x, int y) const {
			return (y*w + x) * 4;
		}

		// flip image data in the y-axis.
		static vector<byte> flip_data_y(const vector<byte>& imgdata, const int imgw, const int imgh) {
			vector<byte> temp(imgdata.size());
			for(int y=0;y<imgh;y++) {
				int i_in = imgw * 4 * (imgh-1-y);
				int i_fl = imgw * 4 * y;
				memcpy(temp.data() + i_fl, imgdata.data() + i_in, imgw * 4 * sizeof(byte));
			}
			return temp;
		}

		static file_image load(string filepath) {
			file_image image;
			byte* imgdata = stbi_load(filepath.c_str(), &image.w, &image.h, &image.ch_orig, 4);
			if(imgdata == NULL) {
				image.w=0;
				image.h=0;
			} else {
				int size = image.w * image.h * 4;
				vector<byte> imgdata_loaded(imgdata, imgdata+size);
				image.data = flip_data_y(imgdata_loaded, image.w, image.h);
				image.path = filepath;
			}
			return image;
		}

		static bool save(const file_image& image, string filepath, const int num_channels) {
			// create parent directories as needed.
			std::error_code ec;
			fs::path parent_dir = fs::path(filepath).parent_path();
			fs::create_directories(parent_dir, ec);
			if(ec) printf("error: %s\n", ec.message().c_str());
			// write file.
			bool allowed = utils::file_io::can_write_file(filepath);
			if(!allowed) return false;
			vector<byte> out_data = flip_data_y(image.data, image.w, image.h);
			int row_stride_in_bytes = image.w * 4 * sizeof(byte);
			int success = stbi_write_png(filepath.c_str(), image.w, image.h, num_channels, out_data.data(), row_stride_in_bytes);
			if(success == 0) {
				printf("failed to save image: %s\n", filepath.c_str());
				printf("errno: %s\n", strerror(errno));
			}
			return success != 0;
		}
	};

	vector<file_image> load_images_in_directory(string dir) {
		// iterate through directory - loading files with recognized exteions.
		vector<file_image> images;
		vector<string> extensions = { ".png", ".jpg", ".jpeg" };
		fs::directory_iterator iter(dir);
		for(const fs::directory_entry entry : iter) {
			bool match = false;
			string f_path = entry.path().string();
			string f_ext  = entry.path().extension().string();
			for(const string& ext : extensions) if(ext.compare(f_ext) == 0) match=true;
			if(match) {
				printf("loading image: %s [%s]\n", f_ext.c_str(), f_path.c_str());
				images.push_back(file_image::load(f_path));
			} else {
				printf("unrecognized extension: %s [%s]\n", f_ext.c_str(), f_path.c_str());
			}
		}
		return images;
	}

	struct sample_image {
		vector<float> data;
		// dimensions of data.
		int w = 0;
		int h = 0;
		// sample area.
		int x0 = 0, x1 = 0;
		int y0 = 0, y1 = 0;

		sample_image(int w, int h) {
			this->w = w;
			this->h = h;
			this->data = vector<float>(w*h*4);
		}

		void clear() {
			for(int x=0;x<data.size();x++) data[x] = 1.0f;// TODO TEST - set this to 0
			x0 = x1 = 0;
			y0 = y1 = 0;
		}

		int get_offset(int x, int y) const {
			return (y*w + x) * 4;
		}

		file_image to_byte_image() {
			file_image image;
			image.data = vector<byte>(data.size());
			image.w = w;
			image.h = h;
			const float m = 255.0f / 1.0f;
			for(int x=0;x<data.size();x++) image.data[x] = byte(data[x] * m);
			return image;
		}
	};

	/*
		generate input values by sampling loaded training images.

		- if the original image is smaller than the input area,
		then the image is just copied to input and centered.

		- if the original image is larger than the input area,
		then it is scaled to fit in input area.

		values outside the sample area are set to 0 and are ignored
		when computing error.
	*/
	void generate_sample(const file_image& image, sample_image& sample) {
		sample.clear();

		printf("0\n");
		// if loaded image is smaller than sample area, then copy.
		if(image.w <= sample.w && image.h <= sample.h) {
			printf("1\n");
			int remaining_w = sample.w - image.w;
			int remaining_h = sample.h - image.h;
			sample.x0 = remaining_w/2;
			sample.y0 = remaining_h/2;
			sample.x1 = sample.x0 + image.w;
			sample.y1 = sample.y0 + image.h;
			const float m = 1.0f / 255.0f;
			for(int y=0;y<image.h;y++) {
			for(int x=0;x<image.w;x++) {
				int i_image  = image.get_offset(x, y);
				int i_sample = sample.get_offset(x+sample.x0, y+sample.y0);
				for(int c=0;c<4;c++) sample.data[i_sample+c] = float(image.data[i_image+c]) * m;
			}}
		}

		// if loaded image is larger than sample area, then scale down.
		else {
			printf("A\n");
			// determine which dimension to scale against.
			float iw_over_sw = float(image.w) / float(sample.w);
			float ih_over_sh = float(image.h) / float(sample.h);
			float scale_factor = 1;
			printf("B\n");
			if(iw_over_sw >= ih_over_sh) {
				// scale to sample width.
				scale_factor = 1.0f / iw_over_sw;
				int scaled_h = std::min(int(image.h * scale_factor), sample.h);
				sample.x0 = 0;
				sample.x1 = sample.w;
				int remaining_h = sample.h  - scaled_h;
				sample.y0 = remaining_h / 2;
				sample.y1 = (sample.y0 + scaled_h);
				printf("C\n");
			} else {
				// scale to sample height.
				scale_factor = 1.0f / ih_over_sh;
				int scaled_w = std::min(int(image.w * scale_factor), sample.w);
				sample.y0 = 0;
				sample.y1 = sample.h;
				int remaining_w = sample.w  - scaled_w;
				sample.x0 = remaining_w / 2;
				sample.x1 = (sample.x0 + scaled_w);
				printf("D\n");
			}
			printf("image  area: %i, %i, %i, %i\n", 0, 0, image.w, image.h);
			printf("sample area: %i, %i, %i, %i\n", sample.x0, sample.y0, sample.x1, sample.y1);

			// sample image data.
			// METHOD: nearest neighbour - TODO: details are very jagged, switch to 3x3 weighted sum.
			printf("E\n");
			const float m = 1.0f / 255.0f;
			const float inv_scale_factor = 1.0f / scale_factor;
			for(int y=sample.y0;y<sample.y1;y++) {
			for(int x=sample.x0;x<sample.x1;x++) {
				int ix = std::clamp(int(float(x-sample.x0) * inv_scale_factor), 0, image.w);
				int iy = std::clamp(int(float(y-sample.y0) * inv_scale_factor), 0, image.h);
				int i_image  = image.get_offset(ix, iy);
				int i_sample = sample.get_offset(x, y);
				for(int c=0;c<4;c++) sample.data[i_sample+c] = float(image.data[i_image+c]) * m;
			}}
			printf("F\n");
		}
	};
}
