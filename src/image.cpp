
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
			int i = ((py*image_w) + px) * 4;
			if(channel == -1) {
				list.push_back(i+0);
				list.push_back(i+1);
				list.push_back(i+2);
				list.push_back(i+3);
			} else {
				list.push_back(i+channel);
			}
		}}
		// assert that all indices are within image bounds.
		// WARNING: this doesnt necessarily mean they are correct.
		for(int x=0;x<list.size();x++) {
			assert(0 <= list[x] && list[x] < (image_w*image_h*4));
		}
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
				delete[] imgdata;
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

	vector<file_image> load_images_in_directory(string dir) {
		vector<fs::directory_entry> entries = get_image_entries_in_directory(dir);
		vector<file_image> images;
		for(const fs::directory_entry entry : entries) {
			images.push_back(file_image::load(entry.path().string()));
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
			for(int x=0;x<data.size();x++) data[x] = 0.0f;
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

	void sample_area_copy(const file_image& image, sample_image& sample) {
		const float m = 1.0f / 255.0f;
		for(int y=0;y<image.h;y++) {
		for(int x=0;x<image.w;x++) {
			int i_image  = image.get_offset(x, y);
			int i_sample = sample.get_offset(x+sample.x0, y+sample.y0);
			for(int c=0;c<4;c++) sample.data[i_sample+c] = float(image.data[i_image+c]) * m;
		}}
	}
	void sample_area_minify_1x1_nearest(const file_image& image, sample_image& sample) {
		const float m = 1.0f / 255.0f;
		const float x_inv_scale_factor = float(image.w) / float(sample.x1 - sample.x0);
		const float y_inv_scale_factor = float(image.h) / float(sample.y1 - sample.y0);
		for(int y=sample.y0;y<sample.y1;y++) {
		for(int x=sample.x0;x<sample.x1;x++) {
			int ix = std::clamp(int(float(x-sample.x0) * x_inv_scale_factor), 0, image.w);
			int iy = std::clamp(int(float(y-sample.y0) * y_inv_scale_factor), 0, image.h);
			int i_image  = image.get_offset(ix, iy);
			int i_sample = sample.get_offset(x, y);
			for(int c=0;c<4;c++) sample.data[i_sample+c] = float(image.data[i_image+c]) * m;
		}}
	}
	void sample_area_minify_WxH_linear(const file_image& image, sample_image& sample) {
		// compute pixel fractions, i.e. area of sample-pixels that are covered when scaling down image-pixels.
		// NOTE: these are linearly seperable, and are thus computed seperately for each dimension.
		float mx = float(sample.x1 - sample.x0) / float(image.w);
		float my = float(sample.y1 - sample.y0) / float(image.h);
		struct pixel_fraction {
			int i0;// index of start pixel in sample.
			int i1;// index of end   pixel in sample.
			float f0;// amount of pixel-area covered on i0 side of integer coordinate boundary.
			float f1;// amount of pixel-area covered on i1 side of integer coordinate boundary.
		};
		vector<pixel_fraction> fractions_x(image.w);
		vector<pixel_fraction> fractions_y(image.h);
		for(int x=0;x<image.w;x++) {
			pixel_fraction pf;
			float p0 = float(x+0) * mx + float(sample.x0);
			float p1 = float(x+1) * mx + float(sample.x0);
			int i0 = pf.i0 = std::floor(p0);
			int i1 = pf.i1 = std::floor(p1);
			if(i0 == i1) {
				pf.f0 = mx;
				pf.f1 = 0;
			} else {
				pf.f0 = float(i1) - p0;
				pf.f1 = p1 - float(i1);
			}
			fractions_x[x] = pf;
		}
		for(int y=0;y<image.h;y++) {
			pixel_fraction pf;
			float p0 = float(y+0) * my + float(sample.y0);
			float p1 = float(y+1) * my + float(sample.y0);
			int i0 = pf.i0 = std::floor(p0);
			int i1 = pf.i1 = std::floor(p1);
			if(i0 == i1) {
				pf.f0 = my;
				pf.f1 = 0;
			} else {
				pf.f0 = float(i1) - p0;
				pf.f1 = p1 - float(i1);
			}
			fractions_y[y] = pf;
		}

		// convert image data to floats.
		vector<float> image_data(image.data.size());
		const float m = 1.0f / 255.0f;
		for(int x=0;x<image.data.size();x++) image_data[x] = float(image.data[x]) * m;

		// project image-area into sample-area.
		for(int y=0;y<image.h;y++) {
		for(int x=0;x<image.w;x++) {
			// get pixel colour.
			float clr[4];// colour of this image-pixel.
			for(int c=0;c<4;c++) clr[c] = image_data[image.get_offset(x, y) + c];
			// determine which sample-pixels the image-pixel intersects.
			const pixel_fraction& pfx = fractions_x[x];
			const pixel_fraction& pfy = fractions_y[y];
			// add pixel colours based on sample-pixel-area covered by image-pixel.
			bool p00 = true;
			bool p10 = pfx.i1 != pfx.i0;
			bool p01 = pfy.i1 != pfy.i0;
			bool p11 = p10 && p01;
			if(p00) {
				int ofs = sample.get_offset(pfx.i0, pfy.i0);
				float area = pfx.f0 * pfy.f0;
				for(int c=0;c<4;c++) sample.data[ofs+c] += clr[c]*area;
			}
			if(p10) {
				int ofs = sample.get_offset(pfx.i1, pfy.i0);
				float area = pfx.f1 * pfy.f0;
				for(int c=0;c<4;c++) sample.data[ofs+c] += clr[c]*area;
			}
			if(p11) {
				int ofs = sample.get_offset(pfx.i1, pfy.i1);
				float area = pfx.f1 * pfy.f1;
				for(int c=0;c<4;c++) sample.data[ofs+c] += clr[c]*area;
			}
			if(p01) {
				int ofs = sample.get_offset(pfx.i0, pfy.i1);
				float area = pfx.f0 * pfy.f1;
				for(int c=0;c<4;c++) sample.data[ofs+c] += clr[c]*area;
			}
		}}

		// clamp sample image colours.
		// sometimes colour values get slightly out of range due to loss-of-precision.
		for(int x=0;x<sample.data.size();x++) if(sample.data[x] > 1.0f || sample.data[x] < 0.0f) {
			//printf("pixel value out of bounds: %f\n", sample.data[x]);
			sample.data[x] = std::clamp(sample.data[x], 0.0f, 1.0f);
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
	void generate_sample_image(const file_image& image, sample_image& sample) {
		sample.clear();

		// if loaded image is smaller than sample area, then copy.
		if(image.w <= sample.w && image.h <= sample.h) {
			int remaining_w = sample.w - image.w;
			int remaining_h = sample.h - image.h;
			sample.x0 = remaining_w/2;
			sample.y0 = remaining_h/2;
			sample.x1 = sample.x0 + image.w;
			sample.y1 = sample.y0 + image.h;
			sample_area_copy(image, sample);
		}

		// if loaded image is larger than sample area, then scale down.
		else {
			// determine which dimension to scale against.
			float iw_over_sw = float(image.w) / float(sample.w);
			float ih_over_sh = float(image.h) / float(sample.h);
			float scale_factor = 1;
			if(iw_over_sw >= ih_over_sh) {
				// scale to sample width.
				int scaled_h = std::min(((image.h * sample.w) / image.w), sample.h);
				sample.x0 = 0;
				sample.x1 = sample.w;
				int remaining_h = sample.h  - scaled_h;
				sample.y0 = remaining_h / 2;
				sample.y1 = (sample.y0 + scaled_h);
			} else {
				// scale to sample height.
				int scaled_w = std::min(((image.w * sample.h) / image.h), sample.w);
				sample.y0 = 0;
				sample.y1 = sample.h;
				int remaining_w = sample.w  - scaled_w;
				sample.x0 = remaining_w / 2;
				sample.x1 = (sample.x0 + scaled_w);
			}

			// sample image data.
			sample_area_minify_WxH_linear(image, sample);
		}
	};

	void generate_error_image(const sample_image& input, const sample_image& output, sample_image& error) {
		for(int x=0;x<error.data.size();x++) {
			error.data[x] = input.data[x] - output.data[x];
		}
	}
}

#endif
