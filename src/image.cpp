
#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "./stb-master/stb_image.h"

/*
	NOTE: this namespace assumes that image-data is in row-major form
	starting from bottom-left corner, and that RGBA values are interleaved.
*/
namespace ML::image {
	using std::vector;
	using std::string;
	using std::unique_ptr;
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

	template<class T>
	struct rectangle {
		vec2<T> p0;
		vec2<T> p1;
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

	struct sample_image {
		unique_ptr<float[]> data;
		// dimensions of sample data.
		int w;
		int h;
		// relevant sample area - only the active sample area is used for computing loss.
		int area_x;
		int area_y;
		int area_w;
		int area_h;

		sample_image(int w, int h, string filepath) {
			this->data = unique_ptr<float[]>(new float[w*h*4]);
			this->w = w;
			this->h = h;

			// load image from file.
			int imgw;
			int imgh;
			int imgc;// number of channels in original image.
			unique_ptr<byte[]> imgdata_loaded(stbi_load(filepath.c_str(), &imgw, &imgh, &imgc, 4));
			unique_ptr<byte[]> imgdata_result = flip_data_y(imgdata_loaded, imgw, imgh);
			unique_ptr<float[]> imgdata_floats = image_bytes_to_floats(imgdata_result, imgw, imgh);

			// sample image.
			sample_1x1(imgdata_floats, imgw, imgh);
		}

	private:
		// flip image data in the y-axis.
		unique_ptr<byte[]> flip_data_y(const unique_ptr<byte[]>& imgdata, const int imgw, const int imgh) {
			unique_ptr<byte[]> temp(new byte[imgw*imgh*4]);
			for(int y=0;y<imgh;y++) {
				int i_in = imgw * 4 * (imgh-1-y);
				int i_fl = imgw * 4 * y;
				memcpy(temp.get() + i_fl, imgdata.get() + i_in, imgw * 4 * sizeof(byte));
			}
			return temp;
		}
		// convert bytes to floats.
		unique_ptr<float[]> image_bytes_to_floats(const unique_ptr<byte[]>& bytes, const int imgw, const int imgh) {
			const int len = imgw*imgh*4;
			const float m = 1.0f / 255.0f;
			unique_ptr<float[]> floats(new float[len]);
			for(int x=0;x<len;x++) floats[x] = float(bytes[x]) * m;
			return floats;
		}
		// convert floats to bytes.
		unique_ptr<byte[]> image_floats_to_bytes(const unique_ptr<float[]>& floats, const int imgw, const int imgh) {
			const int len = imgw*imgh*4;
			const float m = 255.0f;
			unique_ptr<byte[]> bytes(new byte[len]);
			for(int x=0;x<len;x++) bytes[x] = byte(floats[x] * m);
			return bytes;
		}

		// samples individual pixels from input image.
		void sample_1x1(const unique_ptr<float[]>& imgdata, const int imgw, const int imgh) {
			//TODO
			// NOTE: remember to set active sample area as well.
		}
		// samples individual pixels from input image.
		void sample_3x3_weighted_sum() {}//TODO
	};

}
