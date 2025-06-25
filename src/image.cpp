
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
	struct loaded_image {
		string path;
		vector<byte> data;
		int w;
		int h;
		int c;// number of channels the original image has. NOTE: loaded_image always has 4 channels.

		loaded_image(string filepath) {
			path = filepath;
			// load image from file using stb.
			byte* imgdata = stbi_load(filepath.c_str(), &w, &h, &c, 4);
			if(imgdata == NULL) {
				w=0;
				h=0;
			} else {
				int size = w * h * 4;
				vector<byte> imgdata_loaded(imgdata, imgdata+size);
				data = flip_data_y(imgdata_loaded, w, h);
			}
		}

		void print() const {
			printf("<loaded_image>\n");
			printf("path   : %s\n", path.c_str());
			printf("area   : %i x %i (%i channels)\n", w, h, c);
			printf("memsize: %.3f MiB\n", float(data.size()*sizeof(byte))/(1024*1024));
		}

	private:
		// flip image data in the y-axis.
		vector<byte> flip_data_y(const vector<byte>& imgdata, const int imgw, const int imgh) {
			vector<byte> temp(imgdata.size());
			for(int y=0;y<imgh;y++) {
				int i_in = imgw * 4 * (imgh-1-y);
				int i_fl = imgw * 4 * y;
				memcpy(temp.data() + i_fl, imgdata.data() + i_in, imgw * 4 * sizeof(byte));
			}
			return temp;
		}

	public:
		bool save(string filepath) {
			bool allowed = utils::file_io::can_write_file(filepath);
			if(!allowed) return false;
			int status;
			utils::file_io::write_file(filepath, status, data.data(), data.size() * sizeof(byte));
			return status == 0;
		}
	};

	vector<loaded_image> load_images_in_directory(string dir) {
		// iterate through directory - loading files with recognized exteions.
		vector<loaded_image> images;
		vector<string> extensions = { ".png", ".jpg", ".jpeg" };
		fs::directory_iterator iter(dir);
		for(const fs::directory_entry entry : iter) {
			bool match = false;
			string f_path = entry.path().string();
			string f_ext  = entry.path().extension().string();
			for(const string& ext : extensions) if(ext.compare(f_ext) == 0) match=true;
			if(match) {
				printf("loading image: %s [%s]\n", f_ext.c_str(), f_path.c_str());
				images.push_back(loaded_image(f_path));
			} else {
				printf("unrecognized extension: %s [%s]\n", f_ext.c_str(), f_path.c_str());
			}
		}
		return images;
	}

	struct sample_area {
		int x,y,w,h;
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
	sample_area generate_sample(const loaded_image& image, vector<float>& input_values, const int input_w, const int input_h) {
		// TODO
		// - move sampling and float conversion here.
	};


	// OLD CODE ------------------------------------------------------------
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

}
