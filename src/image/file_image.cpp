
#ifndef F_image_file
#define F_image_file

#include <cstddef>
#include <cstring>
#include <vector>
#include "src/utils/file_io.cpp"

#define STB_IMAGE_IMPLEMENTATION
#include "src/stb-master/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "src/stb-master/stb_image_write.h"

namespace ML::image {
	namespace fs = std::filesystem;
	using std::vector;
	using std::string;
	using byte = unsigned char;

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

	// flip image data in the y-axis.
	void flip_data_y(byte* dst, const byte* src, const int width, const int height, const int ch) {
		int row_length = width * ch;
		for(int y=0;y<height;y++) {
			int i_fl = row_length * y;
			int i_in = row_length * (height-1-y);
			memcpy(dst + i_fl, src + i_in, row_length * sizeof(byte));
		}
	}

	/*
		image containing RGBA byte data loaded from image files.
		these are typically all loaded before training.

		due to memory constraints, these are stored as bytes
		and then converted to floats during sampling.
	*/
	struct file_image {
		vector<byte> data = vector<byte>(0);
		int X = 0;
		int Y = 0;
		int C = 0;
		int C_orig = 0;

		int get_offset(int x, int y, int c) const {
			return (y*X + x) * C + c;
		}
	};

	file_image load_file_image(string filepath, int channels) {
		file_image image;
		image.C = channels;
		byte* image_data = stbi_load(filepath.c_str(), &image.X, &image.Y, &image.C_orig, channels);
		if(image_data == NULL) {
			image.X = 0;
			image.Y = 0;
			image.C = 0;
		} else {
			const int size = image.X * image.Y * image.C;
			image.data.resize(size);
			image.C = channels;
			flip_data_y(image.data.data(), image_data, image.X, image.Y, image.C);
			free(image_data);
		}
		return image;
	}

	bool save_file_image(const file_image& image, string filepath, const int channels) {
		// TODO - move this print statement somewhere else.
		printf("save(): X=%i, Y=%i, C=%i, ch=%i, size=%lu, path=%s\n", image.X, image.Y, image.C, channels, image.data.size(), filepath.c_str());
		// create parent directories as needed.
		std::error_code ec;
		fs::path parent_dir = fs::path(filepath).parent_path();
		fs::create_directories(parent_dir, ec);
		if(ec) printf("error: %s\n", ec.message().c_str());
		// write file.
		bool allowed = utils::file_io::can_write_file(filepath);
		if(!allowed) return false;
		int size = image.X * image.Y * image.C;
		int row_length = image.X * image.C;
		vector<byte> out_data(size);
		flip_data_y(out_data.data(), image.data.data(), image.X, image.Y, image.C);
		int success = stbi_write_png(filepath.c_str(), image.X, image.Y, channels, out_data.data(), row_length * sizeof(byte));
		if(success == 0) {
			printf("failed to save image: %s\n", filepath.c_str());
			printf("errno: %s\n", strerror(errno));
		}
		return success != 0;
	}

}

#endif
