#include "../src/utils/commandline.cpp"
#include "../src/image.cpp"
#include <string>
#include <filesystem>

/*
build:
g++ -std=c++23 -O2 -o "./projects/autoencoder.elf" "./projects/autoencoder.cpp"

run:
./projects/autoencoder.elf -m MODELS_DIR -i INPUT_IMAGES_DIR -o OUTPUT_IMAGES_DIR
./projects/autoencoder.elf \
	-m ./data/models \
	-i ./data/images/images_1 \
	-o ./data/output/images_1

*/

using std::string;
namespace fs = std::filesystem;

void print_usage(string msg) {
	printf("%s\n", msg.c_str());
	printf("usage:\n");
	printf("-m MODELS_DIR -i INPUT_IMAGES_DIR -o OUTPUT_IMAGES_DIR\n");
}

int main(const int argc, const char** argv) {
	// parse arguments.
	const utils::commandline::cmd_arguments arguments(argc, argv);
	arguments.print();
	string model_dir;
	string input_dir;
	string output_dir;
	if(arguments.named_arguments.contains("-m")) model_dir  = arguments.named_arguments.at("-m"); else { print_usage("missing MODELS_DIR"); return 1; }
	if(arguments.named_arguments.contains("-i")) input_dir  = arguments.named_arguments.at("-i"); else { print_usage("missing INPUT_IMAGES_DIR"); return 1; }
	if(arguments.named_arguments.contains("-o")) output_dir = arguments.named_arguments.at("-o"); else { print_usage("missing OUTPUT_IMAGES_DIR"); return 1; }
	// TODO - remove trailing slashes ('/') if directory has them.

	// load images.
	std::vector<ML::image::file_image> images = ML::image::load_images_in_directory(input_dir);
	int memsize = 0;
	for(const auto& image : images) {
		image.print();
		memsize += image.data.size() * sizeof(image.data[0]);
	}
	printf("total memsize: %.3f MiB\n", float(memsize)/(1024*1024));

	// generate sample images.
	for(const auto& image : images) {
		printf("generating sample of: %s\n", image.path.c_str());
		ML::image::sample_image sample(768, 768);
		ML::image::generate_sample(image, sample);
		printf("generated sample\n");
		fs::path outpath = fs::path(output_dir) / fs::path(image.path).filename().concat(".png");
		ML::image::file_image output = sample.to_byte_image();
		output.ch_orig = 4;
		output.print();
		ML::image::file_image::save(output, outpath.string(), 4);
		printf("saved sample: %s\n", outpath.c_str());
	}

	return 0;
}
