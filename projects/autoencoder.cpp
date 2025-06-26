#include "../src/utils/commandline.cpp"
#include "../src/image.cpp"
#include "../src/performance.cpp"
#include <string>

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
	std::vector<string> flags { "-m", "-i", "-o" };
	for(string flag : flags) {
		if(!arguments.named_arguments.contains(flag)) {
			print_usage("missing argument: " + flag);
			return 1;
		}
		string arg = arguments.named_arguments.at(flag);
		if(arg.ends_with('/')) {
			print_usage("path shouldnt have trailing slash: " + flag + "\n\t" + arg);
			return 1;
		}
	}
	string model_dir  = arguments.named_arguments.at("-m");
	string input_dir  = arguments.named_arguments.at("-i");
	string output_dir = arguments.named_arguments.at("-o");

	// load images.
	ML::performance::timepoint t0 = ML::performance::now();
	std::vector<ML::image::file_image> images = ML::image::load_images_in_directory(input_dir);
	int memsize = 0;
	for(const auto& image : images) {
		//image.print();
		memsize += image.data.size() * sizeof(image.data[0]);
	}
	printf("total memsize: %.3f MiB\n", float(memsize)/(1024*1024));
	ML::performance::timepoint t1 = ML::performance::now();
	printf("time taken: %li us\n", t1.delta_us(t0));

	///*
	// generate sample images.
	for(const auto& image : images) {
		//printf("generating sample of: %s\n", image.path.c_str());
		ML::image::sample_image sample(768, 768);
		ML::image::generate_sample(image, sample);
		//printf("generated sample\n");
		fs::path outpath = fs::path(output_dir) / fs::path(image.path).filename().concat(".png");
		ML::image::file_image output = sample.to_byte_image();
		output.ch_orig = 4;
		//output.print();
		//ML::image::file_image::save(output, outpath.string(), 4);
		//printf("saved sample: %s\n", outpath.c_str());
	}
	ML::performance::timepoint t2 = ML::performance::now();
	printf("time taken: %li us\n", t2.delta_us(t1));
	//*/

	return 0;
}
