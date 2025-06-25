#include "../src/utils/commandline.cpp"
#include "../src/image.cpp"
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

void print_usage() {
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
	if(arguments.named_arguments.contains("-m")) model_dir  = arguments.named_arguments.at("-m"); else { print_usage(); return 1; }
	if(arguments.named_arguments.contains("-i")) input_dir  = arguments.named_arguments.at("-i"); else { print_usage(); return 1; }
	if(arguments.named_arguments.contains("-o")) output_dir = arguments.named_arguments.at("-o"); else { print_usage(); return 1; }

	// load images.
	std::vector<ML::image::loaded_image> images = ML::image::load_images_in_directory(input_dir);
	int memsize = 0;
	for(const auto& image : images) {
		image.print();
		memsize += image.data.size() * sizeof(image.data[0]);
	}
	printf("total memsize: %.3f MiB\n", float(memsize)/(1024*1024));


	return 0;
}
