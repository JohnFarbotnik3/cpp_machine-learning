#include "../src/utils/commandline.cpp"
#include "../src/utils/random.cpp"
#include "../src/image.cpp"
#include "../src/stats.cpp"
#include "../src/models/autoencoder.cpp"
#include <filesystem>
#include <string>

/*
build:
g++ -std=c++23 -O2 -o "./projects/autoencoder.elf" "./projects/autoencoder.cpp"

run:
./projects/autoencoder.elf -m MODELS_DIR -i INPUT_IMAGES_DIR -o OUTPUT_IMAGES_DIR
./projects/autoencoder.elf \
	-m ./data/models \
	-i ./data/images/images_02 \
	-o ./data/output/images_02 \
	-w 512 \
	-h 512 \
	-tc 200 \
	-tcp 10 \
	-bsz 5 \
	-lr 0.01 \
	-seed 12345

*/

using std::string;
using std::vector;
namespace fs = std::filesystem;

void print_usage(string msg) {
	printf("%s\n", msg.c_str());
	printf("usage:\n");
	printf("-m MODELS_DIR -i INPUT_IMAGES_DIR -o OUTPUT_IMAGES_DIR\n");
}

void training_cycle(
	ML::models::autoencoder& model,
	ML::stats::training_stats& stats,
	vector<fs::directory_entry>& image_entries,
	string output_dir,
	int minibatch_size,
	float learning_rate,
	std::mt19937& gen32
) {
	// divide entries into batches.
	int n_batches = 1 + (image_entries.size() / minibatch_size);
	vector<vector<fs::directory_entry>> image_minibatches(n_batches);
	std::uniform_int_distribution<int> distr(0, n_batches-1);
	for(const auto entry : image_entries) {
		int x = distr(gen32);
		image_minibatches[x].push_back(entry);
	}

	// run training cycle.
	ML::image::sample_image image_input (model.input_w, model.input_h);
	ML::image::sample_image image_output(model.input_w, model.input_h);
	ML::image::sample_image image_error (model.input_w, model.input_h);
	ML::image::sample_image image_temp  (model.input_w, model.input_h);
	for(const auto& minibatch : image_minibatches) {
		// run minibatch.
		for(const auto& entry : minibatch) {
			using timepoint = ML::stats::timepoint;
			timepoint t0;
			timepoint t1;

			// load image.
			t0 = timepoint::now();
			ML::image::file_image loaded_image = ML::image::file_image::load(entry.path().string());
			t1 = timepoint::now();
			stats.push_value("dt load image", t1.delta_us(t0));

			// generate sample.
			t0 = timepoint::now();
			ML::image::generate_sample_image(loaded_image, image_input);
			t1 = timepoint::now();
			stats.push_value("dt gen sample", t1.delta_us(t0));

			// propagate.
			t0 = timepoint::now();
			model.propagate(image_input.data, image_output.data);
			t1 = timepoint::now();
			stats.push_value("dt propagate", t1.delta_us(t0));

			// compute error.
			t0 = timepoint::now();
			ML::image::generate_error_image(image_input, image_output, image_error);
			float avg_error = 0;
			for(int x=0;x<image_error.data.size();x++) avg_error += std::abs(image_error.data[x]);
			avg_error /= image_error.data.size();
			t1 = timepoint::now();
			stats.push_value("dt error", t1.delta_us(t0));
			stats.push_value("avg error", avg_error);

			// backpropagate.
			t0 = timepoint::now();
			model.back_propagate(image_error.data, image_temp.data, image_input.data);
			t1 = timepoint::now();
			stats.push_value("dt backprop", t1.delta_us(t0));
		}
		// adjust model.
		model.apply_batch_error(learning_rate / minibatch.size());
	}
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
	int input_w = arguments.get_named_value("-w", 512);
	int input_h = arguments.get_named_value("-h", 512);
	int n_training_cycles = arguments.get_named_value("-tc", 20);
	int training_print_itv = arguments.get_named_value("-tcp", 1);
	int minibatch_size = arguments.get_named_value("-bsz", 5);
	float learning_rate = arguments.get_named_value("-lr", 0.001f);
	int seed = arguments.get_named_value("-seed", 12345);

	// get list of images in input directory.
	printf("examining input directory.\n");
	vector<fs::directory_entry> image_entries = ML::image::get_image_entries_in_directory(input_dir);
	printf("found %lu images.\n", image_entries.size());

	// create and initialize model.
	printf("initializing model.\n");
	ML::models::autoencoder model(input_w, input_h);
	model.init_model_parameters(seed, 0.0f, 0.3f, 0.0f, 0.2f);

	// train model.
	printf("starting training.\n");
	ML::stats::training_stats stats;
	std::mt19937 gen32 = utils::random::get_generator_32(seed);
	for(int z=1;z<=n_training_cycles;z++) {
		// run training batch.
		printf("training cycle: %i / %i.\n", z, n_training_cycles);
		training_cycle(model, stats, image_entries, output_dir, minibatch_size, learning_rate, gen32);
		// print stats.
		if(z % training_print_itv == 0) {
			stats.print_all();
			stats.clear_all();
		}
	}
	printf("done training.\n");

	// test model by outputting images.
	printf("outputting decoded images.\n");
	ML::image::sample_image image_input (model.input_w, model.input_h);
	ML::image::sample_image image_output(model.input_w, model.input_h);
	for(const fs::directory_entry entry : image_entries) {
		// load and propagate.
		ML::image::file_image loaded_image = ML::image::file_image::load(entry.path().string());
		ML::image::generate_sample_image(loaded_image, image_input);
		model.propagate(image_input.data, image_output.data);
		// output result.
		fs::path outpath = fs::path(output_dir) / fs::path(entry.path()).filename().concat(".png");
		ML::image::file_image output = image_output.to_byte_image();
		ML::image::file_image::save(output, outpath.string(), 4);
	}
	printf("done.\n");

	return 0;
}
