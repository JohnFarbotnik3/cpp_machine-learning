#include "../utils/commandline.cpp"
#include "../utils/random.cpp"
#include "../image.cpp"
#include "../stats.cpp"
#include "../models/autoencoder.cpp"
#include <cstdio>
#include <filesystem>
#include <string>

/*
build:
g++ -std=c++23 -O2 -o "./src/projects/autoencoder.elf" "./src/projects/autoencoder.cpp"
g++ -std=c++23 -O2 -march=native -o "./src/projects/autoencoder.elf" "./src/projects/autoencoder.cpp"

run:
./src/projects/autoencoder.elf \
-m ./data/models \
-i ./data/images/images_02 \
-o ./data/output/images_02 \
-w 512 \
-h 512 \
-tc 500 \
-tcp 10 \
-bsz 5 \
-lr 0.02 \
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
	const vector<fs::directory_entry>& image_entries,
	ML::models::autoencoder& model,
	ML::stats::training_stats& stats,
	int minibatch_size,
	float learning_rate,
	std::mt19937& gen32
) {
	// divide image entries into batches.
	vector<fs::directory_entry> pool = image_entries;
	vector<vector<fs::directory_entry>> image_minibatches;
	while(pool.size() > 0) {
		vector<fs::directory_entry> minibatch;
		for(int z=0;z<minibatch_size;z++) {
			std::uniform_int_distribution<int> distr(0, pool.size()-1);
			int x = distr(gen32);
			minibatch.push_back(pool[x]);
			pool[x] = pool[pool.size()-1];
			pool.pop_back();
		}
		image_minibatches.push_back(minibatch);
	}

	// run training cycle.
	ML::image::sample_image image_input (model.input_w, model.input_h);
	ML::image::sample_image image_output(model.input_w, model.input_h);
	ML::image::sample_image image_error (model.input_w, model.input_h);
	ML::image::sample_image image_temp  (model.input_w, model.input_h);
	for(const auto& minibatch : image_minibatches) {
		// run minibatch.
		assert(minibatch.size() > 0);
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

void update_learning_rate(
	const vector<fs::directory_entry>& image_entries,
	ML::models::autoencoder& model,
	ML::stats::training_stats& stats,
	int minibatch_size,
	float& learning_rate,
	std::mt19937& gen32,
	int cycles
) {
	ML::stats::training_stats stats_0;
	ML::stats::training_stats stats_1;
	ML::stats::training_stats stats_2;
	float lr_0 = learning_rate / 1.2f;
	float lr_1 = learning_rate * 1.0f;
	float lr_2 = learning_rate * 1.2f;
	ML::models::autoencoder model_0 = model;
	ML::models::autoencoder model_1 = model;
	ML::models::autoencoder model_2 = model;
	std::mt19937 gen32_0 = gen32;
	std::mt19937 gen32_1 = gen32;
	std::mt19937 gen32_2 = gen32;
	printf("trying rate = %f\n", lr_0);
	for(int x=0;x<cycles;x++) training_cycle(image_entries, model_0, stats_0, minibatch_size, lr_0, gen32_0);
	printf("trying rate = %f\n", lr_1);
	for(int x=0;x<cycles;x++) training_cycle(image_entries, model_1, stats_1, minibatch_size, lr_1, gen32_1);
	printf("trying rate = %f\n", lr_2);
	for(int x=0;x<cycles;x++) training_cycle(image_entries, model_2, stats_2, minibatch_size, lr_2, gen32_2);
	const vector<int> percentiles { 20 };
	float pct_error_0 = ML::stats::get_percentile_values(percentiles, stats_0.groups.at("avg error"))[0];
	float pct_error_1 = ML::stats::get_percentile_values(percentiles, stats_1.groups.at("avg error"))[0];
	float pct_error_2 = ML::stats::get_percentile_values(percentiles, stats_2.groups.at("avg error"))[0];
	float lowest_error = std::min(std::min(pct_error_0, pct_error_1), pct_error_2);
	if(lowest_error == pct_error_0) {
		learning_rate = lr_0;
		model = model_0;
		stats = stats_0;
		gen32 = gen32_0;
	}
	if(lowest_error == pct_error_1) {
		learning_rate = lr_1;
		model = model_1;
		stats = stats_1;
		gen32 = gen32_1;
	}
	if(lowest_error == pct_error_2) {
		learning_rate = lr_2;
		model = model_2;
		stats = stats_2;
		gen32 = gen32_2;
	}
}

void print_training_stats(ML::stats::training_stats& stats) {
	const vector<int> percentiles { 0, 1, 3, 10, 25, 50, 75, 90, 97, 99, 100 };
	const int FIRST_COLUMN_WIDTH = 20;
	const int COLUMN_WIDTH = 12;
	ML::stats::print_percentiles_header(percentiles, "%", "%i", COLUMN_WIDTH, FIRST_COLUMN_WIDTH);

	printf("STATS\n");
	for(const auto& [name, vec] : stats.groups) {
		ML::stats::print_percentiles(percentiles, name, "%.4f", COLUMN_WIDTH, FIRST_COLUMN_WIDTH, vec);
	}
}

void print_model_parameters(ML::models::autoencoder& model) {
	const vector<int> percentiles { 0, 1, 3, 10, 25, 50, 75, 90, 97, 99, 100 };
	const int FIRST_COLUMN_WIDTH = 20;
	const int COLUMN_WIDTH = 12;
	ML::stats::print_percentiles_header(percentiles, "%", "%i", COLUMN_WIDTH, FIRST_COLUMN_WIDTH);

	vector<float> biases;
	printf("BIASES\n");
	for(int x=0;x<model.layers.size();x++) {
		char buf[64];
		int len = snprintf(buf, 64, "layer %i", x);
		string name = string(buf, len);
		const auto& layer = model.layers[x];
		biases.resize(layer.neurons.size());
		for(int x=0;x<biases.size();x++) biases[x] = layer.neurons[x].bias;
		ML::stats::print_percentiles(percentiles, name, "%.4f", COLUMN_WIDTH, FIRST_COLUMN_WIDTH, biases);
	}

	vector<float> weights;
	printf("WEIGHTS\n");
	for(int x=0;x<model.layers.size();x++) {
		char buf[64];
		int len = snprintf(buf, 64, "layer %i", x);
		string name = string(buf, len);
		const auto& layer = model.layers[x];
		weights.resize(layer.foreward_targets.targets.size());
		for(int x=0;x<weights.size();x++) weights[x] = layer.foreward_targets.targets[x].weight;
		ML::stats::print_percentiles(percentiles, name, "%.4f", COLUMN_WIDTH, FIRST_COLUMN_WIDTH, weights);
	}

	printf("BACKPROP WEIGHTS (should match WEIGHTS)\n");
	for(int x=0;x<model.layers.size();x++) {
		char buf[64];
		int len = snprintf(buf, 64, "layer %i", x);
		string name = string(buf, len);
		const auto& layer = model.layers[x];
		weights.resize(layer.backprop_targets.targets.size());
		for(int x=0;x<weights.size();x++) weights[x] = layer.backprop_targets.targets[x].weight;
		ML::stats::print_percentiles(percentiles, name, "%.4f", COLUMN_WIDTH, FIRST_COLUMN_WIDTH, weights);
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
	printf("==============================\n");
	print_model_parameters(model);
	printf("------------------------------\n");

	// train model.
	printf("starting training.\n");
	ML::stats::training_stats stats;
	std::mt19937 gen32 = utils::random::get_generator_32(seed);
	for(int z=0;z<n_training_cycles;z++) {
		// run training batch.
		printf("training cycle: %i / %i.\n", z, n_training_cycles);
		training_cycle(image_entries, model, stats, minibatch_size, learning_rate, gen32);
		// print stats.
		if(z % training_print_itv == 0) {
			printf("==============================\n");
			print_training_stats(stats);
			stats.clear_all();
			printf("------------------------------\n");
		}
		// update learning rate.
		if(z % 50 == 0) {
			printf("updating learning rate:\n");
			update_learning_rate(image_entries, model, stats, minibatch_size, learning_rate, gen32, 10);
			printf("new learning rate: %f\n", learning_rate);
			z += 10;
		}
	}
	printf("done training.\n");
	printf("==============================\n");
	print_training_stats(stats);
	print_model_parameters(model);
	stats.clear_all();
	printf("------------------------------\n");

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
