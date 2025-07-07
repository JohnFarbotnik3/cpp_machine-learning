#include "../utils/commandline.cpp"
#include "../utils/random.cpp"
#include "../utils/vector_util.cpp"
#include "../image.cpp"
#include "../stats.cpp"
#include "../models/autoencoder.cpp"
#include <cstdio>
#include <filesystem>
#include <string>

/*
debug build:
g++ -std=c++23 -O2 -fsanitize=address -o "./src/projects/autoencoder.elf" "./src/projects/autoencoder.cpp"

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
-tadjustlr_ini 1 \
-tadjustlr_itv 50 \
-tadjustlr_len 10 \
-tcp 50 \
-pmp 1 \
-pmp_debug 0 \
-bsz 5 \
-lr 0.01 \
-seed 12345 \
-n_threads 2

perf:
perf stat -d -d -d -- <RUN COMMAND WITH OPTIONS>
perf record -- COMMAND [OPTIONS...]
perf report -i "./perf.data"

*/

using std::string;
using std::vector;
using timepoint = ML::stats::timepoint;
namespace fs = std::filesystem;
using model_t = ML::models::autoencoder;
using model_image_t = ML::image::variable_image_tiled<float>;

struct training_settings {
	vector<fs::directory_entry> image_entries;
	ML::stats::training_stats stats;
	float learning_rate;
	int n_threads;
	int minibatch_size;
	int seed;
	std::mt19937 gen32;
};

void print_usage(string msg) {
	printf("%s\n", msg.c_str());
	printf("usage:\n");
	printf("-m MODELS_DIR -i INPUT_IMAGES_DIR -o OUTPUT_IMAGES_DIR\n");
}

struct sample_image_cache {
	using file_image_t = ML::image::file_image;

	std::map<string, file_image_t> files;
	std::map<string, model_image_t> samples;

	bool has_file(const string path) {
		return files.contains(path);
	}
	ML::image::file_image& get_file(const string path, int channels) {
		if(files.contains(path)) return files.at(path);
		files.insert_or_assign(path, file_image_t::load(path, channels));
		return files.at(path);
	}

	bool has_sample(const string path) {
		return samples.contains(path);
	}
	model_image_t get_sample(const string path, model_image_t& sample_buffer, const file_image_t& loaded_image) {
		if(samples.contains(path)) return samples.at(path);
		ML::image::generate_sample_image(sample_buffer, loaded_image);
		samples.insert_or_assign(path, sample_buffer);
		return samples.at(path);
	}
};

void training_cycle(ML::models::autoencoder& model, training_settings& settings, sample_image_cache& cache) {
	// divide image entries into batches.
	vector<fs::directory_entry> pool = settings.image_entries;
	vector<vector<fs::directory_entry>> image_minibatches;
	while(pool.size() > 0) {
		vector<fs::directory_entry> minibatch;
		for(int z=0;z<settings.minibatch_size;z++) {
			std::uniform_int_distribution<int> distr(0, pool.size()-1);
			int x = distr(settings.gen32);
			minibatch.push_back(pool[x]);
			pool[x] = pool[pool.size()-1];
			pool.pop_back();
			if(pool.size() == 0) break;
		}
		image_minibatches.push_back(minibatch);
	}

	// run training cycle.
	int X = model.input_dimensions.X;
	int Y = model.input_dimensions.Y;
	int C = model.input_dimensions.C;
	int TX = model.input_dimensions.TX;
	int TY = model.input_dimensions.TY;
	int TC = model.input_dimensions.TC;
	model_image_t image_input_img(X, Y, C, TX, TY, TC);
	const int IMAGE_SIZE = image_input_img.data.size();
	vector<float> image_input (IMAGE_SIZE);
	vector<float> image_output(IMAGE_SIZE);
	vector<float> image_error (IMAGE_SIZE);
	vector<float> image_temp  (IMAGE_SIZE);
	for(const auto& minibatch : image_minibatches) {
		timepoint tb0 = timepoint::now();
		assert(minibatch.size() > 0);

		timepoint t0;
		timepoint t1;

		// clear accumulated error.
		t0 = timepoint::now();
		model.clear_batch_error();
		t1 = timepoint::now();
		settings.stats.push_value("dt clear err", t1.delta_us(t0));

		for(int z=0;z<minibatch.size();z++) {

			// load image.
			const auto& entry = minibatch[z];
			const string path = entry.path().string();
			t0 = timepoint::now();
			ML::image::file_image loaded_image = cache.get_file(path, C);
			t1 = timepoint::now();
			settings.stats.push_value("dt load image", t1.delta_us(t0));

			// generate sample.
			t0 = timepoint::now();
			image_input_img = cache.get_sample(path, image_input_img, loaded_image);
			utils::vector_util::vec_copy(image_input, image_input_img.data, 0, image_input.size());
			t1 = timepoint::now();
			settings.stats.push_value("dt gen sample", t1.delta_us(t0));

			// propagate.
			t0 = timepoint::now();
			model.propagate(settings.n_threads, image_input, image_output);
			t1 = timepoint::now();
			settings.stats.push_value("dt propagate", t1.delta_us(t0));

			// compute error.
			t0 = timepoint::now();
			model.generate_error_image(image_input_img, image_output, image_error, true);
			const float avg_error = utils::vector_util::vec_sum_abs_mt(image_error, 0, image_error.size(), settings.n_threads) / image_error.size();
			t1 = timepoint::now();
			settings.stats.push_value("dt error image", t1.delta_us(t0));
			settings.stats.push_value("avg error", avg_error);
			// TODO TEST
			printf("image: z=%i, avg_error=%f, path=%s\n", z, avg_error, minibatch[z].path().c_str());

			// backpropagate.
			t0 = timepoint::now();
			model.back_propagate(settings.n_threads, image_temp, image_input, image_error);
			t1 = timepoint::now();
			settings.stats.push_value("dt backprop", t1.delta_us(t0));
		}

		// apply accumulated error.
		t0 = timepoint::now();
		model.apply_batch_error(settings.learning_rate, minibatch.size(), settings.n_threads);
		t1 = timepoint::now();
		settings.stats.push_value("dt apply err", t1.delta_us(t0));

		timepoint tb1 = timepoint::now();
		settings.stats.push_value("dt training batch", tb1.delta_us(tb0));
	}
}

void update_learning_rate(ML::models::autoencoder& model, training_settings& settings, int cycles, sample_image_cache& cache) {
	const float LEARNING_RATE_LIMIT = 0.99f;
	float best_pct_error;
	training_settings best_settings = settings;
	ML::models::autoencoder best_model = model;

	vector<float> lr_mults = settings.learning_rate >= LEARNING_RATE_LIMIT
		? vector<float>{ 1.0/1.2, 1.0 }
		: vector<float>{ 1.0/1.2, 1.0, 1.2 };
	for(int z=0;z<lr_mults.size();z++) {
		training_settings test_settings = settings;
		ML::models::autoencoder test_model = model;
		test_settings.learning_rate = std::min(settings.learning_rate * lr_mults[z], LEARNING_RATE_LIMIT);
		printf("trying rate = %f\n", test_settings.learning_rate); for(int x=0;x<cycles;x++) training_cycle(test_model, test_settings, cache);
		const vector<int> percentiles { 20 };
		float pct_error = ML::stats::get_percentile_values(percentiles, test_settings.stats.groups.at("avg error"))[0];
		if(pct_error < best_pct_error || z == 0) {
			best_pct_error = pct_error;
			best_settings = test_settings;
			best_model = test_model;
		}
	}
	model = best_model;
	settings = best_settings;
}

void print_training_stats(ML::stats::training_stats& stats) {
	const vector<int> percentiles { 0, 1, 3, 10, 25, 50, 75, 90, 97, 99, 100 };
	const int FIRST_COLUMN_WIDTH = 20;
	const int SUM_COLUMN_WIDTH = 15;
	const int COLUMN_WIDTH = 12;
	ML::stats::print_percentiles_header(percentiles, "%", "%i", COLUMN_WIDTH, FIRST_COLUMN_WIDTH, SUM_COLUMN_WIDTH);

	printf("STATS\n");
	for(const auto& [name, vec] : stats.groups) {
		ML::stats::print_percentiles(percentiles, name, "%.4f", COLUMN_WIDTH, FIRST_COLUMN_WIDTH, SUM_COLUMN_WIDTH, vec);
	}
}

void print_model_parameters(ML::models::autoencoder& model) {
	const vector<int> percentiles { 0, 1, 3, 10, 25, 50, 75, 90, 97, 99, 100 };
	const int FIRST_COLUMN_WIDTH = 20;
	const int SUM_COLUMN_WIDTH = 15;
	const int COLUMN_WIDTH = 12;
	ML::stats::print_percentiles_header(percentiles, "%", "%i", COLUMN_WIDTH, FIRST_COLUMN_WIDTH, SUM_COLUMN_WIDTH);

	vector<float> biases;
	printf("BIASES\n");
	for(int x=0;x<model.layers.size();x++) {
		char buf[64];
		int len = snprintf(buf, 64, "layer %i", x);
		string name = string(buf, len);
		const auto& layer = model.layers[x];
		biases.resize(layer.biases.size());
		for(int x=0;x<biases.size();x++) biases[x] = layer.biases[x];
		ML::stats::print_percentiles(percentiles, name, "%.4f", COLUMN_WIDTH, FIRST_COLUMN_WIDTH, SUM_COLUMN_WIDTH, biases);
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
		ML::stats::print_percentiles(percentiles, name, "%.4f", COLUMN_WIDTH, FIRST_COLUMN_WIDTH, SUM_COLUMN_WIDTH, weights);
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
	training_settings settings;
	// TODO - move all of these loose parameters to settings as well.
	string model_dir  = arguments.named_arguments.at("-m");
	string input_dir  = arguments.named_arguments.at("-i");
	string output_dir = arguments.named_arguments.at("-o");
	int input_w = arguments.get_named_value("-w", 512);
	int input_h = arguments.get_named_value("-h", 512);
	int input_c = arguments.get_named_value("-channels", 3);
	int n_training_cycles = arguments.get_named_value("-tc", 20);
	int training_print_itv = arguments.get_named_value("-tcp", 1);
	int pmp = arguments.get_named_value("-pmp", 1);
	int pmp_debug = arguments.get_named_value("-pmp_debug", 0);
	int tadjustlr_ini = arguments.get_named_value("-tadjustlr_ini", 1);
	int tadjustlr_itv = arguments.get_named_value("-tadjustlr_itv", 50);
	int tadjustlr_len = arguments.get_named_value("-tadjustlr_len", 10);
	settings.minibatch_size = arguments.get_named_value("-bsz", 5);
	settings.learning_rate = arguments.get_named_value("-lr", 0.001f);
	settings.seed = arguments.get_named_value("-seed", 12345);
	settings.n_threads = arguments.get_named_value("-n_threads", 1);

	// create and initialize model.
	printf("initializing model.\n");
	ML::models::autoencoder::image_dimensions input_dimensions(input_w, input_h, input_c, 4, 4, input_c);
	//ML::models::autoencoder::image_dimensions input_dimensions(input_w, input_h, input_c, input_w/2, input_h/2, input_c);// TEST
	ML::models::autoencoder model(input_dimensions);
	model.init_model_parameters(settings.seed, 0.0f, 0.3f, 0.0f, 0.2f);
	//model.init_model_parameters(settings.seed, 0.0f, 0.0f, 1.0f, 0.5f);// TEST
	printf("==============================\n");
	if(pmp) print_model_parameters(model);
	printf("------------------------------\n");

	// get list of images in input directory.
	printf("examining input directory.\n");
	settings.image_entries = ML::image::get_image_entries_in_directory(input_dir);
	printf("found %lu images.\n", settings.image_entries.size());

	///*
	// train model.
	printf("starting training.\n");
	sample_image_cache cache;
	settings.gen32 = utils::random::get_generator_32(settings.seed);
	for(int z=0;z<n_training_cycles;z++) {
		// print training info.
		printf("training cycle: %i / %i\n", z, n_training_cycles);
		// run training batch.
		training_cycle(model, settings, cache);
		// print stats.
		if(z % training_print_itv == 0) {
			printf("==============================\n");
			print_training_stats(settings.stats);
			if(pmp_debug) print_model_parameters(model);
			settings.stats.clear_all();
			printf("------------------------------\n");
		}
		// update learning rate.
		if(z % tadjustlr_itv == 0 && (z > 0 || tadjustlr_ini)) {
			printf("updating learning rate:\n");
			update_learning_rate(model, settings, tadjustlr_len, cache);
			printf("new learning rate: %f\n", settings.learning_rate);
			z += tadjustlr_len;
		}
		// TODO TEST
		//if(z > 300) print_model_parameters(model);
	}
	printf("done training.\n");
	printf("==============================\n");
	print_training_stats(settings.stats);
	if(pmp) print_model_parameters(model);
	settings.stats.clear_all();
	printf("------------------------------\n");
	//*/

	// test model by outputting images.
	printf("outputting decoded images.\n");
	int X = model.input_dimensions.X;
	int Y = model.input_dimensions.Y;
	int C = model.input_dimensions.C;
	int TX = model.input_dimensions.TX;
	int TY = model.input_dimensions.TY;
	int TC = model.input_dimensions.TC;
	ML::image::variable_image_tiled<float> image_input (X, Y, C, TX, TY, TC);
	ML::image::variable_image_tiled<float> image_output(X, Y, C, TX, TY, TC);
	for(const fs::directory_entry entry : settings.image_entries) {
		// load image.
		ML::image::file_image loaded_image = ML::image::file_image::load(entry.path().string(), C);
		ML::image::generate_sample_image(image_input, loaded_image);
		// propagate.
		model.propagate(settings.n_threads, image_input.data, image_output.data);
		// output result.
		fs::path outpath = fs::path(output_dir) / fs::path(entry.path()).filename().concat(".png");
		ML::image::file_image output = ML::image::to_byte_image(image_output, false);
		//ML::image::file_image output = ML::image::to_byte_image(image_input, false);// TEST
		ML::image::file_image::save(output, outpath.string(), image_input.C);
	}
	printf("done.\n");

	return 0;
}
