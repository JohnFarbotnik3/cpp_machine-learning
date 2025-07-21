
#include <cstdio>
#include <filesystem>
#include <string>
#include "src/utils/commandline.cpp"
#include "src/utils/random.cpp"
#include "src/utils/vector_util.cpp"
#include "src/image/file_image.cpp"
#include "src/image/value_image_lines.cpp"
#include "src/stats.cpp"
#include "src/models/autoencoder_subimages/ae_model.cpp"

/*
debug build:
g++ -std=c++23 -O2 -I "./" -g -fsanitize=address -o "./src/models/autoencoder_subimages/main.elf" "./src/models/autoencoder_subimages/main.cpp"
g++ -std=c++23 -O2 -I "./" -g -o "./src/models/autoencoder_subimages/main.elf" "./src/models/autoencoder_subimages/main.cpp"

build:
g++ -std=c++23 -O2 -I "./" -o "./src/models/autoencoder_subimages/main.elf" "./src/models/autoencoder_subimages/main.cpp"
g++ -std=c++23 -O2 -I "./" -march=native -o "./src/models/autoencoder_subimages/main.elf" "./src/models/autoencoder_subimages/main.cpp"

run:
clear && ./src/models/autoencoder_subimages/main.elf \
-m ./data/models \
-i ./data/images/images_02 \
-o      /dev/shm/images_02_$(date -Iseconds) \
-w 384 \
-h 384 \
-n_training_cycles 500 \
-tadjustlr_ini 1 \
-tadjustlr_itv 50 \
-tadjustlr_len 10 \
-print_interval_stats 50 \
-print_model_params 1 \
-print_model_params_debug 0 \
-bsz 5 \
-lr_b 0.1 \
-lr_w 0.1 \
-seed 12345 \
-n_threads 2 > /dev/shm/log_$(date -Iseconds).txt

debug run (requires compiling with "-g" option):
gdb -ex=r --args ...COMMAND...

perf:
perf stat -d -d -d -- <RUN COMMAND WITH OPTIONS>
perf record -- COMMAND [OPTIONS...]
perf report -i "./perf.data"

*/

using std::string;
using std::vector;
using timepoint = ML::stats::timepoint;
namespace fs = std::filesystem;
using model_t = ML::models::autoencoder_subimage::ae_model;
using model_image_t = ML::models::autoencoder_subimage::simd_image_8f;
using image_dim_t = ML::models::autoencoder_subimage::simd_image_8f_dimensions;
using file_image_t = ML::image::file_image;
using ML::image::value_image::sample_bounds;
using namespace ML::models::autoencoder_subimage;

struct training_settings {
	vector<fs::directory_entry> image_entries;
	ML::stats::training_stats stats;
	vector<float> batch_error_trend;
	int cycle;
	int n_training_cycles;
	float learning_rate_b;
	float learning_rate_w;
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


/*
sample_bounds generate_sample_image_normalized(model_image_t& sample, const file_image_t& image, const int n_threads) {
	sample_bounds bounds = ML::image::value_image::generate_sample_image(sample, image);
	for(int y=bounds.y0;y<bounds.y1;y++) {
	for(int x=bounds.x0;x<bounds.x1;x++) {
	for(int c=0;c<sample.dim.C;c++) {
		const int i = sample.dim.get_offset(x, y, c);
		sample.data[i] = sample.data[i] - 0.5f;
	}}}
	return bounds;
}
file_image_t to_file_image_normalized(const model_image_t& sample, const sample_bounds bounds, const bool clamp_to_sample_area, const int n_threads) {
	model_image_t temp = sample;
	utils::vector_util::vec_fma_mt<float>(temp.data, 1.0f, +0.5f, 0, temp.data.size(), n_threads);
	return ML::image::value_image_lines::to_file_image(temp, bounds, clamp_to_sample_area);
}
*/

void generate_error_image(simd_image_8f& error, const simd_image_8f& input, const simd_image_8f& output, const sample_bounds* bounds, const int N) {
	assert(output.dim.equals(input.dim));
	assert(output.dim.equals(error.dim));
	assert(output.dim.length() > 0);
	assert(0 <= N && N <= 8);
	const simd_image_8f_dimensions dim = output.dim;

	// load bounds into SIMD registers.
	vec8f x0 = _mm256_set1_ps(0);
	vec8f x1 = _mm256_set1_ps(0);
	vec8f y0 = _mm256_set1_ps(0);
	vec8f y1 = _mm256_set1_ps(0);
	for(int n=0;n<N;n++) {
		x0[n] = bounds[n].x0;
		x1[n] = bounds[n].x1;
		y0[n] = bounds[n].y0;
		y1[n] = bounds[n].y1;
	}

	// compute error.
	const vec8f zero = _mm256_setzero_ps();
	for(int y=0;y<dim.Y;y++) {
	for(int x=0;x<dim.X;x++) {
		// get bitmask containing 1's if image[n] is in bounds, and 0's otherwise.
		vec8f mx0 = _mm256_cmp_ps(_mm256_set1_ps(x), x0, _CMP_GE_OS);
		vec8f mx1 = _mm256_cmp_ps(_mm256_set1_ps(x), x1, _CMP_LT_OS);
		vec8f my0 = _mm256_cmp_ps(_mm256_set1_ps(y), y0, _CMP_GE_OS);
		vec8f my1 = _mm256_cmp_ps(_mm256_set1_ps(y), y1, _CMP_LT_OS);
		vec8f mask = _mm256_and_ps(_mm256_and_ps(mx0, mx1), _mm256_and_ps(my0, my1));

		for(int c=0;c<dim.C;c++) {
			const int iofs = dim.get_offset(x, y, c);
			const vec8f diff = _mm256_sub_ps(input.data[x], output.data[x]);
			// error = in_bounds ? (input - output) : 0
			error.data[x] = _mm256_blendv_ps(zero, diff, mask);
			// TODO - sum error?
			// TODO - clamp error?
		}
	}}
}

struct batch_iterator {
	std::mt19937 gen32;
	vector<fs::directory_entry> image_entries;
	vector<int> order;
	int index;

	batch_iterator(const vector<fs::directory_entry> image_entries) : image_entries(image_entries) { reset(); }
	void reset() {
		order = utils::random::generate_shuffle_mapping(gen32, image_entries.size());
		index = 0;
	}
	fs::directory_entry next() {
		if(index == order.size()) reset();
		return image_entries[order[index++]];
	}
};

void training_cycle(model_t& model, training_settings& settings) {
	settings.cycle++;
	// divide image entries into batches.
	// TODO - use batch iterator instead.
	/*
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
	*/

	// run training cycle.
	model_image_t image_input (model.image_dimensions);
	model_image_t image_output(model.image_dimensions);
	model_image_t image_error (model.image_dimensions);
	model_image_t image_temp  (model.image_dimensions);
	float batch_error = 0.0f;
	float batch_count = 0.0f;
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
			const int ch = model.image_dimensions.C;
			ML::image::file_image loaded_image = ML::image::load_file_image(path, ch);
			t1 = timepoint::now();
			settings.stats.push_value("dt load image", t1.delta_us(t0));

			// generate sample.
			t0 = timepoint::now();
			sample_bounds bounds = generate_sample_image_normalized(image_input, loaded_image, settings.n_threads);
			t1 = timepoint::now();
			settings.stats.push_value("dt gen sample", t1.delta_us(t0));

			// propagate.
			t0 = timepoint::now();
			model.propagate(settings.n_threads, image_input, image_output);
			t1 = timepoint::now();
			settings.stats.push_value("dt propagate", t1.delta_us(t0));

			// compute error.
			t0 = timepoint::now();
			generate_error_image(image_input, image_output, image_error, bounds, 1.0f, true);
			const float avg_error = utils::vector_util::vec_sum_abs_mt(image_error.data, 0, image_error.data.size(), settings.n_threads) / image_error.data.size();
			batch_error += avg_error;
			batch_count += 1.0f;
			t1 = timepoint::now();
			settings.stats.push_value("dt error image", t1.delta_us(t0));
			settings.stats.push_value("avg error", avg_error);
			printf("image: z=%i, avg_error=%f, path=%s\n", z, avg_error, minibatch[z].path().c_str());

			// backpropagate.
			t0 = timepoint::now();
			model.back_propagate(settings.n_threads, image_temp, image_error);
			t1 = timepoint::now();
			settings.stats.push_value("dt backprop", t1.delta_us(t0));
		}

		// apply accumulated error.
		t0 = timepoint::now();
		model.apply_batch_error_biases (settings.n_threads, minibatch.size(), settings.learning_rate_b);
		model.apply_batch_error_weights(settings.n_threads, minibatch.size(), settings.learning_rate_w);
		t1 = timepoint::now();
		settings.stats.push_value("dt apply err", t1.delta_us(t0));

		timepoint tb1 = timepoint::now();
		settings.stats.push_value("dt training batch", tb1.delta_us(tb0));
	}
	settings.batch_error_trend.push_back(batch_error / batch_count);
}

void update_learning_rate(model_t& model, training_settings& settings, int cycles, const char mode) {
	const float LEARNING_RATE_LIMIT = 1.0f;
	float best_pct_error;
	training_settings best_settings = settings;
	model_t best_model = model;

	const float rate = (mode == 'b') ? settings.learning_rate_b : settings.learning_rate_w;
	if(rate == 0.0f) return;// ignore learning rate if 0.
	vector<float> lr_mults = rate >= LEARNING_RATE_LIMIT
		? vector<float>{ 0.7, 1.0 }
		: vector<float>{ 0.7, 1.0, 1.2 };
	for(int z=0;z<lr_mults.size();z++) {
		training_settings test_settings = settings;
		model_t test_model = model;
		const float new_rate = std::min(rate * lr_mults[z], LEARNING_RATE_LIMIT);
		if(mode == 'b') test_settings.learning_rate_b = new_rate;
		if(mode == 'w') test_settings.learning_rate_w = new_rate;
		printf("trying rate [mode=%c] = %f\n", mode, new_rate);
		for(int x=0;x<cycles;x++) training_cycle(test_model, test_settings);
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

void print_model_parameters(model_t& model) {
	const vector<int> percentiles { 0, 1, 3, 10, 25, 50, 75, 90, 97, 99, 100 };
	const int FIRST_COLUMN_WIDTH = 20;
	const int SUM_COLUMN_WIDTH = 15;
	const int COLUMN_WIDTH = 12;
	ML::stats::print_percentiles_header(percentiles, "%", "%i", COLUMN_WIDTH, FIRST_COLUMN_WIDTH, SUM_COLUMN_WIDTH);

	printf("BIASES\n");
	for(int x=0;x<model.layers.size();x++) {
		char buf[64];
		int len = snprintf(buf, 64, "layer %i", x);
		string name = string(buf, len);
		const vector<float> biases = model.layers[x].get_biases();
		ML::stats::print_percentiles(percentiles, name, "%.4f", COLUMN_WIDTH, FIRST_COLUMN_WIDTH, SUM_COLUMN_WIDTH, biases);
	}

	printf("WEIGHTS\n");
	for(int x=0;x<model.layers.size();x++) {
		char buf[64];
		int len = snprintf(buf, 64, "layer %i", x);
		string name = string(buf, len);
		const vector<float> weights = model.layers[x].get_weights();
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
	int print_interval_stats = arguments.get_named_value("-print_interval_stats", 1);
	int print_model_params = arguments.get_named_value("-print_model_params", 1);
	int print_model_params_debug = arguments.get_named_value("-print_model_params_debug", 0);
	int tadjustlr_ini = arguments.get_named_value("-tadjustlr_ini", 1);
	int tadjustlr_itv = arguments.get_named_value("-tadjustlr_itv", 50);
	int tadjustlr_len = arguments.get_named_value("-tadjustlr_len", 10);
	settings.cycle = 0;
	settings.n_training_cycles = arguments.get_named_value("-n_training_cycles", 50);
	settings.minibatch_size = arguments.get_named_value("-bsz", 5);
	settings.learning_rate_b = arguments.get_named_value("-lr_b", 0.001f);
	settings.learning_rate_w = arguments.get_named_value("-lr_w", 0.001f);
	settings.seed = arguments.get_named_value("-seed", 12345);
	settings.n_threads = arguments.get_named_value("-n_threads", 1);

	// create and initialize model.
	printf("initializing model.\n");
	image_dim_t input_dimensions(input_w, input_h, input_c);
	//ML::models::autoencoder::image_dimensions input_dimensions(input_w, input_h, input_c, input_w/2, input_h/2, input_c);// TEST
	model_t model(input_dimensions);
	model.init_model_parameters(settings.seed, 0.0f, 0.1f, 0.0f, 1.0f);
	//model.init_model_parameters(settings.seed, 0.0f, 0.0f, 1.0f, 0.5f);// TEST
	printf("==============================\n");
	if(print_model_params) print_model_parameters(model);
	printf("------------------------------\n");

	// get list of images in input directory.
	printf("examining input directory.\n");
	settings.image_entries = ML::image::get_image_entries_in_directory(input_dir);
	printf("found %lu images.\n", settings.image_entries.size());

	///*
	// train model.
	printf("starting training.\n");
	settings.gen32 = utils::random::get_generator_32(settings.seed);
	vector<float>& error_trend = settings.batch_error_trend;
	while(settings.cycle < settings.n_training_cycles) {
		// run training batch.
		training_cycle(model, settings);
		printf("training cycle: %i/%i | error: %f\n", settings.cycle, settings.n_training_cycles, error_trend.back());
		if(settings.cycle % std::max(1, settings.n_training_cycles/100) == 0) {
			fprintf(stderr, "TRAINING: z=%i/%i, lr_b=%f, lr_w=%f, er=%f\n",
				settings.cycle,
				settings.n_training_cycles,
				settings.learning_rate_b,
				settings.learning_rate_w,
				error_trend.back()
			);
		}
		// warn if error starts increasing.
		if(settings.cycle >= 5) {
			const float err_curr = error_trend.back();
			const float err_prev = error_trend[error_trend.size()-5];
			if(err_curr > err_prev) fprintf(stderr, "ERROR RATE INCREASED: z=%i/%i, err_prev=%f, err_curr=%f\n", settings.cycle, settings.n_training_cycles, err_prev, err_curr);
		}
		// print stats.
		if(settings.cycle % print_interval_stats == 0) {
			printf("==============================\n");
			print_training_stats(settings.stats);
			if(print_model_params_debug) print_model_parameters(model);
			settings.stats.clear_all();
			printf("------------------------------\n");
		}
		// update learning rate.
		if(settings.cycle % tadjustlr_itv == 0 && (settings.cycle > 0 || tadjustlr_ini)) {
			printf("updating learning rate:\n");
			update_learning_rate(model, settings, tadjustlr_len, 'b');
			update_learning_rate(model, settings, tadjustlr_len, 'w');
			printf("new learning rate: lr_b=%f, lr_w=%f\n", settings.learning_rate_b, settings.learning_rate_w);
		}
	}
	printf("done training.\n");
	printf("==============================\n");
	print_training_stats(settings.stats);
	if(print_model_params) print_model_parameters(model);
	settings.stats.clear_all();
	printf("------------------------------\n");
	//*/

	// test model by outputting images.
	printf("outputting decoded images.\n");
	model_image_t image_input (model.image_dimensions);
	model_image_t image_output(model.image_dimensions);
	for(const fs::directory_entry entry : settings.image_entries) {
		// load image.
		const int ch = model.image_dimensions.C;
		ML::image::file_image loaded_image = ML::image::load_file_image(entry.path().string(), ch);
		sample_bounds bounds = generate_sample_image_normalized(image_input, loaded_image, settings.n_threads);
		// propagate.
		model.propagate(settings.n_threads, image_input, image_output);
		// output result.
		fs::path outpath = fs::path(output_dir) / fs::path(entry.path()).filename().concat(".png");
		ML::image::file_image output;
		if(settings.n_training_cycles == -1) {
			output = to_file_image_normalized(image_input, bounds, false, settings.n_threads);// TEST
		} else {
			output = to_file_image_normalized(image_output, bounds, false, settings.n_threads);
		}
		ML::image::save_file_image(output, outpath.string(), image_input.dim.C);
	}
	printf("done.\n");

	return 0;
}
