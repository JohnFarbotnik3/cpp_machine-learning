
#include <cstdio>
#include <filesystem>
#include <string>
#include "src/utils/commandline.cpp"
#include "src/utils/random.cpp"
#include "src/image/file_image.cpp"
#include "src/image/value_image.cpp"
#include "src/stats.cpp"
#include "src/models/autoencoder_subimages/ae_model.cpp"

/*
debug build:
g++ -std=c++23 -O2 -march=native -I "./" -g -fsanitize=address -o "./src/models/autoencoder_subimages/main.elf" "./src/models/autoencoder_subimages/main.cpp"
g++ -std=c++23 -O2 -march=native -I "./" -g -o "./src/models/autoencoder_subimages/main.elf" "./src/models/autoencoder_subimages/main.cpp"

build:
g++ -std=c++23 -O2 -march=native -I "./" -o "./src/models/autoencoder_subimages/main.elf" "./src/models/autoencoder_subimages/main.cpp"

run:
clear && ./src/models/autoencoder_subimages/main.elf \
-m ./data/models \
-i ./data/images/images_8 \
-o      /dev/shm/images_8_$(date -Iseconds) \
-w 384 \
-h 384 \
-n_training_cycles 500 \
-tadjustlr_ini 1 \
-tadjustlr_itv 50 \
-tadjustlr_len 10 \
-print_interval_stats 50 \
-print_model_params 1 \
-print_model_params_debug 0 \
-bsz 8 \
-lr_b 0.1 \
-lr_w 0.1 \
-seed 12345 \
-n_threads 2 > /dev/shm/log_$(date -Iseconds).txt

debug run (requires compiling with "-g" option):
gdb -ex=r --args ...COMMAND...

perf:
perf stat -d -d -d -- <RUN COMMAND WITH OPTIONS>
perf record -o "/dev/shm/perf.data" -- COMMAND [OPTIONS...]
perf report -i "/dev/shm/perf.data"

*/

using std::string;
using std::vector;
using timepoint = ML::stats::timepoint;
namespace fs = std::filesystem;
using model_t = ML::models::autoencoder_subimage::ae_model;
using file_image_t = ML::image::file_image;
using sample_image_t = ML::image::value_image::value_image<float>;
using ML::image::value_image::sample_range;
using simd_image_t = ML::models::autoencoder_subimage::simd_image_8f;
using simd_image_dim_t = ML::models::autoencoder_subimage::simd_image_8f_dimensions;
using ML::image::value_image::sample_bounds;
using namespace ML::models::autoencoder_subimage;

struct file_iterator {
	std::mt19937 gen32;
	vector<string> image_paths;
	vector<int> order;
	int index;

	file_iterator() = default;
	file_iterator(const vector<fs::directory_entry> image_entries) {
		for(const fs::directory_entry entry : image_entries) image_paths.push_back(entry.path().string());
		reset();
	}
	void reset() {
		order = utils::random::generate_shuffle_mapping(gen32, image_paths.size());
		index = 0;
	}
	string next() {
		if(index == order.size()) reset();
		return image_paths[order[index++]];
	}
};

struct cache_file_image {
	std::map<string, file_image_t> cache;

	file_image_t& get(string path, const int ch) {
		if(!cache.contains(path)) cache[path] = ML::image::load_file_image(path, ch);
		return cache.at(path);
	}
};
cache_file_image file_image_cache;

// NOTE: this is a temporary measure for using until I write an optimized version of my linear sampling function.
struct sample_pair { sample_image_t image; sample_bounds bounds; };
struct cache_sample_image {
	std::map<string, sample_pair> cache;

	sample_pair& get(string path, const file_image_t& file_image, const int X, const int Y, const int C, const sample_range range) {
		if(!cache.contains(path)) {
			sample_image_t sample_image(X, Y, C);
			sample_bounds bounds = ML::image::value_image::generate_sample_image(sample_image, file_image, range);
			cache[path] = sample_pair{ sample_image, bounds };
		}
		return cache.at(path);
	}
};
cache_sample_image sample_image_cache;


struct training_settings {
	ML::stats::training_stats stats;
	file_iterator file_iter;
	vector<float> batch_error_trend;
	sample_range range;
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

float generate_error_image(simd_image_8f& error_o, const simd_image_8f& value_i, const simd_image_8f& value_o, const sample_bounds* bounds, const int N) {
	assert(value_o.dim.equals(value_i.dim));
	assert(value_o.dim.equals(error_o.dim));
	assert(value_o.dim.length() > 0);
	assert(0 <= N && N <= 8);
	const simd_image_8f_dimensions dim = value_o.dim;

	// load bounds into SIMD registers.
	// TODO - try studying the SIMD interface again to see if there is a consistent way to do int32 GE comparison.
	__m256 x0 = _mm256_setzero_ps();
	__m256 x1 = _mm256_setzero_ps();
	__m256 y0 = _mm256_setzero_ps();
	__m256 y1 = _mm256_setzero_ps();
	for(int n=0;n<N;n++) {
		x0[n] = bounds[n].x0;
		x1[n] = bounds[n].x1;
		y0[n] = bounds[n].y0;
		y1[n] = bounds[n].y1;
	}

	// compute error.
	const vec8f zero = _mm256_setzero_ps();
	vec8f sum = _mm256_setzero_ps();
	for(int oy=0;oy<dim.Y;oy++) {
	for(int ox=0;ox<dim.X;ox++) {
		// get bitmask containing 1's if image[n] is in bounds, and 0's otherwise.
		vec8f vox = _mm256_set1_ps(ox);
		vec8f voy = _mm256_set1_ps(oy);
		vec8f mx0 = _mm256_cmp_ps(vox, x0, _CMP_GE_OS);
		vec8f mx1 = _mm256_cmp_ps(vox, x1, _CMP_LT_OS);
		vec8f my0 = _mm256_cmp_ps(voy, y0, _CMP_GE_OS);
		vec8f my1 = _mm256_cmp_ps(voy, y1, _CMP_LT_OS);
		vec8f mask = _mm256_and_ps(_mm256_and_ps(mx0, mx1), _mm256_and_ps(my0, my1));
	for(int oc=0;oc<dim.C;oc++) {
		const int iofs = dim.get_offset(ox, oy, oc);
		const vec8f diff = _mm256_sub_ps(value_i.data[iofs], value_o.data[iofs]);
		// error = in_bounds ? (input - output) : 0
		error_o.data[iofs] = _mm256_blendv_ps(zero, diff, mask);
		sum = _mm256_add_ps(sum, simd_abs(error_o.data[iofs]));
		// NOTE - clamp error?
	}}}

	return simd_reduce(sum);
}

void training_cycle(model_t& model, training_settings& settings) {
	assert(settings.minibatch_size > 0);
	assert(settings.minibatch_size <= settings.file_iter.image_paths.size());

	const simd_image_dim_t dim = model.image_dimensions;
	float batch_avg_error = 0.0f;

	timepoint tb0 = timepoint::now();
	timepoint t0;
	timepoint t1;

	// clear accumulated error.
	t0 = timepoint::now();
	model.clear_batch_error();
	t1 = timepoint::now();
	settings.stats.push_value("dt clear err", t1.delta_us(t0));

	// get samples.
	const int n_images = settings.minibatch_size;
	assert(5 <= n_images && n_images <= 8);// TODO - rework function to loop for larger minibatch sizes.
	vector<sample_image_t> samples_images;
	vector<sample_bounds> samples_bounds;
	for(int x=0;x<n_images;x++) {
		// load image.
		t0 = timepoint::now();
		const string path = settings.file_iter.next();
		file_image_t file_image = file_image_cache.get(path, dim.C);
		t1 = timepoint::now();
		settings.stats.push_value("dt load image", t1.delta_us(t0));

		// generate sample.
		t0 = timepoint::now();
		const sample_pair& pair = sample_image_cache.get(path, file_image, dim.X, dim.Y, dim.C, settings.range);
		samples_images.push_back(pair.image);
		samples_bounds.push_back(pair.bounds);
		t1 = timepoint::now();
		settings.stats.push_value("dt gen sample", t1.delta_us(t0));
	}

	// pack samples.
	t0 = timepoint::now();
	simd_image_t value_i(dim);
	simd_image_t value_o(dim);
	simd_image_t error_o(dim);
	simd_image_t error_i(dim);
	value_i.pack(samples_images.data(), samples_images.size());
	t1 = timepoint::now();
	settings.stats.push_value("dt sample pack", t1.delta_us(t0));

	// propagate.
	t0 = timepoint::now();
	model.propagate(settings.n_threads, value_i, value_o);
	t1 = timepoint::now();
	settings.stats.push_value("dt propagate", t1.delta_us(t0));

	// compute error.
	t0 = timepoint::now();
	const float total_error = generate_error_image(error_o, value_i, value_o, samples_bounds.data(), n_images);
	int total_length = 0;
	for(int x=0;x<n_images;x++) {
		sample_bounds bounds = samples_bounds[x];
		total_length += (bounds.x1 - bounds.x0) * (bounds.y1 - bounds.y0) * samples_images[x].dim.C;
	}
	const float avg_error = total_error / total_length;
	batch_avg_error += avg_error;
	t1 = timepoint::now();
	settings.stats.push_value("dt error image", t1.delta_us(t0));
	settings.stats.push_value("avg error", avg_error);

	// backpropagate.
	t0 = timepoint::now();
	const float lr_w = settings.learning_rate_w / settings.minibatch_size;
	model.back_propagate(settings.n_threads, error_i, error_o, value_i, lr_w);
	t1 = timepoint::now();
	settings.stats.push_value("dt backprop", t1.delta_us(t0));

	// apply accumulated error.
	t0 = timepoint::now();
	model.apply_batch_error_biases (settings.n_threads, settings.minibatch_size, settings.learning_rate_b);
	t1 = timepoint::now();
	settings.stats.push_value("dt apply err", t1.delta_us(t0));
	timepoint tb1 = timepoint::now();
	settings.stats.push_value("dt training batch", tb1.delta_us(tb0));

	settings.batch_error_trend.push_back(batch_avg_error / settings.minibatch_size);
	settings.cycle++;
}

float rolling_average_last_n(const vector<float> values, const int N) {
	float sum = 0.0f;
	float weight = 0.0f;
	for(int n=N;n>0;n--) {
		const float mult = float(n)/N;
		sum += values[n-1] * mult;
		weight += mult;
	}
	return sum / weight;
}
void update_learning_rate(model_t& model, training_settings& settings, int cycles, const char mode) {
	assert(mode == 'b' || mode == 'w');
	const float LEARNING_RATE_LIMIT = 1.0f;
	float best_avg_error;
	training_settings best_settings = settings;
	model_t best_model = model;

	const float rate = (mode == 'b') ? settings.learning_rate_b : settings.learning_rate_w;
	if(rate == 0.0f) return;// ignore learning rate if 0.
	vector<float> lr_mults = rate >= LEARNING_RATE_LIMIT
		? vector<float>{ 1.0, 0.7 }
		: vector<float>{ 1.0, 0.7, 1.2 };
	for(int z=0;z<lr_mults.size();z++) {
		training_settings test_settings = settings;
		model_t test_model = model;
		const float new_rate = std::min(rate * lr_mults[z], LEARNING_RATE_LIMIT);
		if(mode == 'b') test_settings.learning_rate_b = new_rate;
		if(mode == 'w') test_settings.learning_rate_w = new_rate;
		printf("trying rate [mode=%c] = %f\n", mode, new_rate);
		for(int x=0;x<cycles;x++) training_cycle(test_model, test_settings);
		float avg_error = rolling_average_last_n(test_settings.stats.groups.at("avg error"), cycles);
		fprintf(stderr, "mode=%c, LR=%f, avg_error=%0.10f\n", mode, new_rate, avg_error);
		if(avg_error < best_avg_error || z == 0) {
			best_avg_error = avg_error;
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
	settings.range = sample_range{ -0.5f, 0.5f };
	settings.n_training_cycles = arguments.get_named_value("-n_training_cycles", 50);
	settings.minibatch_size = arguments.get_named_value("-bsz", 8);
	settings.learning_rate_b = arguments.get_named_value("-lr_b", 0.01f);
	settings.learning_rate_w = arguments.get_named_value("-lr_w", 0.01f);
	settings.seed = arguments.get_named_value("-seed", 12345);
	settings.n_threads = arguments.get_named_value("-n_threads", 1);

	// create and initialize model.
	printf("initializing model.\n");
	simd_image_dim_t input_dimensions(input_w, input_h, input_c);
	model_t model(input_dimensions);
	model.init_model_parameters(settings.seed, 0.0f, 0.1f, 0.0f, 1.0f);
	printf("==============================\n");
	if(print_model_params) print_model_parameters(model);
	printf("------------------------------\n");

	// get list of images in input directory.
	printf("examining input directory.\n");
	settings.file_iter = file_iterator(ML::image::get_image_entries_in_directory(input_dir));
	printf("found %lu images.\n", settings.file_iter.image_paths.size());

	///*
	// train model.
	printf("starting training.\n");
	settings.gen32 = utils::random::get_generator_32(settings.seed);
	vector<float>& error_trend = settings.batch_error_trend;
	while(settings.cycle < settings.n_training_cycles) {
		// update learning rate.
		if(settings.cycle % tadjustlr_itv == 0 && (settings.cycle > 0 || tadjustlr_ini)) {
			printf("updating learning rate:\n");
			update_learning_rate(model, settings, tadjustlr_len, 'w');
			update_learning_rate(model, settings, tadjustlr_len, 'b');
			printf("new learning rate: lr_b=%f, lr_w=%f\n", settings.learning_rate_b, settings.learning_rate_w);
		}
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
	sample_image_t sample_i(input_dimensions.X, input_dimensions.Y, input_dimensions.C);
	sample_image_t sample_o(input_dimensions.X, input_dimensions.Y, input_dimensions.C);
	simd_image_t value_i(model.image_dimensions);
	simd_image_t value_o(model.image_dimensions);
	for(const string path : settings.file_iter.image_paths) {
		// load image.
		const int ch = model.image_dimensions.C;
		file_image_t file_image_i = ML::image::load_file_image(path, ch);
		sample_bounds bounds = ML::image::value_image::generate_sample_image(sample_i, file_image_i, settings.range);
		// propagate.
		value_i.pack(&sample_i, 1);
		model.propagate(settings.n_threads, value_i, value_o);
		value_o.unpack(&sample_o, 1);
		// output result.
		fs::path outpath = fs::path(output_dir) / fs::path(path).filename().concat(".png");
		ML::image::file_image file_image_o;
		if(settings.n_training_cycles == -1) {
			file_image_o = ML::image::value_image::to_file_image(sample_i, bounds, settings.range, false);// TEST
		} else {
			file_image_o = ML::image::value_image::to_file_image(sample_o, bounds, settings.range, false);
		}
		ML::image::save_file_image(file_image_o, outpath.string(), value_i.dim.C);
	}
	printf("done.\n");

	return 0;
}
