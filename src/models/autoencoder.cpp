
#include <cassert>
#include <thread>
#include <vector>
#include <algorithm>
#include "./network.cpp"
#include "../image.cpp"
#include "../target_list.cpp"
#include "../utils/random.cpp"
#include "../utils/simd.cpp"
#include "../utils/vector_util.cpp"
#include "../stats.cpp"

namespace ML::models {
	using std::vector;
	using namespace ML::image;
	using namespace ML::target_list;
	using namespace utils::vector_util;

	/*
		a single layer network of forward-connected neurons.
		NOTE: this network is intended to be populated externally.
	*/
	struct layer_network {
		vector<float> biases;
		vector<float> output;// image of output values - used for backprop.
		vector<float> signal;// image of signal values - used for backprop.
		vector<float> biases_error;// accumulated error in biases during minibatch.
		vector<float> weights_error;// accumulated error in weights during minibatch.
		foreward_target_list foreward_targets;
		backprop_target_list backprop_targets;// inverse of this layer's foreward_targets.
		int INPUT_IMAGE_SIZE;
		int OUTPUT_IMAGE_SIZE;

		layer_network(int INPUT_IMAGE_SIZE, int OUTPUT_IMAGE_SIZE) :
			biases(OUTPUT_IMAGE_SIZE),
			output(OUTPUT_IMAGE_SIZE),
			signal(OUTPUT_IMAGE_SIZE),
			biases_error(OUTPUT_IMAGE_SIZE)
		{
			this->INPUT_IMAGE_SIZE = INPUT_IMAGE_SIZE;
			this->OUTPUT_IMAGE_SIZE = OUTPUT_IMAGE_SIZE;
			vec_fill(biases, 0.0f);
			vec_fill(output, 0.0f);
			vec_fill(signal, 0.0f);
			vec_fill(biases_error, 0.0f);
		}

		int input_image_size() {
			return INPUT_IMAGE_SIZE;
		}
		int output_image_size() {
			return OUTPUT_IMAGE_SIZE;
		}

		void sync_targets_list() {
			backprop_targets = foreward_targets.get_inverse(INPUT_IMAGE_SIZE);
			foreward_targets.save_weights(backprop_targets);
			weights_error.resize(foreward_targets.targets.size());
		}

		// ============================================================
		// activation functions.
		// ------------------------------------------------------------

		static float activation_func(float value) {
			/*
			 * NOTE: pure ReLU was causing problems.
			 * now using ReLU with leakage.
			 * //return std::max<float>(value, 0);
			 */
			return value > 0.0f ? value : value * 0.5f;
		}

		static float activation_derivative(float value) {
			/*
			 * NOTE: I deliberately picked 1.0f as the derivative before as I was
			 * worried that weights wouldnt be pushed back up if values got stuck in the negatives.
			 * this was stupid (I didnt pay close attention to the math), and it caused model degeneration.
			 * //return 1.0f;
			 * NOTE: maybe I was right, model seems to get stuck, and when
			 * the signal_error_term hits zero the neuron stops learning.
			 * //return value > 0.0f ? 1.0f : 0.0f;
			 */
			return value > 0.0f ? 1.0f : 0.5f;
		}

		// ============================================================
		// network functions
		// ------------------------------------------------------------

		static vector<int> generate_intervals(int n_threads, int length) {
			vector<int> intervals;
			for(int x=0;x<=n_threads;x++) intervals.push_back((x * length) / n_threads);
			intervals.push_back(length);
			return intervals;
		}

		static void propagate_func(layer_network& layer, const vector<float>& input_values, vector<float>& output_values, const int o_beg, int const o_end) {
			// compute activations.
			const int IMAGE_SIZE = layer.output_image_size();
			for(int n=o_beg;n<o_end;n++) {
				float sum = layer.biases[n];
				const target_itv itv = layer.foreward_targets.get_interval(n);
				for(int i=itv.beg;i<itv.end;i++) {
					const foreward_target ft = layer.foreward_targets.targets[i];
					sum += ft.weight * input_values[ft.neuron_index];
				}
				layer.signal[n] = sum;
				layer.output[n] = activation_func(sum);
			}
			// copy output values.
			vec_copy(output_values, layer.output, o_beg, o_end);
		}

		void propagate(const int n_threads, const std::vector<float>& input_values, std::vector<float>& output_values) {
			// spawn threads.
			vector<int> intervals = generate_intervals(n_threads, output_image_size());
			vector<std::thread> threads;
			for(int x=0;x<n_threads;x++) {
				int n_beg = intervals[x];
				int n_end = intervals[x+1];
				threads.push_back(std::thread(propagate_func, std::ref(*this), std::ref(input_values), std::ref(output_values), n_beg, n_end));
			}
			for(int x=0;x<n_threads;x++) threads[x].join();
		}

		static void back_propagate_func_output_side(layer_network& layer, vector<float>& signal_error_terms, const vector<float>& output_error, const int o_beg, const int o_end) {
			for(int n=o_beg;n<o_end;n++) {
				signal_error_terms[n] = output_error[n] * activation_derivative(layer.signal[n]);
			}
			for(int n=o_beg;n<o_end;n++) {
				layer.biases_error[n] += signal_error_terms[n];
			}
		}

		static void back_propagate_func_input_side(layer_network& layer, vector<float>& input_error, const vector<float>& input_value, const vector<float>& signal_error_terms, const int i_beg, const int i_end) {
			// for each neuron in input...
			for(int n=i_beg;n<i_end;n++) {
				// gather input-error and weight-error.
				const target_itv itv = layer.backprop_targets.get_interval(n);
				for(int x=itv.beg;x<itv.end;x++) {
					const backprop_target bt = layer.backprop_targets.targets[x];
					const int& out_n = bt.neuron_index;
					const float error_term = signal_error_terms[out_n];
					input_error[n] = error_term * bt.weight;
					layer.weights_error[x] += error_term * input_value[n];
				}
			}
		}

		/*
			for backprop derivation, see:
			https://dustinstansbury.github.io/theclevermachine/derivation-backpropagation
		*/
		void back_propagate(
			const int n_threads,
			vector<float>& input_error,
			const vector<float>& input_value,
			const vector<float>& output_error
		) {
			// assertions.
			const int IMAGE_SIZE_I = input_image_size();
			const int IMAGE_SIZE_O = output_image_size();
			assert(IMAGE_SIZE_I > 0);
			assert(IMAGE_SIZE_O > 0);
			assert(input_error.size() == IMAGE_SIZE_I);
			assert(input_value.size() == IMAGE_SIZE_I);
			assert(biases.size() == IMAGE_SIZE_O);
			assert(biases_error.size() == IMAGE_SIZE_O);
			assert(output_error.size() == IMAGE_SIZE_O);

			using timepoint = ML::stats::timepoint;

			// compute signal error terms and adjust biases of output neurons.
			//timepoint t0 = timepoint::now();
			vector<float> signal_error_terms(IMAGE_SIZE_O);
			{
				vector<int> intervals = generate_intervals(n_threads, IMAGE_SIZE_O);
				vector<std::thread> threads;
				for(int x=0;x<n_threads;x++) {
					threads.push_back(std::thread(
						back_propagate_func_output_side,
						std::ref(*this),
						std::ref(signal_error_terms),
						std::ref(output_error),
						intervals[x],
						intervals[x+1]
					));
				}
				for(int x=0;x<n_threads;x++) threads[x].join();
			}

			// back propagate error to input and adjust weights.
			//timepoint t1 = timepoint::now();
			{
				vector<int> intervals = generate_intervals(n_threads, IMAGE_SIZE_I);
				vector<std::thread> threads;
				for(int x=0;x<n_threads;x++) {
					threads.push_back(std::thread(
						back_propagate_func_input_side,
						std::ref(*this),
						std::ref(input_error),
						std::ref(input_value),
						std::ref(signal_error_terms),
						intervals[x],
						intervals[x+1]
					));
				}
				for(int x=0;x<n_threads;x++) threads[x].join();
			}

			// normalize input-error against output-error to have same average gradient per-neuron.
			///*
			//timepoint t2 = timepoint::now();
			const float in_sum  = vec_sum_abs_mt(input_error, 0, input_error.size(), n_threads);
			const float out_sum = vec_sum_abs_mt(output_error, 0, output_error.size(), n_threads);
			float mult = (out_sum / in_sum) * (float(IMAGE_SIZE_I) / float(IMAGE_SIZE_O));
			assert(out_sum > 0.0f);
			assert(in_sum > 0.0f);
			vec_mult_mt(input_error, mult, 0, input_error.size(), n_threads);
			//printf("error: z=%i, isum=%f, osum=%f\n", z, in_sum, out_sum);

			// TEST - print time taken for each part of function.
			//timepoint t3 = timepoint::now();
			//printf("dt:\t%li\t%li\t%li\n", t1.delta_us(t0), t2.delta_us(t1), t3.delta_us(t2));
		}

		void clear_batch_error() {
			// assertions.
			assert(biases_error.size() == biases.size());
			assert(weights_error.size() == foreward_targets.targets.size());
			assert(weights_error.size() == backprop_targets.targets.size());

			vec_fill(biases_error, 0.0f);
			vec_fill(weights_error, 0.0f);
		}

		static void apply_batch_error_biases(layer_network& layer, const int beg, const int end, const float adjustment_rate) {
			const float BIAS_LIMIT = 10.0f;
			const float BIAS_ADJUSTMENT_LIMIT = 0.5f;
			for(int n=beg;n<end;n++) {
				const float adjustment = std::clamp(layer.biases_error[n] * adjustment_rate, -BIAS_ADJUSTMENT_LIMIT, +BIAS_ADJUSTMENT_LIMIT);
				layer.biases[n] = std::clamp(layer.biases[n] + adjustment, -BIAS_LIMIT, +BIAS_LIMIT);
			}
		}
		static void apply_batch_error_weights(layer_network& layer, const int beg, const int end, const float adjustment_rate) {
			const float WEIGHT_LIMIT = 100.0f;
			const float WEIGHT_ADJUSTMENT_LIMIT = 0.5f;
			for(int x=beg;x<end;x++) {
				backprop_target& bt = layer.backprop_targets.targets[x];
				foreward_target& ft = layer.foreward_targets.targets[bt.target_index];
				const float adjustment = std::clamp(layer.weights_error[x] * adjustment_rate, -WEIGHT_ADJUSTMENT_LIMIT, +WEIGHT_ADJUSTMENT_LIMIT);
				bt.weight = std::clamp(bt.weight + adjustment, -WEIGHT_LIMIT, +WEIGHT_LIMIT);
				ft.weight = bt.weight;
			}
		}
		void apply_batch_error(const int n_threads, const int batch_size, const float learning_rate_b, const float learning_rate_w) {
			// assertions.
			assert(biases_error.size() == biases.size());
			assert(weights_error.size() == foreward_targets.targets.size());
			assert(weights_error.size() == backprop_targets.targets.size());

			// adjust biases.
			{
				const float adjustment_rate = learning_rate_b / batch_size;
				vector<std::thread> threads;
				const int len = biases.size();
				for(int x=0;x<n_threads;x++) {
					const int x0 = ((x+0) * len) / n_threads;
					const int x1 = ((x+1) * len) / n_threads;
					threads.push_back(std::thread(apply_batch_error_biases, std::ref(*this), x0, x1, adjustment_rate));
				}
				for(int x=0;x<n_threads;x++) threads[x].join();
			}

			// adjust weights.
			{
				const float adjustment_rate = learning_rate_w / batch_size;
				vector<std::thread> threads;
				const int len = backprop_targets.targets.size();
				for(int x=0;x<n_threads;x++) {
					const int x0 = ((x+0) * len) / n_threads;
					const int x1 = ((x+1) * len) / n_threads;
					threads.push_back(std::thread(apply_batch_error_weights, std::ref(*this), x0, x1, adjustment_rate));
				}
				for(int x=0;x<n_threads;x++) threads[x].join();
			}
		}
	};

	struct autoencoder {
		using layer_image = ML::image::value_image_tiled<float>;
		using layer_image_iterator = ML::image::value_image_tile_iterator;

		// TODO - deduplicate (move this to "image.cpp").
		struct image_dimensions {
			int X = 0;
			int Y = 0;
			int C = 0;
			int TX = 0;
			int TY = 0;
			int TC = 0;

			int length() const {
				return X * Y * C;
			}

			layer_image_iterator get_iterator(int x0, int x1, int y0, int y1, int c0, int c1) const {
				return layer_image_iterator(TX, TY, TC, X, Y, C, x0, x1, y0, y1, c0, c1);
			}
			layer_image_iterator get_iterator() const {
				return layer_image_iterator(TX, TY, TC, X, Y, C, 0, X, 0, Y, 0, C);
			}

			vector<int> generate_target_indices(int x0, int x1, int y0, int y1, int ch=-1) const {
				int c0 = (ch == -1) ? 0 : ch;
				int c1 = (ch == -1) ? C : ch+1;
				layer_image_iterator iter = get_iterator(x0, x1, y0, y1, c0, c1);
				vector<int> list;
				while(iter.has_next()) {
					list.push_back(iter.i);
					iter.next();
				}
				return list;
			}
		};

		image_dimensions input_dimensions;
		vector<layer_network> layers;

		autoencoder(image_dimensions input_dimensions) {
			this->input_dimensions = input_dimensions;
			init_model_topology();
		}

	private:
		/*
		void push_output_layer(const layer_network& layer, const image_dimensions dim) {
			layers.push_back(layer);
			layer_values.emplace_back(dim.X, dim.Y, dim.C, dim.TX, dim.TY, dim.TC);
			layer_errors.emplace_back(dim.X, dim.Y, dim.C, dim.TX, dim.TY, dim.TC);
		}
		*/

		/*
			generates connections from AxA square of pixels (from input)
			to 1x1 square (in output), with AxA input square centered on output pixel.

			NOTE:
			- output layer should have the same neuron-layout as image-data does.
			- odd values of A are preferred (for symmetry).
			- if mix_channels is true, every pixel-value in input is connected to each
			pixel-value-neuron in output.
			- if mix_channels is false, channels are kept seperate.
		*/
		void push_layer_mix_AxA_to_1x1(const image_dimensions dim, const int A, bool mix_channels) {
			// create output layer and generate targets for each neuron (pixel-value) in output.
			// NOTE: the order these target-lists are generated in must match order of neurons.
			layers.push_back(layer_network(dim.length(), dim.length()));
			layer_network& output_layer = layers.back();
			layer_image_iterator output_iter = dim.get_iterator();
			while(output_iter.has_next()) {
				int x0 = std::max(output_iter.x - A/2, 0);
				int y0 = std::max(output_iter.y - A/2, 0);
				int x1 = std::min(x0 + A, dim.X);
				int y1 = std::min(y0 + A, dim.Y);
				vector<int> connection_inds = dim.generate_target_indices(x0, x1, y0, y1, mix_channels ? -1 : output_iter.c);
				output_layer.foreward_targets.push_list(connection_inds);
				output_iter.next();
			}
			output_layer.sync_targets_list();
		}

		/*
			generates connections such that each AxA square from input is scaled
			into a BxB square in output.

			NOTE: image dimensions A and B must be picked such that image scales cleanly.
		*/
		void push_layer_scale_AxA_to_BxB(const image_dimensions idim, image_dimensions& odim, const int A, const int B, const int outC, bool mix_channels) {
			// make sure images will scale cleanly, and have size compatible with choice of A and B.
			// - images dimensions should be divisible by A (input) or B (output).
			// - images should have the same number of AxA or BxB tiles as eachother in each dimension.
			// - if not mixing channels, then output and input should have same number of channels.
			odim = idim;
			odim.X = (odim.X * B) / A;
			odim.Y = (odim.Y * B) / A;
			odim.C = outC;
			odim.TC = outC;
			assert(idim.X % A == 0);
			assert(idim.Y % A == 0);
			assert(odim.X % B == 0);
			assert(odim.Y % B == 0);
			assert(idim.X / A == odim.X / B);
			assert(idim.Y / A == odim.Y / B);
			assert(mix_channels || (odim.C == idim.C));

			// create output layer and generate targets for each neuron (pixel-value) in output.
			// NOTE: the order these target-lists are generated in must match order of neurons.
			layers.push_back(layer_network(idim.length(), odim.length()));
			layer_network& output_layer = layers.back();
			layer_image_iterator output_iter = odim.get_iterator();
			while(output_iter.has_next()) {
				int x0 = (output_iter.x / B) * A;
				int y0 = (output_iter.y / B) * A;
				int x1 = x0 + A;
				int y1 = y0 + A;
				vector<int> connection_inds = idim.generate_target_indices(x0, x1, y0, y1, mix_channels ? -1 : output_iter.c);
				output_layer.foreward_targets.push_list(connection_inds);
				output_iter.next();
			}
			output_layer.sync_targets_list();
		}

		/*
			NOTE: this is called by constructor, so it doesnt need to be called externally
			(since topology is fixed given the same input image resolution).
		*/
		void init_model_topology() {
			// clear.
			layers.clear();

			// input.
			image_dimensions idim = input_dimensions;
			image_dimensions odim = idim;
			const int ch = idim.C;

			// mix and condense image.
			// (w, h) -> (w/8, h/8)
			push_layer_mix_AxA_to_1x1(idim, 3, false);
			push_layer_scale_AxA_to_BxB(idim, odim, 4, 2, 6, true); idim = odim;
			push_layer_mix_AxA_to_1x1(idim, 5, false);
			push_layer_scale_AxA_to_BxB(idim, odim, 4, 2, 8, true); idim = odim;
			push_layer_mix_AxA_to_1x1(idim, 7, false);
			push_layer_scale_AxA_to_BxB(idim, odim, 4, 2, 12, true); idim = odim;
			push_layer_mix_AxA_to_1x1(idim, 9, false);
			push_layer_scale_AxA_to_BxB(idim, odim, 8, 4, 16, true); idim = odim;

			// expand image back to original size.
			push_layer_scale_AxA_to_BxB(idim, odim, 4, 8, 12, true); idim = odim;
			push_layer_mix_AxA_to_1x1(idim, 9, false);
			push_layer_scale_AxA_to_BxB(idim, odim, 2, 4, 8, true); idim = odim;
			push_layer_mix_AxA_to_1x1(idim, 7, false);
			push_layer_scale_AxA_to_BxB(idim, odim, 2, 4, 6, true); idim = odim;
			push_layer_mix_AxA_to_1x1(idim, 5, false);
			push_layer_scale_AxA_to_BxB(idim, odim, 2, 4, ch, true); idim = odim;
			push_layer_mix_AxA_to_1x1(idim, 3, false);

			assert(odim.X == idim.X);
			assert(odim.Y == idim.Y);
			assert(odim.C == idim.C);
			assert(odim.TX == idim.TX);
			assert(odim.TY == idim.TY);
			assert(odim.TC == idim.TC);
		}

	public:
		void init_model_parameters(int seed, float bias_mean, float bias_stddev, float weight_mean, float weight_stddev) {
			std::mt19937 gen32 = utils::random::get_generator_32(seed);
			std::normal_distribution distr_bias = utils::random::rand_normal<float>(bias_mean, bias_stddev);
			std::normal_distribution distr_weight = utils::random::rand_normal<float>(weight_mean, weight_stddev);

			for(int z=0;z<layers.size();z++) {
				auto& layer = layers[z];
				for(auto& bias   : layer.biases) bias = distr_bias(gen32);
				for(auto& target : layer.foreward_targets.targets) target.weight = distr_weight(gen32);
				layer.foreward_targets.save_weights(layer.backprop_targets);
			}
		}

		void propagate(const int n_threads, const std::vector<float>& input_values, std::vector<float>& output_values) {
			// assertions.
			const int n_layers = layers.size();
			assert( input_values.size() == layers[0].output_image_size());
			assert(output_values.size() == layers[n_layers-1].output_image_size());

			vector<float> value_buffer_i;
			vector<float> value_buffer_o;

			// first layer.
			value_buffer_o.resize(layers[0].output_image_size());
			layers[0].propagate(n_threads, input_values, value_buffer_o);
			std::swap(value_buffer_i, value_buffer_o);

			// middle layers.
			for(int L=1;L<layers.size();L++) {
				value_buffer_o.resize(layers[L].output_image_size());
				layers[L].propagate(n_threads, value_buffer_i, value_buffer_o);
				std::swap(value_buffer_i, value_buffer_o);
			}

			// copy output.
			vec_copy(output_values, value_buffer_i, 0, output_values.size());
		}

		/*
			WARNING: loss_squared can lead to error-concentration which causes models to explode
			when training is going well and they are very close to 0 average error.
		*/
		void generate_error_image(const value_image_tiled<float>& input, const vector<float>& output, vector<float>& error, bool loss_squared, bool clamp_error) {
			// assertions.
			const int IMAGE_SIZE = input.X * input.Y * input.C;
			assert(input.data.size() == IMAGE_SIZE);
			assert(output.size() == IMAGE_SIZE);
			assert(error.size() == IMAGE_SIZE);

			// clear error image.
			for(int x=0;x<IMAGE_SIZE;x++) error[x] = 0;

			value_image_tile_iterator sample_iter = input.get_iterator(
				input.x0, input.x1,
				input.y0, input.y1,
				0, input.C
			);

			if(loss_squared) {
				while(sample_iter.has_next()) {
					int i = sample_iter.i;
					const float delta = input.data[i] - output[i];
					error[i] = delta * std::abs(delta);
					sample_iter.next();
				}
			} else {
				while(sample_iter.has_next()) {
					int i = sample_iter.i;
					const float delta = input.data[i] - output[i];
					error[i] = delta;
					sample_iter.next();
				}
			}

			if(clamp_error) {
				for(int x=0;x<error.size();x++) error[x] = std::clamp(error[x], -1.0f, 1.0f);
			}
		}

		void back_propagate(const int n_threads, vector<float>& input_error, const vector<float>& input_value, const vector<float>& output_error) {
			// assertions.
			const int n_layers = layers.size();
			assert( input_value.size() == layers[0].input_image_size());
			assert( input_error.size() == layers[0].input_image_size());
			assert(output_error.size() == layers[n_layers-1].output_image_size());

			vector<float> error_buffer_i;
			vector<float> error_buffer_o;

			// copy output.
			error_buffer_o.resize(layers[n_layers-1].output_image_size());
			vec_copy(error_buffer_o, output_error, 0, output_error.size());

			// middle layers.
			for(int L=n_layers-1;L>0;L--) {
				error_buffer_i.resize(layers[L].input_image_size());
				layers[L].back_propagate(n_threads, error_buffer_i, layers[L-1].output, error_buffer_o);
				std::swap(error_buffer_i, error_buffer_o);
			}

			// first layer.
			layers[0].back_propagate(n_threads, input_error, input_value, error_buffer_o);
		}

		void clear_batch_error() {
			for(int z=0;z<layers.size();z++) layers[z].clear_batch_error();
		}

		void apply_batch_error(const int n_threads, const int batch_size, const float learning_rate_b, const float learning_rate_w) {
			for(int z=0;z<layers.size();z++) layers[z].apply_batch_error(n_threads, batch_size, learning_rate_b, learning_rate_w);
		}
	};
}
