
#include <cassert>
#include <cstring>
#include <thread>
#include <vector>
#include <algorithm>
#include "./network.cpp"
#include "../image.cpp"
#include "../target_list.cpp"
#include "../utils/random.cpp"
#include "../utils/simd.cpp"
#include "../stats.cpp"

namespace ML::models {
	using std::vector;
	using namespace ML::image;
	using namespace ML::target_list;

	/*
	 a * single layer network of forward-connected neurons.
	 NOTE: this network is intended to be populated externally.
	 */
	struct layer_network {
		std::vector<float> biases;
		vector<vector<float>> signal_history;		// sequence of signal images.
		vector<vector<float>> signal_error_terms;	// error terms used for backpropagation.
		foreward_target_list foreward_targets;
		backprop_target_list backprop_targets;// inverse of this layer's foreward_targets.

		// TODO - remember to call this function when creating model.
		void set_history_length(int history_length) {
			int len = biases.size() * history_length;
			signal_history.resize(len);
			signal_error_terms.resize(len);
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
			for(int x=0;x<n_threads;x++) intervals.push_back(x * (length / n_threads));
			intervals.push_back(length);
			return intervals;
		}

		static void propagate_func(layer_network& layer, const vector<float>& input_values, vector<float>& output_values, const int history_index, const int o_beg, int const o_end) {
			// compute activations.
			for(int n=o_beg;n<o_end;n++) {
				float sum = layer.biases[n];
				const target_itv itv = layer.foreward_targets.get_interval(n);
				for(int i=itv.beg;i<itv.end;i++) {
					const foreward_target& target = layer.foreward_targets.targets[i];
					sum += target.weight * input_values[target.neuron_index];
				}
				const int IMAGE_SIZE = layer.biases.size();
				layer.signal_history[history_index][history_index * IMAGE_SIZE + n] = sum;
				output_values[n] = activation_func(sum);
			}
		}

		void propagate(const int n_threads, const int history_index, const std::vector<float>& input_values, std::vector<float>& output_values) {
			// spawn threads.
			const int IMAGE_SIZE = biases.size();
			vector<int> intervals = generate_intervals(n_threads, IMAGE_SIZE);
			vector<std::thread> threads;
			for(int x=0;x<n_threads;x++) {
				int n_beg = intervals[x];
				int n_end = intervals[x+1];
				threads.push_back(std::thread(propagate_func, std::ref(*this), std::ref(input_values), std::ref(output_values), history_index, n_beg, n_end));
			}
			for(int x=0;x<n_threads;x++) threads[x].join();
		}

		static void back_propagate_func_output_side(layer_network& layer, const vector<vector<float>>& output_error_history, const int history_length, const float learning_rate, const int o_beg, const int o_end) {
			// compute signal error terms.
			for(int z=0;z<history_length;z++) {
				const vector<float>& out_error_history = output_error_history[z];
				const vector<float>& out_signal_history = layer.signal_history[z];
				vector<float>& out_error_terms = layer.signal_error_terms[z];
				for(int n=o_beg;n<o_end;n++) {
					out_error_terms[n] = out_error_history[n] * activation_derivative(out_signal_history[n]);
				}
			}

			// adjust biases.
			const float BIAS_LIMIT = 5.0f;
			for(int n=o_beg;n<o_end;n++) {
				float sum = 0;
				for(int z=0;z<history_length;z++) {
					sum += layer.signal_error_terms[z][n];
				}
				layer.biases[n] = std::clamp(layer.biases[n] + (sum * learning_rate), -BIAS_LIMIT, +BIAS_LIMIT);
			}
		}

		static void back_propagate_func_input_side(layer_network& layer, vector<vector<float>>& input_error_history, const int history_length, const float learning_rate, const int i_beg, const int i_end) {
			// for each neuron (or value) in input...
			const float WEIGHT_LIMIT = 100.0f;
			for(int n=i_beg;n<i_end;n++) {
				// adjust weights and gather input error sums.
				vector<float> ie_sums(history_length);
				for(int z=0;z<history_length;z++) ie_sums[z] = 0;
				const target_itv itv = layer.backprop_targets.get_interval(n);
				const float mult = 1.0f / (itv.end - itv.beg);
				for(int x=itv.beg;x<itv.end;x++) {
					backprop_target& bt = layer.backprop_targets.targets[x];
					foreward_target& ft = layer.foreward_targets.targets[bt.target_index];
					const int out_n = bt.neuron_index;
					float w_sum = 0;
					for(int z=0;z<history_length;z++) {
						float error_term = layer.signal_error_terms[z][out_n];
						float out_value = activation_func(layer.signal_history[z][out_n]);
						ie_sums[z] += error_term * mult * ft.weight;
						w_sum      += error_term * mult * out_value;
					}
					// adjust target weight.
					ft.weight = std::clamp(ft.weight + (w_sum * learning_rate), -WEIGHT_LIMIT, +WEIGHT_LIMIT);
				}
				// set input error.
				for(int z=0;z<history_length;z++) {
					input_error_history[z][n] = ie_sums[z];
				}
			}
		}

		// TODO - continue overhaul from here...
		/*
			for backprop derivation, see:
			https://dustinstansbury.github.io/theclevermachine/derivation-backpropagation
		*/
		void back_propagate(
			const int n_threads,
			std::vector<float>& output_error,
			std::vector<float>& input_error,
			std::vector<float>& input_values
		) {
			assert(output_error.size() == neurons.size());
			assert( input_error.size() == input_values.size());
			assert(output_error.size() > 0);
			assert( input_error.size() > 0);

			using timepoint = ML::stats::timepoint;
			timepoint t0 = timepoint::now();

			// compute signal error terms of output neurons.
			//std::vector<float> signal_error_terms(neurons.size());
			if(signal_error_terms.size() < neurons.size()) signal_error_terms.resize(neurons.size());
			{
				vector<int> intervals = generate_intervals(n_threads, neurons.size());
				vector<std::thread> threads;
				for(int x=0;x<n_threads;x++) {
					threads.push_back(std::thread(
						back_propagate_compute_error_terms,
						std::ref(output_error),
						std::ref(signal_error_terms),
						std::ref(this->neurons),
						intervals[x],
						intervals[x+1]
					));
				}
				for(int x=0;x<n_threads;x++) threads[x].join();
			}
			timepoint t1 = timepoint::now();

			// back propagate input error.
			{
				vector<int> intervals = generate_intervals(n_threads, input_error.size());
				vector<std::thread> threads;
				for(int x=0;x<n_threads;x++) {
					threads.push_back(std::thread(
						back_propagate_func_1,
						std::ref(output_error),
						std::ref(input_error),
						std::ref(input_values),
						std::ref(signal_error_terms),
						std::ref(this->backprop_targets),
						intervals[x],
						intervals[x+1]
					));
				}
				for(int x=0;x<n_threads;x++) threads[x].join();
			}
			timepoint t2 = timepoint::now();

			// normalize input-error against output-error to have same average gradient per-neuron.
			///*
			const float DECAY_FACTOR = 1.00f;
			float in_sum = 0;
			float out_sum = 0;
			for(int x=0;x< input_error.size();x++)  in_sum += std::abs( input_error[x]);
			for(int x=0;x<output_error.size();x++) out_sum += std::abs(output_error[x]);
			float mult = (out_sum / in_sum) * (float(input_error.size()) / float(output_error.size())) * DECAY_FACTOR;
			assert(out_sum > 0.0f);
			assert(in_sum > 0.0f);
			for(int x=0;x< input_error.size();x++) input_error[x] *= mult;
			timepoint t3 = timepoint::now();

			// TEST - print time taken for each part of function.
			//printf("dt:\t%li\t%li\t%li\n", t1.delta_us(t0), t2.delta_us(t1), t3.delta_us(t2));
		}
	};

	struct autoencoder {
		using layer_image = ML::image::variable_image_tiled<float>;
		using layer_image_iterator = ML::image::variable_image_tile_iterator;

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
		vector<layer_network>	layers;
		vector<layer_image>		layer_values;	// values output by layers.
		vector<layer_image>		layer_errors;	// errors output by layers.

		autoencoder(image_dimensions input_dimensions) {
			this->input_dimensions = input_dimensions;
			init_model_topology();
		}

	private:
		void push_output_layer(const layer_network& layer, const image_dimensions dim) {
			layers.push_back(layer);
			layer_values.emplace_back(dim.X, dim.Y, dim.C, dim.TX, dim.TY, dim.TC);
			layer_errors.emplace_back(dim.X, dim.Y, dim.C, dim.TX, dim.TY, dim.TC);
		}

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
			layer_network output_layer;
			output_layer.neurons.resize(dim.length());
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

			// generate backprop targets.
			output_layer.backprop_targets = output_layer.foreward_targets.get_inverse(dim.length());

			// push layer into list.
			push_output_layer(output_layer, dim);
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
			layer_network output_layer;
			output_layer.neurons.resize(odim.length());
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

			// generate backprop targets.
			output_layer.backprop_targets = output_layer.foreward_targets.get_inverse(idim.length());

			// push layer into list.
			push_output_layer(output_layer, odim);
		}

		/*
			NOTE: this is called by constructor, so it doesnt need to be called externally
			(since topology is fixed given the same input image resolution).
		*/
		void init_model_topology() {
			// clear.
			layers.clear();
			layer_values.clear();
			layer_errors.clear();

			// input.
			image_dimensions idim = input_dimensions;
			image_dimensions odim = idim;

			// mix and condense image.
			// (w, h) -> (w/32, h/32)
			push_layer_mix_AxA_to_1x1(idim, 3, false);
			push_layer_scale_AxA_to_BxB(idim, odim, 4, 2, 6, true); idim = odim;
			/*
			push_layer_mix_AxA_to_1x1(in_dim, 3, false);
			push_layer_scale_AxA_to_BxB(in_dim, out_dim, 4, 2, true); in_dim = out_dim;
			push_layer_mix_AxA_to_1x1(in_dim, 7, true);
			push_layer_scale_AxA_to_BxB(in_dim, out_dim, 8, 4, true); in_dim = out_dim;
			push_layer_mix_AxA_to_1x1(in_dim, 9, true);
			push_layer_scale_AxA_to_BxB(in_dim, out_dim, 16, 8, true); in_dim = out_dim;
			//*/

			// expand back to original image.
			/*
			push_layer_scale_AxA_to_BxB(in_dim, out_dim, 8, 16, true); in_dim = out_dim;
			push_layer_mix_AxA_to_1x1(in_dim, 9, true);
			push_layer_scale_AxA_to_BxB(in_dim, out_dim, 4, 8, true); in_dim = out_dim;
			push_layer_mix_AxA_to_1x1(in_dim, 7, true);
			push_layer_scale_AxA_to_BxB(in_dim, out_dim, 2, 4, true); in_dim = out_dim;
			push_layer_mix_AxA_to_1x1(in_dim, 3, false);
			//*/

			push_layer_scale_AxA_to_BxB(idim, odim, 2, 4, 4, true); idim = odim;
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
				for(auto& neuron : layer.neurons) neuron.bias = distr_bias(gen32);
				for(auto& target : layer.foreward_targets.targets) target.weight = distr_weight(gen32);
				layer.foreward_targets.store_weights(layer.backprop_targets);
			}
		}

		void apply_batch_error(float rate) {
			for(int x=0;x<layers.size();x++) layers[x].apply_batch_error(rate);
		}

		void propagate(const int n_threads, const std::vector<float>& input_values, std::vector<float>& output_values) {
			// first layer.
			layers[0].propagate(n_threads, input_values, layer_values[0].data);
			// middle layers.
			for(int z=1;z<layer_values.size();z++) {
				layers[z].propagate(n_threads, layer_values[z-1].data, layer_values[z].data);
			}
			// copy output.
			memcpy(output_values.data(), layer_values[layer_values.size()-1].data.data(), output_values.size() * sizeof(float));
		}

		void back_propagate(const int n_threads, std::vector<float>& output_error, std::vector<float>& input_error, std::vector<float>& input_values) {
			// copy output.
			memcpy(layer_errors[layer_errors.size()-1].data.data(), output_error.data(), output_error.size() * sizeof(float));
			// middle layers.
			for(int z=layer_values.size()-1;z>0;z--) layers[z].back_propagate(n_threads, layer_errors[z].data, layer_errors[z-1].data, layer_values[z-1].data);
			// first layer.
			layers[0].back_propagate(n_threads, layer_errors[0].data, input_error, input_values);
		}
	};
}
