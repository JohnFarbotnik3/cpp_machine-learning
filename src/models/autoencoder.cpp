
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
		a single layer network of forward-connected neurons.
		NOTE: this network is intended to be populated externally.
	*/
	struct layer_network {
		// TODO - experiment with using signal_history to store error terms during backprop.
		vector<float> biases;
		vector<vector<float>> output_history; // sequence of output value images.
		vector<vector<float>> signal_history; // sequence of signal images.
		foreward_target_list foreward_targets;
		backprop_target_list backprop_targets;// inverse of this layer's foreward_targets.

		layer_network(int IMAGE_SIZE) {
			biases.resize(IMAGE_SIZE);
		}

		int image_size() {
			return biases.size();
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

		static void propagate_func(layer_network& layer, const vector<float>& input_values, vector<float>& output_values, const int history_index, const int o_beg, int const o_end) {
			// compute activations.
			const int IMAGE_SIZE = layer.image_size();
			vector<float>& output_image = layer.output_history[history_index];
			vector<float>& signal_image = layer.signal_history[history_index];
			for(int n=o_beg;n<o_end;n++) {
				float sum = layer.biases[n];
				const target_itv itv = layer.foreward_targets.get_interval(n);
				for(int i=itv.beg;i<itv.end;i++) {
					const foreward_target ft = layer.foreward_targets.targets[i];
					sum += ft.weight * input_values[ft.neuron_index];
				}
				const float output_value = activation_func(sum);
				signal_image[n] = sum;
				output_image[n] = output_value;
			}
			// copy output values.
			memcpy(output_values.data() + o_beg, output_image.data() + o_beg, (o_end - o_beg) * sizeof(float));
		}

		void propagate(const int n_threads, const int history_index, const std::vector<float>& input_values, std::vector<float>& output_values) {
			// ensure history buffer is large enough.
			const int IMAGE_SIZE = biases.size();
			while(history_index >= output_history.size()) output_history.emplace_back(IMAGE_SIZE);
			while(history_index >= signal_history.size()) signal_history.emplace_back(IMAGE_SIZE);

			// spawn threads.
			vector<int> intervals = generate_intervals(n_threads, IMAGE_SIZE);
			vector<std::thread> threads;
			for(int x=0;x<n_threads;x++) {
				int n_beg = intervals[x];
				int n_end = intervals[x+1];
				threads.push_back(std::thread(propagate_func, std::ref(*this), std::ref(input_values), std::ref(output_values), history_index, n_beg, n_end));
			}
			for(int x=0;x<n_threads;x++) threads[x].join();
		}

		static void back_propagate_func_output_side(layer_network& layer, vector<vector<float>>& signal_error_terms_history, const vector<vector<float>>& output_error_history, const int history_length, const float learning_rate, const int o_beg, const int o_end) {
			// compute signal error terms.
			for(int z=0;z<history_length;z++) {
				const vector<float>& output_error = output_error_history[z];
				const vector<float>& signal_history = layer.signal_history[z];
				vector<float>& signal_error_terms = signal_error_terms_history[z];
				for(int n=o_beg;n<o_end;n++) {
					signal_error_terms[n] = output_error[n] * activation_derivative(signal_history[n]);
				}
			}

			// adjust biases.
			const float BIAS_LIMIT = 5.0f;
			for(int n=o_beg;n<o_end;n++) {
				float sum = 0;
				for(int z=0;z<history_length;z++) {
					sum += signal_error_terms_history[z][n];
				}
				layer.biases[n] = std::clamp(layer.biases[n] + (sum * learning_rate), -BIAS_LIMIT, +BIAS_LIMIT);
			}
		}

		static void back_propagate_func_input_side(layer_network& layer, vector<vector<float>>& input_error_history, const vector<vector<float>>& input_value_history, const vector<vector<float>>& signal_error_terms, const int history_length, const float learning_rate, const int i_beg, const int i_end) {
			// for each neuron (or value) in input...
			const float WEIGHT_LIMIT = 100.0f;
			for(int n=i_beg;n<i_end;n++) {
				const target_itv itv = layer.backprop_targets.get_interval(n);
				const int itv_len = itv.end - itv.beg;

				// gather targets.
				backprop_target bts[itv_len];
				foreward_target fts[itv_len];
				for(int x=0;x<itv_len;x++) {
					backprop_target bt = layer.backprop_targets.targets[x+itv.beg];
					foreward_target ft = layer.foreward_targets.targets[bt.target_index];
					bts[x] = bt;
					fts[x] = ft;
				}

				// gather input-error and weight-adjustment sums.
				float ie_sums[history_length];
				for(int z=0;z<history_length;z++) ie_sums[z] = 0;

				// NOTE: this multiplier prevents late-training spontaneous model degeneration.
				// TODO: figure out why.
				// TODO - IT DOESNT COMPLETELY SOLVE THE PROBLEM - IT ONLY REDUCES IT!
				const float mult = 1.0f / itv_len;
				for(int x=0;x<itv_len;x++) {
					backprop_target& bt = bts[x];
					foreward_target& ft = fts[x];
					const int out_n = bt.neuron_index;
					const int in_n = ft.neuron_index;
					float we_sum = 0;// weight-error sum.
					for(int z=0;z<history_length;z++) {
						float error_term = signal_error_terms[z][out_n];
						float input_value = input_value_history[z][in_n];
						ie_sums[z] += error_term * mult * ft.weight;
						we_sum     += error_term * mult * input_value;
					}
					// adjust target weight.
					ft.weight = std::clamp(ft.weight + (we_sum * learning_rate), -WEIGHT_LIMIT, +WEIGHT_LIMIT);
				}

				// update target weights.
				for(int x=0;x<itv_len;x++) layer.foreward_targets.targets[bts[x].target_index].weight = fts[x].weight;

				// set input error.
				for(int z=0;z<history_length;z++) input_error_history[z][n] = ie_sums[z];
			}
		}

		/*
			for backprop derivation, see:
			https://dustinstansbury.github.io/theclevermachine/derivation-backpropagation
		*/
		void back_propagate(
			const int n_threads,
			const int history_length,
			vector<vector<float>>& input_error_history,
			const vector<vector<float>>& input_value_history,
			const vector<vector<float>>& output_error_history,
			const float learning_rate
		) {
			//printf("back_propagate(): hlen=%i, isize=%lu, osize=%lu\n", history_length, input_error_history[0].size(), output_error_history[0].size());
			if(history_length == 0) return;

			// assertions.
			assert( input_error_history.size() >= history_length);
			assert(output_error_history.size() >= history_length);
			assert(      signal_history.size() >= history_length);
			const int IMAGE_SIZE_I = input_error_history[0].size();
			const int IMAGE_SIZE_O = biases.size();
			assert(IMAGE_SIZE_I > 0);
			assert(IMAGE_SIZE_O > 0);
			for(int z=0;z<history_length;z++) {
				assert( input_error_history[z].size() == IMAGE_SIZE_I);
				assert(output_error_history[z].size() == IMAGE_SIZE_O);
				assert(      signal_history[z].size() == IMAGE_SIZE_O);
			}

			// create buffers.

			using timepoint = ML::stats::timepoint;
			timepoint t0 = timepoint::now();

			// compute signal error terms and adjust biases of output neurons.
			vector<vector<float>> signal_error_terms;
			for(int z=0;z<history_length;z++) {
				// NOTE: I havent explicitly cleared these buffers.
				signal_error_terms.emplace_back(IMAGE_SIZE_O);
			}
			{
				vector<int> intervals = generate_intervals(n_threads, IMAGE_SIZE_O);
				vector<std::thread> threads;
				for(int x=0;x<n_threads;x++) {
					threads.push_back(std::thread(
						back_propagate_func_output_side,
						std::ref(*this),
						std::ref(signal_error_terms),
						std::ref(output_error_history),
						history_length,
						learning_rate,
						intervals[x],
						intervals[x+1]
					));
				}
				for(int x=0;x<n_threads;x++) threads[x].join();
			}
			timepoint t1 = timepoint::now();

			// back propagate error to input and adjust weights.
			{
				vector<int> intervals = generate_intervals(n_threads, IMAGE_SIZE_I);
				vector<std::thread> threads;
				for(int x=0;x<n_threads;x++) {
					threads.push_back(std::thread(
						back_propagate_func_input_side,
						std::ref(*this),
						std::ref(input_error_history),
						std::ref(input_value_history),
						std::ref(signal_error_terms),
						history_length,
						learning_rate,
						intervals[x],
						intervals[x+1]
					));
				}
				for(int x=0;x<n_threads;x++) threads[x].join();
			}
			timepoint t2 = timepoint::now();

			// normalize input-error against output-error to have same average gradient per-neuron.
			///*

			// TODO - revert this back to normalizing against combined batch error (instead of per image error).
			float in_sum = 0;
			float out_sum = 0;
			for(int z=0;z<history_length;z++) {
				for(int x=0;x< input_error_history[z].size();x++)  in_sum += std::abs( input_error_history[z][x]);
				for(int x=0;x<output_error_history[z].size();x++) out_sum += std::abs(output_error_history[z][x]);
			}
			float mult = (out_sum / in_sum) * (float(IMAGE_SIZE_I) / float(IMAGE_SIZE_O));
			assert(out_sum > 0.0f);
			assert(in_sum > 0.0f);
			for(int z=0;z<history_length;z++) {
				for(int x=0;x< input_error_history[z].size();x++) input_error_history[z][x] *= mult;
				//printf("error: z=%i, isum=%f, osum=%f\n", z, in_sum, out_sum);
			}

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
			layer_network output_layer(dim.length());
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
			layers.push_back(output_layer);
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
			layer_network output_layer(odim.length());
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
			layers.push_back(output_layer);
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

			// mix and condense image.
			// (w, h) -> (w/32, h/32)
			push_layer_mix_AxA_to_1x1(idim, 3, false);
			//push_layer_scale_AxA_to_BxB(idim, odim, 4, 2, 4, true); idim = odim;

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

			//push_layer_scale_AxA_to_BxB(idim, odim, 2, 4, 4, true); idim = odim;
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
			}
		}

		void propagate(const int n_threads, const int history_index, const std::vector<float>& input_values, std::vector<float>& output_values) {
			// assertions.
			const int n_layers = layers.size();
			assert( input_values.size() == layers[0].image_size());
			assert(output_values.size() == layers[n_layers-1].image_size());
			bool input_values_contains_nonzero = false;
			for(int x=0;x<input_values.size();x++) {
				if(input_values[x] != 0) {
					input_values_contains_nonzero = true;
					break;
				}
			}
			assert(input_values_contains_nonzero);

			vector<float> value_buffer_i;
			vector<float> value_buffer_o;

			// first layer.
			value_buffer_o.resize(layers[0].image_size());
			layers[0].propagate(n_threads, history_index, input_values, value_buffer_o);
			std::swap(value_buffer_i, value_buffer_o);

			// middle layers.
			for(int L=1;L<layers.size();L++) {
				value_buffer_o.resize(layers[L].image_size());
				layers[L].propagate(n_threads, history_index, value_buffer_i, value_buffer_o);
				std::swap(value_buffer_i, value_buffer_o);
			}

			// copy output.
			memcpy(output_values.data(), value_buffer_i.data(), output_values.size() * sizeof(float));
		}

		/*
			WARNING: loss_squared can lead to error-concentration which causes models to explode
			when training is going well and they are very close to 0 average error.
		*/
		void generate_error_image(const variable_image_tiled<float>& input, const vector<float>& output, vector<float>& error, bool loss_squared) {
			// assertions.
			const int IMAGE_SIZE = input.X * input.Y * input.C;
			assert(input.data.size() == IMAGE_SIZE);
			assert(output.size() == IMAGE_SIZE);
			assert(error.size() == IMAGE_SIZE);

			// clear error image.
			for(int x=0;x<IMAGE_SIZE;x++) error[x] = 0;

			variable_image_tile_iterator sample_iter = input.get_iterator(
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
		}

		void resize_2d(vector<vector<float>>& vec, int X, int Y) {
			vec.resize(X);
			for(int x=0;x<X;x++) vec[x].resize(Y);
		}

		void back_propagate(const int n_threads, const int history_length, vector<vector<float>>& input_error_history, const vector<vector<float>>& input_value_history,  const vector<vector<float>>& output_error_history, const float learning_rate) {
			// assertions.
			if(history_length == 0) return;
			assert(input_error_history.size() >= history_length);
			assert(output_error_history.size() >= history_length);
			const int n_layers = layers.size();
			for(int z=0;z<history_length;z++) {
				const int IMAGE_SIZE_I = layers[0].image_size();
				const int IMAGE_SIZE_O = layers[n_layers-1].image_size();
				assert(IMAGE_SIZE_I == IMAGE_SIZE_O);
				assert( input_error_history[z].size() == IMAGE_SIZE_I);
				assert(output_error_history[z].size() == IMAGE_SIZE_O);
			}
			bool input_history_contains_nonzero = false;
			for(int z=0;z<history_length && !input_history_contains_nonzero;z++) {
				for(int x=0;x<input_value_history[z].size();x++) {
					if(input_value_history[z][x] != 0) {
						input_history_contains_nonzero = true;
						break;
					}
				}
			}
			assert(input_history_contains_nonzero);

			vector<vector<float>> errhist_buffer_i;
			vector<vector<float>> errhist_buffer_o;

			// copy output.
			resize_2d(errhist_buffer_o, history_length, layers[n_layers-1].image_size());
			for(int z=0;z<history_length;z++) {
				const int IMAGE_SIZE = layers[n_layers-1].image_size();
				memcpy(errhist_buffer_o[z].data(), output_error_history[z].data(), IMAGE_SIZE * sizeof(float));
			}

			// middle layers.
			for(int L=n_layers-1;L>0;L--) {
				resize_2d(errhist_buffer_i, history_length, layers[L-1].image_size());
				layers[L].back_propagate(n_threads, history_length, errhist_buffer_i, layers[L-1].output_history, errhist_buffer_o, learning_rate);
				std::swap(errhist_buffer_i, errhist_buffer_o);
			}
			// first layer.
			layers[0].back_propagate(n_threads, history_length, input_error_history, input_value_history, errhist_buffer_o, learning_rate);
		}
	};
}
