
#include <cassert>
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

	using namespace target_list;

	struct layer_neuron {
		float signal 	= 0;// sum of input activations and bias - used for backpropagation.
		float bias		= 0;// bias of neuron[x] in the network.
		float bias_error= 0;// cumulative adjustment to apply to bias.
	};

	/*
	 a * single layer network of forward-connected neurons.
	 NOTE: this network is intended to be populated externally.
	 */
	struct layer_network : ML::models::network {
		std::vector<layer_neuron> neurons;
		std::vector<float> signal_error_terms;
		foreward_target_list foreward_targets;
		backprop_target_list backprop_targets;// inverse of this layer's foreward_targets.

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

		void apply_batch_error(float rate) override {
			const float BIAS_LIMIT = 5.0f;
			const float WEIGHT_LIMIT = 100.0f;
			for(int n=0;n<neurons.size();n++) {
				layer_neuron& neuron = neurons[n];
				neuron.bias += neuron.bias_error * rate;
				neuron.bias  = std::clamp(neuron.bias, -BIAS_LIMIT, +BIAS_LIMIT);
				neuron.bias_error = 0;
			}
			for(int x=0;x<backprop_targets.targets.size();x++) {
				backprop_target& bt = backprop_targets.targets[x];
				bt.weight += bt.weight_error * rate;
				bt.weight  = std::clamp(bt.weight, -WEIGHT_LIMIT, +WEIGHT_LIMIT);
				bt.weight_error = 0;
			}
			// NOTE: load_weights() is quite expensive.
			foreward_targets.load_weights(backprop_targets);
		}

		static vector<int> generate_intervals(int n_threads, int length) {
			vector<int> intervals;
			for(int x=0;x<n_threads;x++) intervals.push_back(x * (length / n_threads));
			intervals.push_back(length);
			return intervals;
		}

		static void propagate_func(
			const vector<float>& input_values,
			vector<float>& output_values,
			vector<layer_neuron>& neurons,
			foreward_target_list& foreward_targets,
			int n_beg,
			int n_end
		) {
			// compute activations.
			for(int n=n_beg;n<n_end;n++) {
				layer_neuron& neuron = neurons[n];
				float sum = neuron.bias;
				const target_itv itv = foreward_targets.get_interval(n);
				for(int i=itv.beg;i<itv.end;i++) {
					const foreward_target& target = foreward_targets.targets[i];
					sum += target.weight * input_values[target.neuron_index];
				}
				neuron.signal = sum;
				output_values[n] = activation_func(sum);
			}
		}

		void propagate(const int n_threads, const std::vector<float>& input_values, std::vector<float>& output_values) override {
			// spawn threads.
			vector<int> intervals = generate_intervals(n_threads, neurons.size());
			vector<std::thread> threads;
			for(int x=0;x<n_threads;x++) {
				threads.push_back(std::thread(
					propagate_func,
					std::ref(input_values),
					std::ref(output_values),
					std::ref(this->neurons),
					std::ref(this->foreward_targets),
					intervals[x],
					intervals[x+1]
				));
			}
			for(int x=0;x<n_threads;x++) threads[x].join();
		}

		static void back_propagate_func_0(
			const vector<float>& output_error,
			vector<float>& signal_error_terms,
			vector<layer_neuron>& neurons,
			const int n_beg,
			const int n_end
		) {
			// compute signal error terms of output neurons.
			for(int n=n_beg;n<n_end;n++) {
				const float signal_error_term = output_error[n] * activation_derivative(neurons[n].signal);
				signal_error_terms[n] = signal_error_term;
				neurons[n].bias_error += signal_error_term;
			}
		}

		static void back_propagate_func_1(
			const vector<float>& output_error,
			vector<float>& input_error,
			const vector<float>& input_values,
			const vector<float>& signal_error_terms,
			backprop_target_list& backprop_targets,
			const int n_beg,
			const int n_end
		) {
			// back propagate input error.
			for(int n=n_beg;n<n_end;n++) {
				float input_error_sum = 0;
				const target_itv itv = backprop_targets.get_interval(n);
				const float mult = 1.0f / (itv.end - itv.beg);
				const float value = input_values[n];
				for(int x=itv.beg;x<itv.end;x++) {
					backprop_target& bt = backprop_targets.targets[x];
					const float signal_error_term = signal_error_terms[bt.neuron_index];
					input_error_sum += signal_error_term * mult * bt.weight;
				}
				for(int x=itv.beg;x<itv.end;x++) {
					backprop_target& bt = backprop_targets.targets[x];
					const float signal_error_term = signal_error_terms[bt.neuron_index];
					bt.weight_error += signal_error_term * mult * value;
				}
				input_error[n] = input_error_sum;
			}
		}

		/*
			for backprop derivation, see:
			https://dustinstansbury.github.io/theclevermachine/derivation-backpropagation
		*/
		void back_propagate(
			const int n_threads,
			std::vector<float>& output_error,
			std::vector<float>& input_error,
			std::vector<float>& input_values
		) override {
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
						back_propagate_func_0,
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

	struct autoencoder : network {

		int input_w;
		int input_h;
		int middle_layer_index;
		vector<layer_network>	layers;
		vector<vector<float>>	layer_values;	// values output by layers - in row-major-RGBA format.
		vector<vector<float>>	layer_errors;	// errors output by layers.

		autoencoder(int w, int h) {
			this->input_w = w;
			this->input_h = h;
			init_model_topology();
		}

	private:
		struct image_dimensions {
			int w;
			int h;
		};

		void push_output_layer(const layer_network& output_layer) {
			layers.push_back(output_layer);
			layer_values.push_back(vector<float>(output_layer.neurons.size()));
			layer_errors.push_back(vector<float>(output_layer.neurons.size()));
		}

		/*
			generates connections from AxA square of pixels (from input)
			to 1x1 square (in output), with AxA input square centered on output pixel.

			NOTE:
			- output layer should have the same neuron-layout as image-data does (row-major RGBA).
			- odd values of A are preferred (for symmetry).
			- if mix_channels is true, every pixel-value in input is connected to each
			pixel-value-neuron in output (4x as many connections).
			- if mix_channels is false, channels are kept seperate.
		*/
		void push_layer_mix_AxA_to_1x1(const image_dimensions dim, const int A, bool mix_channels) {
			// create output layer and generate targets for each neuron (pixel-value) in output.
			// NOTE: the order these target-lists are generated in must match order of neurons.
			layer_network output_layer;
			output_layer.neurons.resize(ML::image::get_image_data_length(dim.w, dim.h));
			for(int y=0;y<dim.h;y++) {
			for(int x=0;x<dim.w;x++) {
			for(int c=0;c<4;c++) {
				int in_x0 = x - A/2;
				int in_y0 = y - A/2;
				vector<int> connection_inds = ML::image::generate_image_data_indices(dim.w, dim.h, in_x0, in_y0, A, A, mix_channels ? -1 : c);
				output_layer.foreward_targets.push_list(connection_inds);
			}}}

			// generate backprop targets.
			output_layer.backprop_targets = output_layer.foreward_targets.get_inverse(ML::image::get_image_data_length(dim.w, dim.h));

			// push layer into list.
			push_output_layer(output_layer);
		}

		/*
			generates connections such that each AxA square from input is scaled
			into a BxB square in output.

			NOTE: image dimensions, A, and B must be picked such that image scales cleanly.
		*/
		void push_layer_scale_AxA_to_BxB(
			const image_dimensions idim,
			image_dimensions& odim,
			const int A,
			const int B,
			bool mix_channels
		) {
			// make sure images will scale cleanly, and have size compatible with choice of A and B.
			// - images dimensions should be divisible by A (input) or B (output).
			// - images should have the same number of AxA or BxB tiles as eachother in each dimension.
			odim = image_dimensions(
				(idim.w * B) / A,
				(idim.h * B) / A
			);
			assert(idim.w % A == 0);
			assert(idim.h % A == 0);
			assert(odim.w % B == 0);
			assert(odim.h % B == 0);
			assert(idim.w / A == odim.w / B);
			assert(idim.h / A == odim.h / B);

			// create output layer and generate targets for each neuron (pixel-value) in output.
			// NOTE: the order these target-lists are generated in must match order of neurons.
			layer_network output_layer;
			output_layer.neurons.resize(ML::image::get_image_data_length(odim.w, odim.h));
			for(int y=0;y<odim.h;y++) {
			for(int x=0;x<odim.w;x++) {
			for(int c=0;c<4;c++) {
				// get tile coordinates.
				int tx = x / B;
				int ty = y / B;
				// get input area.
				int in_x0 = tx * A;
				int in_y0 = ty * A;
				vector<int> connection_inds = ML::image::generate_image_data_indices(idim.w, idim.h, in_x0, in_y0, A, A, mix_channels ? -1 : c);
				output_layer.foreward_targets.push_list(connection_inds);
			}}}

			// generate backprop targets.
			output_layer.backprop_targets = output_layer.foreward_targets.get_inverse(ML::image::get_image_data_length(idim.w, idim.h));

			// push layer into list.
			push_output_layer(output_layer);
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
			image_dimensions  in_dim = image_dimensions(input_w, input_h);
			image_dimensions out_dim(0, 0);

			// mix and condense image.
			// (w, h) -> (w/32, h/32)
			push_layer_mix_AxA_to_1x1(in_dim, 3, false);
			push_layer_scale_AxA_to_BxB(in_dim, out_dim, 4, 2, true); in_dim = out_dim;
			/*
			push_layer_mix_AxA_to_1x1(in_dim, 3, false);
			push_layer_scale_AxA_to_BxB(in_dim, out_dim, 4, 2, true); in_dim = out_dim;
			push_layer_mix_AxA_to_1x1(in_dim, 7, true);
			push_layer_scale_AxA_to_BxB(in_dim, out_dim, 8, 4, true); in_dim = out_dim;
			push_layer_mix_AxA_to_1x1(in_dim, 9, true);
			push_layer_scale_AxA_to_BxB(in_dim, out_dim, 16, 8, true); in_dim = out_dim;
			//*/

			// middle layer - embedding values will be obtained from this image.
			middle_layer_index = layers.size();

			// expand back to original image.
			/*
			push_layer_scale_AxA_to_BxB(in_dim, out_dim, 8, 16, true); in_dim = out_dim;
			push_layer_mix_AxA_to_1x1(in_dim, 9, true);
			push_layer_scale_AxA_to_BxB(in_dim, out_dim, 4, 8, true); in_dim = out_dim;
			push_layer_mix_AxA_to_1x1(in_dim, 7, true);
			push_layer_scale_AxA_to_BxB(in_dim, out_dim, 2, 4, true); in_dim = out_dim;
			push_layer_mix_AxA_to_1x1(in_dim, 3, false);
			//*/
			push_layer_scale_AxA_to_BxB(in_dim, out_dim, 2, 4, true); in_dim = out_dim;
			push_layer_mix_AxA_to_1x1(in_dim, 3, false);

			assert(out_dim.w == input_w);
			assert(out_dim.h == input_h);
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

		void apply_batch_error(float rate) override {
			for(int x=0;x<layers.size();x++) layers[x].apply_batch_error(rate);
		}

		void propagate(const int n_threads, const std::vector<float>& input_values, std::vector<float>& output_values) override {
			// first layer.
			layers[0].propagate(n_threads, input_values, layer_values[0]);
			// middle layers.
			for(int z=1;z<layer_values.size();z++) {
				layers[z].propagate(n_threads, layer_values[z-1], layer_values[z]);
			}
			// copy output.
			output_values = layer_values[layer_values.size()-1];
		}

		void back_propagate(const int n_threads, std::vector<float>& output_error, std::vector<float>& input_error, std::vector<float>& input_values) override {
			// copy output.
			layer_errors[layer_errors.size()-1] = output_error;
			// middle layers.
			for(int z=layer_values.size()-1;z>0;z--) layers[z].back_propagate(n_threads, layer_errors[z], layer_errors[z-1], layer_values[z-1]);
			// first layer.
			layers[0].back_propagate(n_threads, layer_errors[0], input_error, input_values);
		}
	};
}
