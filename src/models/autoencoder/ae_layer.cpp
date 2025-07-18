
#include <vector>
#include "src/utils/vector_util.cpp"
#include "./target_list.cpp"

namespace ML::models::autoencoder {
	using std::vector;
	using namespace utils::vector_util;

	/*
		a single layer network of forward-connected neurons.
		NOTE: this network is intended to be populated externally.
	*/
	struct ae_layer {
		vector<float> biases;
		vector<float> output;// image of output values - used for backprop.
		vector<float> signal;// image of signal values - used for backprop.
		vector<float> biases_error;// accumulated error in biases during minibatch.
		vector<float> weights_error;// accumulated error in weights during minibatch.
		foreward_target_list foreward_targets;
		backprop_target_list backprop_targets;// inverse of this layer's foreward_targets.
		int INPUT_IMAGE_SIZE;
		int OUTPUT_IMAGE_SIZE;

		ae_layer(int INPUT_IMAGE_SIZE, int OUTPUT_IMAGE_SIZE) :
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

		static float activation_func(const float value) {
			const float sign = value >= 0.0f ? 1.0f : -1.0f;
			const float mag = std::abs(value);
			if(mag < 0.5f) return value * 1.0f;						// [0.0, 0.5] 0.00 -> 0.50
			if(mag < 1.0f) return value * 0.7f + (sign * 0.15f);	// [0.5, 1.0] 0.50 -> 0.85
			if(mag < 2.0f) return value * 0.5f + (sign * 0.35f);	// [1.0, 2.0] 0.85 -> 1.35
			if(mag < 4.0f) return value * 0.3f + (sign * 0.75f);	// [2.0, 4.0] 1.35 -> 1.95
			return value * 0.1f + (sign * 1.55f);					// [4.0, inf] 1.95 -> inf.
		}

		static float activation_derivative(const float value) {
			const float mag = std::abs(value);
			if(mag < 0.5f) return 1.0f;
			if(mag < 1.0f) return 0.7f;
			if(mag < 2.0f) return 0.5f;
			if(mag < 4.0f) return 0.3f;
			return 0.1f;
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

		static void propagate_func(ae_layer& layer, const vector<float>& input_values, vector<float>& output_values, const int o_beg, int const o_end) {
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

		static void back_propagate_func_output_side(ae_layer& layer, vector<float>& signal_error_terms, const vector<float>& output_error, const int o_beg, const int o_end) {
			for(int n=o_beg;n<o_end;n++) {
				signal_error_terms[n] = output_error[n] * activation_derivative(layer.signal[n]);
			}
			for(int n=o_beg;n<o_end;n++) {
				layer.biases_error[n] += signal_error_terms[n];
			}
		}

		static void back_propagate_func_input_side(ae_layer& layer, vector<float>& input_error, const vector<float>& input_value, const vector<float>& signal_error_terms, const int i_beg, const int i_end) {
			// for each neuron in input...
			for(int in_n=i_beg;in_n<i_end;in_n++) {
				// gather input-error and weight-error.
				float input_error_sum = 0;
				const target_itv itv = layer.backprop_targets.get_interval(in_n);
				for(int x=itv.beg;x<itv.end;x++) {
					const backprop_target bt = layer.backprop_targets.targets[x];
					const int& out_n = bt.neuron_index;
					const float error_term = signal_error_terms[out_n];
					const float num_inputs = layer.foreward_targets.offsets[out_n+1] - layer.foreward_targets.offsets[out_n];
					const float mult = 1.0f / num_inputs;
					// TODO - test using "mult=sqrtf(1.0f / num_inputs);".
					input_error_sum        += error_term * mult * bt.weight;
					layer.weights_error[x] += error_term * mult * input_value[in_n];
				}
				input_error[in_n] = input_error_sum;
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

		static void apply_batch_error_biases(ae_layer& layer, const int beg, const int end, const float adjustment_rate) {
			const float BIAS_LIMIT = 10.0f;
			const float BIAS_ADJUSTMENT_LIMIT = 0.5f;
			for(int n=beg;n<end;n++) {
				const float adjustment = std::clamp(layer.biases_error[n] * adjustment_rate, -BIAS_ADJUSTMENT_LIMIT, +BIAS_ADJUSTMENT_LIMIT);
				layer.biases[n] = std::clamp(layer.biases[n] + adjustment, -BIAS_LIMIT, +BIAS_LIMIT);
			}
		}
		static void apply_batch_error_weights(ae_layer& layer, const int beg, const int end, const float adjustment_rate) {
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

}
