
#include <algorithm>
#include <cassert>
#include <vector>
#include "./network.cpp"
#include "../utils/simd.cpp"

/*
	fixed networks are assumed to be made of simple layers
	which propagate from input to output in a single step.

	for backprop derivation, see:
	https://dustinstansbury.github.io/theclevermachine/derivation-backpropagation
*/
namespace ML::networks {

	struct layer_neuron {
		float signal 	= 0;// sum of input activations and bias - used for backpropagation.
		float bias		= 0;// bias of neuron[x] in the network.
		float bias_error= 0;// cumulative adjustment to apply to bias.
		int targets_len	= 0;// number of input targets.
		int targets_ofs	= 0;// start position of input-targets list.
	};

	struct layer_target {
		int index;			// index of target value/neuron.
		float weight;
		float weight_error;	// cumulative adjustment to apply to weight.

		layer_target(int index) {
			this->index = index;
			this->weight = 0;
			this->weight_error = 0;
		}
	};

	/*
		a single layer network of forward-connected neurons.

		this network is intended to be populated externally.
	*/
	struct layer_network : network {
		using neuron_t = layer_neuron;
		using target_t = layer_target;

		std::vector<neuron_t>	neurons;
		std::vector<target_t>	targets;

		// ============================================================
		// activation functions.
		// ------------------------------------------------------------

		float activation_func(float value) {
			/*
			NOTE: pure ReLU was causing problems.
			now using ReLU with leakage.
			//return std::max<float>(value, 0);
			*/
			return value > 0.0f ? value : value * 0.5f;
		}

		float activation_derivative(float value) {
			/*
			NOTE: I deliberately picked 1.0f as the derivative before as I was
			worried that weights wouldnt be pushed back up if values got stuck in the negatives.
			this was stupid (I didnt pay close attention to the math), and it caused model degeneration.
			//return 1.0f;
			NOTE: maybe I was right, model seems to get stuck, and when
			the signal_error_term hits zero the neuron stops learning.
			//return value > 0.0f ? 1.0f : 0.0f;
			*/
			return value > 0.0f ? 1.0f : 0.5f;
		}

		// ============================================================
		// network functions
		// ------------------------------------------------------------

		void apply_batch_error(float rate) override {
			// TODO TEST - troubleshoot learning problems.
			const float BIAS_LIMIT = 3.0f;
			const float BIAS_RATE = 1.0f;
			const float WEIGHT_LIMIT = 100.0f;
			const float WEIGHT_RATE = 1.0f;
			const float ADJUSTMENT_LIMIT = 0.5f;
			for(int n=0;n<neurons.size();n++) {
				neuron_t& neuron = neurons[n];
				neuron.bias += std::clamp(neuron.bias_error * rate, -ADJUSTMENT_LIMIT, +ADJUSTMENT_LIMIT) * BIAS_RATE;
				neuron.bias  = std::clamp(neuron.bias, -BIAS_LIMIT, +BIAS_LIMIT);
				neuron.bias_error = 0;
				for(int i=0;i<neuron.targets_len;i++) {
					target_t& target = targets[neuron.targets_ofs + i];
					target.weight += std::clamp(target.weight_error * rate, -ADJUSTMENT_LIMIT, ADJUSTMENT_LIMIT) * WEIGHT_RATE;
					target.weight  = std::clamp(target.weight, -WEIGHT_LIMIT, +WEIGHT_LIMIT);
					target.weight_error = 0;
				}
			}
		}

		void propagate(const std::vector<float>& input_values, std::vector<float>& output_values) override {
			// compute activations.
			for(int n=0;n<neurons.size();n++) {
				neuron_t& neuron = neurons[n];
				float sum = neuron.bias;
				for(int i=0;i<neuron.targets_len;i++) {
					target_t& target = targets[neuron.targets_ofs + i];
					sum += target.weight * input_values[target.index];
				}
				neuron.signal = sum;
				output_values[n] = activation_func(sum);
			}
		}

		void back_propagate(
			std::vector<float>& output_error,
			std::vector<float>& input_error,
			std::vector<float>& input_values
		) override {
			assert(output_error.size() == neurons.size());
			assert(input_error.size() == input_values.size());

			// clear input error.
			for(int x=0;x<input_error.size();x++) input_error[x] = 0;

			// propagate error.
			for(int n=0;n<neurons.size();n++) {
				assert(-100.0f < output_error[n] && output_error[n] < 100.0f);
				neuron_t& neuron = neurons[n];
				const float signal_error_term = output_error[n] * activation_derivative(neuron.signal);
				const float mult = 1.0f / neuron.targets_len;
				neuron.bias_error += signal_error_term;
				for(int i=0;i<neuron.targets_len;i++) {
					target_t& target = targets[neuron.targets_ofs + i];
					target.weight_error += signal_error_term * input_values[target.index] * mult;
					input_error[target.index] += signal_error_term * target.weight * mult;
				}
			}
		}
	};
}







