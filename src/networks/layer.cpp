
#include <algorithm>
#include <vector>
#include <random>
#include "./network.cpp"

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

		layer_neuron() = default;
		layer_neuron(int len, int ofs) {
			this->targets_len = len;
			this->targets_ofs = ofs;
		}
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
		std::mt19937 gen32;

		layer_network(int seed) {
			this->gen32 = std::mt19937(seed);
		}

		// ============================================================
		// helpers
		// ------------------------------------------------------------

		// TODO - rand generator reference should be provided to functions instead of being class member.
		float rand(float a, float b) {
			std::uniform_real_distribution<float> distr(a, b);
			return distr(gen32);
		}

		float activation_func(float value) {
			return std::max<float>(value, 0);
		}

		float activation_derivative(float value) {
			return 1.0f;
		}

		// ============================================================
		// network functions
		// ------------------------------------------------------------

		void apply_batch_error(float rate) override {
			for(int n=0;n<neurons.size();n++) {
				neuron_t& neuron = neurons[n];
				neuron.bias += neuron.bias_error * rate;
				neuron.bias_error = 0;
				for(int i=0;i<neuron.targets_len;i++) {
					target_t& target = targets[neuron.targets_ofs + i];
					target.weight += target.weight_error * rate;
					target.weight_error = 0;
				}
			}
		}

		void propagate(std::vector<float>& input_values, std::vector<float>& output_values) override {
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

		void back_propagate(std::vector<float>& input_error, std::vector<float>& output_error, std::vector<float>& input_values) override {
			// clear input error.
			for(int x=0;x<input_error.size();x++) input_error[x] = 0;

			// propagate error.
			for(int n=0;n<neurons.size();n++) {
				neuron_t& neuron = neurons[n];
				const float signal_error_term = output_error[n] * activation_derivative(neuron.signal);
				neuron.bias_error += signal_error_term;
				for(int i=0;i<neuron.targets_len;i++) {
					target_t& target = targets[neuron.targets_ofs + i];
					target.weight_error += signal_error_term * input_values[target.index];
					input_error[target.index] += signal_error_term * target.weight;
				}
			}

			// normalize input error to match output error.
			float isum = 0;
			float osum = 0;
			for(int x=0;x< input_error.size();x++) isum +=  input_error[x];
			for(int x=0;x<output_error.size();x++) osum += output_error[x];
			float mult = osum / isum;
			for(int x=0;x< input_error.size();x++) input_error[x] *= mult;
		}
	};
}







