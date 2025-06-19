
#include <algorithm>
#include <cstdio>
#include <vector>
#include <random>
#include "./network.cpp"

/*
	fixed networks are assumed to be made of simple layers
	which propagate from input to output in a single step.
*/
namespace ML::networks {
	//using namespace ML::networks;
	using index_t = unsigned int;
	using std::vector;

	template<index_t MAX_CONNECTIONS>
	struct layer_neuron {
		float	value = 0;
		float	bias = 0;
		float	weights[MAX_CONNECTIONS];
		index_t	targets_len = 0;
		index_t	targets_arr[MAX_CONNECTIONS];
		float	loss_sum = 0;					// storage value for backpropagation.
		float	loss_bias = 0;					// cumulative adjustment to apply to bias.
		float	loss_weights[MAX_CONNECTIONS];	// cumulative adjustment to apply to weights.
	};

	/*
		a network of forward-connected neurons with the following layout:
		|...input...|...hidden...|...output...|

		this network is intended to be populated externally.
	*/
	template<index_t MAX_CONNECTIONS>
	struct layer_network : network {
		using neuron_t = layer_neuron<MAX_CONNECTIONS>;

		std::mt19937 gen32;
		vector<vector<neuron_t>> layers;

		layer_network(int seed) {
			this->gen32 = std::mt19937(seed);
		}

		// ============================================================
		// helpers
		// ------------------------------------------------------------

		float rand(float a, float b) {
			std::uniform_real_distribution<float> distr(a, b);
			return distr(gen32);
		}

		// ============================================================
		// network functions
		// ------------------------------------------------------------

		void propagate(std::vector<float>& input, std::vector<float>& output) override {
			// make sure lengths match.
			vector<neuron_t>  input_layer = layers[0];
			vector<neuron_t> output_layer = layers[layers.size() - 1];
			if( input.size() !=  input_layer.size()) { fprintf(stderr, "ERROR:  input lengths dont match: %u != %u\n",  input.size(),  input_layer.size()); return; }
			if(output.size() != output_layer.size()) { fprintf(stderr, "ERROR: output lengths dont match: %u != %u\n", output.size(), output_layer.size()); return; }

			// copy input values into network.
			for(int x=0;x<input_layer.size();x++) input_layer[x].value = input[x];

			// propagate signal through network.
			for(int z=1;z<layers.size();z++) {
				vector<neuron_t>& layer = layers[z];
				vector<neuron_t>& prev_layer = layers[z-1];
				for(int x=0;x<layer.size();x++) {
					neuron_t& neuron = layer[x];
					float value = neuron.bias;
					for(int t=0;t<neuron.targets_len;t++) {
						neuron_t& target = prev_layer[neuron.targets_arr[t]];
						value += neuron.weights[t] * target.value;
					}
					// ReLU.
					neuron.value = std::max<float>(value, 0);
				}
			}

			// copy output values from network.
			for(int x=0;x<output_layer.size();x++) output[x] = output_layer[x].value;
		}

		void apply_batch_loss() override {}// TODO

		void reset_batch_loss() override {}// TODO

		void anneal(float rate) override {}// TODO

		void back_propagate(float rate, std::vector<float>& input_loss, std::vector<float>& output_loss) {
			// make sure lengths match.
			vector<neuron_t>  input_layer = layers[0];
			vector<neuron_t> output_layer = layers[layers.size() - 1];
			if( input_loss.size() !=  input_layer.size()) { fprintf(stderr, "ERROR:  input lengths dont match: %u != %u\n",  input_loss.size(),  input_layer.size()); return; }
			if(output_loss.size() != output_layer.size()) { fprintf(stderr, "ERROR: output lengths dont match: %u != %u\n", output_loss.size(), output_layer.size()); return; }

			// TODO - continue from here...
		}
		/*
		void back_anneal_get_loss(float loss) {
			float sign = loss >= 0.0f ? 1.0f : -1.0f;
			float fabs = loss >= 0.0f ? loss : -loss;
			//float mult = fabs >= 1.0f ? 1.0f : 0.2f;
			float mult = sqrtf(fabs);
			return rand(-0.2f, 1.0f) * sign * mult;
		}
		void back_anneal(float annealing_rate, std::vector<float>& input_loss, std::vector<float>& output_loss) override {
			// clear old values.
			for(int x=0;x<total_length;x++) neurons[x].loss_sum = 0;

			// TODO - rethink this, and draw some pictures.
			// ^ it may be easier to implement regular backpropagation first.

			// copy loss values from output.
			for(int x=0;x<output_length;x++) {
				neuron_t& neuron = neurons[x + output_offset];
				float loss = output_loss[x];
				neuron.loss_sum += back_anneal_get_loss(loss);
			}

			// back-propagate loss.
			for(int x=total_length;x>=input_length;x--) {
				neuron_t& neuron = neurons[x];
				float loss = neuron.loss_sum[x];
				for(int t=0;t<neuron.targets_len;t++) {
					neuron_t& target = neurons[neuron.targets_arr[t]];

				}
			}

			// add to batch loss.
			// TODO - remember to multiply by annealing_rate.
		}
		*/

	};
}







