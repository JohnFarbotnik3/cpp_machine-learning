
#include "types.cpp"
#include "patterns.cpp"
#include <cassert>
#include "src/utils/vector_util.cpp"
#include "src/utils/random.cpp"

namespace ML::models::autoencoder_subimage {
	struct subimage {
		simple_image_f biases;
		simple_image_f biases_error;	// accumulated error in biases during minibatch.
		simple_image_f signal;			// image of signal values - used for backprop.
		padded_image_f value_image_i;	// input values - populated externally.
		simple_image_f value_image_o;
		padded_image_f error_image_i;
		simple_image_f error_image_o;	// output-side error - populated externally.
		input_neuron_offset_struct fw_offsets;
		vector<float> weights;
		vector<float> weights_error;

		subimage() = default;
		subimage(const padded_dim_t idim, const simple_dim_t odim, const layer_pattern pattern) :
			biases			(odim),
			biases_error	(odim),
			signal			(odim),
			value_image_i	(idim),
			value_image_o	(odim),
			error_image_i	(idim),
			error_image_o	(odim)
		{
			assert(idim.inner_length() > 0);
			assert(odim.length() > 0);
			fw_offsets = get_input_neuron_offsets_kernel(pattern, idim, odim);
			const int n_weights = fw_offsets.kernel.size() * odim.length();
			weights.resize(n_weights, 0.0f);
			weights_error.resize(n_weights, 0.0f);
		}

		void init_model_parameters(int seed, float bias_mean, float bias_stddev, float weight_mean, float weight_stddev) {
			std::mt19937 gen32 = utils::random::get_generator_32(seed);
			std::normal_distribution distr_bias = utils::random::rand_normal<float>(bias_mean, bias_stddev);
			std::normal_distribution distr_weight = utils::random::rand_normal<float>(weight_mean, weight_stddev);

			const int WEIGHTS_PER_OUTPUT_NEURON = fw_offsets.kernel.size();
			const float mult = sqrtf(1.0f / WEIGHTS_PER_OUTPUT_NEURON);
			for(int n=0;n<biases.data.size();n++) biases.data[n] = distr_bias(gen32);
			for(int x=0;x<weights.size();x++) weights[x] = distr_weight(gen32) * mult;
		}

		void foreward_propagate() {
			const padded_dim_t idim = value_image_i.dim;
			const simple_dim_t odim = value_image_o.dim;
			const int WEIGHTS_PER_OUTPUT_NEURON = fw_offsets.kernel.size();

			for(int out_n=0;out_n<odim.length();out_n++) {
				const int kofs = fw_offsets.kernel_offsets.data[out_n];
				const int wofs = out_n * WEIGHTS_PER_OUTPUT_NEURON;
				float sum = biases.data[out_n];
				for(int x=0;x<WEIGHTS_PER_OUTPUT_NEURON;x++) {
					const int in_n = fw_offsets.kernel[x] + kofs;
					sum += weights[wofs + x] * value_image_i.data[in_n];
				}
				signal.data[out_n] = sum;
			}
			for(int out_n=0;out_n<odim.length();out_n++) {
				value_image_o.data[out_n] = activation_func(signal.data[out_n]);
			}
		}

		void backward_propagate() {
			const padded_dim_t idim = error_image_i.dim;
			const simple_dim_t odim = error_image_o.dim;
			const int WEIGHTS_PER_OUTPUT_NEURON = fw_offsets.kernel.size();
			//const float mult = sqrtf(1.0f / WEIGHTS_PER_OUTPUT_NEURON);
			const float mult = 1.0f / WEIGHTS_PER_OUTPUT_NEURON;

			error_image_i.clear();
			for(int out_n=0;out_n<odim.length();out_n++) {
				if(error_image_o.data[out_n] == 0.0f) continue;// OPTIMIZATION: skip if no error.
				const int kofs = fw_offsets.kernel_offsets.data[out_n];
				const int wofs = out_n * WEIGHTS_PER_OUTPUT_NEURON;
				const float signal_error_term = error_image_o.data[out_n] * activation_derivative(signal.data[out_n]);
				const float signal_error_term_w = signal_error_term * mult;
				biases_error.data[out_n] += signal_error_term;
				for(int x=0;x<WEIGHTS_PER_OUTPUT_NEURON;x++) {
					const int in_n = fw_offsets.kernel[x] + kofs;
					weights_error[wofs + x]		+= signal_error_term_w * value_image_i.data[in_n];
					error_image_i.data[in_n]	+= signal_error_term * weights[wofs + x];
				}
			}
		}

		void apply_batch_error_biases(const float adjustment_rate) {
			const float BIAS_LIMIT = 100.0f;
			const float BIAS_ADJUSTMENT_LIMIT = 0.5f;
			for(int n=0;n<biases.data.size();n++) {
				const float adjustment = std::clamp(biases_error.data[n] * adjustment_rate, -BIAS_ADJUSTMENT_LIMIT, +BIAS_ADJUSTMENT_LIMIT);
				biases.data[n] = std::clamp(biases.data[n] + adjustment, -BIAS_LIMIT, +BIAS_LIMIT);
			}
		}
		void apply_batch_error_weights(const float adjustment_rate) {
			const float WEIGHT_LIMIT = 100.0f;
			const float WEIGHT_ADJUSTMENT_LIMIT = 0.5f;
			for(int x=0;x<weights.size();x++) {
				const float adjustment = std::clamp(weights_error[x] * adjustment_rate, -WEIGHT_ADJUSTMENT_LIMIT, +WEIGHT_ADJUSTMENT_LIMIT);
				weights[x] = std::clamp(weights[x] + adjustment, -WEIGHT_LIMIT, +WEIGHT_LIMIT);
			}
		}

		void clear_batch_error_biases() { biases_error.data.assign(biases_error.data.size(), 0.0f); }
		void clear_batch_error_weights() { weights_error.assign(weights_error.size(), 0.0f); }
	};
}






