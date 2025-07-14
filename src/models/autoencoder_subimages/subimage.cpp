
#include <vector>
#include "types.cpp"

namespace ML::models::autoencoder_subimage {
	using std::vector;

	struct subimage {
		image_t input;
		image_t biases;
		image_t biases_error;// accumulated error in biases during minibatch.
		image_t output;// image of output values - used for backprop.
		image_t signal;// image of signal values - used for backprop.
		vector<fw_target> fw_targets;// foreward-propagation targets.
		vector<bp_target> bp_targets;// backward-propagation targets.

		subimage() = default;
		subimage(const dim_t idim, const dim_t odim, const layer_pattern pattern) :
		input(idim),
		biases(odim),
		biases_error(odim),
		output(odim),
		signal(odim)
		{
			const int IMAGE_SIZE_I = idim.inner_length();
			const int IMAGE_SIZE_O = odim.inner_length();
			int n_weights = 0;
			if(pattern.type == LAYER_TYPE::ENCODE		) n_weights = IMAGE_SIZE_O * pattern.A * pattern.A * idim.innerC();
			if(pattern.type == LAYER_TYPE::SPATIAL_MIX	) n_weights = IMAGE_SIZE_O * pattern.N * pattern.N;
			if(pattern.type == LAYER_TYPE::ENCODE_MIX	) n_weights = IMAGE_SIZE_O * pattern.N * pattern.N * idim.innerC();
			if(pattern.type == LAYER_TYPE::DENSE		) n_weights = IMAGE_SIZE_O * IMAGE_SIZE_I;
			assert(n_weights > 0);
			fw_targets.resize(n_weights, fw_target{ 0.0f });
			bp_targets.resize(n_weights, bp_target{ 0.0f, 0.0f, 0 });
			// TODO
		}

		int input_image_size() const  {
			return input.dim.inner_length();
		}
		int output_image_size() const {
			return output.dim.inner_length();
		}
		int weights_per_output_neuron() const {
			return fw_targets.size() / output_image_size();
		}
		int weights_per_input_neuron() const {
			return bp_targets.size() / input_image_size();
		}
	};

}
