
#include "types.cpp"
#include "patterns.cpp"

namespace ML::models::autoencoder_subimage {
	struct subimage {
		image_f input;
		image_f biases;
		image_f biases_error;// accumulated error in biases during minibatch.
		image_f signal;// image of signal values - used for backprop.
		image_f output;// image of output values - used for backprop.
		neuron_offset_struct fw_offsets;
		fw_target_list fw_targets;
		bp_target_list bp_targets;

		subimage() = default;
		subimage(const dim_t idim, const dim_t odim, const layer_pattern pattern) :
		input		(idim),
		biases		(odim),
		biases_error(odim),
		signal		(odim),
		output		(odim)
		{
			assert(idim.inner_length() > 0);
			assert(odim.inner_length() > 0);
			assert(odim.padX == 0);
			assert(odim.padY == 0);
			fw_offsets = get_input_neuron_offsets_kernel(pattern, idim, odim);
			fw_targets = init_fw_targets(fw_offsets, odim);
			bp_targets = init_bp_targets(fw_offsets, idim, odim);
		}
	};
}






