
#include "types.cpp"

namespace ML::models::autoencoder_subimage {
	struct subimage {
		image_f input;
		image_f biases;
		image_f biases_error;// accumulated error in biases during minibatch.
		image_f signal;// image of signal values - used for backprop.
		image_f output;// image of output values - used for backprop.
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
			const int IMAGE_SIZE_I = idim.inner_length();
			const int IMAGE_SIZE_O = odim.inner_length();
			int n_weights = 0;
			if(pattern.type == LAYER_TYPE::ENCODE		) n_weights = IMAGE_SIZE_O * pattern.A * pattern.A * idim.innerC();
			if(pattern.type == LAYER_TYPE::SPATIAL_MIX	) n_weights = IMAGE_SIZE_O * pattern.N * pattern.N;
			if(pattern.type == LAYER_TYPE::ENCODE_MIX	) n_weights = IMAGE_SIZE_O * pattern.N * pattern.N * idim.innerC();
			if(pattern.type == LAYER_TYPE::DENSE		) n_weights = IMAGE_SIZE_O * IMAGE_SIZE_I;
			assert(n_weights > 0);
			// TODO - initialize bp target indices.
		}

		int input_image_size() const  {
			return input.dim.inner_length();
		}
		int output_image_size() const {
			return output.dim.inner_length();
		}
	};

}
