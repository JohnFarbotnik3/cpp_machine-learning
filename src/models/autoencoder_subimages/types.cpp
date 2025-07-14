
#ifndef F_ae_subimage_types
#define F_ae_subimage_types

#include "src/image/value_image_padded.cpp"

namespace ML::models::autoencoder_subimage {
	using namespace ML::image;
	using image_t = ML::image::value_image_padded<float>;
	using iter_t = ML::image::value_image_padded_iterator;
	using dim_t = ML::image::value_image_padded_dimensions;

	struct fw_target { float weight=0.0f; };
	struct bp_target { float weight=0.0f, weight_error=0.0f; int output_neuron_index; };

	enum LAYER_TYPE {
		NONE,
		/*
			scale image from AxA squares to BxB squares,
			mixing colour channels as well.
		*/
		ENCODE,
		/*
			mix pixels (channel-isolated) from centered-NxN squares
			to centered-MxM squares, leaving image size the same.
		*/
		SPATIAL_MIX,
		/*
			scale image from AxA squares to BxB squares,
			mixing pixel-values from centered-NxN squares to centered-BxB squares.
			(where N is non-zero and is divisible by A.)

			this dramatically increases number of parameters,
			but may be capable of encoding information that ENCODE and SPATIAL_MIX struggle to.

			an example use case would be adding channels and spatially-mixing,
			before condensing image in next layer.
		*/
		ENCODE_MIX,
		/* connect all input neurons to output neurons. */
		DENSE,
	};

	struct layer_pattern {
		LAYER_TYPE type;
		int A,B;
		int N,M;
	};

}

#endif
