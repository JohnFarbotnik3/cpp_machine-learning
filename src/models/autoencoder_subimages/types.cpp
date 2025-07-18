
#ifndef F_ae_subimage_types
#define F_ae_subimage_types

#include <vector>
#include "src/image/value_image_lines.cpp"
#include "src/image/value_image_padded.cpp"

namespace ML::models::autoencoder_subimage {
	//using namespace ML::image;
	using std::vector;
	using padded_image_f = ML::image::value_image_padded::value_image_padded<float>;
	using padded_image_i = ML::image::value_image_padded::value_image_padded<int>;
	using padded_dim_t = ML::image::value_image_padded::value_image_padded_dimensions;
	using simple_image_f = ML::image::value_image_lines::value_image_lines<float>;
	using simple_image_i = ML::image::value_image_lines::value_image_lines<int>;
	using simple_dim_t = ML::image::value_image_lines::value_image_lines_dimensions;

	enum LAYER_TYPE {
		NONE,
		/*
			scale image from AxA squares to BxB squares,
			mixing colour channels as well.
		*/
		ENCODE,
		/*
			mix pixels (channel-isolated) from centered-NxN squares
			to centered-BxB squares, leaving image size the same.
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
		int A,B,N;

		static layer_pattern dense() {
			return layer_pattern{ LAYER_TYPE::DENSE, 0, 0, 0 };
		}
		static layer_pattern spatial_mix(const int N, const int B) {
			return layer_pattern{ LAYER_TYPE::SPATIAL_MIX, 0, B, N };
		}
		static layer_pattern encode(const int A, const int B) {
			return layer_pattern{ LAYER_TYPE::ENCODE, A, B, 0 };
		}
		static layer_pattern encode_mix(const int A, const int B, const int N) {
			return layer_pattern{ LAYER_TYPE::ENCODE_MIX, A, B, N };
		}

	};

	struct fw_target { float weight=0.0f; };
	struct bp_target { float weight=0.0f, weight_error=0.0f; int output_neuron_index; };
	struct fw_target_list {
		int weights_per_output_neuron;
		vector<fw_target> targets;
	};
	struct bp_target_list {
		padded_image_i intervals;// postfix-intervals of targets belonging to each input-neuron.
		vector<bp_target> targets;
	};

	struct input_neuron_offset_struct {
		/*
			each output-neuron reads from a similar arrangement of input-values (for example, an NxN square);
			this arrangement is stored as a re-usable kernel of offsets.
		*/
		vector<int> kernel;
		/*
			each output-neuron may read from a different area of the input-image;
			these offsets are meant to be combined with the kernel to get input-value indices.
		*/
		simple_image_i kernel_offsets;
	};

	///*
	float activation_func(const float value) {
		const float sign = value >= 0.0f ? 1.0f : -1.0f;
		const float mag = std::abs(value);
		if(mag < 0.25f) return value * 1.0f;					// [0.00, 0.25] 0.000 -> 0.250
		if(mag < 0.50f) return value * 0.7f + (sign * 0.075f);	// [0.25, 0.50] 0.250 -> 0.425
		if(mag < 1.00f) return value * 0.5f + (sign * 0.175f);	// [0.50, 1.00] 0.425 -> 0.675
		if(mag < 2.00f) return value * 0.3f + (sign * 0.375f);	// [1.00, 2.00] 0.675 -> 0.975
		return value * 0.1f + (sign * 0.775f);					// [2.00,  inf] 0.975 -> inf.
	}

	float activation_derivative(const float value) {
		const float mag = std::abs(value);
		if(mag < 0.25f) return 1.0f;
		if(mag < 0.50f) return 0.7f;
		if(mag < 1.00f) return 0.5f;
		if(mag < 2.00f) return 0.3f;
		return 0.1f;
	}
	//*/

	/*
	float activation_func(const float value) {
		return value / (1.0f + std::abs(value));
	}

	float activation_derivative(const float value) {
		return (activation_func(value+0.001f) - activation_func(value)) * 1000.0f;
	}
	//*/

}

#endif
