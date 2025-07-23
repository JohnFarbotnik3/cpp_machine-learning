
#ifndef F_ae_subimage_types
#define F_ae_subimage_types

#include "simd.cpp"
#include "simd_image.cpp"

namespace ML::models::autoencoder_subimage {
	struct image_bounds {
		int x0,x1;
		int y0,y1;

		int X() const { return x1-x0; }
		int Y() const { return y1-y0; }
	};

	enum LAYER_TYPE {
		NONE,

		/* connect all input neurons from input-bounds to all output neurons in output-bounds. */
		DENSE,

		/* scale image from AxA squares to BxB squares, mixing colour channels as well. */
		ENCODE,

		/*
			scale image from AxA squares to BxB squares,
			mixing pixel-values from centered-NxN squares to centered-BxB squares.
			(where N is non-zero and is divisible by A.)

			this dramatically increases number of parameters compared to SPATIAL_MIX,
			but may be capable of encoding information that ENCODE and SPATIAL_MIX struggle to.

			an example use case would be adding channels and spatially-mixing,
			before condensing image in next layer.
		*/
		ENCODE_MIX,

		/*
			mix pixels (channel-isolated) from centered-NxN squares
			to centered-BxB squares, leaving image size the same.
		*/
		SPATIAL_MIX,
	};

	struct layer_pattern {
		LAYER_TYPE type;
		int A,B,N;
		int WEIGHTS_PER_OUTPUT_NEURON;

		static layer_pattern dense(simd_image_8f_dimensions idim, const image_bounds bounds_i) {
			const int wpon = bounds_i.X() *bounds_i.Y() * idim.C;
			return layer_pattern{ LAYER_TYPE::DENSE, 0, 0, 0, wpon };
		}
		static layer_pattern encode(simd_image_8f_dimensions idim, const int A, const int B) {
			const int wpon = A * A * idim.C;
			return layer_pattern{ LAYER_TYPE::ENCODE, A, B, A, wpon };
		}
		static layer_pattern encode_mix(simd_image_8f_dimensions idim, const int A, const int B, const int N) {
			const int wpon = N * N * idim.C;
			return layer_pattern{ LAYER_TYPE::ENCODE_MIX, A, B, N, wpon };
		}
		static layer_pattern spatial_mix(simd_image_8f_dimensions idim, const int N, const int B) {
			const int wpon = N * N;
			return layer_pattern{ LAYER_TYPE::SPATIAL_MIX, B, B, N, wpon };
		}
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

	vec8f simd_activation_func(vec8f signal) {
		const vec8f mag = simd_abs(signal);
		vec8f mult = _mm256_set1_ps(1.0f);
		mult = simd_gte_cmov(mag, 0.25f, 0.7f, mult);
		mult = simd_gte_cmov(mag, 0.50f, 0.5f, mult);
		mult = simd_gte_cmov(mag, 1.00f, 0.3f, mult);
		mult = simd_gte_cmov(mag, 2.00f, 0.1f, mult);
		vec8f ofs = _mm256_setzero_ps();
		ofs = simd_gte_cmov(mag, 0.25f, 0.075f, ofs);
		ofs = simd_gte_cmov(mag, 0.50f, 0.175f, ofs);
		ofs = simd_gte_cmov(mag, 1.00f, 0.375f, ofs);
		ofs = simd_gte_cmov(mag, 2.00f, 0.775f, ofs);
		const vec8f product = _mm256_fmadd_ps(mag,  mult, ofs);
		return simd_gte_cmov(signal, _mm256_setzero_ps(), product, simd_negative(product));
	}
	vec8f simd_activation_derivative(vec8f signal) {
		const vec8f mag = simd_abs(signal);
		vec8f mult = _mm256_set1_ps(1.0f);
		mult = simd_gte_cmov(mag, 0.25f, 0.7f, mult);
		mult = simd_gte_cmov(mag, 0.50f, 0.5f, mult);
		mult = simd_gte_cmov(mag, 1.00f, 0.3f, mult);
		mult = simd_gte_cmov(mag, 2.00f, 0.1f, mult);
		return mult;
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
