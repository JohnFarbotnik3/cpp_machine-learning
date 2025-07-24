
#include "src/image/value_image.cpp"
#include "types.cpp"
#include <cassert>
#include "src/utils/random.cpp"
#include "src/image/extra_image_types.cpp"
#include "simd.cpp"
#include "simd_image.cpp"

namespace ML::models::autoencoder_subimage {
	using ML::image::value_image::value_image_dimensions;
	using ML::image::value_image::value_image;

	struct subimage {
		value_image<float> biases;
		vector<float> weights;
		layer_pattern pattern;
		image_bounds bounds_i;			// corrosponding soft boundary area of input image.
		image_bounds bounds_o;			// corrosponding soft boundary area of output image.

		subimage() = default;
		subimage(const layer_pattern pattern, const simd_image_8f_dimensions idim, const value_image_dimensions subodim, const image_bounds bounds_i, const image_bounds bounds_o) :
			biases(subodim),
			pattern(pattern),
			bounds_i(bounds_i),
			bounds_o(bounds_o)
		{
			const int n_weights = pattern.WEIGHTS_PER_OUTPUT_NEURON * biases.dim.length();
			weights.resize(n_weights, 0.0f);
		}

		void init_model_parameters(int seed, float bias_mean, float bias_stddev, float weight_mean, float weight_stddev) {
			std::mt19937 gen32 = utils::random::get_generator_32(seed);
			std::normal_distribution distr_bias = utils::random::rand_normal<float>(bias_mean, bias_stddev);
			std::normal_distribution distr_weight = utils::random::rand_normal<float>(weight_mean, weight_stddev);
			const float mult = sqrtf(1.0f / pattern.WEIGHTS_PER_OUTPUT_NEURON);
			for(int n=0;n<biases.data.size();n++) biases.data[n] = distr_bias(gen32);
			for(int x=0;x<weights.size();x++) weights[x] = distr_weight(gen32) * mult;
		}

		void foreward_propagate_dense(const simd_image_8f& value_i, simd_image_8f& value_o, simd_image_8f& signal_o) {
			const simd_image_8f_dimensions idim = value_i.dim;
			const simd_image_8f_dimensions odim = value_o.dim;
			const int WEIGHTS_PER_OUTPUT_NEURON = pattern.WEIGHTS_PER_OUTPUT_NEURON;
			for(int oy=bounds_o.y0;oy<bounds_o.y1;oy++) {
			for(int ox=bounds_o.x0;ox<bounds_o.x1;ox++) {
			for(int oc=0;oc<odim.C;oc++) {
				const int out_n = value_o.dim.get_offset(ox, oy, oc);
				const int bias_n = biases.dim.get_offset(ox - bounds_o.x0, oy - bounds_o.y0, oc);
				const int w_ofs = bias_n * WEIGHTS_PER_OUTPUT_NEURON;
				vec8f sum = simd_value(biases.data[bias_n]);

				int w = w_ofs;
				for(int y=bounds_i.y0;y<bounds_i.y1;y++) {
				for(int x=bounds_i.x0;x<bounds_i.x1;x++) {
				for(int c=0;c<idim.C;c++) {
					const int in_n = idim.get_offset(x, y, c);
					sum = _mm256_fmadd_ps(value_i.data[in_n], _mm256_set1_ps(weights[w]), sum);
					w++;
				}}}

				signal_o.data[out_n] = sum;
				value_o.data[out_n] = simd_activation_func(sum);
			}}}
		}
		void foreward_propagate_encode(const simd_image_8f& value_i, simd_image_8f& value_o, simd_image_8f& signal_o) {
			const simd_image_8f_dimensions idim = value_i.dim;
			const simd_image_8f_dimensions odim = value_o.dim;
			const int A = pattern.A;
			const int B = pattern.B;
			const int WEIGHTS_PER_OUTPUT_NEURON = pattern.WEIGHTS_PER_OUTPUT_NEURON;
			const int W_STRIDE_X = idim.C;
			const int W_STRIDE_Y = idim.C * A;
			for(int oy=bounds_o.y0;oy<bounds_o.y1;oy++) { const int iy0=(oy/B)*A;
			for(int ox=bounds_o.x0;ox<bounds_o.x1;ox++) { const int ix0=(ox/B)*A;
			for(int oc=0;oc<odim.C;oc++) {
				const int out_n = value_o.dim.get_offset(ox, oy, oc);
				const int bias_n = biases.dim.get_offset(ox - bounds_o.x0, oy - bounds_o.y0, oc);
				const int w_ofs = bias_n * WEIGHTS_PER_OUTPUT_NEURON;
				vec8f sum = simd_value(biases.data[bias_n]);

				for(int y=0;y<A;y++) { const int iy = y + iy0;
				for(int x=0;x<A;x++) { const int ix = x + ix0;
				for(int c=0;c<idim.C;c++) {
					const int in_n = idim.get_offset(ix, iy, c);
					const int w = w_ofs + y*W_STRIDE_Y + x*W_STRIDE_X + c;
					sum = _mm256_fmadd_ps(value_i.data[in_n], _mm256_set1_ps(weights[w]), sum);
				}}}

				signal_o.data[out_n] = sum;
				value_o.data[out_n] = simd_activation_func(sum);
			}}}
		}
		void foreward_propagate_encode_mix(const simd_image_8f& value_i, simd_image_8f& value_o, simd_image_8f& signal_o) {
			const simd_image_8f_dimensions idim = value_i.dim;
			const simd_image_8f_dimensions odim = value_o.dim;
			const int A = pattern.A;
			const int B = pattern.B;
			const int N = pattern.N;
			const int WEIGHTS_PER_OUTPUT_NEURON = pattern.WEIGHTS_PER_OUTPUT_NEURON;
			const int W_STRIDE_X = idim.C;
			const int W_STRIDE_Y = idim.C * N;
			for(int oy=bounds_o.y0;oy<bounds_o.y1;oy++) { const int iy0=(oy/B)*A + A/2 - N/2;
			for(int ox=bounds_o.x0;ox<bounds_o.x1;ox++) { const int ix0=(ox/B)*A + A/2 - N/2;
			for(int oc=0;oc<odim.C;oc++) {
				const int out_n = value_o.dim.get_offset(ox, oy, oc);
				const int bias_n = biases.dim.get_offset(ox - bounds_o.x0, oy - bounds_o.y0, oc);
				const int w_ofs = bias_n * WEIGHTS_PER_OUTPUT_NEURON;
				vec8f sum = simd_value(biases.data[bias_n]);

				for(int y=0;y<N;y++) { const int iy = y + iy0; if(iy<0 || iy>=idim.Y) continue;
				for(int x=0;x<N;x++) { const int ix = x + ix0; if(ix<0 || ix>=idim.X) continue;
				for(int c=0;c<idim.C;c++) {
					const int in_n = idim.get_offset(ix, iy, c);
					const int w = w_ofs + y*W_STRIDE_Y + x*W_STRIDE_X + c;
					sum = _mm256_fmadd_ps(value_i.data[in_n], _mm256_set1_ps(weights[w]), sum);
				}}}

				signal_o.data[out_n] = sum;
				value_o.data[out_n] = simd_activation_func(sum);
			}}}
		}
		void foreward_propagate_spatial_mix(const simd_image_8f& value_i, simd_image_8f& value_o, simd_image_8f& signal_o) {
			const simd_image_8f_dimensions idim = value_i.dim;
			const simd_image_8f_dimensions odim = value_o.dim;
			const int B = pattern.B;
			const int N = pattern.N;
			const int WEIGHTS_PER_OUTPUT_NEURON = pattern.WEIGHTS_PER_OUTPUT_NEURON;
			const int W_STRIDE_X = 1;
			const int W_STRIDE_Y = N;
			for(int oy=bounds_o.y0;oy<bounds_o.y1;oy++) { const int iy0=(oy/B)*B + B/2 - N/2;
			for(int ox=bounds_o.x0;ox<bounds_o.x1;ox++) { const int ix0=(ox/B)*B + B/2 - N/2;
			for(int oc=0;oc<odim.C;oc++) {
				const int out_n = value_o.dim.get_offset(ox, oy, oc);
				const int bias_n = biases.dim.get_offset(ox - bounds_o.x0, oy - bounds_o.y0, oc);
				const int w_ofs = bias_n * WEIGHTS_PER_OUTPUT_NEURON;
				vec8f sum = simd_value(biases.data[bias_n]);

				for(int y=0;y<N;y++) { const int iy = y + iy0; if(iy<0 || iy>=idim.Y) continue;
				for(int x=0;x<N;x++) { const int ix = x + ix0; if(ix<0 || ix>=idim.X) continue;
					const int in_n = idim.get_offset(ix, iy, oc);
					const int w = w_ofs + y*W_STRIDE_Y + x*W_STRIDE_X;
					sum = _mm256_fmadd_ps(value_i.data[in_n], _mm256_set1_ps(weights[w]), sum);
				}}

				signal_o.data[out_n] = sum;
				value_o.data[out_n] = simd_activation_func(sum);
			}}}
		}
		void foreward_propagate(const simd_image_8f& value_i, simd_image_8f& value_o, simd_image_8f& signal_o) {
			if(pattern.type == LAYER_TYPE::DENSE) foreward_propagate_dense(value_i, value_o, signal_o);
			if(pattern.type == LAYER_TYPE::ENCODE) foreward_propagate_encode(value_i, value_o, signal_o);
			if(pattern.type == LAYER_TYPE::ENCODE_MIX) foreward_propagate_encode_mix(value_i, value_o, signal_o);
			if(pattern.type == LAYER_TYPE::SPATIAL_MIX) foreward_propagate_spatial_mix(value_i, value_o, signal_o);
		}

		void backward_propagate_dense(simd_image_8f& error_i, const simd_image_8f& error_o, const simd_image_8f& value_i, const simd_image_8f& signal_o, const float adjustment_rate_w, const float adjustment_rate_b) {
			const simd_image_8f_dimensions idim = error_i.dim;
			const simd_image_8f_dimensions odim = error_o.dim;
			const int WEIGHTS_PER_OUTPUT_NEURON = pattern.WEIGHTS_PER_OUTPUT_NEURON;
			for(int oy=bounds_o.y0;oy<bounds_o.y1;oy++) {
			for(int ox=bounds_o.x0;ox<bounds_o.x1;ox++) {
			for(int oc=0;oc<odim.C;oc++) {
				const int bias_n = biases.dim.get_offset(ox - bounds_o.x0, oy - bounds_o.y0, oc);
				const int out_n = error_o.dim.get_offset(ox, oy, oc);
				const int w_ofs = bias_n * WEIGHTS_PER_OUTPUT_NEURON;
				const vec8f signal_error_term = _mm256_mul_ps(error_o.data[out_n], simd_activation_derivative(signal_o.data[out_n]));
				biases.data[bias_n] += std::clamp(simd_reduce(signal_error_term) * adjustment_rate_b, -0.5f, +0.5f);

				int w = w_ofs;
				for(int y=bounds_i.y0;y<bounds_i.y1;y++) {
				for(int x=bounds_i.x0;x<bounds_i.x1;x++) {
				for(int c=0;c<idim.C;c++) {
					const int in_n = idim.get_offset(x, y, c);
					const vec8f weight_error = _mm256_mul_ps(signal_error_term, value_i.data[in_n]);
					const vec8f input_error  = _mm256_mul_ps(signal_error_term, simd_value(weights[w]));
					weights[w] += std::clamp(simd_reduce(weight_error) * adjustment_rate_w, -1.0f, +1.0f);
					simd_incr(error_i.data[in_n], input_error);
					w++;
				}}}
			}}}
		}
		void backward_propagate_encode(simd_image_8f& error_i, const simd_image_8f& error_o, const simd_image_8f& value_i, const simd_image_8f& signal_o, const float adjustment_rate_w, const float adjustment_rate_b) {
			const simd_image_8f_dimensions idim = error_i.dim;
			const simd_image_8f_dimensions odim = error_o.dim;
			const int A = pattern.A;
			const int B = pattern.B;
			const int WEIGHTS_PER_OUTPUT_NEURON = pattern.WEIGHTS_PER_OUTPUT_NEURON;
			const int W_STRIDE_X = idim.C;
			const int W_STRIDE_Y = idim.C * A;
			for(int oy=bounds_o.y0;oy<bounds_o.y1;oy++) { const int iy0=(oy/B)*A;
			for(int ox=bounds_o.x0;ox<bounds_o.x1;ox++) { const int ix0=(ox/B)*A;
			for(int oc=0;oc<odim.C;oc++) {
				const int bias_n = biases.dim.get_offset(ox - bounds_o.x0, oy - bounds_o.y0, oc);
				const int out_n = error_o.dim.get_offset(ox, oy, oc);
				const int w_ofs = bias_n * WEIGHTS_PER_OUTPUT_NEURON;
				const vec8f signal_error_term = _mm256_mul_ps(error_o.data[out_n], simd_activation_derivative(signal_o.data[out_n]));
				biases.data[bias_n] += std::clamp(simd_reduce(signal_error_term) * adjustment_rate_b, -0.5f, +0.5f);

				for(int y=0;y<A;y++) { const int iy = y + iy0;
				for(int x=0;x<A;x++) { const int ix = x + ix0;
				for(int c=0;c<idim.C;c++) {
					const int in_n = idim.get_offset(ix, iy, c);
					const int w = w_ofs + y*W_STRIDE_Y + x*W_STRIDE_X + c;
					const vec8f weight_error = _mm256_mul_ps(signal_error_term, value_i.data[in_n]);
					const vec8f input_error  = _mm256_mul_ps(signal_error_term, simd_value(weights[w]));
					weights[w] += std::clamp(simd_reduce(weight_error) * adjustment_rate_w, -1.0f, +1.0f);
					simd_incr(error_i.data[in_n], input_error);
				}}}
			}}}
		}
		void backward_propagate_encode_mix(simd_image_8f& error_i, const simd_image_8f& error_o, const simd_image_8f& value_i, const simd_image_8f& signal_o, const float adjustment_rate_w, const float adjustment_rate_b) {
			const simd_image_8f_dimensions idim = error_i.dim;
			const simd_image_8f_dimensions odim = error_o.dim;
			const int A = pattern.A;
			const int B = pattern.B;
			const int N = pattern.N;
			const int WEIGHTS_PER_OUTPUT_NEURON = pattern.WEIGHTS_PER_OUTPUT_NEURON;
			const int W_STRIDE_X = idim.C;
			const int W_STRIDE_Y = idim.C * N;
			for(int oy=bounds_o.y0;oy<bounds_o.y1;oy++) { const int iy0=(oy/B)*A + A/2 - N/2;
			for(int ox=bounds_o.x0;ox<bounds_o.x1;ox++) { const int ix0=(ox/B)*A + A/2 - N/2;
			for(int oc=0;oc<odim.C;oc++) {
				const int bias_n = biases.dim.get_offset(ox - bounds_o.x0, oy - bounds_o.y0, oc);
				const int out_n = error_o.dim.get_offset(ox, oy, oc);
				const int w_ofs = bias_n * WEIGHTS_PER_OUTPUT_NEURON;
				const vec8f signal_error_term = _mm256_mul_ps(error_o.data[out_n], simd_activation_derivative(signal_o.data[out_n]));
				biases.data[bias_n] += std::clamp(simd_reduce(signal_error_term) * adjustment_rate_b, -0.5f, +0.5f);

				for(int y=0;y<N;y++) { const int iy = y + iy0; if(iy<0 || iy>=idim.Y) continue;
				for(int x=0;x<N;x++) { const int ix = x + ix0; if(ix<0 || ix>=idim.X) continue;
				const bool within_write_bounds =
					(bounds_i.x0 <= ix) & (ix < bounds_i.x1) &
					(bounds_i.y0 <= iy) & (iy < bounds_i.y1);
				if(within_write_bounds) {
				for(int c=0;c<idim.C;c++) {
					const int in_n = idim.get_offset(ix, iy, c);
					const int w = w_ofs + y*W_STRIDE_Y + x*W_STRIDE_X + c;
					const vec8f weight_error = _mm256_mul_ps(signal_error_term, value_i.data[in_n]);
					const vec8f input_error  = _mm256_mul_ps(signal_error_term, simd_value(weights[w]));
					weights[w] += std::clamp(simd_reduce(weight_error) * adjustment_rate_w, -1.0f, +1.0f);
					simd_incr(error_i.data[in_n], input_error);
				}}
				else {
				for(int c=0;c<idim.C;c++) {
					const int in_n = idim.get_offset(ix, iy, c);
					const int w = w_ofs + y*W_STRIDE_Y + x*W_STRIDE_X + c;
					const vec8f weight_error = _mm256_mul_ps(signal_error_term, value_i.data[in_n]);
					weights[w] += std::clamp(simd_reduce(weight_error) * adjustment_rate_w, -1.0f, +1.0f);
				}}
				}}
			}}}
		}
		void backward_propagate_spatial_mix(simd_image_8f& error_i, const simd_image_8f& error_o, const simd_image_8f& value_i, const simd_image_8f& signal_o, const float adjustment_rate_w, const float adjustment_rate_b) {
			const simd_image_8f_dimensions idim = error_i.dim;
			const simd_image_8f_dimensions odim = error_o.dim;
			const int A = pattern.A;
			const int B = pattern.B;
			const int N = pattern.N;
			const int WEIGHTS_PER_OUTPUT_NEURON = pattern.WEIGHTS_PER_OUTPUT_NEURON;
			const int W_STRIDE_X = 1;
			const int W_STRIDE_Y = N;
			for(int oy=bounds_o.y0;oy<bounds_o.y1;oy++) { const int iy0=(oy/B)*B + B/2 - N/2;
			for(int ox=bounds_o.x0;ox<bounds_o.x1;ox++) { const int ix0=(ox/B)*B + B/2 - N/2;
			for(int oc=0;oc<odim.C;oc++) {
				const int bias_n = biases.dim.get_offset(ox - bounds_o.x0, oy - bounds_o.y0, oc);
				const int out_n = error_o.dim.get_offset(ox, oy, oc);
				const int w_ofs = bias_n * WEIGHTS_PER_OUTPUT_NEURON;
				const vec8f signal_error_term = _mm256_mul_ps(error_o.data[out_n], simd_activation_derivative(signal_o.data[out_n]));
				biases.data[bias_n] += std::clamp(simd_reduce(signal_error_term) * adjustment_rate_b, -0.5f, +0.5f);

				for(int y=0;y<N;y++) { const int iy = y + iy0; if(iy<0 || iy>=idim.Y) continue;
				for(int x=0;x<N;x++) { const int ix = x + ix0; if(ix<0 || ix>=idim.X) continue;
				const bool within_write_bounds =
					(bounds_i.x0 <= ix) & (ix < bounds_i.x1) &
					(bounds_i.y0 <= iy) & (iy < bounds_i.y1);
				if(within_write_bounds) {
					const int in_n = idim.get_offset(ix, iy, oc);
					const int w = w_ofs + y*W_STRIDE_Y + x*W_STRIDE_X;
					const vec8f weight_error = _mm256_mul_ps(signal_error_term, value_i.data[in_n]);
					const vec8f input_error  = _mm256_mul_ps(signal_error_term, simd_value(weights[w]));
					weights[w] += std::clamp(simd_reduce(weight_error) * adjustment_rate_w, -1.0f, +1.0f);
					simd_incr(error_i.data[in_n], input_error);
				}
				else {
					const int in_n = idim.get_offset(ix, iy, oc);
					const int w = w_ofs + y*W_STRIDE_Y + x*W_STRIDE_X;
					const vec8f weight_error = _mm256_mul_ps(signal_error_term, value_i.data[in_n]);
					weights[w] += std::clamp(simd_reduce(weight_error) * adjustment_rate_w, -1.0f, +1.0f);
				}
				}}
			}}}
		}
		void backward_propagate(simd_image_8f& error_i, const simd_image_8f& error_o, const simd_image_8f& value_i, const simd_image_8f& signal_o, const float adjustment_rate_w, const float adjustment_rate_b) {
			// clear related area of input image.
			const vec8f zero = simd_value(0);
			for(int iy=bounds_i.y0;iy<bounds_i.y1;iy++) {
				const int i0 = error_i.dim.get_offset(bounds_i.x0, iy, 0);
				const int i1 = error_i.dim.get_offset(bounds_i.x1, iy, 0);
				for(int iofs=i0;iofs<i1;iofs++) error_i.data[iofs] = zero;
			}

			if(pattern.type == LAYER_TYPE::DENSE) backward_propagate_dense(error_i, error_o, value_i, signal_o, adjustment_rate_w, adjustment_rate_b);
			if(pattern.type == LAYER_TYPE::ENCODE) backward_propagate_encode(error_i, error_o, value_i, signal_o, adjustment_rate_w, adjustment_rate_b);
			if(pattern.type == LAYER_TYPE::ENCODE_MIX) backward_propagate_encode_mix(error_i, error_o, value_i, signal_o, adjustment_rate_w, adjustment_rate_b);
			if(pattern.type == LAYER_TYPE::SPATIAL_MIX) backward_propagate_spatial_mix(error_i, error_o, value_i, signal_o, adjustment_rate_w, adjustment_rate_b);
		}
	};
}






