
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

	struct read_coords { int x,y,c,iofs; };
	struct read_pattern {
		vector<read_coords> list;
		read_pattern() = default;
		read_pattern(const layer_pattern pattern, const simd_image_8f_dimensions idim) {
			assert(pattern.type != LAYER_TYPE::NONE);
			const int A = pattern.A;
			const int B = pattern.B;
			const int N = pattern.N;
			if(pattern.type == LAYER_TYPE::DENSE) {
				for(int iy=0;iy<idim.Y;iy++) {
				for(int ix=0;ix<idim.X;ix++) {
				for(int ic=0;ic<idim.C;ic++) {
					list.push_back({ ix, iy, ic, idim.get_offset(ix, iy, ic) });
				}}}
			}
			if(pattern.type == LAYER_TYPE::ENCODE) {
				for(int iy=0;iy<A;iy++) {
				for(int ix=0;ix<A;ix++) {
				for(int ic=0;ic<idim.C;ic++) {
					list.push_back({ ix, iy, ic, idim.get_offset(ix, iy, ic) });
				}}}
			}
			if(pattern.type == LAYER_TYPE::ENCODE_MIX) {
				assert(N % A == 0);
				const int p0 = (A/2) - (N/2);
				const int p1 = p0 + N;
				for(int iy=p0;iy<p1;iy++) {
				for(int ix=p0;ix<p1;ix++) {
				for(int ic=0;ic<idim.C;ic++) {
					list.push_back({ ix, iy, ic, idim.get_offset(ix, iy, ic) });
				}}}
			}
			if(pattern.type == LAYER_TYPE::SPATIAL_MIX) {
				assert(N % B == 0);
				const int p0 = (B/2) - (N/2);
				const int p1 = p0 + N;
				for(int iy=p0;iy<p1;iy++) {
				for(int ix=p0;ix<p1;ix++) {
					const int ic = 0;
					list.push_back({ ix, iy, ic, idim.get_offset(ix, iy, ic) });
				}}
			}
		}
		read_coords get_offset(const layer_pattern pattern, const simd_image_8f_dimensions idim, const int ox, const int oy, const int oc) {
			const int A = pattern.A;
			const int B = pattern.B;
			const int N = pattern.N;
			if(pattern.type == LAYER_TYPE::ENCODE || pattern.type == LAYER_TYPE::ENCODE_MIX) {
				const int tx = (ox / B) * A;
				const int ty = (oy / B) * A;
				const int iofs = idim.get_offset(tx, ty, 0);
				return read_coords{ tx, ty, 0, iofs };
			}
			if(pattern.type == LAYER_TYPE::SPATIAL_MIX) {
				const int tx = (ox / B) * B;
				const int ty = (oy / B) * B;
				const int iofs = idim.get_offset(tx, ty, oc);
				return read_coords{ tx, ty, oc, iofs };
			}
			return read_coords{ 0, 0, 0, 0 };
		}
		bool is_in_read_bounds(const read_coords pos, const read_coords ofs, const simd_image_8f_dimensions idim) {
			const int ix = pos.x + ofs.x;
			const int iy = pos.y + ofs.y;
			return (
				(ix >= 0) & (ix < idim.X) &
				(iy >= 0) & (iy < idim.Y)
			);
		}
		bool is_in_write_bounds(const read_coords pos, const read_coords ofs, image_bounds bounds_i) {
			const int ix = pos.x + ofs.x;
			const int iy = pos.y + ofs.y;
			return (
				(ix >= bounds_i.x0) & (ix < bounds_i.x1) &
				(iy >= bounds_i.y0) & (iy < bounds_i.y1)
			);
		}
	};

	struct subimage {
		value_image<float> biases;
		vector<float> weights;
		layer_pattern pattern;
		read_pattern kernel;
		image_bounds bounds_i;			// corrosponding soft boundary area of input image.
		image_bounds bounds_o;			// corrosponding soft boundary area of output image.

		subimage() = default;
		subimage(const layer_pattern pattern, const simd_image_8f_dimensions idim, const value_image_dimensions subodim, const image_bounds bounds_i, const image_bounds bounds_o) :
			biases(subodim),
			pattern(pattern),
			kernel(pattern, idim),
			bounds_i(bounds_i),
			bounds_o(bounds_o)
		{
			const int n_weights_per_output_neuron = kernel.list.size();
			const int n_weights = n_weights_per_output_neuron * biases.dim.length();
			weights.resize(n_weights, 0.0f);
			//printf("nw=%i, wpn=%i\n", n_weights, n_weights_per_output_neuron);
		}

		void init_model_parameters(int seed, float bias_mean, float bias_stddev, float weight_mean, float weight_stddev) {
			std::mt19937 gen32 = utils::random::get_generator_32(seed);
			std::normal_distribution distr_bias = utils::random::rand_normal<float>(bias_mean, bias_stddev);
			std::normal_distribution distr_weight = utils::random::rand_normal<float>(weight_mean, weight_stddev);
			const int n_weights_per_output_neuron = kernel.list.size();
			const float mult = sqrtf(1.0f / n_weights_per_output_neuron);
			for(int n=0;n<biases.data.size();n++) biases.data[n] = distr_bias(gen32);
			for(int x=0;x<weights.size();x++) weights[x] = distr_weight(gen32) * mult;
		}

		void foreward_propagate_general(const simd_image_8f& value_i, simd_image_8f& value_o, simd_image_8f& signal_o) {
			const simd_image_8f_dimensions idim = value_i.dim;
			const simd_image_8f_dimensions odim = value_o.dim;
			const int WEIGHTS_PER_OUTPUT_NEURON = kernel.list.size();
			for(int oy=bounds_o.y0;oy<bounds_o.y1;oy++) {
			for(int ox=bounds_o.x0;ox<bounds_o.x1;ox++) {
			for(int oc=0;oc<odim.C;oc++) {
				const int out_n = value_o.dim.get_offset(ox, oy, oc);
				const int bias_n = biases.dim.get_offset(ox - bounds_o.x0, oy - bounds_o.y0, oc);
				const int w_ofs = bias_n * WEIGHTS_PER_OUTPUT_NEURON;
				const float bias = biases.data[bias_n];
				vec8f sum = simd_value(bias);
				const read_coords coords_offset = kernel.get_offset(pattern, idim, ox, oy, oc);
				for(int w=0;w<WEIGHTS_PER_OUTPUT_NEURON;w++) {
					const read_coords coords = kernel.list[w];
					if(!kernel.is_in_read_bounds(coords, coords_offset, idim)) continue;
					const float weight = weights[w_ofs + w];
					const int in_n = coords.iofs + coords_offset.iofs;
					sum = _mm256_fmadd_ps(value_i.data[in_n], _mm256_set1_ps(weight), sum);
				}
				signal_o.data[out_n] = sum;
				value_o.data[out_n] = simd_activation_func(sum);
			}}}
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
			//foreward_propagate_general(value_i, value_o, signal_o);
			///*
			if(pattern.type == LAYER_TYPE::DENSE) foreward_propagate_dense(value_i, value_o, signal_o);
			if(pattern.type == LAYER_TYPE::ENCODE) foreward_propagate_encode(value_i, value_o, signal_o);
			if(pattern.type == LAYER_TYPE::ENCODE_MIX) foreward_propagate_encode_mix(value_i, value_o, signal_o);
			if(pattern.type == LAYER_TYPE::SPATIAL_MIX) foreward_propagate_spatial_mix(value_i, value_o, signal_o);
			//*/
		}

		void backward_propagate_general(simd_image_8f& error_i, const simd_image_8f& error_o, const simd_image_8f& value_i, const simd_image_8f& signal_o, const float adjustment_rate_w, const float adjustment_rate_b) {
			const simd_image_8f_dimensions idim = error_i.dim;
			const simd_image_8f_dimensions odim = error_o.dim;
			const int WEIGHTS_PER_OUTPUT_NEURON = kernel.list.size();
			for(int oy=bounds_o.y0;oy<bounds_o.y1;oy++) {
			for(int ox=bounds_o.x0;ox<bounds_o.x1;ox++) {
			for(int oc=0;oc<odim.C;oc++) {
				const int bias_n = biases.dim.get_offset(ox - bounds_o.x0, oy - bounds_o.y0, oc);
				const int out_n = error_o.dim.get_offset(ox, oy, oc);
				const int wofs = bias_n * WEIGHTS_PER_OUTPUT_NEURON;
				const vec8f signal_error_term = _mm256_mul_ps(error_o.data[out_n], simd_activation_derivative(signal_o.data[out_n]));
				biases.data[bias_n] += std::clamp(simd_reduce(signal_error_term) * adjustment_rate_b, -0.5f, +0.5f);

				const read_coords coords_offset = kernel.get_offset(pattern, idim, ox, oy, oc);
				for(int w=0;w<WEIGHTS_PER_OUTPUT_NEURON;w++) {
					const read_coords coords = kernel.list[w];
					if(!kernel.is_in_read_bounds(coords, coords_offset, idim)) continue;
					const int in_n = coords.iofs + coords_offset.iofs;
					vec8f weight_error = _mm256_mul_ps(signal_error_term, value_i.data[in_n]);
					vec8f input_error  = _mm256_mul_ps(signal_error_term, simd_value(weights[wofs + w]));
					weights[w] += std::clamp(simd_reduce(weight_error) * adjustment_rate_w, -1.0f, +1.0f);
					if(kernel.is_in_write_bounds(coords, coords_offset, bounds_i)) simd_incr(error_i.data[in_n], input_error);
				}
			}}}
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

			//backward_propagate_general(error_i, error_o, value_i, signal_o, adjustment_rate_w);
			///*
			if(pattern.type == LAYER_TYPE::DENSE) backward_propagate_dense(error_i, error_o, value_i, signal_o, adjustment_rate_w, adjustment_rate_b);
			if(pattern.type == LAYER_TYPE::ENCODE) backward_propagate_encode(error_i, error_o, value_i, signal_o, adjustment_rate_w, adjustment_rate_b);
			if(pattern.type == LAYER_TYPE::ENCODE_MIX) backward_propagate_encode_mix(error_i, error_o, value_i, signal_o, adjustment_rate_w, adjustment_rate_b);
			if(pattern.type == LAYER_TYPE::SPATIAL_MIX) backward_propagate_spatial_mix(error_i, error_o, value_i, signal_o, adjustment_rate_w, adjustment_rate_b);
			//*/
		}
	};
}






