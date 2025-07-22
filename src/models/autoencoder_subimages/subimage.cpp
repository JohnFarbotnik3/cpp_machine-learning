
#include "src/image/value_image.cpp"
#include "types.cpp"
#include <cassert>
#include <map>
#include "src/utils/random.cpp"
#include "src/image/extra_image_types.cpp"
#include "simd.cpp"
#include "simd_image.cpp"

namespace ML::models::autoencoder_subimage {
	using ML::image::value_image::value_image_dimensions;
	using ML::image::value_image::value_image;

	/*
		a map for storing extra error values that cannot be safely written to input-error image
		due to intersecting write-area of another thread.
	*/
	struct input_error_map {
		std::map<int, int> pixel_offsets;// Map<image_ofs, data_ofs>.
		vector<vec8f> data;

		void clear() {
			data.assign(data.size(), simd_value(0.0f));
		}
		void apply(simd_image_8f& error_i) {
			for(const auto& [iofs, dofs] : pixel_offsets) {
				simd_incr(error_i.data[iofs], data[dofs]);
			}
		}

		int get_offset(const int iofs) {
			if(!pixel_offsets.contains(iofs)) {
				pixel_offsets[iofs] = data.size();
				data.push_back(simd_value(0.0f));
			}
			return pixel_offsets[iofs];
		}
	};

	struct image_bounds {
		int x0,x1;
		int y0,y1;
	};

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
		value_image<float> biases_error;// accumulated error in biases during minibatch.
		vector<float> weights;
		vector<float> weights_error;
		layer_pattern pattern;
		read_pattern kernel;
		image_bounds bounds_i;			// corrosponding soft boundary area of input image.
		image_bounds bounds_o;			// corrosponding soft boundary area of output image.
		input_error_map error_map;		// extra input-error that couldnt be safely written to input-error-image.

		subimage() = default;
		subimage(const layer_pattern pattern, const simd_image_8f_dimensions idim, const value_image_dimensions subodim, const image_bounds bounds_i, const image_bounds bounds_o) :
			biases(subodim),
			biases_error(subodim),
			pattern(pattern),
			kernel(pattern, idim),
			bounds_i(bounds_i),
			bounds_o(bounds_o)
		{
			const int n_weights_per_output_neuron = kernel.list.size();
			const int n_weights = n_weights_per_output_neuron * biases.dim.length();
			weights.resize(n_weights, 0.0f);
			weights_error.resize(n_weights, 0.0f);
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

		void foreward_propagate(
			const simd_image_8f& value_i,
			simd_image_8f& value_o,
			simd_image_8f& signal_o
		) {
			const simd_image_8f_dimensions idim = value_i.dim;
			const simd_image_8f_dimensions odim = value_o.dim;
			const int WEIGHTS_PER_OUTPUT_NEURON = kernel.list.size();
			for(int oy=bounds_o.y0;oy<bounds_o.y1;oy++) {
			for(int ox=bounds_o.x0;ox<bounds_o.x1;ox++) {
			for(int oc=0;oc<odim.C;oc++) {
				const int bias_n = biases.dim.get_offset(ox - bounds_o.x0, oy - bounds_o.y0, oc);
				const int wofs = bias_n * WEIGHTS_PER_OUTPUT_NEURON;
				const float bias = biases.data[bias_n];
				vec8f sum = simd_value(bias);
				const read_coords coords_offset = kernel.get_offset(pattern, idim, ox, oy, oc);
				for(int w=0;w<WEIGHTS_PER_OUTPUT_NEURON;w++) {
					const read_coords coords = kernel.list[w];
					if(!kernel.is_in_read_bounds(coords, coords_offset, idim)) continue;
					const float weight = weights[wofs + w];
					const int in_n = coords.iofs + coords_offset.iofs;
					sum = _mm256_fmadd_ps(value_i.data[in_n], _mm256_set1_ps(weight), sum);
				}
				const int out_n = value_o.dim.get_offset(ox, oy, oc);
				signal_o.data[out_n] = sum;
				value_o.data[out_n] = simd_activation_func(sum);
			}}}
		}

		void backward_propagate(
			simd_image_8f& error_i,
			const simd_image_8f& error_o,
			const simd_image_8f& value_i,
			const simd_image_8f& signal_o
		) {
			const simd_image_8f_dimensions idim = error_i.dim;
			const simd_image_8f_dimensions odim = error_o.dim;
			const int WEIGHTS_PER_OUTPUT_NEURON = kernel.list.size();

			// clear related area of input image.
			const vec8f zero = simd_value(0);
			for(int iy=bounds_i.y0;iy<bounds_i.y1;iy++) {
			for(int ix=bounds_i.x0;ix<bounds_i.x1;ix++) {
			for(int ic=0;ic<idim.C;ic++) {
				error_i.data[error_i.dim.get_offset(ix, iy, ic)] = zero;
			}}}
			error_map.clear();

			// backprop.
			//const float mult = sqrtf(1.0f / WEIGHTS_PER_OUTPUT_NEURON);
			const vec8f mult = simd_value(1.0f / WEIGHTS_PER_OUTPUT_NEURON);
			for(int oy=bounds_o.y0;oy<bounds_o.y1;oy++) {
			for(int ox=bounds_o.x0;ox<bounds_o.x1;ox++) {
			for(int oc=0;oc<odim.C;oc++) {
				const int bias_n = biases.dim.get_offset(ox - bounds_o.x0, oy - bounds_o.y0, oc);
				//if(simd_eq(error_o.data[out_n], simd_value(0.0f))) continue;// OPTIMIZATION: skip if no error.
				const int wofs = bias_n * WEIGHTS_PER_OUTPUT_NEURON;
				const int out_n = error_o.dim.get_offset(ox, oy, oc);
				const vec8f signal_error_term_i = _mm256_mul_ps(error_o.data[out_n], simd_activation_derivative(signal_o.data[out_n]));
				const vec8f signal_error_term_w = _mm256_mul_ps(signal_error_term_i, mult);
				const read_coords coords_offset = kernel.get_offset(pattern, idim, ox, oy, oc);
				for(int w=0;w<WEIGHTS_PER_OUTPUT_NEURON;w++) {
					const read_coords coords = kernel.list[w];
					if(!kernel.is_in_read_bounds(coords, coords_offset, idim)) continue;
					const float weight = weights[wofs + w];
					const int in_n = coords.iofs + coords_offset.iofs;
					vec8f weight_error = _mm256_mul_ps(signal_error_term_w, value_i.data[in_n]);
					vec8f input_error  = _mm256_mul_ps(signal_error_term_i, simd_value(weight));
					weights_error[wofs + w] += simd_reduce(weight_error);
					if(kernel.is_in_write_bounds(coords, coords_offset, bounds_i)) {
						simd_incr(error_i.data[in_n], input_error);
					} else {
						simd_incr(error_map.data[error_map.get_offset(in_n)], input_error);
					}
				}
			}}}
		}

		void commit_extra_error(simd_image_8f& error_i) {
			error_map.apply(error_i);
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






