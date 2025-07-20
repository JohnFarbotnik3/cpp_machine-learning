
#include "types.cpp"
#include "patterns.cpp"
#include <cassert>
#include <map>
#include "src/utils/random.cpp"
#include "src/image/extra_image_types.cpp"
#include "simd.cpp"

namespace ML::models::autoencoder_subimage {
	using ML::image::value_image::value_image_dimensions;
	using ML::image::value_image::value_image;
	using ML::image::extra_image_types::interleaved_image_dimensions;
	using ML::image::extra_image_types::interleaved_image;

	/*
		a map for storing extra error values that cannot be safely written to input-error image
		due to intersecting write-area of another thread.
	*/
	struct input_error_map {
		std::map<int, int> pixel_offsets;// Map<image_ofs, data_ofs>.
		vector<float> data;

		void clear() {
			data.assign(data.size(), 0.0f);
		}

		int get_offset(const interleaved_image_dimensions idim, const int x, const int y) {
			const int pixel_offset = idim.get_offset(x,y,0);
			if(!pixel_offsets.contains(pixel_offset)) {
				pixel_offsets[pixel_offset] = data.size();
				data.resize(data.size() + idim.pixel_length(), 0.0f);
			}
			return pixel_offsets[pixel_offset];
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
		read_pattern(const layer_pattern pattern, const interleaved_image_dimensions idim) {
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
				const int p0 = (B/2) - (N/2);
				const int p1 = p0 + N;
				for(int iy=p0;iy<p1;iy++) {
				for(int ix=p0;ix<p1;ix++) {
					const int ic = 0;
					list.push_back({ ix, iy, ic, idim.get_offset(ix, iy, ic) });
				}}
			}
		}
		read_coords get_offset(const layer_pattern pattern, const interleaved_image_dimensions idim, const int ox, const int oy, const int oc) {
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
				const int tx = (ox / B) * A;
				const int ty = (oy / B) * A;
				const int iofs = idim.get_offset(tx, ty, oc);
				return read_coords{ tx, ty, oc, iofs };
			}
			return read_coords{ 0, 0, 0, 0 };
		}
		bool is_in_read_bounds(const read_coords pos, const read_coords ofs, const interleaved_image_dimensions idim) {
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
		read_pattern kernel;
		image_bounds bounds_i;			// corrosponding soft boundary area of input image.
		image_bounds bounds_o;			// corrosponding soft boundary area of output image.
		input_error_map error_map;		// extra input-error that couldnt be safely written to input-error-image.

		subimage() = default;
		subimage(const int X, const int Y, const int C, const int D, const layer_pattern pattern, const interleaved_image_dimensions idim, const image_bounds bounds_i, const image_bounds bounds_o) :
			biases(X,Y,C),
			biases_error(X,Y,C),
			kernel(pattern, idim),
			bounds_i(bounds_i),
			bounds_o(bounds_o)
		{
			const int n_weights_per_output_neuron = kernel.list.size();
			const int n_weights = n_weights_per_output_neuron * biases.dim.length();
			weights.resize(n_weights, 0.0f);
			weights_error.resize(n_weights, 0.0f);
		}

		void init_model_parameters(int seed, float bias_mean, float bias_stddev, float weight_mean, float weight_stddev) {
			std::mt19937 gen32 = utils::random::get_generator_32(seed);
			std::normal_distribution distr_bias = utils::random::rand_normal<float>(bias_mean, bias_stddev);
			std::normal_distribution distr_weight = utils::random::rand_normal<float>(weight_mean, weight_stddev);
			const float mult = sqrtf(1.0f / n_weights_per_output_neuron);
			for(int n=0;n<biases.data.size();n++) biases.data[n] = distr_bias(gen32);
			for(int x=0;x<weights.size();x++) weights[x] = distr_weight(gen32) * mult;
		}

		// TODO - if this doesnt work, write a non-SIMD reference implementation and test that first.
		void foreward_propagate(const interleaved_image<float>& value_i, interleaved_image<float>& value_o, interleaved_image<float>& signal, const layer_pattern pattern) {
			const interleaved_image_dimensions idim = value_i.dim;
			const interleaved_image_dimensions odim = value_o.dim;
			const int C = odim.C;
			const int D = odim.D;
			assert(D == vec8f_LENGTH);
			const int WEIGHTS_PER_OUTPUT_NEURON = kernel.list.size();
			for(int oy=bounds_o.y0;oy<bounds_o.y1;oy++) {
			for(int ox=bounds_o.x0;ox<bounds_o.x1;ox++) {
			for(int oc=0;oc<C;oc++) {
				const int out_n = biases.dim.get_offset(ox, oy, oc);
				const int wofs = out_n * WEIGHTS_PER_OUTPUT_NEURON;
				const float bias = biases.data[out_n];
				vec8f sum = _mm256_set1_ps(bias);
				const read_coords coords_offset = kernel.get_offset(pattern, idim, ox, oy, oc);
				for(int w=0;w<WEIGHTS_PER_OUTPUT_NEURON;w++) {
					const read_coords coords = kernel.list[w];
					if(!kernel.is_in_read_bounds(coords, coords_offset, idim)) continue;
					const float weight = weights[wofs + w];
					const int iofs = coords.iofs + coords_offset.iofs;
					sum = _mm256_fmadd_ps(_mm256_loadu_ps(value_i.data.data()+iofs), _mm256_set1_ps(weight), sum);
				}
				const int oofs = value_o.dim.get_offset(ox, oy, oc);
				_mm256_storeu_ps(signal .data.data() + oofs, sum);
				_mm256_storeu_ps(value_o.data.data() + oofs, simd_activation_func(sum));
			}}}
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

		void commit_extra_error() {}// TODO

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






