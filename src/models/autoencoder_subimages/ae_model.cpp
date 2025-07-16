
#include <cassert>
#include <vector>
#include <algorithm>
#include "src/image/value_image_lines.cpp"
#include "src/utils/random.cpp"
#include "src/utils/simd.cpp"
#include "src/utils/vector_util.cpp"
#include "./ae_layer.cpp"

namespace ML::models::autoencoder_subimage {
	using std::vector;
	using namespace ML::image;
	using namespace utils::vector_util;
	using dimensions_t = simple_dim_t;

	struct ae_model {
		dimensions_t image_dimensions;
		vector<ae_layer> layers;

		void push_layer_dense		(const dimensions_t idim, const dimensions_t odim) {
			assert(idim.length() > 0);
			assert(odim.length() > 0);
			layers.push_back(ae_layer(idim, odim, layer_pattern::dense(), 1, 1));
		}
		void push_layer_spatial_mix	(const dimensions_t idim, const dimensions_t odim, const int GX, const int GY, const int N, const int B) {
			assert(idim.length() > 0);
			assert(odim.length() > 0);
			assert(odim.equals(idim));
			layers.push_back(ae_layer(idim, odim, layer_pattern::spatial_mix(N, B), GX, GY));
		}
		void push_layer_encode		(const dimensions_t idim, dimensions_t odim, const int GX, const int GY, const int A, const int B, const int out_ch) {
			assert(idim.X % A == 0);
			assert(idim.Y % A == 0);
			assert(odim.X % B == 0);
			assert(odim.Y % B == 0);
			assert((idim.X / A) == (odim.X / B));
			assert((idim.Y / A) == (odim.Y / B));
			odim.X = (idim.X * B) / A;
			odim.Y = (idim.Y * B) / A;
			odim.C = out_ch;
			assert(idim.length() > 0);
			assert(odim.length() > 0);
			layers.push_back(ae_layer(idim, odim, layer_pattern::encode(A, B), GX, GY));
		}
		void push_layer_encode_mix	(const dimensions_t idim, dimensions_t odim, const int GX, const int GY, const int A, const int B, const int N, const int out_ch) {
			assert(idim.X % A == 0);
			assert(idim.Y % A == 0);
			assert(odim.X % B == 0);
			assert(odim.Y % B == 0);
			assert((idim.X / A) == (odim.X / B));
			assert((idim.Y / A) == (odim.Y / B));
			odim.X = (idim.X * B) / A;
			odim.Y = (idim.Y * B) / A;
			odim.C = out_ch;
			assert(idim.length() > 0);
			assert(odim.length() > 0);
			layers.push_back(ae_layer(idim, odim, layer_pattern::encode_mix(A, B, N), GX, GY));
		}

		ae_model(dimensions_t input_dimensions) {
			this->image_dimensions = input_dimensions;

			// init_model_topology.
			dimensions_t idim = input_dimensions;
			dimensions_t odim = idim;
			const int ch = idim.C;

			// encoder: mix and condense image.
			push_layer_encode_mix(idim, odim, 8,8,	8,2, 8,	 8); idim = odim;
			push_layer_encode_mix(idim, odim, 4,4,	8,2,16,	24); idim = odim;
			push_layer_encode_mix(idim, odim, 2,2,	8,2,16,	72); idim = odim;

			// decoder: expand image back to original size.
			push_layer_encode_mix(idim, odim, 2,2,	2,8,4,	24); idim = odim;
			push_layer_encode_mix(idim, odim, 4,4,	2,8,4,	 8); idim = odim;
			push_layer_encode_mix(idim, odim, 8,8,	2,8,4,	ch); idim = odim;

			assert(odim.X == input_dimensions.X);
			assert(odim.Y == input_dimensions.Y);
			assert(odim.C == input_dimensions.C);
		}

		void init_model_parameters(int seed, float bias_mean, float bias_stddev, float weight_mean, float weight_stddev) {
			std::mt19937 gen32 = utils::random::get_generator_32(seed);
			std::uniform_int_distribution<int> distr = utils::random::rand_uniform_int<int>(INT_MIN, INT_MAX);
			for(int z=0;z<layers.size();z++) {
				const int new_seed = distr(gen32);
				layers[z].init_model_parameters(new_seed, bias_mean, bias_stddev, weight_mean, weight_stddev);
			}
		}

		void propagate(const int n_threads, const simple_image_f& input_values, simple_image_f& output_values) {
			assert(output_values.dim.equals(input_values.dim));
			assert(output_values.dim.length() > 0);
			assert(layers.size() > 0);
			const int n_layers = layers.size();
			simple_image_f value_buffer_i;
			simple_image_f value_buffer_o;

			// first layer.
			value_buffer_o = simple_image_f(layers.front().odim);
			layers.front().propagate(n_threads, input_values, value_buffer_o);
			std::swap(value_buffer_i, value_buffer_o);

			// middle layers.
			for(int L=1;L<n_layers-1;L++) {
				value_buffer_o = simple_image_f(layers[L].odim);
				layers[L].propagate(n_threads, value_buffer_i, value_buffer_o);
				std::swap(value_buffer_i, value_buffer_o);
			}

			// last layer.
			layers.back().propagate(n_threads, value_buffer_i, output_values);
		}

		/*
			WARNING: loss_squared can lead to error-concentration which causes models to explode
			when training is going well and they are very close to 0 average error.
		*/
		void generate_error_image(const simple_image_f& input, const simple_image_f& output, simple_image_f& error, bool loss_squared, bool clamp_error) {
			assert(output.dim.equals(input.dim));
			assert(output.dim.equals(error.dim));
			assert(output.dim.length() > 0);
			assert(input.x1 > input.x0);
			assert(input.y1 > input.y0);

			error.clear();

			// sample area bounds.
			const int ix0 = input.x0;
			const int iy0 = input.y0;
			const int ix1 = input.x1;
			const int iy1 = input.y1;
			const int ic0 = 0;
			const int ic1 = input.dim.C;

			if(loss_squared) {
				float sum_e1 = 0;
				float sum_e2 = 0;
				for(int iy=iy0;iy<iy1;iy++) {
				for(int ix=ix0;ix<ix1;ix++) {
				for(int ic=ic0;ic<ic1;ic++) {
					const int i = error.dim.get_offset(ix, iy, ic);
					const float e1 = input.data[i] - output.data[i];
					const float e2 = e1 * std::abs(e1);
					error.data[i] = e2;
					sum_e1 += e1;
					sum_e2 += e2;
				}}}
				// normalize to match original total error.
				float mult = sum_e1 / sum_e2;
				for(int iy=iy0;iy<iy1;iy++) {
				for(int ix=ix0;ix<ix1;ix++) {
				for(int ic=ic0;ic<ic1;ic++) {
					const int i = error.dim.get_offset(ix, iy, ic);
					error.data[i] *= mult;
				}}}
			} else {
				for(int iy=iy0;iy<iy1;iy++) {
				for(int ix=ix0;ix<ix1;ix++) {
				for(int ic=ic0;ic<ic1;ic++) {
					const int i = error.dim.get_offset(ix, iy, ic);
					error.data[i] = input.data[i] - output.data[i];
				}}}
			}

			if(clamp_error) {
				for(int iy=iy0;iy<iy1;iy++) {
				for(int ix=ix0;ix<ix1;ix++) {
				for(int ic=ic0;ic<ic1;ic++) {
					const int i = error.dim.get_offset(ix, iy, ic);
					error.data[i] = std::clamp(error.data[i], -1.0f, 1.0f);
				}}}
			}
		}

		void back_propagate(const int n_threads, simple_image_f& input_error, const simple_image_f& output_error) {
			assert(input_error.dim.equals(output_error.dim));
			assert(input_error.dim.length() > 0);
			assert(layers.size() > 0);
			const int n_layers = layers.size();
			simple_image_f error_buffer_i;
			simple_image_f error_buffer_o;

			// last layer.
			error_buffer_i = simple_image_f(layers.back().idim);
			layers.back().back_propagate(n_threads, error_buffer_i, output_error);
			std::swap(error_buffer_i, error_buffer_o);

			// middle layers.
			for(int L=n_layers-2;L>0;L--) {
				error_buffer_i = simple_image_f(layers[L].idim);
				layers[L].back_propagate(n_threads, error_buffer_i, error_buffer_o);
				std::swap(error_buffer_i, error_buffer_o);
			}

			// first layer.
			layers.front().back_propagate(n_threads, input_error, error_buffer_o);
		}

		void apply_batch_error_biases(const int n_threads, const int batch_size, const float learning_rate) {
			for(int z=0;z<layers.size();z++) layers[z].apply_batch_error_biases(n_threads, batch_size, learning_rate);
		}
		void apply_batch_error_weights(const int n_threads, const int batch_size, const float learning_rate) {
			for(int z=0;z<layers.size();z++) layers[z].apply_batch_error_weights(n_threads, batch_size, learning_rate);
		}
		void clear_batch_error() {
			for(int z=0;z<layers.size();z++) layers[z].clear_batch_error();
		}
	};
}
