
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

		void push_layer_dense				(const dimensions_t idim, const dimensions_t odim) {
			assert(idim.length() > 0);
			assert(odim.length() > 0);
			layers.push_back(ae_layer(idim, odim, layer_pattern::dense(), 1, 1));
		}
		void push_layer_spatial_mix			(const dimensions_t idim, const int GX, const int GY, const int N, const int B) {
			assert(idim.length() > 0);
			const dimensions_t odim = idim;
			layers.push_back(ae_layer(idim, odim, layer_pattern::spatial_mix(N, B), GX, GY));
		}
		dimensions_t push_layer_encode		(const dimensions_t idim, const int GX, const int GY, const int A, const int B, const int out_ch) {
			assert(idim.length() > 0);
			assert(idim.X % A == 0);
			assert(idim.Y % A == 0);
			dimensions_t odim;
			odim.X = (idim.X / A) * B;
			odim.Y = (idim.Y / A) * B;
			odim.C = out_ch;
			assert(odim.length() > 0);
			assert(odim.X % B == 0);
			assert(odim.Y % B == 0);
			assert((idim.X / A) == (odim.X / B));
			assert((idim.Y / A) == (odim.Y / B));
			layers.push_back(ae_layer(idim, odim, layer_pattern::encode(A, B), GX, GY));
			return odim;
		}
		dimensions_t push_layer_encode_mix	(const dimensions_t idim, const int GX, const int GY, const int A, const int B, const int N, const int out_ch) {
			assert(idim.length() > 0);
			assert(idim.X % A == 0);
			assert(idim.Y % A == 0);
			dimensions_t odim;
			odim.X = (idim.X / A) * B;
			odim.Y = (idim.Y / A) * B;
			odim.C = out_ch;
			assert(odim.length() > 0);
			assert(odim.X % B == 0);
			assert(odim.Y % B == 0);
			assert((idim.X / A) == (odim.X / B));
			assert((idim.Y / A) == (odim.Y / B));
			layers.push_back(ae_layer(idim, odim, layer_pattern::encode_mix(A, B, N), GX, GY));
			return odim;
		}

		ae_model(dimensions_t input_dimensions) {
			this->image_dimensions = input_dimensions;

			// init_model_topology.
			dimensions_t idim = input_dimensions;
			dimensions_t odim = idim;
			const int ch = idim.C;

			// encoder: mix and condense image.
			odim = push_layer_encode	(idim, 8,8,	8,2,	16); idim = odim;
			odim = push_layer_encode	(idim, 4,4,	8,2,	48); idim = odim;

			// decoder: expand image back to original size.
			odim = push_layer_encode	(idim, 4,4,	2,8,	16); idim = odim;
			odim = push_layer_encode	(idim, 8,8,	2,8,	ch); idim = odim;

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
