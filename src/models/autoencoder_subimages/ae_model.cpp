
#include <cassert>
#include <vector>
#include "src/utils/random.cpp"
#include "src/utils/simd.cpp"
#include "./ae_layer.cpp"

namespace ML::models::autoencoder_subimage {
	using std::vector;
	using namespace ML::image;
	using dimensions_t = simd_image_8f_dimensions;

	struct ae_model {
		dimensions_t image_dimensions;
		vector<ae_layer> layers;

		void push_layer_dense				(const dimensions_t idim, const dimensions_t odim) {
			assert(idim.length() > 0);
			assert(odim.length() > 0);
			layers.push_back(ae_layer(idim, odim, 1, 1, layer_pattern::dense()));
		}
		void push_layer_spatial_mix			(const dimensions_t idim, const int GX, const int GY, const int N, const int B) {
			assert(idim.length() > 0);
			const dimensions_t odim = idim;
			layers.push_back(ae_layer(idim, odim, GX, GY, layer_pattern::spatial_mix(N, B)));
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
			layers.push_back(ae_layer(idim, odim, GX, GY, layer_pattern::encode(A, B)));
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
			layers.push_back(ae_layer(idim, odim, GX, GY, layer_pattern::encode_mix(A, B, N)));
			return odim;
		}

		ae_model(dimensions_t input_dimensions) {
			this->image_dimensions = input_dimensions;

			// init_model_topology.
			dimensions_t idim = input_dimensions;
			dimensions_t odim = idim;
			const int ch = idim.C;

			// encoder: mix and condense image.
			odim = push_layer_encode	(idim, 8,8,	8,2,	12); idim = odim;
			push_layer_spatial_mix		(idim, 4,4, 10,2);
			odim = push_layer_encode	(idim, 4,4,	8,2,	36); idim = odim;
			push_layer_spatial_mix		(idim, 2,2, 6,2);

			// decoder: expand image back to original size.
			push_layer_spatial_mix		(idim, 2,2, 6,2);
			odim = push_layer_encode	(idim, 4,4,	2,8,	12); idim = odim;
			push_layer_spatial_mix		(idim, 4,4, 10,2);
			odim = push_layer_encode	(idim, 8,8,	2,8,	ch); idim = odim;

			//push_layer_spatial_mix(idim, 8,8,	1,1); idim = odim;// TODO TEST

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

		void propagate(const int n_threads, const simd_image_8f& value_i, simd_image_8f& value_o) {
			assert(value_i.dim.equals(value_o.dim));
			assert(value_i.dim.length() > 0);
			assert(layers.size() > 0);
			const int n_layers = layers.size();
			// first layer.
			layers.front().propagate(n_threads, value_i);
			// middle layers.
			for(int L=1;L<n_layers;L++) layers[L].propagate(n_threads, layers[L-1].value_o);
			// last layer.
			memcpy(value_o.data.data(), layers.back().value_o.data.data(), value_o.data.size()*sizeof(value_o.data[0]));
		}

		void back_propagate(const int n_threads, simd_image_8f& error_i, const simd_image_8f& error_o, const simd_image_8f& value_i) {
			assert(error_i.dim.equals(error_o.dim));
			assert(error_i.dim.length() > 0);
			assert(layers.size() > 0);
			const int n_layers = layers.size();
			// last layer.
			memcpy(layers.back().error_o.data.data(), error_o.data.data(), error_o.data.size()*sizeof(error_o.data[0]));
			// middle layers.
			for(int L=n_layers-1;L>0;L--) layers[L].back_propagate(n_threads, layers[L-1].error_o, layers[L-1].value_o);
			// first layer.
			layers.front().back_propagate(n_threads, error_i, value_i);
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
