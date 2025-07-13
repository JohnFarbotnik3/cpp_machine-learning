
#include <cassert>
#include <vector>
#include <algorithm>
#include "src/image/value_image_lines.cpp"
#include "src/utils/random.cpp"
#include "src/utils/simd.cpp"
#include "src/utils/vector_util.cpp"
#include "./ae_layer.cpp"

namespace ML::models::autoencoder_fixed {
	using std::vector;
	using namespace ML::image;
	using namespace utils::vector_util;

	struct ae_model {
		using layer_image = ML::image::value_image_lines<float>;
		using dimensions_t = value_image_lines_dimensions;

		dimensions_t input_dimensions;
		vector<ae_layer> layers;

		ae_model(value_image_lines_dimensions input_dimensions) {
			this->input_dimensions = input_dimensions;
			init_model_topology();
		}

	private:
		void push_layer_scale_AxA_to_BxB(const dimensions_t idim, dimensions_t& odim, const int A, const int B, const int M) {
			odim.X = (idim.X * B) / A;
			odim.Y = (idim.Y * B) / A;
			scale_ratio scale = { A, B, M };
			//printf("idim: X=%i, Y=%i, C=%i, odim: X=%i, Y=%i, C=%i\n", idim.X, idim.Y, idim.C, odim.X, odim.Y, odim.C);
			layers.push_back(ae_layer(idim, odim, scale));
		}

		/*
			NOTE: this is called by constructor, so it doesnt need to be called externally
			(since topology is fixed given the same input image resolution).
		*/
		void init_model_topology() {
			// clear.
			layers.clear();

			// input.
			dimensions_t idim = input_dimensions;
			dimensions_t odim = idim;
			const int ch = idim.C;

			// mix and condense image.
			odim.C= 8; push_layer_scale_AxA_to_BxB(idim, odim, 8, 2,  8); idim = odim;
			odim.C=24; push_layer_scale_AxA_to_BxB(idim, odim, 8, 2, 16); idim = odim;
			odim.C=72; push_layer_scale_AxA_to_BxB(idim, odim, 8, 2, 16); idim = odim;

			// expand image back to original size.
			odim.C=24; push_layer_scale_AxA_to_BxB(idim, odim, 2, 8, 4); idim = odim;
			odim.C= 8; push_layer_scale_AxA_to_BxB(idim, odim, 2, 8, 4); idim = odim;
			odim.C=ch; push_layer_scale_AxA_to_BxB(idim, odim, 2, 8, 2); idim = odim;

			assert(odim.X == input_dimensions.X);
			assert(odim.Y == input_dimensions.Y);
			assert(odim.C == input_dimensions.C);
		}

	public:
		void init_model_parameters(int seed, float bias_mean, float bias_stddev, float weight_mean, float weight_stddev) {
			std::mt19937 gen32 = utils::random::get_generator_32(seed);
			std::normal_distribution distr_bias = utils::random::rand_normal<float>(bias_mean, bias_stddev);
			std::normal_distribution distr_weight = utils::random::rand_normal<float>(weight_mean, weight_stddev);

			for(int z=0;z<layers.size();z++) {
				auto& layer = layers[z];
				const float mult = sqrtf(1.0f / layer.weights_per_output_neuron());
				for(int n=0;n<layer.biases.size();n++) layer.biases[n] = distr_bias(gen32);
				for(int x=0;x<layer.weights.size();x++) layer.weights[x] = distr_weight(gen32) * mult;
			}
		}

		void propagate(const int n_threads, const std::vector<float>& input_values, std::vector<float>& output_values) {
			// assertions.
			const int n_layers = layers.size();
			assert( input_values.size() == layers[0].input_image_size());
			assert(output_values.size() == layers[n_layers-1].output_image_size());

			vector<float> value_buffer_i;
			vector<float> value_buffer_o;

			// first layer.
			value_buffer_o.resize(layers[0].output_image_size());
			layers[0].propagate(n_threads, input_values, value_buffer_o);
			std::swap(value_buffer_i, value_buffer_o);

			// middle layers.
			for(int L=1;L<layers.size();L++) {
				value_buffer_o.resize(layers[L].output_image_size());
				layers[L].propagate(n_threads, value_buffer_i, value_buffer_o);
				std::swap(value_buffer_i, value_buffer_o);
			}

			// copy output.
			vec_copy(output_values, value_buffer_i, 0, output_values.size());
		}

		/*
			WARNING: loss_squared can lead to error-concentration which causes models to explode
			when training is going well and they are very close to 0 average error.
		*/
		void generate_error_image(const value_image_lines<float>& input, const vector<float>& output, vector<float>& error, bool loss_squared, bool clamp_error) {
			// assertions.
			const int IMAGE_SIZE = input.X * input.Y * input.C;
			assert(input.data.size() == IMAGE_SIZE);
			assert(output.size() == IMAGE_SIZE);
			assert(error.size() == IMAGE_SIZE);

			// clear error image.
			for(int x=0;x<IMAGE_SIZE;x++) error[x] = 0;

			value_image_lines_iterator sample_iter = input.get_iterator(
				input.x0, input.x1,
				input.y0, input.y1,
				0, input.C
			);

			if(loss_squared) {
				while(sample_iter.has_next()) {
					int i = sample_iter.i;
					const float delta = input.data[i] - output[i];
					error[i] = delta * std::abs(delta);
					sample_iter.next();
				}
			} else {
				while(sample_iter.has_next()) {
					int i = sample_iter.i;
					const float delta = input.data[i] - output[i];
					error[i] = delta;
					sample_iter.next();
				}
			}

			if(clamp_error) {
				for(int x=0;x<error.size();x++) error[x] = std::clamp(error[x], -1.0f, 1.0f);
			}
		}

		void back_propagate(const int n_threads, vector<float>& input_error, const vector<float>& input_value, const vector<float>& output_error) {
			// assertions.
			const int n_layers = layers.size();
			assert( input_value.size() == layers[0].input_image_size());
			assert( input_error.size() == layers[0].input_image_size());
			assert(output_error.size() == layers[n_layers-1].output_image_size());

			vector<float> error_buffer_i;
			vector<float> error_buffer_o;

			// copy output.
			error_buffer_o.resize(layers[n_layers-1].output_image_size());
			vec_copy(error_buffer_o, output_error, 0, output_error.size());

			// middle layers.
			for(int L=n_layers-1;L>0;L--) {
				error_buffer_i.resize(layers[L].input_image_size());
				layers[L].back_propagate(n_threads, error_buffer_i, layers[L-1].output, error_buffer_o);
				std::swap(error_buffer_i, error_buffer_o);
			}

			// first layer.
			layers[0].back_propagate(n_threads, input_error, input_value, error_buffer_o);
		}

		void clear_batch_error() {
			for(int z=0;z<layers.size();z++) layers[z].clear_batch_error();
		}

		void apply_batch_error(const int n_threads, const int batch_size, const float learning_rate_b, const float learning_rate_w) {
			for(int z=0;z<layers.size();z++) layers[z].apply_batch_error(n_threads, batch_size, learning_rate_b, learning_rate_w);
		}
	};
}
