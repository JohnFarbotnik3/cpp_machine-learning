
#include <cassert>
#include <vector>
#include <algorithm>
#include "src/image/value_image_tiles.cpp"
#include "src/utils/random.cpp"
#include "src/utils/simd.cpp"
#include "src/utils/vector_util.cpp"
#include "./ae_layer.cpp"

namespace ML::models::autoencoder {
	using std::vector;
	using namespace ML::image;
	using namespace utils::vector_util;

	struct ae_model {
		using layer_image = ML::image::value_image_tiles<float>;
		using layer_image_iterator = ML::image::value_image_tiles_iterator;

		value_image_tiles_dimensions input_dimensions;
		vector<ae_layer> layers;

		ae_model(value_image_tiles_dimensions input_dimensions) {
			this->input_dimensions = input_dimensions;
			init_model_topology();
		}

	private:
		/*
			generates connections from AxA square of pixels (from input)
			to 1x1 square (in output), with AxA input square centered on output pixel.

			NOTE:
			- output layer should have the same neuron-layout as image-data does.
			- odd values of A are preferred (for symmetry).
			- if mix_channels is true, every pixel-value in input is connected to each
			pixel-value-neuron in output.
			- if mix_channels is false, channels are kept seperate.
		*/
		void push_layer_mix_AxA_to_1x1(const value_image_tiles_dimensions dim, const int A, bool mix_channels) {
			// create output layer and generate targets for each neuron (pixel-value) in output.
			// NOTE: the order these target-lists are generated in must match order of neurons.
			layers.push_back(ae_layer(dim.length(), dim.length()));
			ae_layer& output_layer = layers.back();
			layer_image_iterator output_iter = dim.get_iterator();
			while(output_iter.has_next()) {
				int x0 = std::max(output_iter.x - A/2, 0);
				int y0 = std::max(output_iter.y - A/2, 0);
				int x1 = std::min(x0 + A, dim.X);
				int y1 = std::min(y0 + A, dim.Y);
				vector<int> connection_inds = dim.generate_target_indices(x0, x1, y0, y1, mix_channels ? -1 : output_iter.c);
				output_layer.foreward_targets.push_list(connection_inds);
				output_iter.next();
			}
			output_layer.sync_targets_list();
		}

		/*
			generates connections such that each AxA square from input is scaled
			into a BxB square in output.

			NOTE: image dimensions A and B must be picked such that image scales cleanly.
		*/
		void push_layer_scale_AxA_to_BxB(const value_image_tiles_dimensions idim, value_image_tiles_dimensions& odim, const int A, const int B, const int inC, const int outC, bool mix_channels) {
			// make sure images will scale cleanly, and have size compatible with choice of A and B.
			// - images dimensions should be divisible by A (input) or B (output).
			// - images should have the same number of AxA or BxB tiles as eachother in each dimension.
			// - if not mixing channels, then output and input should have same number of channels.
			odim = idim;
			odim.X = (odim.X * B) / A;
			odim.Y = (odim.Y * B) / A;
			odim.C = outC;
			odim.TC = outC;
			assert(idim.X % A == 0);
			assert(idim.Y % A == 0);
			assert(odim.X % B == 0);
			assert(odim.Y % B == 0);
			assert(idim.X / A == odim.X / B);
			assert(idim.Y / A == odim.Y / B);
			assert(mix_channels || (odim.C == idim.C));

			// create output layer and generate targets for each neuron (pixel-value) in output.
			// NOTE: the order these target-lists are generated in must match order of neurons.
			layers.push_back(ae_layer(idim.length(), odim.length()));
			ae_layer& output_layer = layers.back();
			layer_image_iterator output_iter = odim.get_iterator();
			while(output_iter.has_next()) {
				int x0 = (output_iter.x / B) * A;
				int y0 = (output_iter.y / B) * A;
				int x1 = x0 + A;
				int y1 = y0 + A;
				vector<int> connection_inds = idim.generate_target_indices(x0, x1, y0, y1, mix_channels ? -1 : output_iter.c);
				output_layer.foreward_targets.push_list(connection_inds);
				output_iter.next();
			}
			output_layer.sync_targets_list();
		}

		/*
			NOTE: this is called by constructor, so it doesnt need to be called externally
			(since topology is fixed given the same input image resolution).
		*/
		void init_model_topology() {
			// clear.
			layers.clear();

			// input.
			value_image_tiles_dimensions idim = input_dimensions;
			value_image_tiles_dimensions odim = idim;
			const int ch = idim.C;

			// mix and condense image.
			// (w, h) -> (w/8, h/8)
			push_layer_mix_AxA_to_1x1(idim, 3, false);
			push_layer_scale_AxA_to_BxB(idim, odim, 8, 2, ch, 12, true); idim = odim;
			push_layer_scale_AxA_to_BxB(idim, odim, 8, 2, 12, 16, true); idim = odim;

			// expand image back to original size.
			push_layer_scale_AxA_to_BxB(idim, odim, 2, 8, 16, 12, true); idim = odim;
			push_layer_scale_AxA_to_BxB(idim, odim, 2, 8, 12, ch, true); idim = odim;
			push_layer_mix_AxA_to_1x1(idim, 3, false);

			assert(odim.X == input_dimensions.X);
			assert(odim.Y == input_dimensions.Y);
			assert(odim.C == input_dimensions.C);
			assert(odim.TX == input_dimensions.TX);
			assert(odim.TY == input_dimensions.TY);
			assert(odim.TC == input_dimensions.TC);
		}

	public:
		void init_model_parameters(int seed, float bias_mean, float bias_stddev, float weight_mean, float weight_stddev) {
			std::mt19937 gen32 = utils::random::get_generator_32(seed);
			std::normal_distribution distr_bias = utils::random::rand_normal<float>(bias_mean, bias_stddev);
			std::normal_distribution distr_weight = utils::random::rand_normal<float>(weight_mean, weight_stddev);

			for(int z=0;z<layers.size();z++) {
				auto& layer = layers[z];
				//for(auto& bias   : layer.biases) bias = distr_bias(gen32);
				//for(auto& target : layer.foreward_targets.targets) target.weight = distr_weight(gen32);
				///*
				for(int n=0;n<layer.biases.size();n++) {
					layer.biases[n] = distr_bias(gen32);
					target_itv itv = layer.foreward_targets.get_interval(n);
					const float mult = sqrtf(1.0f / (itv.end - itv.beg));
					for(int x=itv.beg;x<itv.end;x++) {
						layer.foreward_targets.targets[x].weight = distr_weight(gen32) * mult;
						//layer.foreward_targets.targets[x].weight = distr_weight(gen32);
					}
				}
				//*/
				layer.foreward_targets.save_weights(layer.backprop_targets);
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
		void generate_error_image(const value_image_tiles<float>& input, const vector<float>& output, vector<float>& error, bool loss_squared, bool clamp_error) {
			// assertions.
			const int IMAGE_SIZE = input.X * input.Y * input.C;
			assert(input.data.size() == IMAGE_SIZE);
			assert(output.size() == IMAGE_SIZE);
			assert(error.size() == IMAGE_SIZE);

			// clear error image.
			for(int x=0;x<IMAGE_SIZE;x++) error[x] = 0;

			value_image_tiles_iterator sample_iter = input.get_iterator(
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
