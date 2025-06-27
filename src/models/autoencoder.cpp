
#include <cassert>
#include <vector>
#include "../networks/layer.cpp"
#include "../utils/random.cpp"
#include "../image.cpp"

namespace ML::models {
	using std::vector;
	using namespace ML::networks;
	using namespace ML::image;

	struct autoencoder {

		int input_w;
		int input_h;
		int middle_layer_index;
		vector<layer_network>	layers;
		vector<vector<float>>	layer_values;	// values output by layers - in row-major-RGBA format.
		vector<vector<float>>	layer_errors;	// errors output by layers.

		autoencoder(int w, int h) {
			this->input_w = w;
			this->input_h = h;
			init_model_topology();
		}

	private:
		struct image_dimensions {
			int w;
			int h;
		};

		/*
			generates connections from AxA square of pixels (from input)
			to 1x1 square (in output), with AxA input square centered on output pixel.

			NOTE:
			- output layer should have the same neuron-layout as image-data does (row-major RGBA).
			- odd values of A are preferred (for symmetry).
			- if mix_channels is true, every pixel-value in input is connected to each
			pixel-value-neuron in output (4x as many connections).
			- if mix_channels is false, channels are kept seperate.
		*/
		void push_layer_mix_AxA_to_1x1(const image_dimensions dim, const int A, bool mix_channels) {
			// create output layer.
			layer_network output_layer;
			output_layer.neurons.resize(dim.w * dim.h * 4);

			// for each pixel-value in output...
			for(int y=0;y<dim.h;y++) {
			for(int x=0;x<dim.w;x++) {
			for(int c=0;c<4;c++) {
				int in_x0 = x - A/2;
				int in_y0 = y - A/2;
				vector<int> connection_inds = ML::image::generate_image_data_indices(dim.w, dim.h, in_x0, in_y0, A, A, mix_channels ? -1 : c);
				// update neuron.
				layer_neuron& neuron = output_layer.neurons[(y*dim.w + x)*4 + c];
				neuron.targets_len = connection_inds.size();
				neuron.targets_ofs = output_layer.targets.size();
				// add connections.
				for(int ci : connection_inds) output_layer.targets.push_back(ci);
			}}}

			// push layer into list.
			layers.push_back(output_layer);
			layer_values.push_back(vector<float>(output_layer.neurons.size()));
			layer_errors.push_back(vector<float>(output_layer.neurons.size()));
		}

		/*
			generates connections such that each AxA square from input is scaled
			into a BxB square in output.

			NOTE: image dimensions, A, and B must be picked such that image scales cleanly.
		*/
		void push_layer_scale_AxA_to_BxB(
			const image_dimensions idim,
			image_dimensions& odim,
			const int A,
			const int B,
			bool mix_channels
		) {
			// make sure images will scale cleanly, and have size compatible with choice of A and B.
			// - images dimensions should be divisible by A (input) or B (output).
			// - images should have the same number of AxA or BxB tiles as eachother in each dimension.
			odim = image_dimensions(
				(idim.w * B) / A,
				(idim.h * B) / A
			);
			assert(idim.w % A == 0);
			assert(idim.h % A == 0);
			assert(odim.w % B == 0);
			assert(odim.h % B == 0);
			assert(idim.w / A == odim.w / B);
			assert(idim.h / A == odim.h / B);

			// create output layer.
			layer_network output_layer;
			output_layer.neurons.resize(ML::image::get_image_data_length(odim.w, odim.h));

			// for each pixel-value in output...
			for(int y=0;y<odim.h;y++) {
			for(int x=0;x<odim.w;x++) {
			for(int c=0;c<4;c++) {
				// get tile coordinates.
				int tx = x / B;
				int ty = y / B;
				// get input area.
				int in_x0 = tx * A;
				int in_y0 = ty * A;
				vector<int> connection_inds = ML::image::generate_image_data_indices(idim.w, idim.h, in_x0, in_y0, A, A, mix_channels ? -1 : c);
				// update neuron.
				layer_neuron& neuron = output_layer.neurons[ML::image::get_image_data_offset(odim.w, odim.h, x, y) + c];
				neuron.targets_len = connection_inds.size();
				neuron.targets_ofs = output_layer.targets.size();
				// add connections.
				for(int ci : connection_inds) output_layer.targets.push_back(ci);
			}}}

			// push layer into list.
			layers.push_back(output_layer);
			layer_values.push_back(vector<float>(output_layer.neurons.size()));
			layer_errors.push_back(vector<float>(output_layer.neurons.size()));
		}

		/*
			NOTE: this is called by constructor, so it doesnt need to be called externally
			(since topology is fixed given the same input image resolution).
		*/
		void init_model_topology() {
			// clear.
			layers.clear();
			layer_values.clear();
			layer_errors.clear();

			// input.
			image_dimensions  in_dim = image_dimensions(input_w, input_h);
			image_dimensions out_dim(0, 0);

			// mix and condense image.
			// (w, h) -> (w/32, h/32)
			push_layer_mix_AxA_to_1x1(in_dim, 3, false);
			push_layer_scale_AxA_to_BxB(in_dim, out_dim, 4, 2, true); in_dim = out_dim;
			/*
			push_layer_mix_AxA_to_1x1(in_dim, 3, false);
			push_layer_scale_AxA_to_BxB(in_dim, out_dim, 4, 2, true); in_dim = out_dim;
			push_layer_mix_AxA_to_1x1(in_dim, 7, true);
			push_layer_scale_AxA_to_BxB(in_dim, out_dim, 8, 4, true); in_dim = out_dim;
			push_layer_mix_AxA_to_1x1(in_dim, 9, true);
			push_layer_scale_AxA_to_BxB(in_dim, out_dim, 16, 8, true); in_dim = out_dim;
			//*/

			// middle layer - embedding values will be obtained from this image.
			middle_layer_index = layers.size();

			// expand back to original image.
			/*
			push_layer_scale_AxA_to_BxB(in_dim, out_dim, 8, 16, true); in_dim = out_dim;
			push_layer_mix_AxA_to_1x1(in_dim, 9, true);
			push_layer_scale_AxA_to_BxB(in_dim, out_dim, 4, 8, true); in_dim = out_dim;
			push_layer_mix_AxA_to_1x1(in_dim, 7, true);
			push_layer_scale_AxA_to_BxB(in_dim, out_dim, 2, 4, true); in_dim = out_dim;
			push_layer_mix_AxA_to_1x1(in_dim, 3, false);
			//*/
			push_layer_scale_AxA_to_BxB(in_dim, out_dim, 2, 4, true); in_dim = out_dim;
			push_layer_mix_AxA_to_1x1(in_dim, 3, false);

			assert(out_dim.w == input_w);
			assert(out_dim.h == input_h);
		}

	public:
		void init_model_parameters(int seed, float bias_mean, float bias_stddev, float weight_mean, float weight_stddev) {
			std::mt19937 gen32 = utils::random::get_generator_32(seed);
			std::normal_distribution distr_bias = utils::random::rand_normal<float>(bias_mean, bias_stddev);
			std::normal_distribution distr_weight = utils::random::rand_normal<float>(weight_mean, weight_stddev);

			for(int z=0;z<layers.size();z++) {
				auto& layer = layers[z];
				for(auto& neuron : layer.neurons) neuron.bias = distr_bias(gen32);
				for(auto& target : layer.targets) target.weight = distr_weight(gen32);
			}
		}

		void apply_batch_error(float rate) {
			for(int x=0;x<layers.size();x++) layers[x].apply_batch_error(rate);
		}

		void propagate(std::vector<float>& input_values, std::vector<float>& output_values) {
			// first layer.
			layers[0].propagate(input_values, layer_values[0]);
			// middle layers.
			for(int z=1;z<layer_values.size();z++) {
				layers[z].propagate(layer_values[z-1], layer_values[z]);
			}
			// copy output.
			output_values = layer_values[layer_values.size()-1];
		}

		void back_propagate(std::vector<float>& output_error, std::vector<float>& input_error, std::vector<float>& input_values) {
			// copy output.
			layer_errors[layer_errors.size()-1] = output_error;
			// middle layers.
			for(int z=layer_values.size()-1;z>0;z--) layers[z].back_propagate(layer_errors[z], layer_errors[z-1], layer_values[z-1]);
			// first layer.
			layers[0].back_propagate(layer_errors[0], input_error, input_values);
		}
	};
}
