
#include <vector>
#include "../networks/layer.cpp"
#include "../utils/random.cpp"
#include "../geometry.cpp"

namespace ML::models {
	using std::vector;
	using namespace ML::networks;
	using namespace ML::geometry;

	struct autoencoder {

		image_dimensions input_dimensions;
		int middle_layer_index;
		vector<layer_network>	layers;
		vector<image_dimensions>layer_dims;		// dimensions of layer outputs.
		vector<vector<float>>	layer_values;	// values output by layers.
		vector<vector<float>>	layer_errors;	// errors output by layers.

		autoencoder(image_dimensions input_dimensions) {
			this->input_dimensions = input_dimensions;
			init_model_topology();
		}

	private:

		// connect AxA squares to BxB squares.
		// neurons or connections that end up out of bounds during tiling are omitted.
		// TODO: create visual explanation of this algorithm.
		void push_layer_AxA_to_BxB(
			const image_dimensions input_dim,
			const image_dimensions output_dim,
			const int A,
			const int B,
			const vec2<int> input_advance,
			const vec2<int> output_advance,
			const vec2<int> input_offset,
			const vec2<int> output_offset
		) {
			layer_network output_layer;

			// generate tile coordinates such that entire output will be tiled.
			// output-tiles and input-tiles advance together, at their respective rates.
			// NOTE: coordinates are clamped to input/output dimensions as needed.
			vector<int> out_tx;
			vector<int> out_ty;
			vector<int> in_tx;
			vector<int> in_ty;
			for(int t=-1;;t++) {
				int xo = t * output_advance.x + output_offset.x;
				int xi = t *  input_advance.x +  input_offset.x;
				if(xo >= output_dim.X) break;
				out_tx.push_back(std::clamp(xo, 0, output_dim.X));
				 in_tx.push_back(std::clamp(xi, 0,  input_dim.X));
			}
			for(int t=-1;;t++) {
				int yo = t * output_advance.y + output_offset.y;
				int yi = t *  input_advance.y +  input_offset.y;
				if(yo >= output_dim.Y) break;
				out_ty.push_back(std::clamp(yo, 0, output_dim.Y));
				 in_ty.push_back(std::clamp(yi, 0,  input_dim.Y));
			}

			// for each tile in output layer...
			for(int tx=0; tx<out_tx.size()-1; tx++) {
			for(int ty=0; ty<out_ty.size()-1; ty++) {
				// generate connection target indices pointing to AxA square in input.
				vector<int> connection_inds;
				for(int x=in_tx[tx]; x<in_tx[tx+1]; x++) {
				for(int y=in_ty[ty]; y<in_ty[ty+1]; y++) {
					connection_inds.push_back(input_dim.get_index(x,y));
				}}
				// generate BxB square of neurons in output.
				for(int x=out_tx[tx]; x<out_tx[tx+1]; x++) {
				for(int y=out_ty[ty]; y<out_ty[ty+1]; y++) {
					// create neuron.
					int len = connection_inds.size();
					int ofs = output_layer.targets.size();
					output_layer.neurons.push_back(layer_neuron(len, ofs));
					// add connections.
					for(int ci : connection_inds) output_layer.targets.push_back(ci);
				}}
			}}

			// push layer into list.
			layers.push_back(output_layer);
			layer_dims.push_back(output_dim);
			layer_values.push_back(vector<float>(output_dim.get_area()));
			layer_errors.push_back(vector<float>(output_dim.get_area()));
		}
		// mix - 4x4 to 2x2 with overlapping input.
		void push_layer_mix_4x4(const image_dimensions in_dim) {
			const image_dimensions out_dim = in_dim;
			push_layer_AxA_to_BxB(in_dim, out_dim, 4, 2, vec2(2,2), vec2(2,2), vec2(-1,-1), vec2(0,0));
		}
		// mix - 8x8 to 2x2 with overlapping input.
		void push_layer_mix_8x8(const image_dimensions in_dim) {
			const image_dimensions out_dim = in_dim;
			push_layer_AxA_to_BxB(in_dim, out_dim, 8, 2, vec2(2,2), vec2(2,2), vec2(-3,-3), vec2(0,0));
		}
		// condense.
		void push_layer_condense_4x4_to_2x2(const image_dimensions in_dim, const image_dimensions out_dim) {
			push_layer_AxA_to_BxB(in_dim, out_dim, 4, 2, vec2(4,4), vec2(2,2), vec2(0,0), vec2(0,0));
		}
		void push_layer_condense_8x8_to_4x4(const image_dimensions in_dim, const image_dimensions out_dim) {
			push_layer_AxA_to_BxB(in_dim, out_dim, 8, 4, vec2(8,8), vec2(4,4), vec2(0,0), vec2(0,0));
		}
		// expand.
		void push_layer_expand_2x2_to_4x4(const image_dimensions in_dim, const image_dimensions out_dim) {
			push_layer_AxA_to_BxB(in_dim, out_dim, 2, 4, vec2(2,2), vec2(4,4), vec2(0,0), vec2(0,0));
		}
		void push_layer_expand_4x4_to_8x8(const image_dimensions in_dim, const image_dimensions out_dim) {
			push_layer_AxA_to_BxB(in_dim, out_dim, 4, 8, vec2(4,4), vec2(8,8), vec2(0,0), vec2(0,0));
		}

		/*
			NOTE: this is called by constructor, so it doesnt need to be called externally
			(since topology is fixed given the same input image resolution).
		*/
		void init_model_topology() {
			// clear.
			layers.clear();
			layer_dims.clear();
			layer_values.clear();
			layer_errors.clear();

			// input.
			// W x H x 4 image (r,g,b,a) - 4 colour channels are treated as 2x2 square.
			image_dimensions  in_dim = image_dimensions(input_dimensions.X * 2, input_dimensions.Y * 2);
			image_dimensions out_dim(0, 0);

			// mix and condense image.
			// (w, h) -> (w/64, h/64)
			for(int z=0;z<3;z++) {
				// mix
				push_layer_mix_4x4(in_dim);
				push_layer_mix_4x4(in_dim);
				// condense
				out_dim = image_dimensions(in_dim.X/2, in_dim.Y/2);
				push_layer_condense_4x4_to_2x2(in_dim, out_dim);
				in_dim = out_dim;
			}
			for(int z=0;z<3;z++) {
				// mix
				push_layer_mix_8x8(in_dim);
				push_layer_mix_8x8(in_dim);
				// condense
				out_dim = image_dimensions(in_dim.X/2, in_dim.Y/2);
				push_layer_condense_8x8_to_4x4(in_dim, out_dim);
				in_dim = out_dim;
			}

			// middle layer - embedding values will be obtained from this image.
			middle_layer_index = layers.size();

			// expand and mix image.
			// (w/64, h/64) -> (w, h)
			for(int z=0;z<3;z++) {
				// expand
				out_dim = image_dimensions(in_dim.X*2, in_dim.Y*2);
				push_layer_expand_4x4_to_8x8(in_dim, out_dim);
				in_dim = out_dim;
				// mix
				push_layer_mix_8x8(in_dim);
				push_layer_mix_8x8(in_dim);
			}
			for(int z=0;z<3;z++) {
				// expand
				out_dim = image_dimensions(in_dim.X*2, in_dim.Y*2);
				push_layer_expand_2x2_to_4x4(in_dim, out_dim);
				in_dim = out_dim;
				// mix
				push_layer_mix_4x4(in_dim);
				push_layer_mix_4x4(in_dim);
			}
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

		// TODO
		void propagate(std::vector<float>& input_values, std::vector<float>& output_values);

		// TODO
		void back_propagate(std::vector<float>& input_error, std::vector<float>& output_error, std::vector<float>& input_values);
	};
}
