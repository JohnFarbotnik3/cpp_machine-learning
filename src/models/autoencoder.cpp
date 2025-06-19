
#include <vector>
#include "../networks/layer.cpp"

namespace ML::models {
	using std::vector;
	using namespace ML::networks;

	struct autoencoder {
		struct dimensions {
			int w;// layer width.
			int h;// layer height.
			dimensions(int w, int h) {
				this->w = w;
				this->h = h;
			}
			int  get_index(int x, int y) const { return x*h + y; }
			bool in_bounds(int x, int y) const { return (0 <= x) & (x < w) & (0 <= y) & (y < h); }
		};

		int input_w;
		int input_h;
		int middle_layer_index;
		vector<layer_network>	layers;
		vector<dimensions>		layer_dims;		// dimensions between outputs.
		vector<vector<float>>	layer_values;	// values between layers.
		vector<vector<float>>	layer_errors;	// errors between layers.

		autoencoder(int input_w, int input_h, int middle_w, int middle_h) {
			this->input_w = input_w;
			this->input_h = input_h;
			create_layers();
		}

		// ============================================================
		// tiling functions.
		// NOTE: connections and neurons that are out of bounds are truncated/ignored
		//       (on both input and output side of tiling).
		// ------------------------------------------------------------
		// connect NxN squares to NxN squares.
		template<int N>
		void push_layer_NxN_to_NxN(const dimensions input_dim, dimensions& output_dim, const int tile_offset_xy) {
			output_dim = dimensions(input_dim.w, input_dim.h);
			layer_network output_layer(12345);
			// for each tile in input...
			const int tx = tile_offset_xy-N;
			const int ty = tile_offset_xy-N;
			for(int x=tx;x<output_dim.w;x+=N) {
			for(int y=ty;y<output_dim.h;y+=N) {
				// create NxN square of output neurons.
				for(int nx=x;nx<x+N;nx++) {
				for(int ny=y;ny<y+N;ny++) {
				if(output_dim.in_bounds(nx, ny)) {
					// create NxN square of input connections for each neuron.
					auto& targets = output_layer.targets;
					const int ofs = targets.size();
					for(int cx=x;cx<x+N;cx++) {
					for(int cy=y;cy<y+N;cy++) {
					if(input_dim.in_bounds(cx, cy)) {
						targets.push_back(layer_target(input_dim.get_index(cx, cy)));
					}}}
					const int len = targets.size() - ofs;
					output_layer.neurons.push_back(layer_neuron(len, ofs));
				}}}
			}}
			// push layer into list.
			layers.push_back(output_layer);
		}
		void push_layer_8x8_to_8x8(const dimensions input_dim, dimensions& output_dim, const int tile_offset_xy) {}// TODO
		// condense from (w,h) to (w/2, h/2), with input  tiles offset by tile_offset.
		void push_layer_4x4_to_2x2(const dimensions input_dim, dimensions& output_dim, const int tile_offset_xy) {}// TODO
		void push_layer_8x8_to_4x4(const dimensions input_dim, dimensions& output_dim, const int tile_offset_xy) {}// TODO
		// expand   from (w,h) to (w*2, h*2), with output tiles offset by tile_offset.
		void push_layer_2x2_to_4x4(const dimensions input_dim, dimensions& output_dim, const int tile_offset_xy) {}// TODO
		void push_layer_4x4_to_8x8(const dimensions input_dim, dimensions& output_dim, const int tile_offset_xy) {}// TODO

		void create_layers() {
			// input.
			// W x H x 4 image (r,g,b,a) - 4 colour channels are treated as 2x2 square.
			dimensions dim(input_w, input_h);
			layer_dims	.push_back(dim);
			layer_values.push_back(vector<float>(input_w*input_h));
			layer_errors.push_back(vector<float>(input_w*input_h));

			// mix and condense image.
			// (w, h) -> (w/64, h/64)
			dimensions outdim(0, 0);
			for(int z=0;z<3;z++) {
				push_layer_4x4_to_4x4(dim, outdim, 0); dim=outdim;// mix
				push_layer_4x4_to_4x4(dim, outdim, 2); dim=outdim;// mix
				push_layer_4x4_to_2x2(dim, outdim, 0); dim=outdim;// condense
			}
			for(int z=0;z<3;z++) {
				push_layer_8x8_to_8x8(dim, outdim, 0); dim=outdim;// mix
				push_layer_8x8_to_8x8(dim, outdim, 4); dim=outdim;// mix
				push_layer_8x8_to_4x4(dim, outdim, 0); dim=outdim;// condense
			}

			// middle layer - embedding values will be obtained from this image.
			middle_layer_index = layers.size();

			// mix and expand image.
			for(int z=0;z<3;z++) {
				push_layer_4x4_to_8x8(dim, outdim, 0); dim=outdim;// expand
				push_layer_8x8_to_8x8(dim, outdim, 4); dim=outdim;// mix
				push_layer_8x8_to_8x8(dim, outdim, 0); dim=outdim;// mix
			}
			for(int z=0;z<3;z++) {
				push_layer_2x2_to_4x4(dim, outdim, 0); dim=outdim;// expand
				push_layer_4x4_to_4x4(dim, outdim, 2); dim=outdim;// mix
				push_layer_4x4_to_4x4(dim, outdim, 0); dim=outdim;// mix
			}
		}

		// TODO...
	};
}
