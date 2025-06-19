
#include <vector>
#include "../networks/layer.cpp"

namespace ML::models {
	using std::vector;
	using namespace ML::networks;

	struct autoencoder {
		struct dimensions {
			int X;// layer width.
			int Y;// layer height.
			dimensions(int X, int Y) {
				this->X = X;
				this->Y = Y;
			}
			int  get_index(int x, int y) const { return x*Y + y; }
			bool in_bounds(int x, int y) const { return (0 <= x) & (x < X) & (0 <= y) & (y < Y); }
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

		struct vec2 {
			int x;
			int y;
		};

		// connect AxA squares to BxB squares.
		// neurons or connections that end up out of bounds during tiling are omitted.
		void push_layer_AxA_to_BxB(
			const dimensions input_dim,
			const dimensions output_dim,
			const int A,
			const int B,
			const vec2 input_advance,
			const vec2 output_advance,
			const vec2 input_offset,
			const vec2 output_offset
		) {
			layer_network output_layer(12345);
			// for each tile in output layer...
			for(int tx=-1, out_tx = (tx * output_advance.x + output_offset.x); out_tx < output_dim.X; tx++) {
			for(int ty=-1, out_ty = (ty * output_advance.y + output_offset.y); out_ty < output_dim.Y; ty++) {
				int in_tx = (tx * input_advance.x + input_offset.x);
				int in_ty = (ty * input_advance.y + input_offset.y);
				// create a BxB square of neurons.
				for(int nx=out_tx;nx<out_tx+B;nx++) {
				for(int ny=out_ty;ny<out_ty+B;ny++) {
				if(output_dim.in_bounds(nx, ny)) {
					// create connections to AxA square in input-layer for each neuron.
					auto& targets = output_layer.targets;
					const int ofs = targets.size();
					for(int cx=in_tx;cx<in_tx+A;cx++) {
					for(int cy=in_ty;cy<in_ty+A;cy++) {
					if(input_dim.in_bounds(cx, cy)) {
						targets.push_back(layer_target(input_dim.get_index(cx, cy)));
					}
					}}
					const int len = targets.size() - ofs;
					output_layer.neurons.push_back(layer_neuron(len, ofs));
				}
				}}
			}}
			// push layer into list.
			layers.push_back(output_layer);
			layer_dims.push_back(output_dim);
		}
		// mix - 4x4 to 2x2 with overlapping input.
		void push_layer_mix_4x4(const dimensions in_dim) {
			const dimensions out_dim = in_dim;
			push_layer_AxA_to_BxB(in_dim, out_dim, 4, 2, vec2(2,2), vec2(2,2), vec2(-1,-1), vec2(0,0));
		}
		// mix - 8x8 to 2x2 with overlapping input.
		void push_layer_mix_8x8(const dimensions in_dim) {
			const dimensions out_dim = in_dim;
			push_layer_AxA_to_BxB(in_dim, out_dim, 8, 2, vec2(2,2), vec2(2,2), vec2(-3,-3), vec2(0,0));
		}
		// condense.
		void push_layer_condense_4x4_to_2x2(const dimensions in_dim, const dimensions out_dim) {
			push_layer_AxA_to_BxB(in_dim, out_dim, 4, 2, vec2(4,4), vec2(2,2), vec2(0,0), vec2(0,0));
		}
		void push_layer_condense_8x8_to_4x4(const dimensions in_dim, const dimensions out_dim) {
			push_layer_AxA_to_BxB(in_dim, out_dim, 8, 4, vec2(8,8), vec2(4,4), vec2(0,0), vec2(0,0));
		}
		// expand.
		void push_layer_expand_2x2_to_4x4(const dimensions in_dim, const dimensions out_dim) {
			push_layer_AxA_to_BxB(in_dim, out_dim, 2, 4, vec2(2,2), vec2(4,4), vec2(0,0), vec2(0,0));
		}
		void push_layer_expand_4x4_to_8x8(const dimensions in_dim, const dimensions out_dim) {
			push_layer_AxA_to_BxB(in_dim, out_dim, 4, 8, vec2(4,4), vec2(8,8), vec2(0,0), vec2(0,0));
		}

		void create_layers() {
			// input.
			// W x H x 4 image (r,g,b,a) - 4 colour channels are treated as 2x2 square.
			dimensions  in_dim(input_w, input_h);
			dimensions out_dim(0, 0);
			layer_dims	.push_back(in_dim);
			layer_values.push_back(vector<float>(input_w*input_h));
			layer_errors.push_back(vector<float>(input_w*input_h));

			// mix and condense image.
			// (w, h) -> (w/64, h/64)
			for(int z=0;z<3;z++) {
				// mix
				push_layer_mix_4x4(in_dim);
				push_layer_mix_4x4(in_dim);
				// condense
				out_dim = dimensions(in_dim.X/2, in_dim.Y/2);
				push_layer_condense_4x4_to_2x2(in_dim, out_dim);
				in_dim = out_dim;
			}
			for(int z=0;z<3;z++) {
				// mix
				push_layer_mix_8x8(in_dim);
				push_layer_mix_8x8(in_dim);
				// condense
				out_dim = dimensions(in_dim.X/2, in_dim.Y/2);
				push_layer_condense_8x8_to_4x4(in_dim, out_dim);
				in_dim = out_dim;
			}

			// middle layer - embedding values will be obtained from this image.
			middle_layer_index = layers.size();

			// expand and mix image.
			// (w/64, h/64) -> (w, h)
			for(int z=0;z<3;z++) {
				// expand
				out_dim = dimensions(in_dim.X*2, in_dim.Y*2);
				push_layer_expand_4x4_to_8x8(in_dim, out_dim);
				in_dim = out_dim;
				// mix
				push_layer_mix_8x8(in_dim);
				push_layer_mix_8x8(in_dim);
			}
			for(int z=0;z<3;z++) {
				// expand
				out_dim = dimensions(in_dim.X*2, in_dim.Y*2);
				push_layer_expand_2x2_to_4x4(in_dim, out_dim);
				in_dim = out_dim;
				// mix
				push_layer_mix_4x4(in_dim);
				push_layer_mix_4x4(in_dim);
			}

			// populate other vectors (values, errors).
			// TODO...
		}
	};
}
