
#include <vector>
#include "types.cpp"

namespace ML::models::autoencoder_subimage {
	using std::vector;

	/*
		generate a list-kernel of input-neuron offsets
		- from perspective of output-neuron [0,0,0]
		- with respect to input-neuron [0,0,0]
		for use during foreward-propagation.
	*/
	vector<int> get_input_neuron_offsets_kernel(const layer_pattern pattern, const dim_t idim) {
		assert(pattern.type != LAYER_TYPE::NONE);
		assert(idim.innerX() > 0);
		assert(idim.innerY() > 0);
		assert(idim.innerC() > 0);
		vector<int> offsets;
		const int ox = 0;
		const int oy = 0;
		const int oc = 0;
		if(pattern.type == LAYER_TYPE::DENSE) {
			for(int iy=0;iy<idim.innerY();iy++) {
			for(int ix=0;ix<idim.innerX();ix++) {
			for(int ic=0;ic<idim.innerC();ic++) {
				offsets.push_back(idim.get_offset_padded(ix, iy, ic));
			}}}
		}
		if(pattern.type == LAYER_TYPE::ENCODE) {
			const int A = pattern.A;
			const int B = pattern.B;
			assert(A > 0);
			assert(B > 0);
			for(int iy=0;iy<A;iy++) {
			for(int ix=0;ix<A;ix++) {
			for(int ic=0;ic<idim.innerC();ic++) {
				offsets.push_back(idim.get_offset_padded(ix, iy, ic));
			}}}
		}
		if(pattern.type == LAYER_TYPE::SPATIAL_MIX) {
			const int N = pattern.N;
			const int M = pattern.M;
			assert(N > 0);
			assert(M > 0);
			const int p0 = (ox - (ox % M)) + (M/2) - (N/2);
			const int p1 = p0 + N;
			assert(p0 + idim.padX >= 0);
			assert(p0 + idim.padY >= 0);
			for(int iy=p0;iy<p1;iy++) {
			for(int ix=p0;ix<p1;ix++) {
				const int ic = oc;
				offsets.push_back(idim.get_offset_padded(ix, iy, ic));
			}}
		}
		if(pattern.type == LAYER_TYPE::ENCODE_MIX) {
			const int A = pattern.A;
			const int B = pattern.B;
			const int N = pattern.N;
			assert(A > 0);
			assert(B > 0);
			assert(N > 0);
			assert(N % A == 0);
			const int p0 = (ox / B) * A + (A/2) - (N/2);
			const int p1 = p0 + N;
			assert(p0 + idim.padX >= 0);
			assert(p0 + idim.padY >= 0);
			for(int iy=p0;iy<p1;iy++) {
			for(int ix=p0;ix<p1;ix++) {
			for(int ic=0;ic<idim.innerC();ic++) {
				offsets.push_back(idim.get_offset_padded(ix, iy, ic));
			}}}
		}
		assert(offsets.size() > 0);
		return offsets;
	}

	/*
		this is combined with the offsets kernel to get
		the input-indices of neurons to read from.
	*/
	int get_input_kernel_offset(const layer_pattern pattern, const dim_t idim, const int ox, const int oy, const int oc) {
		assert(pattern.type != LAYER_TYPE::NONE);
		const int origin = idim.get_offset_padded(0, 0, 0);
		int ix0;
		int iy0;
		int ic0;
		if(pattern.type == LAYER_TYPE::DENSE) {
			ix0 = 0;
			iy0 = 0;
			ic0 = 0;
		}
		if(pattern.type == LAYER_TYPE::ENCODE) {
			const int A = pattern.A;
			const int B = pattern.B;
			ix0 = (ox / B) * A;
			iy0 = (oy / B) * A;
			ic0 = 0;
		}
		if(pattern.type == LAYER_TYPE::SPATIAL_MIX) {
			const int N = pattern.N;
			const int M = pattern.M;
			ix0 = (ox - (ox % M)) + (M/2) - (N/2);
			iy0 = (oy - (oy % M)) + (M/2) - (N/2);
			ic0 = oc;
		}
		if(pattern.type == LAYER_TYPE::ENCODE_MIX) {
			const int A = pattern.A;
			const int B = pattern.B;
			const int N = pattern.N;
			ix0 = (ox / B) * A + (A/2) - (N/2);
			iy0 = (oy / B) * A + (A/2) - (N/2);
			ic0 = 0;
		}
		return idim.get_offset_padded(ix0, iy0, ic0) - origin;
	}

	struct init_targets_return_t {
		fw_target_list fw_targets;
		bp_target_list bp_targets;
	};
	// TODO - review this algorithm for logic errors.
	init_targets_return_t init_targets(const layer_pattern pattern, const dim_t idim, const dim_t odim) {
		fw_target_list fw_targets;
		bp_target_list bp_targets;

		// create targets, counting how many belong to each input-neuron.
		struct Pair {
			bp_target target;
			int input_neuron;
			int ft_index;
		};
		vector<Pair> pairs;
		vector<int> counts(idim.outer_length(), 0);// number of targets per input-neuron.
		const vector<int> kernel = get_input_neuron_offsets_kernel(pattern, idim);
		for(int oy=0;oy<odim.innerY();oy++) {
		for(int ox=0;ox<odim.innerX();ox++) {
		for(int oc=0;oc<odim.innerC();oc++) {
			const int out_n = odim.get_offset_padded(ox, oy, oc);
			const int koffset = get_input_kernel_offset(pattern, idim, ox, oy, oc);
			for(int i=0;i<kernel.size();i++) {
				const int in_n = kernel[i] + koffset;
				Pair pair;
				pair.target = bp_target{ 0.0f, 0.0f, out_n };
				pair.input_neuron = in_n;
				pair.ft_index = fw_targets.targets.size();
				pairs.push_back(pair);
				fw_targets.targets.push_back(fw_target{ 0.0f });
				counts[in_n]++;
			}
		}}}

		// compute postfix sums.
		bp_targets.intervals = image_i(idim);
		int s = 0;
		for(int x=0;x<counts.size();x++) { s+=counts[x]; bp_targets.intervals.data[x] = s; }

		// gather targets.
		bp_targets.targets.resize(s);
		bp_targets.ft_indices.resize(s);
		for(const Pair& pair : pairs) {
			const int end = bp_targets.intervals.data[pair.input_neuron];
			const int ind = end - counts[pair.input_neuron];
			bp_targets.targets[ind] = pair.target;
			bp_targets.ft_indices[ind] = pair.ft_index;
			counts[pair.input_neuron]--;
		}

		return init_targets_return_t{ fw_targets, bp_targets };
	}

}
