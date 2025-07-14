
#include <vector>
#include "types.cpp"

namespace ML::models::autoencoder_subimage {
	using std::vector;

	/*
		generate list of input-neuron offsets from perspective of output-neuron [0,0,0];
		for use during foreward-propagation.
	*/
	vector<int> get_input_neuron_offsets(const layer_pattern pattern, const dim_t idim) {
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
				offsets.push_back(idim.get_offset(ix, iy, ic));
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
				offsets.push_back(idim.get_offset(ix, iy, ic));
			}}}
		}
		if(pattern.type == LAYER_TYPE::SPATIAL_MIX) {
			const int N = pattern.N;
			const int M = pattern.M;
			assert(N > 0);
			assert(M > 0);
			const int p0 = (ox / M) * M + (M/2) - (N/2);
			const int p1 = p0 + N;
			assert(p0 + idim.padX >= 0);
			assert(p0 + idim.padY >= 0);
			for(int iy=p0;iy<p1;iy++) {
			for(int ix=p0;ix<p1;ix++) {
				const int ic = oc;
				offsets.push_back(idim.get_offset(ix, iy, ic));
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
			const int p0 = (ox / B) * B + (B/2) - (N/2);
			const int p1 = p0 + N;
			assert(p0 + idim.padX >= 0);
			assert(p0 + idim.padY >= 0);
			for(int iy=p0;iy<p1;iy++) {
			for(int ix=p0;ix<p1;ix++) {
			for(int ic=0;ic<idim.innerC();ic++) {
				offsets.push_back(idim.get_offset(ix, iy, ic));
			}}}
		}
		assert(offsets.size() > 0);
		return offsets;
	}

	/* populate output-neuron indices of backprop targets. */
	vector<int> init_output_neuron_indices(const layer_pattern pattern, const dim_t idim, const dim_t odim, vector<bp_target>& bp_targets) {
		vector<int> counts(N_INPUT_NEURONS, 0);// current number of targets each input-neuron has.
		vector<int> offsets = get_input_neuron_offsets(pattern, idim);
		// TODO
	}
}
