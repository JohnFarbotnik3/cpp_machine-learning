
#include <vector>
#include "types.cpp"

namespace ML::models::autoencoder_subimage {
	using std::vector;

	input_neuron_offset_struct get_input_neuron_offsets_kernel(const layer_pattern pattern, const padded_dim_t idim, const simple_dim_t odim) {
		const int A = pattern.A;
		const int B = pattern.B;
		const int N = pattern.N;

		input_neuron_offset_struct data;
		vector<int>&	kernel  = data.kernel;
		simple_image_i&	offsets = data.kernel_offsets;
		offsets = simple_image_i(odim);

		assert(pattern.type != LAYER_TYPE::NONE);

		if(pattern.type == LAYER_TYPE::DENSE) {
			for(int iy=0;iy<idim.innerY();iy++) {
			for(int ix=0;ix<idim.innerX();ix++) {
			for(int ic=0;ic<idim.C;ic++) {
				kernel.push_back(idim.get_offset_padded(ix, iy, ic));
			}}}

			offsets.clear();
		}

		if(pattern.type == LAYER_TYPE::ENCODE) {
			for(int iy=0;iy<A;iy++) {
			for(int ix=0;ix<A;ix++) {
			for(int ic=0;ic<idim.C;ic++) {
				kernel.push_back(idim.get_offset_padded(ix, iy, ic));
			}}}

			const int origin = idim.get_offset_padded(0, 0, 0);
			for(int oy=0;oy<odim.Y;oy++) {
			for(int ox=0;ox<odim.X;ox++) {
			for(int oc=0;oc<odim.C;oc++) {
				const int ix0 = (ox / B) * A;
				const int iy0 = (oy / B) * A;
				const int ic0 = 0;
				offsets.data[offsets.dim.get_offset(ox, oy, oc)] = idim.get_offset_padded(ix0, iy0, ic0) - origin;
			}}}
		}

		if(pattern.type == LAYER_TYPE::SPATIAL_MIX) {
			assert(idim.C == odim.C);
			const int p0 = (B/2) - (N/2);
			const int p1 = p0 + N;
			assert(p0 + idim.pad == 0);// assert that there is exactly the correct amount of padding.

			for(int iy=p0;iy<p1;iy++) {
			for(int ix=p0;ix<p1;ix++) {
				const int ic = 0;
				kernel.push_back(idim.get_offset_padded(ix, iy, ic));
			}}

			const int origin = idim.get_offset_padded(p0, p0, 0);
			for(int oy=0;oy<odim.Y;oy++) {
			for(int ox=0;ox<odim.X;ox++) {
			for(int oc=0;oc<odim.C;oc++) {
				const int ix0 = (ox / B) * B + (B/2) - (N/2);
				const int iy0 = (oy / B) * B + (B/2) - (N/2);
				const int ic0 = oc;
				offsets.data[offsets.dim.get_offset(ox, oy, oc)] = idim.get_offset_padded(ix0, iy0, ic0) - origin;
			}}}
		}

		if(pattern.type == LAYER_TYPE::ENCODE_MIX) {
			assert(N % A == 0);
			const int p0 = (A/2) - (N/2);
			const int p1 = p0 + N;
			assert(p0 + idim.pad == 0);// assert that there is exactly the correct amount of padding.

			for(int iy=p0;iy<p1;iy++) {
			for(int ix=p0;ix<p1;ix++) {
			for(int ic=0;ic<idim.C;ic++) {
				kernel.push_back(idim.get_offset_padded(ix, iy, ic));
			}}}

			const int origin = idim.get_offset_padded(p0, p0, 0);
			for(int oy=0;oy<odim.Y;oy++) {
			for(int ox=0;ox<odim.X;ox++) {
			for(int oc=0;oc<odim.C;oc++) {
				const int ix0 = (ox / B) * A + (A/2) - (N/2);
				const int iy0 = (oy / B) * A + (A/2) - (N/2);
				const int ic0 = 0;
				offsets.data[offsets.dim.get_offset(ox, oy, oc)] = idim.get_offset_padded(ix0, iy0, ic0) - origin;
			}}}
		}

		return data;
	}
}






