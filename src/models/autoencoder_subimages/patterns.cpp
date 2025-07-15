
#include <vector>
#include "types.cpp"

namespace ML::models::autoencoder_subimage {
	using std::vector;

	neuron_offset_struct get_input_neuron_offsets_kernel(const layer_pattern pattern, const dim_t idim, const dim_t odim) {
		const int A = pattern.A;
		const int B = pattern.B;
		const int N = pattern.N;
		const int M = pattern.M;

		neuron_offset_struct data;
		vector<int>&	kernel  = data.kernel;
		image_i&		offsets = data.kernel_offsets;
		offsets = image_i(odim);

		assert(pattern.type != LAYER_TYPE::NONE);

		if(pattern.type == LAYER_TYPE::DENSE) {
			assert(A == 0);
			assert(B == 0);
			assert(N == 0);
			assert(M == 0);

			for(int iy=0;iy<idim.innerY();iy++) {
			for(int ix=0;ix<idim.innerX();ix++) {
			for(int ic=0;ic<idim.innerC();ic++) {
				kernel.push_back(idim.get_offset_padded(ix, iy, ic));
			}}}

			offsets.clear();
		}

		if(pattern.type == LAYER_TYPE::ENCODE) {
			assert(A > 0);
			assert(B > 0);
			assert(N == 0);
			assert(M == 0);

			for(int iy=0;iy<A;iy++) {
			for(int ix=0;ix<A;ix++) {
			for(int ic=0;ic<idim.innerC();ic++) {
				kernel.push_back(idim.get_offset_padded(ix, iy, ic));
			}}}

			const int origin = idim.get_offset_padded(0, 0, 0);
			for(int oy=0;oy<odim.innerY();oy++) {
			for(int ox=0;ox<odim.innerX();ox++) {
			for(int oc=0;oc<odim.innerC();oc++) {
				const int ix0 = (ox / B) * A;
				const int iy0 = (oy / B) * A;
				const int ic0 = 0;
				offsets.data[offsets.dim.get_offset_padded(ox, oy, oc)] = idim.get_offset_padded(ix0, iy0, ic0) - origin;
			}}}
		}

		if(pattern.type == LAYER_TYPE::SPATIAL_MIX) {
			assert(A == 0);
			assert(B == 0);
			assert(N > 0);
			assert(M > 0);
			assert(idim.innerC() == odim.innerC());
			const int p0 = (M/2) - (N/2);
			const int p1 = p0 + N;
			assert(p0 + idim.padX == 0);// assert that there is exactly the correct amount of padding.
			assert(p0 + idim.padY == 0);

			for(int iy=p0;iy<p1;iy++) {
			for(int ix=p0;ix<p1;ix++) {
				const int ic = 0;
				kernel.push_back(idim.get_offset_padded(ix, iy, ic));
			}}

			const int origin = idim.get_offset_padded(p0, p0, 0);
			for(int oy=0;oy<odim.innerY();oy++) {
			for(int ox=0;ox<odim.innerX();ox++) {
			for(int oc=0;oc<odim.innerC();oc++) {
				const int ix0 = (ox - (ox%M)) + (M/2) - (N/2);
				const int iy0 = (oy - (oy%M)) + (M/2) - (N/2);
				const int ic0 = oc;
				offsets.data[offsets.dim.get_offset_padded(ox, oy, oc)] = idim.get_offset_padded(ix0, iy0, ic0) - origin;
			}}}
		}

		if(pattern.type == LAYER_TYPE::ENCODE_MIX) {
			assert(A > 0);
			assert(B > 0);
			assert(N > 0);
			assert(M == 0);
			assert(N % A == 0);
			const int p0 = (A/2) - (N/2);
			const int p1 = p0 + N;
			assert(p0 + idim.padX == 0);// assert that there is exactly the correct amount of padding.
			assert(p0 + idim.padY == 0);

			for(int iy=p0;iy<p1;iy++) {
			for(int ix=p0;ix<p1;ix++) {
			for(int ic=0;ic<idim.innerC();ic++) {
				kernel.push_back(idim.get_offset_padded(ix, iy, ic));
			}}}

			const int origin = idim.get_offset_padded(p0, p0, 0);
			for(int oy=0;oy<odim.innerY();oy++) {
			for(int ox=0;ox<odim.innerX();ox++) {
			for(int oc=0;oc<odim.innerC();oc++) {
				const int ix0 = (ox / B) * A + (A/2) - (N/2);
				const int iy0 = (oy / B) * A + (A/2) - (N/2);
				const int ic0 = 0;
				offsets.data[offsets.dim.get_offset_padded(ox, oy, oc)] = idim.get_offset_padded(ix0, iy0, ic0) - origin;
			}}}
		}

		return data;
	}

	fw_target_list init_fw_targets(const neuron_offset_struct& offset_struct, const dim_t odim) {
		const vector<int>&	kernel = offset_struct.kernel;
		const vector<int>&	kernel_offsets = offset_struct.kernel_offsets.data;
		fw_target_list list;
		list.weights_per_output_neuron = kernel.size();

		for(int oy=0;oy<odim.innerY();oy++) {
		for(int ox=0;ox<odim.innerX();ox++) {
		for(int oc=0;oc<odim.innerC();oc++) {
			const int out_n = odim.get_offset_padded(ox, oy, oc);
			const int k_ofs = kernel_offsets[out_n];
			for(int i=0;i<kernel.size();i++) list.targets.push_back(fw_target{ 0.0f });
		}}}
		return list;
	}

	bp_target_list init_bp_targets(const neuron_offset_struct& offset_struct, const dim_t idim, const dim_t odim) {
		const vector<int>&	kernel = offset_struct.kernel;
		const vector<int>&	kernel_offsets = offset_struct.kernel_offsets.data;
		bp_target_list list;

		// counting how many targets will belong to each input-neuron.
		vector<int> counts(idim.outer_length(), 0);// number of targets per input-neuron.
		for(int oy=0;oy<odim.innerY();oy++) {
		for(int ox=0;ox<odim.innerX();ox++) {
		for(int oc=0;oc<odim.innerC();oc++) {
			const int out_n = odim.get_offset_padded(ox, oy, oc);
			const int k_ofs = kernel_offsets[out_n];
			for(int i=0;i<kernel.size();i++) {
				const int in_n = kernel[i] + k_ofs;
				counts[in_n]++;
			}
		}}}

		// compute postfix sums.
		list.intervals = image_i(idim);
		int e = 0;
		for(int x=0;x<counts.size();x++) { e+=counts[x]; list.intervals.data[x] = e; }
		list.targets.resize(e);

		// initialize array of target interval start pointers.
		vector<int> ptrs(idim.outer_length());
		int b = 0;
		for(int x=0;x<list.intervals.data.size();x++) { ptrs[x]=b; b=list.intervals.data[x]; }

		// create targets.
		for(int oy=0;oy<odim.innerY();oy++) {
		for(int ox=0;ox<odim.innerX();ox++) {
		for(int oc=0;oc<odim.innerC();oc++) {
			const int out_n = odim.get_offset_padded(ox, oy, oc);
			const int k_ofs = kernel_offsets[out_n];
			for(int i=0;i<kernel.size();i++) {
				const int in_n = kernel[i] + k_ofs;
				list.targets[ptrs[in_n]] = bp_target{ 0.0f, 0.0f, out_n };
				ptrs[in_n]++;
			}
		}}}

		return list;
	}

	void sync_weights_fw_to_bp(const neuron_offset_struct& offset_struct, const dim_t idim, const dim_t odim, const fw_target_list& fw_targets, bp_target_list& bp_targets) {
		// initialize array of target interval start pointers.
		vector<int> ptrs(idim.outer_length());
		int b = 0;
		for(int x=0;x<bp_targets.intervals.data.size();x++) { ptrs[x]=b; b=bp_targets.intervals.data[x]; }

		// create targets.
		const vector<int>&	kernel = offset_struct.kernel;
		const vector<int>&	kernel_offsets = offset_struct.kernel_offsets.data;
		for(int oy=0;oy<odim.innerY();oy++) {
		for(int ox=0;ox<odim.innerX();ox++) {
		for(int oc=0;oc<odim.innerC();oc++) {
			const int out_n = odim.get_offset_padded(ox, oy, oc);
			const int k_ofs = kernel_offsets[out_n];
			int ft_n = out_n * kernel.size();
			for(int i=0;i<kernel.size();i++) {
				const int in_n = kernel[i] + k_ofs;
				const fw_target& ft = fw_targets.targets[ft_n++];
				bp_target& bt = bp_targets.targets[ptrs[in_n]++];
				bt.weight = ft.weight;
			}
		}}}
	}

	void sync_weights_bp_to_fw(const neuron_offset_struct& offset_struct, const dim_t idim, const dim_t odim, fw_target_list& fw_targets, const bp_target_list& bp_targets) {
		// initialize array of target interval start pointers.
		vector<int> ptrs(idim.outer_length());
		int b = 0;
		for(int x=0;x<bp_targets.intervals.data.size();x++) { ptrs[x]=b; b=bp_targets.intervals.data[x]; }

		// create targets.
		const vector<int>&	kernel = offset_struct.kernel;
		const vector<int>&	kernel_offsets = offset_struct.kernel_offsets.data;
		for(int oy=0;oy<odim.innerY();oy++) {
		for(int ox=0;ox<odim.innerX();ox++) {
		for(int oc=0;oc<odim.innerC();oc++) {
			const int out_n = odim.get_offset_padded(ox, oy, oc);
			const int k_ofs = kernel_offsets[out_n];
			int ft_n = out_n * kernel.size();
			for(int i=0;i<kernel.size();i++) {
				const int in_n = kernel[i] + k_ofs;
				fw_target& ft = fw_targets.targets[ft_n++];
				const bp_target& bt = bp_targets.targets[ptrs[in_n]++];
				ft.weight = bt.weight;
			}
		}}}
	}
}






