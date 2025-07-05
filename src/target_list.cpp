
#include <vector>

namespace ML::target_list {

	// target pointing to neuron in previous layer, used for foreward-propagation.
	struct foreward_target {
		int neuron_index	= 0;// index of input neuron.
		float weight		= 0;
	};

	// inverse of foreward_target, used for back-propagation.
	// points to neuron in next layer, as well as its associated foreward_target.
	struct backprop_target {
		int neuron_index	= 0;	// index of output neuron.
		int target_index	= 0;	// index of related foreward_target.
		float weight		= 0;	// copy of weight from corresponding foreward_target.
		float weight_error	= 0;	// error accumulator (total adjustment to apply to foreward_target weight).
	};

	// interval of targets in a target list.
	struct target_itv {
		int beg;
		int end;
	};

	// struct for managing targets.
	template<class target_t>
	struct target_list {
		/*
			offset of neuron[i]'s targets in target list.
			length of intervals are inferred by: LENGTH = offsets[i+1] - offsets[i].

			NOTE: length of this struct will be 1 greater than number of neuron-target-lists.
		*/
		std::vector<int>		offsets { 0 };
		std::vector<target_t>	targets;

		target_itv get_interval(int neuron_index) const {
			int beg = offsets[neuron_index];
			int end = offsets[neuron_index + 1];
			return target_itv { beg, end };
		}

		void push_list(std::vector<int> src_neuron_indices) {
			for(int index : src_neuron_indices) {
				foreward_target ft;
				ft.neuron_index = index;
				targets.push_back(ft);
			}
			offsets.push_back(targets.size());
		}

		/*
		TODO:
		- add functions for adding and removing targets from lists.
		* due to way offsets are stored, target list must stay consolidated,
			so ideally target lists will be modified in batches to reduce number of consolidations needed.
		* if foreward targets are updated, then backprop targets need to be updated too.
		...
		*/
	};

	// inverse of corresponding foreward_target_list.
	struct backprop_target_list : target_list<backprop_target> {};

	// list of targets used for foreward propagation.
	struct foreward_target_list : target_list<foreward_target> {
		// synchronize weights of backprop targets with foreward targets.
		void store_weights(backprop_target_list& backprop_list) {
			for(int x=0;x<backprop_list.targets.size();x++) {
				backprop_target& bt = backprop_list.targets[x];
				foreward_target& ft = this->targets[bt.target_index];
				bt.weight = ft.weight;
			}
		}

		// commit weights from backprop targets to foreward targets.
		void load_weights(backprop_target_list& backprop_list) {
			for(int x=0;x<backprop_list.targets.size();x++) {
				backprop_target& bt = backprop_list.targets[x];
				foreward_target& ft = this->targets[bt.target_index];
				ft.weight = bt.weight;
			}
		}

		backprop_target_list get_inverse(const int input_length) {
			// get number of back-targets for each input neuron.
			std::vector<int> lengths(input_length);
			for(int x=0;x<input_length;x++) lengths[x] = 0;
			for(int x=0;x<this->targets.size();x++) lengths[this->targets[x].neuron_index]++;

			// initialize backprop target list.
			backprop_target_list list;
			list.offsets.resize(input_length + 1);
			list.targets.resize(this->targets.size());
			int position = 0;
			for(int x=0;x<input_length;x++) {
				list.offsets[x] = position;
				position += lengths[x];
			}

			// invert foreward targets map.
			for(int n=0;n<this->offsets.size();n++) {
				target_itv itv = this->get_interval(n);
				for(int x=itv.beg;x<itv.end;x++) {
					foreward_target& ft = this->targets[x];
					backprop_target& bt = list.targets[list.offsets[ft.neuron_index]++];
					bt.neuron_index = n;
					bt.target_index = x;
				}
			}

			// set offsets to their correct values.
			position = 0;
			for(int x=0;x<input_length;x++) {
				list.offsets[x] = position;
				position += lengths[x];
			}
			list.offsets[input_length] = list.targets.size();

			return list;
		}
	};

};

