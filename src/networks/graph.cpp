
#include <map>
#include <vector>
#include "./network.cpp"

/*
	graph networks act as a single layer of interconnected neurons,
	without any constraint on which direction signals propagate.

	neurons use the previous activation value of target-neurons
	instead of the current value in order to support recurrence cleanly.

	NOTES:
	- unlike the simple layer-network (see "layer.cpp"), this struct
	stores values/value-history internally in order to support multiple
	propagation steps per call to "propagate()".
	- the value history is needed so that backpropagation knows the value
	at each propagation step, in order to accurately compute errors & adjustments.
	- in much the same way as errors are normalized such that input error and output error match
	in layer-network implementation, the graph should keep track of total error distributed per-step
	so that it can be normalized against the error from previous step / output.
	this may benefit from a decaying scale factor, in similar fashion to reinforcement learning.
*/
namespace ML::networks {
	using index_t = unsigned int;
	using std::vector;
	using std::map;

	template<index_t MAX_CONNECTIONS>
	struct graph_neuron {
		float	value = 0;
		float	value_prev = 0;
		float	bias = 0;
		float	weights[MAX_CONNECTIONS];
		index_t	targets_len = 0;
		index_t	targets[MAX_CONNECTIONS];
	};

	/*
		a graph of inter-connected neurons with the following layout:
		|...input (fixed)...|...output (fixed)...|...general (resizable)...|
	*/
	template<index_t MAX_CONNECTIONS, int N_PROP_ITERATIONS>
	struct graph_network : network {
		index_t	input_beg;	// start index of input neurons.
		index_t	output_beg;	// start index of output neurons.
		index_t	general_beg;// start index of general neurons.
		index_t	length;		// number of neurons in network.
		vector<graph_neuron<MAX_CONNECTIONS>> neurons;

		graph_network(index_t input_len, index_t output_len, index_t general_len) {
			input_beg	= 0;
			output_beg	= input_beg + input_len;
			general_beg	= output_beg + output_len;
			length		= general_beg + general_len;
			neurons.resize(length);
		}

		void resize(const index_t newlen) {
			neurons.resize(newlen);
			length = newlen;
		}

		/* re-arranges neurons given an index-mapping: map<from,to>. */
		void move_neurons(const map<index_t,index_t>& mapping) {
			// copy neurons to new positions.
			index_t num = mapping.size();
			graph_neuron<MAX_CONNECTIONS> temp[num];
			{ int x = 0; for(const auto& [from, to]: mapping) temp[x++] = neurons[from]; }
			{ int x = 0; for(const auto& [from, to]: mapping) neurons[to] = temp[x++]; }

			// remap target indices.
			for(int x=0;x<length;x++) {
				graph_neuron<MAX_CONNECTIONS>& neuron = neurons[x];
				for(int y=0;y<neuron.targets_len;y++) neuron.targets[y] = mapping.at(neuron.targets[y]);
			}
		}

		//TODO
		//void create_neurons(int N) {}

		/* removes neurons from the graph, and consolidates remaining neurons. */
		void delete_neurons(vector<index_t> indices) {
			// create mapping such that neurons at end will overwrite neurons to remove.
			index_t end = length;
			map<index_t,index_t> mapping;
			for(const index_t index : indices) { end--; mapping[end] = index; }

			move_neurons(mapping);
			resize(end);
		}

		void propagate(std::vector<float>& input, std::vector<float>& output) override {
			/*
				TODO

				NOTE: the input neurons dont need to be wasted space;
				we start be reading the running the input-section of the graph,
				reading from connections by index like the rest of the graph neurons do
				but targeting input vector instead.
				NOTE: since the input wont change during multiple internal propagation-steps,
				we only need to update the rest of the graph when in the stepping loop.
			*/

			/*
			// make sure lengths match. - NOTE: we dont need to do that, see above...
			vector<neuron_t>  input_layer = layers[0];
			vector<neuron_t> output_layer = layers[layers.size() - 1];
			if( input.size() !=  input_layer.size()) { fprintf(stderr, "ERROR:  input lengths dont match: %u != %u\n",  input.size(),  input_layer.size()); return; }
			if(output.size() != output_layer.size()) { fprintf(stderr, "ERROR: output lengths dont match: %u != %u\n", output.size(), output_layer.size()); return; }
			*/
		}

		// TODO
		/*
			for backprop, instead of using depth first search, have the following values for each neuron:
				loss_sum - loss accumulated during this iteration.
				loss_sum_prev - loss accumulated during previous iteration.
				loss_sum_total - loss accumulated across all iterations.

			some combination of these values can be normalized or scaled
			(ex. exponential decay w.r.t iteration count) in order to produce well behaved training adjustments.

			WARNING: Im pretty sure that a value-history will be required for backpropagation.
			^ the value history can be stored directly in neurons for now.
			^ in the case of video, picking a small enough number of iterations per frame (ex. 4) should make this feasable.
				(this assumes that loss_backprop is ran every frame, which I'm thinking is the proper way to do it)

			TODO: start with annealing for now - in particular: use naive backprop,
			then use direction and rough magnitude for choosing random annealing value with multiplier [-0.2, 1.0].
		*/
	};
}












