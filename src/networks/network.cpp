
#include <vector>
namespace ML::networks {
	struct network {
		/* run forward-propagation in the network. */
		virtual void propagate(std::vector<float>& input, std::vector<float>& output);

		/* adjust weights and biases according to batch loss. */
		virtual void apply_batch_loss();
		/* clear accumulated batch loss before running next batch. */
		virtual void reset_batch_loss();
		/* randomly modify model parameters. */
		virtual void anneal(float rate);
		/* back-propagate loss through network stochastically. */
		virtual void back_anneal(float rate, std::vector<float>& input_loss, std::vector<float>& output_loss);
		/* back-propagate loss through network. */
		virtual void back_propagate(float rate, std::vector<float>& input_loss, std::vector<float>& output_loss);

		/* remove unneeded connections. */
		virtual void prune_connections();
		/* remove unneeded neurons. */
		virtual void prune_neurons();
	};
}














