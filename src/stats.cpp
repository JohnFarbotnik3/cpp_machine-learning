
#include <vector>

namespace ML {
	using std::vector;

	struct loss_stats {
		vector<float> loss_iteration;
		vector<float> loss_minibatch;
		vector<float> loss_batch;
	};
};
