
#include <cassert>
#include <random>
#include <thread>
#include "types.cpp"
#include "subimage.cpp"

namespace ML::models::autoencoder_subimage {
	struct ae_layer {
		vector<subimage> subimages;
		int subimage_grid_X;// number of subimages along X-axis.
		int subimage_grid_Y;// number of subimages along Y-axis.
		layer_pattern pattern;
		simd_image_8f signal_o;
		simd_image_8f value_o;
		simd_image_8f error_o;

		ae_layer(
			const simd_image_8f_dimensions idim,
			const simd_image_8f_dimensions odim,
			const int grid_X,
			const int grid_Y,
			const layer_pattern pattern
		) :
			signal_o(odim),
			value_o(odim),
			error_o(odim)
		{
			this->pattern = pattern;
			this->subimage_grid_X = grid_X;
			this->subimage_grid_Y = grid_Y;

			// assert that image cleanly divides into subimages.
			assert(idim.X % grid_X == 0);
			assert(idim.Y % grid_Y == 0);
			assert(odim.X % grid_X == 0);
			assert(odim.Y % grid_Y == 0);

			// create grid of subimages.
			value_image_dimensions subodim(odim.X/grid_X, odim.Y/grid_Y, odim.C);
			for(int y=0;y<grid_Y;y++) {
			for(int x=0;x<grid_X;x++) {
				image_bounds bounds_i;
				bounds_i.x0 = ((x+0) * idim.X) / grid_X;
				bounds_i.x1 = ((x+1) * idim.X) / grid_X;
				bounds_i.y0 = ((y+0) * idim.Y) / grid_Y;
				bounds_i.y1 = ((y+1) * idim.Y) / grid_Y;
				image_bounds bounds_o;
				bounds_o.x0 = ((x+0) * odim.X) / grid_X;
				bounds_o.x1 = ((x+1) * odim.X) / grid_X;
				bounds_o.y0 = ((y+0) * odim.Y) / grid_Y;
				bounds_o.y1 = ((y+1) * odim.Y) / grid_Y;
				subimages.push_back(subimage(pattern, idim, subodim, bounds_i, bounds_o));
			}}
		}

		// ============================================================
		// network functions
		// ------------------------------------------------------------

		void init_model_parameters(int seed, float bias_mean, float bias_stddev, float weight_mean, float weight_stddev) {
			std::mt19937 gen32 = utils::random::get_generator_32(seed);
			std::uniform_int_distribution<int> distr = utils::random::rand_uniform_int<int>(INT_MIN, INT_MAX);
			for(int x=0;x<subimages.size();x++) {
				const int new_seed = distr(gen32);
				subimages[x].init_model_parameters(new_seed, bias_mean, bias_stddev, weight_mean, weight_stddev);
			}
		}

		static void propagate_func(ae_layer& layer, const int beg, const int end, const simd_image_8f& value_i) {
			for(int x=beg;x<end;x++) layer.subimages[x].foreward_propagate(value_i, layer.value_o, layer.signal_o);
		}
		void propagate(const int n_threads, const simd_image_8f& value_i) {
			vector<std::thread> threads;
			for(int z=0;z<n_threads;z++) {
				const int beg = (subimages.size() * (z+0)) / n_threads;
				const int end = (subimages.size() * (z+1)) / n_threads;
				threads.push_back(std::thread(propagate_func, std::ref(*this), beg, end, std::ref(value_i)));
			}
			for(int z=0;z<n_threads;z++) threads[z].join();
		}

		static void back_propagate_func(ae_layer& layer, const int beg, const int end, simd_image_8f& error_i, const simd_image_8f& value_i) {
			for(int x=beg;x<end;x++) layer.subimages[x].backward_propagate(error_i, layer.error_o, value_i, layer.signal_o);
		}
		void back_propagate(const int n_threads, simd_image_8f& error_i, const simd_image_8f& value_i) {
			vector<std::thread> threads;
			for(int z=0;z<n_threads;z++) {
				const int beg = (subimages.size() * (z+0)) / n_threads;
				const int end = (subimages.size() * (z+1)) / n_threads;
				threads.push_back(std::thread(back_propagate_func, std::ref(*this), beg, end, std::ref(error_i), std::ref(value_i)));
			}
			for(int z=0;z<n_threads;z++) threads[z].join();
			for(int x=0;x<subimages.size();x++) subimages[x].commit_extra_error(error_i);

			// normalize input-error against output-error to have same average gradient per-neuron.
			const float  in_sum = simd_reduce_abs_mt(error_i.data.data(), error_i.data.size(), n_threads);
			const float out_sum = simd_reduce_abs_mt(error_o.data.data(), error_o.data.size(), n_threads);
			float mult = (out_sum / in_sum) * (float(error_i.data.size()) / float(error_o.data.size()));
			printf("error: in_sum=%f, out_sum=%f\n", in_sum, out_sum);
			assert(out_sum > 0.0f);
			assert( in_sum > 0.0f);
			simd_scale_mt(error_i.data.data(), error_i.data.size(), n_threads, mult);
		}

		static void apply_batch_error_biases_func(ae_layer& layer, const int beg, const int end, const float adjustment_rate) {
			for(int x=beg;x<end;x++) layer.subimages[x].apply_batch_error_biases(adjustment_rate);
		}
		static void apply_batch_error_weights_func(ae_layer& layer, const int beg, const int end, const float adjustment_rate) {
			for(int x=beg;x<end;x++) layer.subimages[x].apply_batch_error_weights(adjustment_rate);
		}
		void apply_batch_error_biases(const int n_threads, const int batch_size, const float learning_rate) {
			const float adjustment_rate = learning_rate / batch_size;
			vector<std::thread> threads;
			for(int z=0;z<n_threads;z++) {
				const int beg = (subimages.size() * (z+0)) / n_threads;
				const int end = (subimages.size() * (z+1)) / n_threads;
				threads.push_back(std::thread(apply_batch_error_biases_func, std::ref(*this), beg, end, adjustment_rate));
			}
			for(int z=0;z<n_threads;z++) threads[z].join();
		}
		void apply_batch_error_weights(const int n_threads, const int batch_size, const float learning_rate) {
			const float adjustment_rate = learning_rate / batch_size;
			vector<std::thread> threads;
			for(int z=0;z<n_threads;z++) {
				const int beg = (subimages.size() * (z+0)) / n_threads;
				const int end = (subimages.size() * (z+1)) / n_threads;
				threads.push_back(std::thread(apply_batch_error_weights_func, std::ref(*this), beg, end, adjustment_rate));
			}
			for(int z=0;z<n_threads;z++) threads[z].join();
		}

		void clear_batch_error() {
			for(int x=0;x<subimages.size();x++) {
				subimages[x].clear_batch_error_biases();
				subimages[x].clear_batch_error_weights();
			}
		}

		// ============================================================
		// stats helper functions.
		// ------------------------------------------------------------

		vector<float> get_biases() {
			vector<float> arr(subimages[0].biases.data.size() * subimage_grid_X * subimage_grid_Y);
			for(int y=0;y<subimage_grid_Y;y++) {
			for(int x=0;x<subimage_grid_X;x++) {
				const value_image<float>& biases = subimages[y*subimage_grid_X + x].biases;
				const int i0 = (y*subimage_grid_X + x) * biases.data.size();
				memcpy(arr.data()+i0, biases.data.data(), biases.data.size() * sizeof(float));
			}}
			return arr;
		}

		vector<float> get_weights() {
			vector<float> arr(subimages[0].weights.size() * subimage_grid_X * subimage_grid_Y);
			for(int y=0;y<subimage_grid_Y;y++) {
			for(int x=0;x<subimage_grid_X;x++) {
				const vector<float>& weights = subimages[y*subimage_grid_X + x].weights;
				const int i0 = (y*subimage_grid_X + x) * weights.size();
				memcpy(arr.data()+i0, weights.data(), weights.size() * sizeof(float));
			}}
			return arr;
		}
	};

}
