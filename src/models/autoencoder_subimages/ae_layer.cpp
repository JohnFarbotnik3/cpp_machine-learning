
#include <cassert>
#include <random>
#include <thread>
#include "src/utils/vector_util.cpp"
#include "types.cpp"
#include "subimage.cpp"

namespace ML::models::autoencoder_subimage {
	using namespace utils::vector_util;

	struct ae_layer {
		layer_pattern pattern;
		simple_dim_t idim;// overall dimensions of input.
		simple_dim_t odim;// overall dimensions of output.
		padded_dim_t subidim;// input  dimensions of subimages.
		simple_dim_t subodim;// output dimensions of subimages.
		vector<subimage> subimages;
		int subimage_grid_X;// number of subimages along X-axis.
		int subimage_grid_Y;// number of subimages along Y-axis.

		ae_layer(const simple_dim_t idim, const simple_dim_t odim, const layer_pattern pattern, const int grid_X, const int grid_Y) :
			idim(idim), odim(odim), pattern(pattern), subimage_grid_X(grid_X), subimage_grid_Y(grid_Y)
		{
			// assert that image cleanly divides into subimages.
			assert(idim.X % grid_X == 0);
			assert(idim.Y % grid_Y == 0);
			assert(odim.X % grid_X == 0);
			assert(odim.Y % grid_Y == 0);

			// determine amount of padding.
			const int A = pattern.A;
			const int B = pattern.B;
			const int N = pattern.N;
			int pad = 0;
			if(pattern.type == LAYER_TYPE::DENSE) {
				assert(odim.equals(idim));
			}
			if(pattern.type == LAYER_TYPE::ENCODE) {
				assert((odim.X / B) == (idim.X / A));
				assert((odim.Y / B) == (idim.Y / A));
			}
			if(pattern.type == LAYER_TYPE::SPATIAL_MIX) {
				assert(odim.equals(idim));
				assert((N - B) % 2 == 0);
				pad = (N/2) - (B/2);
			}
			if(pattern.type == LAYER_TYPE::ENCODE_MIX) {
				assert((odim.X / B) == (idim.X / A));
				assert((odim.Y / B) == (idim.Y / A));
				assert((N - A) % 2 == 0);
				pad = (N/2) - (A/2);
			}

			// create grid of subimages.
			subidim = padded_dim_t(idim.X/grid_X, idim.Y/grid_Y, idim.C, pad);
			subodim = simple_dim_t(odim.X/grid_X, odim.Y/grid_Y, odim.C);
			for(int y=0;y<grid_Y;y++) {
			for(int x=0;x<grid_X;x++) {
				subimages.push_back(subimage(subidim, subodim, pattern));
			}}
		}

		// ============================================================
		// subimage manipulation.
		// ------------------------------------------------------------

		static void subimage_load(const simple_image_f& img, padded_image_f& sub, const int image_x0, const int image_y0) {
			const simple_dim_t imgdim = img.dim;
			const padded_dim_t subdim = sub.dim;
			const int ch = img.dim.C;
			      float* subdata = sub.data.data();
			const float* imgdata = img.data.data();

			// load inner part of subimage.
			const int ROW_SZ = sizeof(imgdata[0]) * ch * subdim.innerX();
			for(int sy=0;sy<subdim.innerY();sy++) {
				const int i0 = imgdim.get_offset(image_x0, image_y0+sy, 0);
				const int s0 = subdim.get_offset_padded(0, sy, 0);
				memcpy(subdata+s0, imgdata+i0, ROW_SZ);
			}

			// load padding part of subimage.
			const int pad = subdim.pad;
			const int PIXEL_SZ = sizeof(imgdata[0]) * ch;
			for(int sy=-pad;sy<subdim.innerY()+pad;sy++) {
			for(int sx=-pad;sx<subdim.innerX()+pad;sx++) {
				// NOTE: skip to end of inner-row when arriving at start of inner-row.
				if(subdim.is_within_inner_bounds(sx, sy) && sx==0) sx=subdim.innerX();
				const int ix = image_x0 + sx;
				const int iy = image_y0 + sy;
				if(imgdim.is_within_bounds(ix, iy)) {
					const int i0 = imgdim.get_offset(ix, iy, 0);
					const int s0 = subdim.get_offset_padded(sx, sy, 0);
					memcpy(subdata+s0, imgdata+i0, PIXEL_SZ);
				}
			}}
		}
		static void subimage_load(const simple_image_f& img, simple_image_f& sub, const int image_x0, const int image_y0) {
			const simple_dim_t imgdim = img.dim;
			const simple_dim_t subdim = sub.dim;
			const int ch = img.dim.C;
			      float* subdata = sub.data.data();
			const float* imgdata = img.data.data();

			// load inner part of subimage.
			const int ROW_SZ = sizeof(imgdata[0]) * ch * subdim.X;
			for(int sy=0;sy<subdim.Y;sy++) {
				const int i0 = imgdim.get_offset(image_x0, image_y0+sy, 0);
				const int s0 = subdim.get_offset(0, sy, 0);
				memcpy(subdata+s0, imgdata+i0, ROW_SZ);
			}
		}
		static void subimage_save_inner(simple_image_f& img, const padded_image_f& sub, const int image_x0, const int image_y0) {
			const simple_dim_t imgdim = img.dim;
			const padded_dim_t subdim = sub.dim;
			const int ch = img.dim.C;
			const float* subdata = sub.data.data();
			      float* imgdata = img.data.data();

			// save inner part of subimage.
			const int ROW_SZ = sizeof(imgdata[0]) * ch * subdim.innerX();
			for(int sy=0;sy<subdim.innerY();sy++) {
				const int i0 = imgdim.get_offset(image_x0, image_y0+sy, 0);
				const int s0 = subdim.get_offset_padded(0, sy, 0);
				memcpy(imgdata+i0, subdata+s0, ROW_SZ);
			}
		}
		static void subimage_save_inner(simple_image_f& img, const simple_image_f& sub, const int image_x0, const int image_y0) {
			const simple_dim_t imgdim = img.dim;
			const simple_dim_t subdim = sub.dim;
			const int ch = img.dim.C;
			const float* subdata = sub.data.data();
			      float* imgdata = img.data.data();

			// save inner part of subimage.
			const int ROW_SZ = sizeof(imgdata[0]) * ch * subdim.X;
			for(int sy=0;sy<subdim.Y;sy++) {
				const int i0 = imgdim.get_offset(image_x0, image_y0+sy, 0);
				const int s0 = subdim.get_offset(0, sy, 0);
				memcpy(imgdata+i0, subdata+s0, ROW_SZ);
			}
		}
		static void subimage_save_outer(simple_image_f& img, const padded_image_f& sub, const int image_x0, const int image_y0) {
			const simple_dim_t imgdim = img.dim;
			const padded_dim_t subdim = sub.dim;
			const int ch = img.dim.C;
			const float* subdata = sub.data.data();
			      float* imgdata = img.data.data();

			// save padding part of subimage.
			// NOTE: this increments rather than overwrites content.
			const int pad = subdim.pad;
			for(int sy=-pad;sy<subdim.innerY()+pad;sy++) {
			for(int sx=-pad;sx<subdim.innerX()+pad;sx++) {
				// NOTE: skip to end of inner-row when arriving at start of inner-row.
				if(subdim.is_within_inner_bounds(sx, sy) && sx==0) sx=subdim.innerX();
				const int ix = image_x0 + sx;
				const int iy = image_y0 + sy;
				if(imgdim.is_within_bounds(ix, iy)) {
					const int i0 = imgdim.get_offset(ix, iy, 0);
					const int s0 = subdim.get_offset_padded(sx, sy, 0);
					for(int c=0;c<ch;c++) imgdata[i0+c] += subdata[s0+c];
				}
			}}
		}

		void foreward_propagate_subimages_load(const simple_image_f& input_values) {
			for(int gy=0;gy<subimage_grid_Y;gy++) {
			for(int gx=0;gx<subimage_grid_X;gx++) {
				const int image_x0 = (gx * idim.X) / subimage_grid_X;
				const int image_y0 = (gy * idim.Y) / subimage_grid_Y;
				padded_image_f& sub = subimages[gy*subimage_grid_X + gx].value_image_i;
				subimage_load(input_values, sub, image_x0, image_y0);
			}}
		}
		void foreward_propagate_subimages_save(simple_image_f& output_values) {
			for(int gy=0;gy<subimage_grid_Y;gy++) {
			for(int gx=0;gx<subimage_grid_X;gx++) {
				const int image_x0 = (gx * idim.X) / subimage_grid_X;
				const int image_y0 = (gy * idim.Y) / subimage_grid_Y;
				const simple_image_f& sub = subimages[gy*subimage_grid_X + gx].value_image_o;
				subimage_save_inner(output_values, sub, image_x0, image_y0);
			}}
		}

		void backward_propagate_subimages_load(const simple_image_f& output_error) {
			for(int gy=0;gy<subimage_grid_Y;gy++) {
			for(int gx=0;gx<subimage_grid_X;gx++) {
				const int image_x0 = (gx * idim.X) / subimage_grid_X;
				const int image_y0 = (gy * idim.Y) / subimage_grid_Y;
				simple_image_f& sub = subimages[gy*subimage_grid_X + gx].error_image_o;
				subimage_load(output_error, sub, image_x0, image_y0);
			}}
		}
		void backward_propagate_subimages_save(simple_image_f& input_error) {
			for(int gy=0;gy<subimage_grid_Y;gy++) {
			for(int gx=0;gx<subimage_grid_X;gx++) {
				const int image_x0 = (gx * idim.X) / subimage_grid_X;
				const int image_y0 = (gy * idim.Y) / subimage_grid_Y;
				const padded_image_f& sub = subimages[gy*subimage_grid_X + gx].error_image_i;
				subimage_save_inner(input_error, sub, image_x0, image_y0);
			}}
			for(int gy=0;gy<subimage_grid_Y;gy++) {
			for(int gx=0;gx<subimage_grid_X;gx++) {
				const int image_x0 = (gx * idim.X) / subimage_grid_X;
				const int image_y0 = (gy * idim.Y) / subimage_grid_Y;
				const padded_image_f& sub = subimages[gy*subimage_grid_X + gx].error_image_i;
				subimage_save_outer(input_error, sub, image_x0, image_y0);
			}}
		}

		//TODO (but only if its worth the significant increase in complexity)
		void foreward_propagate_subimages_load_middle(const ae_layer& prev_layer) {}
		void backward_propagate_subimages_load_middle(const ae_layer& next_layer) {}

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

		static void propagate_func(ae_layer& layer, const int beg, const int end) {
			for(int x=beg;x<end;x++) layer.subimages[x].foreward_propagate();
		}
		void propagate(const int n_threads, const simple_image_f& input_values, simple_image_f& output_values) {
			foreward_propagate_subimages_load(input_values);
			vector<std::thread> threads;
			for(int z=0;z<n_threads;z++) {
				const int beg = (subimages.size() * (z+0)) / n_threads;
				const int end = (subimages.size() * (z+1)) / n_threads;
				threads.push_back(std::thread(propagate_func, std::ref(*this), beg, end));
			}
			for(int z=0;z<n_threads;z++) threads[z].join();
			foreward_propagate_subimages_save(output_values);
		}

		static void back_propagate_func(ae_layer& layer, const int beg, const int end) {
			for(int x=beg;x<end;x++) layer.subimages[x].backward_propagate();
		}
		void back_propagate(const int n_threads, simple_image_f& input_error, const simple_image_f& output_error) {
			backward_propagate_subimages_load(output_error);
			vector<std::thread> threads;
			for(int z=0;z<n_threads;z++) {
				const int beg = (subimages.size() * (z+0)) / n_threads;
				const int end = (subimages.size() * (z+1)) / n_threads;
				threads.push_back(std::thread(back_propagate_func, std::ref(*this), beg, end));
			}
			for(int z=0;z<n_threads;z++) threads[z].join();
			backward_propagate_subimages_save(input_error);

			// normalize input-error against output-error to have same average gradient per-neuron.
			const float  in_sum = vec_sum_abs_mt( input_error.data, 0,  input_error.data.size(), n_threads);
			const float out_sum = vec_sum_abs_mt(output_error.data, 0, output_error.data.size(), n_threads);
			float mult = (out_sum / in_sum) * (float(input_error.data.size()) / float(output_error.data.size()));
			//printf("error: isum=%f, osum=%f\n", in_sum, out_sum);
			assert(out_sum > 0.0f);
			assert( in_sum > 0.0f);
			vec_mult_mt(input_error.data, mult, 0, input_error.data.size(), n_threads);
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
	};

}
