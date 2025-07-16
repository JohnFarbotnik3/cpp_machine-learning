
#include <thread>
#include <vector>
#include "src/utils/vector_util.cpp"
#include "types.cpp"
#include "subimage.cpp"

namespace ML::models::autoencoder_subimage {
	using namespace utils::vector_util;

	struct ae_layer {
		layer_pattern pattern;
		dim_t idim;// overall dimensions of input.
		dim_t odim;// overall dimensions of output.
		dim_t subidim;// input  dimensions of subimages.
		dim_t subodim;// output dimensions of subimages.
		vector<subimage> subimages;
		int subimage_grid_X;// number of subimages along X-axis.
		int subimage_grid_Y;// number of subimages along Y-axis.

		// TODO - continue from here...

		ae_layer(const dim_t idim, const dim_t odim, const layer_pattern pattern, const int grid_X, const int grid_Y) : idim(idim), odim(odim), pattern(pattern) {
			// assert that overall input and output dimensions dont have padding.
			assert(!idim.has_padding());
			assert(!odim.has_padding());
			// assert that image cleanly divides into subimages.
			assert(idim.innerX() % grid_X == 0);
			assert(idim.innerY() % grid_Y == 0);
			assert(odim.innerX() % grid_X == 0);
			assert(odim.innerY() % grid_Y == 0);

			// TODO - create grid of subimages.

		}

		// ============================================================
		// subimage manipulation.
		// ------------------------------------------------------------

		static void subimage_load(const image_f& image, image_f& sub, const int image_x0, const int image_y0) {
			assert(image.dim.outerC() == sub.dim.outerC());
			const dim_t imgdim = image.dim;
			const dim_t subdim = sub.dim;
			const int ch = imgdim.outerC();
			float* subdata = sub.data.data();
			const float* imgdata = image.data.data();

			// load inner part of subimage.
			const int ROW_SZ = sizeof(imgdata[0]) * ch * subdim.innerX();
			for(int sy=0;sy<subdim.innerY();sy++) {
				const int i0 = imgdim.get_offset_padded(image_x0, image_y0+sy, 0);
				const int s0 = subdim.get_offset_padded(0, sy, 0);
				memcpy(subdata+s0, imgdata+i0, ROW_SZ);
			}

			// load padding part of subimage.
			const int pad = subdim.pad;
			const int PIXEL_SZ = sizeof(imgdata[0]) * ch;
			for(int sy=-pad;sy<subdim.innerY()+pad;sy++) {
			for(int sx=-pad;sx<subdim.innerX()+pad;sx++) {
				// NOTE: skip to end of inner-row when arriving at start of inner-row.
				if(subdim.is_within_inner_bounds(sx, sy, 0) && sx==0) sx=subdim.innerX();
				const int ix = image_x0 + sx;
				const int iy = image_y0 + sy;
				if(imgdim.is_within_inner_bounds(ix, iy, 0)) {
					const int i0 = imgdim.get_offset_padded(ix, iy, 0);
					const int s0 = subdim.get_offset_padded(sx, sy, 0);
					memcpy(subdata+s0, imgdata+i0, PIXEL_SZ);
				}
			}}
		}
		static void subimage_save_inner(image_f& image, const image_f& sub, const int image_x0, const int image_y0) {
			assert(image.dim.outerC() == sub.dim.outerC());
			const dim_t imgdim = image.dim;
			const dim_t subdim = sub.dim;
			const int ch = imgdim.outerC();
			const float* subdata = sub.data.data();
			float* imgdata = image.data.data();

			// save inner part of subimage.
			const int ROW_SZ = sizeof(imgdata[0]) * ch * subdim.innerX();
			for(int sy=0;sy<subdim.innerY();sy++) {
				const int i0 = imgdim.get_offset_padded(image_x0, image_y0+sy, 0);
				const int s0 = subdim.get_offset_padded(0, sy, 0);
				memcpy(imgdata+i0, subdata+s0, ROW_SZ);
			}
		}
		static void subimage_save_outer(image_f& image, const image_f& sub, const int image_x0, const int image_y0) {
			assert(image.dim.outerC() == sub.dim.outerC());
			const dim_t imgdim = image.dim;
			const dim_t subdim = sub.dim;
			const int ch = imgdim.outerC();
			const float* subdata = sub.data.data();
			float* imgdata = image.data.data();

			// save padding part of subimage.
			// NOTE: this increments rather than overwrites content.
			const int pad = subdim.pad;
			for(int sy=-pad;sy<subdim.innerY()+pad;sy++) {
			for(int sx=-pad;sx<subdim.innerX()+pad;sx++) {
				// NOTE: skip to end of inner-row when arriving at start of inner-row.
				if(subdim.is_within_inner_bounds(sx, sy, 0) && sx==0) sx=subdim.innerX();
				const int ix = image_x0 + sx;
				const int iy = image_y0 + sy;
				if(imgdim.is_within_inner_bounds(ix, iy, 0)) {
					const int i0 = imgdim.get_offset_padded(ix, iy, 0);
					const int s0 = subdim.get_offset_padded(sx, sy, 0);
					for(int c=0;c<ch;c++) imgdata[i0+c] += subdata[s0+c];
				}
			}}
		}

		void foreward_propagate_subimages_load(const image_f& input_values) {
			for(int gy=0;gy<subimage_grid_Y;gy++) {
			for(int gx=0;gx<subimage_grid_X;gx++) {
				const int image_x0 = (gx * idim.innerX()) / subimage_grid_X;
				const int image_y0 = (gy * idim.innerY()) / subimage_grid_Y;
				image_f& sub = subimages[gy*subimage_grid_X + gx].value_image_i;
				subimage_load(input_values, sub, image_x0, image_y0);
			}}
		}
		void foreward_propagate_subimages_save(image_f& output_values) {
			for(int gy=0;gy<subimage_grid_Y;gy++) {
			for(int gx=0;gx<subimage_grid_X;gx++) {
				const int image_x0 = (gx * idim.innerX()) / subimage_grid_X;
				const int image_y0 = (gy * idim.innerY()) / subimage_grid_Y;
				const image_f& sub = subimages[gy*subimage_grid_X + gx].value_image_o;
				subimage_save_inner(output_values, sub, image_x0, image_y0);
			}}
		}

		void backward_propagate_subimages_load(const image_f& output_error) {
			for(int gy=0;gy<subimage_grid_Y;gy++) {
			for(int gx=0;gx<subimage_grid_X;gx++) {
				const int image_x0 = (gx * idim.innerX()) / subimage_grid_X;
				const int image_y0 = (gy * idim.innerY()) / subimage_grid_Y;
				image_f& sub = subimages[gy*subimage_grid_X + gx].error_image_o;
				subimage_load(output_error, sub, image_x0, image_y0);
			}}
		}
		void backward_propagate_subimages_save(image_f& input_error) {
			for(int gy=0;gy<subimage_grid_Y;gy++) {
			for(int gx=0;gx<subimage_grid_X;gx++) {
				const int image_x0 = (gx * idim.innerX()) / subimage_grid_X;
				const int image_y0 = (gy * idim.innerY()) / subimage_grid_Y;
				const image_f& sub = subimages[gy*subimage_grid_X + gx].error_image_i;
				subimage_save_inner(input_error, sub, image_x0, image_y0);
			}}
			for(int gy=0;gy<subimage_grid_Y;gy++) {
			for(int gx=0;gx<subimage_grid_X;gx++) {
				const int image_x0 = (gx * idim.innerX()) / subimage_grid_X;
				const int image_y0 = (gy * idim.innerY()) / subimage_grid_Y;
				const image_f& sub = subimages[gy*subimage_grid_X + gx].error_image_i;
				subimage_save_outer(input_error, sub, image_x0, image_y0);
			}}
		}

		//TODO
		void foreward_propagate_subimages_load_middle(const ae_layer& prev_layer) {}
		void backward_propagate_subimages_load_middle(const ae_layer& next_layer) {}

		// ============================================================
		// network functions
		// ------------------------------------------------------------


		static void propagate_func(ae_layer& layer, const vector<float>& input_values, vector<float>& output_values, const image_area o_area) {
			const value_image_lines_dimensions& idim = layer.idim;
			const value_image_lines_dimensions& odim = layer.odim;
			const layer_pattern& scale = layer.pattern;
			const vector<float>& biases = layer.biases;
			vector<float>& signal = layer.signal;
			vector<float>& output = layer.output;
			const vector<float>& weights = layer.weights;
			const int WEIGHTS_PER_OUTPUT_NEURON = layer.weights_per_output_neuron();

			for(int oy=o_area.y0;oy<o_area.y1;oy++) {
			for(int ox=o_area.x0;ox<o_area.x1;ox++) {
				const image_area i_area = layer.table.get_input_area(ox, oy);
				for(int oc=0;oc<odim.C;oc++) {
					const int out_n = odim.get_offset(ox, oy, oc);
					float sum = biases[out_n];
					int w = out_n * WEIGHTS_PER_OUTPUT_NEURON;// initial weight index.
					for(int iy=i_area.y0;iy<i_area.y1;iy++) {
					for(int ix=i_area.x0;ix<i_area.x1;ix++) {
					if(!idim.is_within_bounds(ix, iy)) { w+=idim.C; continue; }
					for(int ic=0;ic<idim.C;ic++) {
						const int in_n = idim.get_offset(ix, iy, ic);
						sum += weights[w] * input_values[in_n];
						w++;
					}}}
					signal[out_n] = sum;
					output[out_n] = activation_func(sum);
				}
			}}

			// copy output values.
			for(int oy=o_area.y0;oy<o_area.y1;oy++) {
				const int i0 = odim.get_offset(o_area.x0, oy, 0);
				const int i1 = odim.get_offset(o_area.x1, oy, 0);
				vec_copy(output_values, layer.output, i0, i1);
			}
		}

		void propagate(const int n_threads, const std::vector<float>& input_values, std::vector<float>& output_values) {
			// spawn threads.
			vector<image_area> intervals = generate_intervals(n_threads, odim);
			vector<std::thread> threads;
			for(int z=0;z<intervals.size();z++) {
				threads.push_back(std::thread(propagate_func, std::ref(*this), std::ref(input_values), std::ref(output_values), intervals[z]));
			}
			for(int z=0;z<intervals.size();z++) threads[z].join();
		}

		static void back_propagate_func_output_side(ae_layer& layer, vector<float>& signal_error_terms, const vector<float>& output_error, const image_area o_area) {
			for(int y=o_area.y0;y<o_area.y1;y++) {
				const int i0 = layer.odim.get_offset_padded(o_area.x0, y, 0);
				const int i1 = layer.odim.get_offset_padded(o_area.x1, y, 0);
				for(int n=i0;n<i1;n++) signal_error_terms[n] = output_error[n] * activation_derivative(layer.signal[n]);
				for(int n=i0;n<i1;n++) layer.biases_error[n] += signal_error_terms[n];
			}
		}

		static void back_propagate_func_input_side(ae_layer& layer, vector<float>& input_error, const vector<float>& input_value, const vector<float>& signal_error_terms, const image_area i_area) {
			const value_image_lines_dimensions& idim = layer.idim;
			const value_image_lines_dimensions& odim = layer.odim;
			const layer_pattern& scale = layer.pattern;
			const area_table& table = layer.table;
			const int WEIGHTS_PER_OUTPUT_NEURON = layer.weights_per_output_neuron();
			const int WEIGHTS_PER_INPUT_NEURON = layer.weights_per_input_neuron();
			const int W_STRIDE_X = idim.C;
			const int W_STRIDE_Y = idim.C * scale.M;
			const int E_STRIDE_X = odim.C;
			const int E_STRIDE_Y = odim.C * scale.M * scale.B / scale.A;
			const float mult = sqrtf(1.0f / WEIGHTS_PER_OUTPUT_NEURON);// TODO - test if this helps or hinders deep autoencoders.
			//const float mult = 1.0f / WEIGHTS_PER_OUTPUT_NEURON;

			// for each input neuron in input-area...
			for(int iy=i_area.y0;iy<i_area.y1;iy++) {
			for(int ix=i_area.x0;ix<i_area.x1;ix++) {
			for(int ic=0;ic<idim.C;ic++) {
				const int in_n = idim.get_offset(ix, iy, ic);
				float input_error_sum = 0;
				// for each output neuron that may read from it...
				const image_area o_area = table.get_output_area_clamped(ix, iy);
				for(int oy=o_area.y0;oy<o_area.y1;oy++) {
				for(int ox=o_area.x0;ox<o_area.x1;ox++) {
					const int e = (in_n * WEIGHTS_PER_INPUT_NEURON) + ((ox - o_area.x0) * E_STRIDE_X) + ((oy - o_area.y0) * E_STRIDE_Y);// weight-error index.
					const image_area r_area = table.get_input_area(ox, oy);
					const int wofs = ((iy - r_area.y0) * W_STRIDE_Y) + ((ix - r_area.x0) * W_STRIDE_X) + ic;
					const int nofs = odim.get_offset(ox, oy, 0);
				// accumulate input & weight error.
				for(int oc=0;oc<odim.C;oc++) {
					const int out_n = nofs + oc;
					const int w = out_n * WEIGHTS_PER_OUTPUT_NEURON + wofs;
					const float error_term = signal_error_terms[out_n];
					layer.weights_error[e+oc] += error_term * mult * input_value[in_n];
					input_error_sum           += error_term * mult * layer.weights[w];
				}
				}}
				input_error[in_n] = input_error_sum;
			}}}
		}

		void back_propagate(const int n_threads, vector<float>& input_error, const vector<float>& input_value, const vector<float>& output_error) {
			// assertions.
			const int IMAGE_SIZE_I = input_image_size();
			const int IMAGE_SIZE_O = output_image_size();
			assert(IMAGE_SIZE_I > 0);
			assert(IMAGE_SIZE_O > 0);
			assert(input_error.size() == IMAGE_SIZE_I);
			assert(input_value.size() == IMAGE_SIZE_I);
			assert(biases.size() == IMAGE_SIZE_O);
			assert(biases_error.size() == IMAGE_SIZE_O);
			assert(output_error.size() == IMAGE_SIZE_O);

			// compute signal error terms and adjust biases of output neurons.
			//timepoint t0 = timepoint::now();
			vector<float> signal_error_terms(IMAGE_SIZE_O);
			{
				vector<image_area> intervals = generate_intervals(n_threads, odim);
				vector<std::thread> threads;
				for(int x=0;x<n_threads;x++) {
					threads.push_back(std::thread(
						back_propagate_func_output_side,
						std::ref(*this),
						std::ref(signal_error_terms),
						std::ref(output_error),
						intervals[x]
					));
				}
				for(int x=0;x<n_threads;x++) threads[x].join();
			}

			// back propagate error to input and adjust weights.
			//timepoint t1 = timepoint::now();
			{
				vector<image_area> intervals = generate_intervals(n_threads, idim);
				vector<std::thread> threads;
				for(int x=0;x<n_threads;x++) {
					threads.push_back(std::thread(
						back_propagate_func_input_side,
						std::ref(*this),
						std::ref(input_error),
						std::ref(input_value),
						std::ref(signal_error_terms),
						intervals[x]
					));
				}
				for(int x=0;x<n_threads;x++) threads[x].join();
			}

			// normalize input-error against output-error to have same average gradient per-neuron.
			///*
			//timepoint t2 = timepoint::now();
			const float in_sum  = vec_sum_abs_mt(input_error, 0, input_error.size(), n_threads);
			const float out_sum = vec_sum_abs_mt(output_error, 0, output_error.size(), n_threads);
			float mult = (out_sum / in_sum) * (float(IMAGE_SIZE_I) / float(IMAGE_SIZE_O));
			//printf("error: isum=%f, osum=%f\n", in_sum, out_sum);
			assert(out_sum > 0.0f);
			assert(in_sum > 0.0f);
			vec_mult_mt(input_error, mult, 0, input_error.size(), n_threads);

			// TEST - print time taken for each part of function.
			//timepoint t3 = timepoint::now();
			//printf("dt:\t%li\t%li\t%li\n", t1.delta_us(t0), t2.delta_us(t1), t3.delta_us(t2));
		}

		void clear_batch_error() {
			assert(biases_error.size() == biases.size());
			assert(weights_error.size() == weights.size());
			vec_fill(biases_error, 0.0f);
			vec_fill(weights_error, 0.0f);
		}

		static void apply_batch_error_biases(ae_layer& layer, const int beg, const int end, const float adjustment_rate) {
			const float BIAS_LIMIT = 100.0f;
			const float BIAS_ADJUSTMENT_LIMIT = 0.5f;
			for(int n=beg;n<end;n++) {
				const float adjustment = std::clamp(layer.biases_error[n] * adjustment_rate, -BIAS_ADJUSTMENT_LIMIT, +BIAS_ADJUSTMENT_LIMIT);
				layer.biases[n] = std::clamp(layer.biases[n] + adjustment, -BIAS_LIMIT, +BIAS_LIMIT);
			}
		}
		static void apply_batch_error_weights(ae_layer& layer, const image_area i_area, const float adjustment_rate) {
			const float WEIGHT_LIMIT = 100.0f;
			const float WEIGHT_ADJUSTMENT_LIMIT = 0.5f;
			/*
			for(int x=beg;x<end;x++) {
				const float adjustment = std::clamp(layer.weights_error[x] * adjustment_rate, -WEIGHT_ADJUSTMENT_LIMIT, +WEIGHT_ADJUSTMENT_LIMIT);
				layer.weights[x] = std::clamp(layer.weights[x] + adjustment, -WEIGHT_LIMIT, +WEIGHT_LIMIT);
			}
			*/

			// NOTE: this algorithm is copy-pasted from backprop, because inverting from
			// backprop indices to foreward indices is quite hard, and not really worth the effort.
			// WARNING: this algorithm is write-style and not thread-safe.
			const value_image_lines_dimensions& idim = layer.idim;
			const value_image_lines_dimensions& odim = layer.odim;
			const layer_pattern& scale = layer.pattern;
			const area_table& table = layer.table;
			const int WEIGHTS_PER_OUTPUT_NEURON = layer.weights_per_output_neuron();
			const int WEIGHTS_PER_INPUT_NEURON = layer.weights_per_input_neuron();
			const int W_STRIDE_X = idim.C;
			const int W_STRIDE_Y = idim.C * scale.M;

			// for each input neuron in input-area...
			for(int iy=i_area.y0;iy<i_area.y1;iy++) {
			for(int ix=i_area.x0;ix<i_area.x1;ix++) { const image_area o_area = table.get_output_area(ix, iy);
			for(int ic=0;ic<idim.C;ic++) {
				const int in_n = idim.get_offset(ix, iy, ic);
				int e = in_n * WEIGHTS_PER_INPUT_NEURON;// weight-error index.
				// for each output neuron that may read from it...
				for(int oy=o_area.y0;oy<o_area.y1;oy++) {
				for(int ox=o_area.x0;ox<o_area.x1;ox++) {
				if(odim.is_within_bounds(ox, oy)) {
					const image_area r_area = table.get_input_area(ox, oy);
					const int wofs_y = (iy - r_area.y0) * W_STRIDE_Y;
					const int wofs_x = (ix - r_area.x0) * W_STRIDE_X;
				for(int oc=0;oc<odim.C;oc++) {
					// compute the weight index that would be used to read this input-neuron.
					const int out_n = odim.get_offset(ox, oy, oc);
					const int w = (out_n * WEIGHTS_PER_OUTPUT_NEURON) + wofs_x + wofs_y + ic;
					// accumulate error.
					const float adjustment = std::clamp(layer.weights_error[e] * adjustment_rate, -WEIGHT_ADJUSTMENT_LIMIT, +WEIGHT_ADJUSTMENT_LIMIT);
					layer.weights[w] = std::clamp(layer.weights[w] + adjustment, -WEIGHT_LIMIT, +WEIGHT_LIMIT);
					e++;
				}} else {
					e += odim.C;
				}
				}}
				assert(e == (in_n * WEIGHTS_PER_INPUT_NEURON + WEIGHTS_PER_INPUT_NEURON));
			}}}
		}
		void apply_batch_error(const int n_threads, const int batch_size, const float learning_rate_b, const float learning_rate_w) {
			// assertions.
			assert(biases_error.size() == biases.size());
			assert(weights_error.size() == weights.size());

			// adjust biases.
			{
				const float adjustment_rate = learning_rate_b / batch_size;
				vector<std::thread> threads;
				const int len = biases.size();
				for(int x=0;x<n_threads;x++) {
					const int x0 = ((x+0) * len) / n_threads;
					const int x1 = ((x+1) * len) / n_threads;
					threads.push_back(std::thread(apply_batch_error_biases, std::ref(*this), x0, x1, adjustment_rate));
				}
				for(int x=0;x<n_threads;x++) threads[x].join();
			}

			// adjust weights.
			{
				const float adjustment_rate = learning_rate_w / batch_size;
				vector<image_area> intervals = generate_intervals(n_threads, idim);
				vector<std::thread> threads;
				const int len = weights.size();
				for(int x=0;x<n_threads;x++) {
					threads.push_back(std::thread(apply_batch_error_weights, std::ref(*this), intervals[x], adjustment_rate));
				}
				for(int x=0;x<n_threads;x++) threads[x].join();
			}
		}
	};

}
