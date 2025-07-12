
#include <vector>
#include "src/utils/vector_util.cpp"
#include "src/image/value_image_lines.cpp"

namespace ML::models::autoencoder {
	using std::vector;
	using namespace utils::vector_util;
	using namespace ML::image;

	/*
		each layer condenses or expands the previous layer from AxA squares to BxB squares.

		neurons inside each BxB square read from a CxC square in the input layer which is
		centered on the related AxA square.
	*/
	struct scale_ratio {
		int A;// input  - AxA square.
		int B;// output - BxB square.
		int C;// input area - a CxC square centered on scaled position in previous layer.
	};

	struct image_area {
		int x0, x1;
		int y0, y1;

		bool is_within_image_bounds(const value_image_lines_dimensions dim) const {
			return (
				(x0 > 0) & (x1 <= dim.X) &
				(y0 > 0) & (y1 <= dim.Y)
			);
		}
	};

	struct image_itv { int p0, p1; };

	struct area_table {
		vector<image_itv> input_area_x;
		vector<image_itv> input_area_y;
		vector<image_itv> output_area_x;
		vector<image_itv> output_area_y;
		const int WEIGHTS_PER_OUTPUT_NEURON;
		const int WEIGHTS_PER_INPUT_NEURON;

		area_table(const value_image_lines_dimensions idim, const value_image_lines_dimensions odim, const scale_ratio scale) {
			assert(idim.length() > 0);
			assert(odim.length() > 0);
			// assert that input/output are tiled cleanly by NxN squares.
			assert(idim.X % scale.A == 0);
			assert(idim.Y % scale.A == 0);
			assert(odim.X % scale.B == 0);
			assert(odim.Y % scale.B == 0);
			// assert that input and output have same number of NxN squares.
			assert((idim.X / scale.A) == (odim.X / scale.B));
			assert((idim.Y / scale.A) == (odim.Y / scale.B));
			// assert that overlapping factor is consistent.
			assert(scale.C % scale.A);

			input_area_x.resize(odim.X);
			input_area_y.resize(odim.Y);
			output_area_x.resize(idim.X);
			output_area_y.resize(idim.Y);

			// compute input areas of output neurons.
			for(int ox=0;ox<odim.X;ox++) {
				const int ix0 = (ox / scale.B) * scale.A + (scale.A / 2) - (scale.C / 2);
				const int ix1 = ix0 + scale.C;
				input_area_x[ox] = image_itv{ ix0, ix1 };
			}
			for(int oy=0;oy<odim.Y;oy++) {
				const int iy0 = (oy / scale.B) * scale.A + (scale.A / 2) - (scale.C / 2);
				const int iy1 = iy0 + scale.C;
				input_area_y[oy] = image_itv{ iy0, iy1 };
			}

			// determine output areas of input neurons (using input areas).
			const int MIN_INT = 0;
			const int MAX_INT = std::max(odim.X, odim.Y);
			for(int p=0;p<output_area_x.size();p++) output_area_x[p] = image_itv{ MAX_INT, MIN_INT };
			for(int p=0;p<output_area_y.size();p++) output_area_y[p] = image_itv{ MAX_INT, MIN_INT };
			for(int ox=0;ox<odim.X;ox++) {
				const image_itv itv_i = input_area_x[ox];
				for(int p=itv_i.p0;p<itv_i.p1;p++) {
					if(p < 0 || p > idim.X) continue;
					image_itv& itv_o = output_area_x[p];
					itv_o.p0 = std::min(itv_o.p0, ox);
					itv_o.p1 = std::max(itv_o.p1, ox);
				}
			}
			for(int oy=0;oy<odim.Y;oy++) {
				const image_itv itv_i = input_area_y[oy];
				for(int p=itv_i.p0;p<itv_i.p1;p++) {
					if(p < 0 || p > idim.Y) continue;
					image_itv& itv_o = output_area_y[p];
					itv_o.p0 = std::min(itv_o.p0, oy);
					itv_o.p1 = std::max(itv_o.p1, oy);
				}
			}
		}

		// get the (unclamped) area of input-neurons this output-neuron reads from.
		image_area get_input_area(int ox, int oy) const {
			const image_itv itv_x = input_area_x[ox];
			const image_itv itv_y = input_area_y[oy];
			return image_area {
				itv_x.p0, itv_x.p1,
				itv_y.p0, itv_y.p1,
			};
		}

		// get the area of output-neurons read from this input-neuron.
		image_area get_output_area(int ix, int iy) const {
			const image_itv itv_x = output_area_x[ix];
			const image_itv itv_y = output_area_y[iy];
			return image_area {
				itv_x.p0, itv_x.p1,
				itv_y.p0, itv_y.p1,
			};
		}

		/*
			TODO: it would be super useful if there was an easy way to
			compute indices which relate foreward-targets to backprop-targets,
			and vice-versa.

			this would enable read-style gathering of weight-errors during backprop
			as if I were still using backprop targets, and having both conversions would allow
			multithreading the apply_batch_error function as well.

			it may be possible to generate lookup tables for both of these,
			and there is a lot of linear separability to be had here,
			but this is by far some of the most complex coordinate manipulation I have done thusfar.
		*/

		// TODO
		// get index of weight for foreward propagation.
		//int get_foreward_weight_index(const int out_n, const int ix, const int iy, const int ic) const {}

		// TODO
		// get index of weight-error for backpropagation.
		//int get_backprop_weight_error_index() const {}

		// TODO
		//int get_related_fw_target_index() const {}
		//int get_realted_bp_target_index() const {}
	};

	struct ae_layer {
		vector<float> biases;
		vector<float> output;// image of output values - used for backprop.
		vector<float> signal;// image of signal values - used for backprop.
		vector<float> weights;
		vector<float> biases_error;// accumulated error in biases during minibatch.
		vector<float> weights_error;// accumulated error in weights during minibatch.
		const value_image_lines_dimensions idim;// input image dimensions.
		const value_image_lines_dimensions odim;// output image dimensions.
		const scale_ratio scale;
		const area_table table;

		ae_layer(
			const value_image_lines_dimensions idim,
			const value_image_lines_dimensions odim,
			scale_ratio scale
		) : idim(idim), odim(odim), scale(scale), table(idim, odim, scale) {
			const int OUTPUT_IMAGE_SIZE = output_image_size();
			biases.resize(OUTPUT_IMAGE_SIZE);
			output.resize(OUTPUT_IMAGE_SIZE);
			signal.resize(OUTPUT_IMAGE_SIZE);
			weights.resize(OUTPUT_IMAGE_SIZE * weights_per_output_neuron());
			biases_error.resize(OUTPUT_IMAGE_SIZE);
			weights_error.resize(OUTPUT_IMAGE_SIZE * weights_per_output_neuron());
			vec_fill(biases, 0.0f);
			vec_fill(output, 0.0f);
			vec_fill(signal, 0.0f);
			vec_fill(weights, 0.0f);
			vec_fill(biases_error, 0.0f);
			vec_fill(weights_error, 0.0f);
		}

		int input_image_size() const {
			return idim.length();
		}
		int output_image_size() const {
			return odim.length();
		}
		int weights_per_output_neuron() const {
			return scale.C * scale.C * idim.C;
		}
		int weights_per_input_neuron() const {
			return weights_per_output_neuron() * (scale.B * scale.B) / (scale.A * scale.A);
		};

		// ============================================================
		// activation functions.
		// ------------------------------------------------------------

		static float activation_func(const float value) {
			const float sign = value >= 0.0f ? 1.0f : -1.0f;
			const float mag = std::abs(value);
			if(mag < 0.5f) return value * 1.0f;						// [0.0, 0.5] 0.00 -> 0.50
			if(mag < 1.0f) return value * 0.7f + (sign * 0.15f);	// [0.5, 1.0] 0.50 -> 0.85
			if(mag < 2.0f) return value * 0.5f + (sign * 0.35f);	// [1.0, 2.0] 0.85 -> 1.35
			if(mag < 4.0f) return value * 0.3f + (sign * 0.75f);	// [2.0, 4.0] 1.35 -> 1.95
			return value * 0.1f + (sign * 1.55f);					// [4.0, inf] 1.95 -> inf.
		}

		static float activation_derivative(const float value) {
			const float mag = std::abs(value);
			if(mag < 0.5f) return 1.0f;
			if(mag < 1.0f) return 0.7f;
			if(mag < 2.0f) return 0.5f;
			if(mag < 4.0f) return 0.3f;
			return 0.1f;
		}

		// ============================================================
		// network functions
		// ------------------------------------------------------------

		static vector<image_area> generate_intervals(const int n_threads, const value_image_lines_dimensions dim) {
			vector<image_area> intervals;
			for(int z=0;z<n_threads;z++) {
				const int x0 = 0;
				const int x1 = dim.X;
				const int y0 = ((z+0) * dim.Y) / n_threads;
				const int y1 = ((z+1) * dim.Y) / n_threads;
				intervals.push_back(image_area{ x0, x1, y0, y1});
			}
			return intervals;
		}

		static void propagate_func(ae_layer& layer, const vector<float>& input_values, vector<float>& output_values, const image_area o_area) {
			const value_image_lines_dimensions& idim = layer.idim;
			const value_image_lines_dimensions& odim = layer.odim;
			const scale_ratio& scale = layer.scale;
			const vector<float>& biases = layer.biases;
			vector<float>& signal = layer.signal;
			vector<float>& output = layer.output;
			const vector<float>& weights = layer.weights;
			const int WEIGHTS_PER_NEURON = layer.weights_per_output_neuron();

			for(int oy=o_area.y0;oy<o_area.y1;oy++) {
			for(int ox=o_area.x0;ox<o_area.x1;ox++) {
				const image_area i_area = layer.table.get_input_area(ox, oy);

				if(i_area.is_within_image_bounds(idim)) {
					for(int oc=0;oc<odim.C;oc++) {
						const int out_n = odim.get_offset(ox, oy, oc);
						float sum = biases[out_n];
						int w = out_n * WEIGHTS_PER_NEURON;// initial weight index.
						for(int iy=i_area.y0;iy<i_area.y1;iy++) {
						for(int ix=i_area.x0;ix<i_area.x1;ix++) {
						for(int ic=0;ic<idim.C;ic++) {
							const int in_n = idim.get_offset(ix, iy, ic);
							sum += weights[w] * input_values[in_n];
							w++;
						}}}
						assert(w == ((out_n * WEIGHTS_PER_NEURON) + WEIGHTS_PER_NEURON));
						signal[out_n] = sum;
						output[out_n] = activation_func(sum);
					}
				} else {
					for(int oc=0;oc<odim.C;oc++) {
						const int o_n = odim.get_offset(ox, oy, oc);
						float sum = biases[o_n];
						int w = o_n * WEIGHTS_PER_NEURON;// initial weight index.
						for(int iy=i_area.y0;iy<i_area.y1;iy++) {
						for(int ix=i_area.x0;ix<i_area.x1;ix++) {
						for(int ic=0;ic<idim.C;ic++) {
							const int i_n = idim.get_offset(ix, iy, ic);
							sum += weights[w] * input_values[i_n];
							w++;
						}}}
						assert(w == ((o_n * WEIGHTS_PER_NEURON) + WEIGHTS_PER_NEURON));
						signal[o_n] = sum;
						output[o_n] = activation_func(sum);
					}
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
				const int i0 = layer.odim.get_offset(o_area.x0, y, 0);
				const int i1 = layer.odim.get_offset(o_area.x1, y, 0);
				for(int n=i0;n<i1;n++) signal_error_terms[n] = output_error[n] * activation_derivative(layer.signal[n]);
				for(int n=i0;n<i1;n++) layer.biases_error[n] += signal_error_terms[n];
			}
		}

		static void back_propagate_func_input_side(ae_layer& layer, vector<float>& input_error, const vector<float>& input_value, const vector<float>& signal_error_terms, const image_area i_area) {
			const value_image_lines_dimensions& idim = layer.idim;
			const value_image_lines_dimensions& odim = layer.odim;
			const scale_ratio& scale = layer.scale;
			const int WEIGHTS_PER_NEURON = layer.weights_per_output_neuron();
			const int W_STRIDE_X = idim.C;
			const int W_STRIDE_Y = idim.C * scale.C;
			const float mult = 1.0f / WEIGHTS_PER_NEURON;

			for(int iy=i_area.y0;iy<i_area.y1;iy++) {
			for(int ix=i_area.x0;ix<i_area.x1;ix++) {
				// get area of output-neurons that read from this input-neuron.
				const image_area o_area = layer.table.get_output_area(ix, iy);

				// for each neuron in input: gather input-error and weight-error from output-neurons.
				for(int ic=0;ic<idim.C;ic++) {
					const int in_n = idim.get_offset(ix, iy, ic);
					float input_error_sum = 0;
					for(int ox=o_area.x0;ox<o_area.x1;ox++) {
					for(int oy=o_area.y0;oy<o_area.y1;oy++) {
						const image_area r_area = layer.table.get_input_area(ox, oy);
						const int dx = ix - r_area.x0;
						const int dy = iy - r_area.y0;
						const int wofs = (dx * W_STRIDE_X) + (dy * W_STRIDE_Y) + ic;

						for(int oc=0;oc<odim.C;oc++) {
							// get the weight-index corrosponding to this input-neuron.
							const int out_n = odim.get_offset(ox, oy, oc);
							const int w = out_n * WEIGHTS_PER_NEURON + wofs;

							const float error_term = signal_error_terms[out_n];
							input_error_sum        += error_term * mult * layer.weights[w];
							// WARNING: this is a write-style operation, and is thus not thread safe.
							// TODO: find a way to fix this.
							layer.weights_error[w] += error_term * mult * input_value[in_n];
						}
					}}
					input_error[in_n] = input_error_sum;
				}
			}}
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
			assert(out_sum > 0.0f);
			assert(in_sum > 0.0f);
			vec_mult_mt(input_error, mult, 0, input_error.size(), n_threads);
			//printf("error: z=%i, isum=%f, osum=%f\n", z, in_sum, out_sum);

			// TEST - print time taken for each part of function.
			//timepoint t3 = timepoint::now();
			//printf("dt:\t%li\t%li\t%li\n", t1.delta_us(t0), t2.delta_us(t1), t3.delta_us(t2));
		}

		// TODO - continue from here...
		void clear_batch_error() {
			assert(biases_error.size() == biases.size());
			assert(weights_error.size() == weights.size());
			vec_fill(biases_error, 0.0f);
			vec_fill(weights_error, 0.0f);
		}

		static void apply_batch_error_biases(ae_layer& layer, const int beg, const int end, const float adjustment_rate) {
			const float BIAS_LIMIT = 10.0f;
			const float BIAS_ADJUSTMENT_LIMIT = 0.5f;
			for(int n=beg;n<end;n++) {
				const float adjustment = std::clamp(layer.biases_error[n] * adjustment_rate, -BIAS_ADJUSTMENT_LIMIT, +BIAS_ADJUSTMENT_LIMIT);
				layer.biases[n] = std::clamp(layer.biases[n] + adjustment, -BIAS_LIMIT, +BIAS_LIMIT);
			}
		}
		static void apply_batch_error_weights(ae_layer& layer, const int beg, const int end, const float adjustment_rate) {
			const float WEIGHT_LIMIT = 100.0f;
			const float WEIGHT_ADJUSTMENT_LIMIT = 0.5f;
			for(int x=beg;x<end;x++) {
				backprop_target& bt = layer.backprop_targets.targets[x];
				foreward_target& ft = layer.foreward_targets.targets[bt.target_index];
				const float adjustment = std::clamp(layer.weights_error[x] * adjustment_rate, -WEIGHT_ADJUSTMENT_LIMIT, +WEIGHT_ADJUSTMENT_LIMIT);
				bt.weight = std::clamp(bt.weight + adjustment, -WEIGHT_LIMIT, +WEIGHT_LIMIT);
				ft.weight = bt.weight;
			}
		}
		void apply_batch_error(const int n_threads, const int batch_size, const float learning_rate_b, const float learning_rate_w) {
			// assertions.
			assert(biases_error.size() == biases.size());
			assert(weights_error.size() == foreward_targets.targets.size());
			assert(weights_error.size() == backprop_targets.targets.size());

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
				vector<std::thread> threads;
				const int len = backprop_targets.targets.size();
				for(int x=0;x<n_threads;x++) {
					const int x0 = ((x+0) * len) / n_threads;
					const int x1 = ((x+1) * len) / n_threads;
					threads.push_back(std::thread(apply_batch_error_weights, std::ref(*this), x0, x1, adjustment_rate));
				}
				for(int x=0;x<n_threads;x++) threads[x].join();
			}
		}
	};

}
