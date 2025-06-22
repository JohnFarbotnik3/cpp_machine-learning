
#include <algorithm>
#include <vector>

/*
	NOTE: this namespace assumes that image-data is in row-major form,
	and that RGBA values are interleaved.
*/
namespace ML::image {
	using std::vector;

	vector<int> generate_image_data_indices(int image_w, int image_h, int x, int y, int w, int h, int channel=-1) {
		// clip to image area.
		int x0 = std::max(x, 0);
		int y0 = std::max(y, 0);
		int x1 = std::min(x+w, image_w);
		int y1 = std::min(y+h, image_h);
		// generate indices.
		vector<int> list;
		for(int py=y0;py<y1;py++) {
		for(int px=x0;px<x1;px++) {
			if(channel == -1) {
				int i = ((py*image_w) + px) * 4;
				list.push_back(i+0);
				list.push_back(i+1);
				list.push_back(i+2);
				list.push_back(i+3);
			} else {
				int i = ((py*image_w) + px) * 4 + channel;
				list.push_back(i);
			}
		}}
		return list;
	}




	/*
		iterator for generating image-data indices.
		this is mostly used for building network layers.

		NOTE:
		- this expects RGBA image data to be in row-major form, treating pixels as 4x1 rectangles.
		- this advances by 4 index units each time next() is called.
	*/
	struct pixel_iterator_rgba_4x1_rows {
		private:
			// image dimensions.
			int w, h;
			// iterator region bounds.
			int x0,x1;
			int y0,y1;
			// index distance to advance each iteration.
			int adv;
		public:
			int c;// channel.
			int x;// y position.
			int y;// x position.
			int i;// index in image-data array.
		pixel_iterator_rgba_4x1_rows(int imageW, int imageH, int px, int py, int pw, int ph, int advance=1, int channel=0) {
			w = imageW;
			h = imageH;
			x0 = std::max(px, 0);
			x1 = std::min(px + pw, imageW);
			y0 = std::max(py, 0);
			y1 = std::min(py + ph, imageH);
			c = channel;
			x = x0;
			y = y0;
			i = get_index(x, y, c, w);
		}
		static int get_index(int x, int y, int c, int width) {
			return (y*width + x) * 4 + c;
		}
		bool has_next() {
			return (x < x1) & (y < y1);
		}
		int next() {
			c += adv;
			if(c >=  4) { c%=4; x++; }
			if(x >= x1) { x=x0; y++; }
			return i = get_index(x, y, c, w);
		}
	};







	template<class T>
	struct vec2 {
		T x,y;

		static vec2 linear_combination(vec2<T> a, T a_mult, vec2<T> b, T b_mult) {
			T x = (a.x * a_mult) + (b.x * b_mult);
			T y = (a.y * a_mult) + (b.y * b_mult);
			return vec2<T>{ x,y };
		}
	};

	template<class T>
	struct rectangle {
		vec2<T> p0;
		vec2<T> p1;
	};

	struct image_dimensions {
		int w;// layer width.
		int h;// layer height.
		image_dimensions() = default;
		image_dimensions(int w, int h) {
			this->w = w;
			this->h = h;
		}
		int  get_area() const { return w*h; }
		int  get_index(int x, int y) const { return x*h + y; }
		bool in_bounds(int x, int y) const { return (0 <= x) & (x < w) & (0 <= y) & (y < h); }
		// clip rectangle to dimensions.
		rectangle<int> clip(rectangle<int> rect) const {
			rectangle<int> out;
			out.p0.x = std::max(rect.p0.x, 0);
			out.p0.y = std::max(rect.p0.y, 0);
			out.p1.x = std::min(rect.p1.x, this->w);
			out.p1.y = std::min(rect.p1.y, this->h);
			return out;
		};
		// generates indices for points inside given rectangle, omitting indices outside image dimensions.
		std::vector<int> get_indices_in_rectangle(rectangle<int> rect) const {
			rect = clip(rect);
			const auto p0 = rect.p0;
			const auto p1 = rect.p1;
			std::vector<int> list;
			for(int x=p0.x;x<p1.x;x++) {
			for(int y=p0.y;y<p1.y;y++) {
				list.push_back(get_index(x, y));
			}}
			return list;
		}
	};

}
