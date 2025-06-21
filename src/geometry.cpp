
#include <algorithm>
#include <vector>
namespace ML::geometry {

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
		int X;// layer width.
		int Y;// layer height.
		image_dimensions() = default;
		image_dimensions(int X, int Y) {
			this->X = X;
			this->Y = Y;
		}
		int  get_area() const { return X*Y; }
		int  get_index(int x, int y) const { return x*Y + y; }
		bool in_bounds(int x, int y) const { return (0 <= x) & (x < X) & (0 <= y) & (y < Y); }
		// clip rectangle to dimensions.
		rectangle<int> clip(rectangle<int> rect) const {
			rectangle<int> out;
			out.p0.x = std::max(rect.p0.x, 0);
			out.p0.y = std::max(rect.p0.y, 0);
			out.p1.x = std::min(rect.p1.x, this->X);
			out.p1.y = std::min(rect.p1.y, this->Y);
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
