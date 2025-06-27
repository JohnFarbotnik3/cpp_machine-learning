
#include <algorithm>
#include <cstdio>
#include <ctime>
#include <map>
#include <string>
#include <vector>
#include "./utils/string_manip.cpp"

namespace ML::stats {
	using std::vector;
	using std::string;
	using std::map;

	struct timepoint {
		std::timespec ts;

		static timepoint now() {
			timepoint tp;
			std::timespec_get(&tp.ts, TIME_UTC);
			return tp;
		}

		int64_t delta_ns(timepoint t0) {
			int64_t s = ts.tv_sec  - t0.ts.tv_sec;
			int64_t n = ts.tv_nsec - t0.ts.tv_nsec;
			return s*1000000000 + n;
		}
		int64_t delta_us(timepoint t0) {
			return delta_ns(t0) / 1000;
		}
		int64_t delta_ms(timepoint t0) {
			return delta_ns(t0) / 1000000;
		}
	};

	struct training_stats {
		map<string, vector<float>> groups;

		void push_value(string name, float value) {
			if(!groups.contains(name)) groups[name] = vector<float>();
			groups[name].push_back(value);
		}
		void clear_group(string name) {
			if(groups.contains(name)) groups[name].clear();
		}
		void clear_all() {
			groups.clear();
		}

		struct vec_stats {
			int count;
			float min;
			float mean;
			float max;
			float sum;
		};
		vec_stats get_vec_stats(const vector<float>& vec) {
			if(vec.empty()) return vec_stats { 0,0,0,0 };
			vec_stats stats;
			stats.count	= vec.size();
			stats.min	= vec[0];
			stats.max	= vec[0];
			stats.sum	= vec[0];
			for(int x=1;x<vec.size();x++) {
				stats.min = std::min(vec[x], stats.min);
				stats.max = std::max(vec[x], stats.max);
				stats.sum += vec[x];
			}
			stats.mean = stats.sum / stats.count;
			return stats;
		}

		void print_group(string name) {
			if(!groups.contains(name)) return;
			const vector<float>& vec = groups[name];
			vec_stats stats = get_vec_stats(vec);
			printf("%s\t%i\t%f\t%f\t%f\t%f\n", name.c_str(), stats.count, stats.min, stats.mean, stats.max, stats.sum);
		}
		void print_all() {
			for(const auto& [name, vec] : groups) print_group(name);
		}
	};

	void print_percentiles_header(const vector<int>& percentiles, string name, string fmt, int column_width) {
		printf("PERCENTILES:\n");
		{
			string str = utils::string_manip::get_padded_number_string("%s", name.c_str(), column_width, false);
			printf("%s", str.c_str());
		}
		for(int x=0;x<percentiles.size();x++) {
			string str = utils::string_manip::get_padded_number_string(fmt, percentiles[x], column_width, true);
			printf("%s", str.c_str());
		}
		printf("\n");
	}

	void print_percentiles(const vector<int>& percentiles, string name, string fmt, int column_width, vector<float> values) {
		if(values.size() < 1) return;
		vector<float> sorted = values;
		std::sort(sorted.begin(), sorted.end());
		{
			string str = utils::string_manip::get_padded_number_string("%s", name.c_str(), column_width, false);
			printf("%s", str.c_str());
		}
		for(int x=0;x<percentiles.size();x++) {
			int pct = percentiles[x];
			size_t index = std::clamp<size_t>((sorted.size() * pct) / 100, 0, sorted.size()-1);
			string str = utils::string_manip::get_padded_number_string(fmt, sorted[index], column_width, true);
			printf("%s", str.c_str());
		}
		printf("\n");
	}
};
