
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
	};

	vector<float> get_percentile_values(const vector<int>& percentiles, const vector<float> values) {
		vector<float> sorted = values;
		std::sort(sorted.begin(), sorted.end());
		vector<float> pct_values(percentiles.size());
		for(int x=0;x<percentiles.size();x++) {
			size_t index = std::clamp<size_t>((sorted.size() * percentiles[x]) / 100, 0, sorted.size()-1);
			pct_values[x] = sorted[index];
		}
		return pct_values;
	}

	void print_percentiles_header(const vector<int>& percentiles, string name, string fmt, int column_width, int first_column_width) {
		printf("PERCENTILES:\n");
		{
			string str = utils::string_manip::get_padded_number_string("%s", name.c_str(), first_column_width, false);
			printf("%s", str.c_str());
		}
		for(int x=0;x<percentiles.size();x++) {
			string str = utils::string_manip::get_padded_number_string(fmt, percentiles[x], column_width, true);
			printf("%s", str.c_str());
		}
		printf("\n");
	}

	void print_percentiles(const vector<int>& percentiles, string name, string fmt, int column_width, int first_column_width, vector<float> values) {
		if(values.size() < 1) return;
		{
			string str = utils::string_manip::get_padded_number_string("%s", name.c_str(), first_column_width, false);
			printf("%s", str.c_str());
		}
		const vector<float> pct_values = get_percentile_values(percentiles, values);
		for(int x=0;x<percentiles.size();x++) {
			string str = utils::string_manip::get_padded_number_string(fmt, pct_values[x], column_width, true);
			printf("%s", str.c_str());
		}
		printf("\n");
	}
};
