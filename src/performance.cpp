
#include <cstdint>
#include <ctime>

namespace ML::performance {
	struct timepoint {
		std::timespec ts;

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
	timepoint now() {
		timepoint tp;
		std::timespec_get(&tp.ts, TIME_UTC);
		return tp;
	}
}
