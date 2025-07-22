
#ifndef F_simd_cpp
#define F_simd_cpp

#include <cassert>
#include <cstdint>
#include <immintrin.h>
#include <thread>

namespace ML::models::autoencoder_subimage {

	// https://stackoverflow.com/questions/79539019/how-to-fix-a-warning-ignoring-attributes-with-a-vector-of-m256
	//using vec8f = __m256;
	struct vec8f {
		__m256 data;
		vec8f(__m256 x) : data(x) {}
		operator __m256() const { return data; }
	};
	struct vec8i {
		__m256i data;
		vec8i(__m256i x) : data(x) {}
		operator __m256i() const { return data; }
	};

	const int vec8f_LENGTH = 8;


	vec8f simd_value(const float value) {
		return _mm256_set1_ps(value);
	}


	vec8f simd_gte_cmov(vec8f a, vec8f b, vec8f va, vec8f vb) {
		const vec8f mask = _mm256_cmp_ps(a, b, _CMP_GE_OS);
		return _mm256_blendv_ps(vb, va, mask);
	}
	vec8f simd_gte_cmov(vec8f a, float b, float va, vec8f vb) {
		return simd_gte_cmov(a, _mm256_set1_ps(b), _mm256_set1_ps(va), vb);
	}
	vec8f simd_negative(vec8f a) {
		return _mm256_sub_ps(_mm256_setzero_ps(), a);
	}
	vec8f simd_sign(vec8f a) {
		return simd_gte_cmov(a, _mm256_setzero_ps(), _mm256_set1_ps(1.0f), _mm256_set1_ps(-1.0f));
	}
	vec8f simd_abs(vec8f a) {
		// https://stackoverflow.com/questions/23847377/how-does-this-function-compute-the-absolute-value-of-a-float-through-a-not-and-a
		return _mm256_andnot_ps(_mm256_set1_ps(-0.0f), a);
		//return simd_gte_cmov(a, _mm256_setzero_ps(), a, simd_negative(a));
	}
	void simd_incr(vec8f& a, const vec8f b) {
		a = _mm256_add_ps(a, b);
	}


	bool simd_eq(vec8f a, vec8f b) {
		uint32_t data[vec8f_LENGTH];
		_mm256_storeu_ps((float*)data, _mm256_cmp_ps(a, b, _CMP_NEQ_OS));
		uint32_t result = 0;
		for(int x=0;x<vec8f_LENGTH;x++) result |= data[x];
		return result == 0;
	}
	float simd_reduce(vec8f a) {
		float sum = 0;
		for(int x=0;x<vec8f_LENGTH;x++) sum += a.data[x];
		return sum;
	}
	float simd_reduce(const vec8f* data, const size_t len) {
		vec8f sum = simd_value(0);
		for(int x=0;x<len;x++) sum = _mm256_add_ps(sum, data[x]);
		return simd_reduce(sum);
	}
	float simd_reduce_abs(const vec8f* data, const size_t len) {
		vec8f sum = simd_value(0);
		for(int x=0;x<len;x++) sum = _mm256_add_ps(sum, simd_abs(data[x]));
		return simd_reduce(sum);
	}
	void simd_reduce_abs_mt_func(const vec8f* data, const size_t len, float& result) {
		result = simd_reduce_abs(data, len);
	}
	float simd_reduce_abs_mt(const vec8f* data, const size_t len, const int n_threads) {
		float results[n_threads];
		std::thread threads[n_threads];
		for(int x=0;x<n_threads;x++) {
			const int b = ((x+0) * len) / n_threads;
			const int e = ((x+1) * len) / n_threads;
			threads[x] = std::thread(simd_reduce_abs_mt_func, data+b, e-b, std::ref(results[x]));
		}
		for(int x=0;x<n_threads;x++) threads[x].join();
		float sum = 0;
		for(int x=0;x<n_threads;x++) sum += results[x];
		return sum;
	}
	void simd_scale_mt_func(vec8f* data, const size_t len, float mult) {
		vec8f m = _mm256_set1_ps(mult);
		for(int x=0;x<len;x++) data[x] = _mm256_mul_ps(data[x], m);
	}
	void simd_scale_mt(vec8f* data, const size_t len, const int n_threads, float mult) {
		std::thread threads[n_threads];
		for(int x=0;x<n_threads;x++) {
			const int b = ((x+0) * len) / n_threads;
			const int e = ((x+1) * len) / n_threads;
			threads[x] = std::thread(simd_scale_mt_func, data+b, e-b, mult);
		}
		for(int x=0;x<n_threads;x++) threads[x].join();
	}
}

#endif
