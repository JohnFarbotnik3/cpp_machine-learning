
#ifndef F_simd_cpp
#define F_simd_cpp

#include <cassert>
#include <cstdint>
#include <immintrin.h>

namespace ML::models::autoencoder_subimage {

	using vec8f = __m256;
	const int vec8f_LENGTH = 8;


	vec8f simd_value(const float value) {
		return _mm256_set1_ps(value);
	}
	vec8f simd_loadu(const float* src) {
		return _mm256_loadu_ps(src);
	}
	void simd_storeu(float* dst, vec8f src) {
		_mm256_storeu_ps(dst, src);
	}


	vec8f simd_gte_cmov(vec8f a, vec8f b, vec8f va, vec8f vb) {
		vec8f mask = _mm256_cmp_ps(a, b, _CMP_GE_OS);
		return _mm256_or_ps(_mm256_and_ps(mask, va), _mm256_andnot_ps(mask, va));
	}
	vec8f simd_gte_cmov(vec8f a, float b, float va, vec8f vb) {
		return simd_gte_cmov(a, _mm256_set1_ps(b), _mm256_set1_ps(va), vb);
	}
	vec8f simd_sign(vec8f a) {
		return simd_gte_cmov(a, _mm256_set1_ps(0.0f), _mm256_set1_ps(1.0f), _mm256_set1_ps(-1.0f));
	}
	vec8f simd_negative(vec8f a) {
		return _mm256_sub_ps(_mm256_set1_ps(0.0f), a);
	}
	vec8f simd_abs(vec8f a) {
		return simd_gte_cmov(a, _mm256_set1_ps(0.0f), a, simd_negative(a));
	}
	void simd_incr(vec8f& a, const vec8f b) {
		a = _mm256_add_ps(a, b);
	}


	bool simd_eq(vec8f a, vec8f b) {
		vec8f mask = _mm256_cmp_ps(a, b, _CMP_EQ_OS);
		uint32_t data[vec8f_LENGTH];
		_mm256_storeu_ps((float*)data, mask);
		for(int x=0;x<vec8f_LENGTH;x++) if(data[x] == 0) return false;
		return true;
	}
	float simd_reduce(vec8f a) {
		float data[vec8f_LENGTH];
		_mm256_storeu_ps(data, a);
		float sum = 0;
		for(int x=0;x<vec8f_LENGTH;x++) sum += data[x];
		return sum;
	}
}

#endif
