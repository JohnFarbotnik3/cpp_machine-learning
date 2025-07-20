
#include <immintrin.h>

namespace ML::models::autoencoder_subimage {
	//using _mm256_add_ps
	using vec8f = __m256;
	const int vec8f_LENGTH = 8;


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


	void simd_load(vec8f* dst, const int simd_len, const float* src) {
		for(int x=0;x<simd_len;x++) dst[x] = _mm256_loadu_ps(src + (x*vec8f_LENGTH));
	}
	void simd_store(const vec8f* src, const int simd_len, float* dst) {
		for(int x=0;x<simd_len;x++) _mm256_storeu_ps(dst + (x*vec8f_LENGTH), src[x]);
	}
	void simd_fill(vec8f* dst, const int simd_len, const float value) {
		for(int x=0;x<simd_len;x++) dst[x] = _mm256_set1_ps(value);
	}


	void simd_mult_accumulate(vec8f* dst, const int simd_len, const float* values, const float mult) {
		for(int x=0;x<simd_len;x++) {
			dst[x] = _mm256_fmadd_ps(
				_mm256_loadu_ps(values + (x*vec8f_LENGTH)),
				_mm256_set1_ps(mult),
				dst[x]
			);
		}
	}

}
