#include "include/utils.h"
#include<immintrin.h>
#include<iostream>
#include<iomanip>
#include <inttypes.h> // For PRId64
#include<vector>

#define setPD(x) _mm256_set1_pd(x)

double normal_cdf(double value)
{
	return 0.5 * std::erfc(-value * M_SQRT1_2);
}

// std :: ostream& operator<<(std :: ostream& out, __m256i x){
// 	long long result[4];
//     _mm256_storeu_epi64(result, x);
    
//     out << "[" << result[0] << ", " << result[1] << ", " 
//         << result[2] << ", " << result[3] << "]";
//     return out;
// };

// std :: ostream& operator<<(std :: ostream& out, __m256d x){
//     double result[4];
//     _mm256_storeu_pd(result, x);
    
//     out << "[" << result[0] << ", " << result[1] << ", " 
//         << result[2] << ", " << result[3] << "]";
//     return out;
// };


__m256d log_avx(__m256d x){
	// constants 
	    
    const __m256d one = setPD(1.0);
    const __m256d half = setPD(0.5);
    
    // ln(2) with full double precision
    const __m256d log2_const = setPD(0.693147180559945309417232121458176568);

    // POLYNOMIAL COEFFICIENTS (12 terms, minimax optimized for double precision)
    // For P(z) = (log(1+z) - z) / z^2
    const __m256d p0  = setPD(-0.50000000000000000000);
    const __m256d p1  = setPD(0.33333333333332931888);
    const __m256d p2  = setPD(-0.24999999999732151675);
    const __m256d p3  = setPD(0.19999999997186632439);
    const __m256d p4  = setPD(-0.16666666649282846147);
    const __m256d p5  = setPD(0.14285714013327126139);
    const __m256d p6  = setPD(-0.12500000185585285493);
    const __m256d p7  = setPD(0.11111096773344331233);
    const __m256d p8  = setPD(-0.09999912071983321162);
    const __m256d p9  = setPD(0.09090908869853962393);
    const __m256d p10 = setPD(-0.08333332768783457174);
    const __m256d p11 = setPD(0.07692307694389229853);

	// x = 2^E * M
	// x = 2^(E+0.5) * (M / sqrt(2))
	// -0.396 <= ln (M / sqrt(2)) < 0.396

	__m256i x_int = _mm256_castpd_si256(x);
	__m256i shifted = _mm256_srli_epi64(x_int, 52);
	__m256i biasedExp = _mm256_and_si256(shifted, _mm256_set1_epi64x(0x7FF));
	__m256i exp = _mm256_sub_epi64(biasedExp, _mm256_set1_epi64x(1023));

	// NICE int_64 to double conversion

	// we want the lower 32 bit out of each of the 64 bits of the oprignal numbers
	// so shuffle, this treats the input as 8 sets of 32 bits
	// now at the end we have all the lower 32-bits for the 4 int64_t essentially

    const __m256i permute_mask = _mm256_set_epi32(0, 0, 0, 0, 6, 4, 2, 0);
    __m256i packed_lanes = _mm256_permutevar8x32_epi32(exp, permute_mask);
    
	// treat __m256i as __m128i (ignore the first 128 bits)
    __m128i exp_packed = _mm256_castsi256_si128(packed_lanes);

    // Convert the packed 32-bit integers to doubles.
    __m256d exp_f = _mm256_cvtepi32_pd(exp_packed);

	// Mantissa
	const __m256d mMask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x000FFFFFFFFFFFFFULL));
	__m256d mantissa = _mm256_and_pd(x, mMask);
	// mantissa obtained, now setting exp to 1023 
	const __m256d exponent_one   = _mm256_castsi256_pd(_mm256_set1_epi64x(0x3FF0000000000000LL));
	mantissa = _mm256_or_pd(mantissa, exponent_one);

	// mantissa range reduction because 
	const __m256d sqrt2 = setPD(1.4142135623730951);
    __m256d mask = _mm256_cmp_pd(mantissa, sqrt2, _CMP_GT_OQ);
    // If m > sqrt(2), then m_adj = m/2 and exp_adj = exp+1
    // Otherwise, m_adj = m and exp_adj = exp
	// choose a or b based on the mask
    __m256d mant_adjusted = _mm256_blendv_pd(mantissa, _mm256_mul_pd(mantissa, half), mask);
    exp_f = _mm256_blendv_pd(exp_f, _mm256_add_pd(exp_f, one), mask);

	// approximate polynomial
    __m256d z = _mm256_sub_pd(mant_adjusted, one);
    __m256d z2 = _mm256_mul_pd(z, z);
    // HORNER'S METHOD to evaluate P(z)
    __m256d poly = p11;
    poly = _mm256_fmadd_pd(poly, z, p10);
    poly = _mm256_fmadd_pd(poly, z, p9);
    poly = _mm256_fmadd_pd(poly, z, p8);
    poly = _mm256_fmadd_pd(poly, z, p7);
    poly = _mm256_fmadd_pd(poly, z, p6);
    poly = _mm256_fmadd_pd(poly, z, p5);
    poly = _mm256_fmadd_pd(poly, z, p4);
    poly = _mm256_fmadd_pd(poly, z, p3);
    poly = _mm256_fmadd_pd(poly, z, p2);
    poly = _mm256_fmadd_pd(poly, z, p1);
    poly = _mm256_fmadd_pd(poly, z, p0);
    
    // Reconstruct the final approximation from P(z): log(1+z) ≈ z + z^2 * P(z)
    poly = _mm256_fmadd_pd(poly, z2, z);

    __m256d result = _mm256_fmadd_pd(exp_f, log2_const, poly);
    

    __m256d zero = _mm256_setzero_pd();
    __m256d inf = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FF0000000000000LL));
    __m256d neg_inf = _mm256_castsi256_pd(_mm256_set1_epi64x(0xFFF0000000000000LL));
    __m256d nan = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FF8000000000000LL));
    
    // x < 0 -> NaN
    __m256d neg_mask = _mm256_cmp_pd(x, zero, _CMP_LT_OQ);
    result = _mm256_blendv_pd(result, nan, neg_mask);

    // x == 0 -> -inf
    __m256d zero_mask = _mm256_cmp_pd(x, zero, _CMP_EQ_OQ);
    result = _mm256_blendv_pd(result, neg_inf, zero_mask);
    
    // x == inf -> inf
    __m256d inf_mask = _mm256_cmp_pd(x, inf, _CMP_EQ_OQ);
    result = _mm256_blendv_pd(result, inf, inf_mask);
    
    // x == NaN -> NaN
    __m256d nan_mask = _mm256_cmp_pd(x, x, _CMP_UNORD_Q);
    result = _mm256_blendv_pd(result, nan, nan_mask);
    
    return result;
}

__m256d exp_avx(__m256d x){

    const __m256d log2_e = setPD(1.4426950408889634); // log2(e)
    const __m256d A      = setPD(1LL << 52);         // 2^52, the shift factor
    const __m256d B      = setPD(1023.0);            // IEEE 754 bias
    const __m256d one    = setPD(1);
    // x = x * log2(e)
    x = _mm256_mul_pd(x, log2_e);
    __m256d x_i = _mm256_floor_pd(x);
    __m256d xf = _mm256_sub_pd(x, x_i);

    // Your coefficients for K_n(xf) = 1 + xf - 2^xf
    const __m256d p0 = setPD(8.96880707e-9);
    const __m256d p1 = setPD(3.06852825e-1);
    const __m256d p2 = setPD(-2.40226805e-1);
    const __m256d p3 = setPD(-5.55041739e-2);
    const __m256d p4 = setPD(-9.61658346e-3);
    const __m256d p5 = setPD(-1.33314802e-3);
    const __m256d p6 = setPD(-1.56598145e-4);
    const __m256d p7 = setPD(-1.55032043e-5);

    // Evaluate the polynomial K_n(xf) using Horner's method
    __m256d poly = p7;
    poly = _mm256_fmadd_pd(poly, xf, p6);
    poly = _mm256_fmadd_pd(poly, xf, p5);
    poly = _mm256_fmadd_pd(poly, xf, p4);
    poly = _mm256_fmadd_pd(poly, xf, p3);
    poly = _mm256_fmadd_pd(poly, xf, p2);
    poly = _mm256_fmadd_pd(poly, xf, p1);
    poly = _mm256_fmadd_pd(poly, xf, p0);

    //x = x - K_n(xf)

    x = _mm256_sub_pd(x, poly);

    // now we have out answer hidden in x, the exponent part is xi and fractional part is xf (our mantissa)
    
    __m256d i_double = _mm256_fmadd_pd(A, x, _mm256_mul_pd(A, B));
    // // avx-512 method
    __m256i i_long = _mm256_cvtpd_epi64(i_double);

    //  step by step conversion
    // alignas(32) double temp_d[4];
    // alignas(32) long long temp_ll[4];
    // _mm256_store_pd(temp_d, i_double);
    // for(int i = 0; i < 4; ++i) {
    //     temp_ll[i] = (long long)temp_d[i];
    // }
    // __m256i i_long = _mm256_load_si256((__m256i*)temp_ll);

    __m256d result = _mm256_castsi256_pd(i_long);

    return result;
}

__m256d erfc_avx(__m256d x){
    __m256d zero = setPD(0);
    __m256d two = setPD(2);
    __m256d mask = _mm256_cmp_pd(x, zero, _CMP_GT_OQ);

    __m256d x_abs = _mm256_blendv_pd(-x, x, mask);

    __m256d p = setPD(0.3275911);
    __m256d a[5] = {setPD(0.254829592), setPD(-0.284496736), setPD(1.421413741), -setPD(1.453152027), setPD(1.061405429)};
    __m256d one = setPD(1);
    __m256d px = _mm256_mul_pd(x_abs, p);
    __m256d t = _mm256_div_pd(one, _mm256_add_pd(one, px));
    __m256d s = setPD(0);

    s = _mm256_mul_pd(t, _mm256_add_pd(s, a[4]));
    s = _mm256_mul_pd(t, _mm256_add_pd(s, a[3]));
    s = _mm256_mul_pd(t, _mm256_add_pd(s, a[2]));
    s = _mm256_mul_pd(t, _mm256_add_pd(s, a[1]));
    s = _mm256_mul_pd(t, _mm256_add_pd(s, a[0]));

    __m256d x_sqared = _mm256_mul_pd(x, x);
    s = _mm256_mul_pd(s, exp_avx(-x_sqared));
    s = _mm256_blendv_pd(two - s, s, mask);
    return s;
}

__m256d normal_cdf_avx(__m256d x){
    // 0.5 * std::erfc(-value * M_SQRT1_2);

    __m256d half = setPD(0.5);
    __m256d invSQRT2 = setPD(M_SQRT1_2);

    return _mm256_mul_pd(half, erfc_avx(_mm256_mul_pd(-x, invSQRT2)));
}