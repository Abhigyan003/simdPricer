#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h> // For PRId64

__m256d log_pd_avx2(__m256d x) {
    // STEP 1: DEFINE CONSTANTS FOR DOUBLE PRECISION
    // ==============================================
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d log2_const = _mm256_set1_pd(0.693147180559945309417);
    
    // POLYNOMIAL COEFFICIENTS (12 terms, minimax optimized for double precision)
    // For P(z) = (log(1+z) - z) / z^2
    // ========================================================================
    const __m256d p0  = _mm256_set1_pd(-0.50000000000000000000);
    const __m256d p1  = _mm256_set1_pd(0.33333333333332931888);
    const __m256d p2  = _mm256_set1_pd(-0.24999999999732151675);
    const __m256d p3  = _mm256_set1_pd(0.19999999997186632439);
    const __m256d p4  = _mm256_set1_pd(-0.16666666649282846147);
    const __m256d p5  = _mm256_set1_pd(0.14285714013327126139);
    const __m256d p6  = _mm256_set1_pd(-0.12500000185585285493);
    const __m256d p7  = _mm256_set1_pd(0.11111096773344331233);
    const __m256d p8  = _mm256_set1_pd(-0.09999912071983321162);
    const __m256d p9  = _mm256_set1_pd(0.09090908869853962393);
    const __m256d p10 = _mm256_set1_pd(-0.08333332768783457174);
    const __m256d p11 = _mm256_set1_pd(0.07692307694389229853);

    // STEP 2: EXTRACT EXPONENT
    // =====================================
    __m256i x_int = _mm256_castpd_si256(x);
    __m256i exp_i = _mm256_srli_epi64(x_int, 52);
    exp_i = _mm256_and_si256(exp_i, _mm256_set1_epi64x(0x7FF));
    __m256i exp_unbiased = _mm256_sub_epi64(exp_i, _mm256_set1_epi64x(1023));

    // Convert 64-bit integer exponent to double using SIMD
    // Create a shuffle mask to select the low 32 bits of each 64-bit lane.
    // The source register has 8 x 32-bit lanes: [7, 6, 5, 4, 3, 2, 1, 0].
    // We want to gather lanes 0, 2, 4, and 6 into the bottom half of a new register.
    const __m256i permute_mask = _mm256_set_epi32(0, 0, 0, 0, 6, 4, 2, 0);

    // Apply the mask to permute the 32-bit lanes across the register.
    __m256i packed_lanes = _mm256_permutevar8x32_epi32(exp_unbiased, permute_mask);
    
    // The low 128 bits of packed_lanes now hold our 4 desired int32 values.
    // Cast the 256-bit vector to a 128-bit one to isolate them.
    __m128i exp_packed = _mm256_castsi256_si128(packed_lanes);

    // Convert the packed 32-bit integers to doubles.
    __m256d exp_f = _mm256_cvtepi32_pd(exp_packed);
    
    // STEP 3: EXTRACT AND NORMALIZE MANTISSA
    // =======================================
    // Get mantissa m in [1.0, 2.0) by forcing the exponent to 0 (biased 1023)
    const __m256d mant_mask_nosign = _mm256_castsi256_pd(_mm256_set1_epi64x(0x000FFFFFFFFFFFFFULL));
    const __m256d exponent_one   = _mm256_castsi256_pd(_mm256_set1_epi64x(0x3FF0000000000000LL));
    __m256d m = _mm256_or_pd(_mm256_and_pd(x, mant_mask_nosign), exponent_one);
    
    // STEP 4: RANGE REDUCTION
    // ========================
    const __m256d sqrt2 = _mm256_set1_pd(1.4142135623730951);
    __m256d mask = _mm256_cmp_pd(m, sqrt2, _CMP_GT_OQ);

    // If m > sqrt(2), then m_adj = m/2 and exp_adj = exp+1
    // Otherwise, m_adj = m and exp_adj = exp
    __m256d mant_adjusted = _mm256_blendv_pd(m, _mm256_mul_pd(m, half), mask);
    exp_f = _mm256_blendv_pd(exp_f, _mm256_add_pd(exp_f, one), mask);

    // STEP 5: COMPUTE ln(mantissa) WITH MINIMAX POLYNOMIAL
    // =======================================================
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

    // STEP 6: COMBINE RESULTS
    // =======================
    __m256d result = _mm256_fmadd_pd(exp_f, log2_const, poly);
    
    // STEP 7: HANDLE SPECIAL CASES
    // =============================
    __m256d zero = _mm256_setzero_pd();
    __m256d inf = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FF0000000000000LL));
    __m256d nan = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FF8000000000000LL));
    
    // x <= 0 -> NaN
    __m256d invalid_mask = _mm256_cmp_pd(x, zero, _CMP_LE_OQ);
    result = _mm256_blendv_pd(result, nan, invalid_mask);
    
    // x == inf -> inf
    __m256d inf_mask = _mm256_cmp_pd(x, inf, _CMP_EQ_OQ);
    result = _mm256_blendv_pd(result, inf, inf_mask);
    
    // x == NaN -> NaN
    __m256d nan_mask = _mm256_cmp_pd(x, x, _CMP_UNORD_Q);
    result = _mm256_blendv_pd(result, nan, nan_mask);
    
    return result;
}

// Log base 10 for doubles
__m256d log10_pd_avx2(__m256d x) {
    const __m256d log10_e = _mm256_set1_pd(0.43429448190325182765);
    return _mm256_mul_pd(log_pd_avx2(x), log10_e);
}

// Log base 2 for doubles
__m256d log2_pd_avx2(__m256d x) {
    const __m256d log2_e = _mm256_set1_pd(1.4426950408889634073);
    return _mm256_mul_pd(log_pd_avx2(x), log2_e);
}

void test_log_avx2_double() {
    double inputs[4] = {0.5, M_E, 10.0, 100.0};
    double results[4];
    
    __m256d x = _mm256_loadu_pd(inputs);
    __m256d y = log_pd_avx2(x);
    _mm256_storeu_pd(results, y);
    
    printf("AVX2 log(x) DOUBLE PRECISION results:\n");
    printf("%-15s %-25s %-25s %-15s %-12s\n", 
           "Input", "AVX2 Result", "Reference", "Error", "ULP Error");
    printf("------------------------------------------------------------------------------------------\n");
    
    for (int i = 0; i < 4; i++) {
        double reference = log(inputs[i]);
        double error = fabs(results[i] - reference);
        
        union { double d; int64_t i; } ref_u = {reference};
        union { double d; int64_t i; } res_u = {results[i]};
        int64_t ulp_error = llabs(ref_u.i - res_u.i);
        
        printf("%-15.10f %-25.17f %-25.17f %-15.2e %-12" PRId64 "\n", 
               inputs[i], results[i], reference, error, ulp_error);
    }
    
    printf("\nExtensive testing:\n");
    double test_vals[4];
    for (int batch = 0; batch < 5; batch++) {
        switch(batch) {
            case 0: 
                test_vals[0] = 0.1; test_vals[1] = 1.0; 
                test_vals[2] = 2.0; test_vals[3] = 7.389;
                break;
            case 1:
                test_vals[0] = 1e-10; test_vals[1] = 1e-5;
                test_vals[2] = 1e5; test_vals[3] = 1e10;
                break;
            case 2:
                test_vals[0] = 0.707; test_vals[1] = 1.414;
                test_vals[2] = 0.5; test_vals[3] = 2.0;
                break;
            case 3:
                test_vals[0] = M_PI; test_vals[1] = M_E * M_E;
                test_vals[2] = sqrt(2.0); test_vals[3] = 1.0 / M_E;
                break;
            case 4:
                test_vals[0] = 1.0 + 1e-15; test_vals[1] = 1.0 - 1e-15;
                test_vals[2] = 2.0 - 1e-15; test_vals[3] = 2.0 + 1e-15;
                break;
        }
        
        __m256d x_test = _mm256_loadu_pd(test_vals);
        __m256d y_test = log_pd_avx2(x_test);
        _mm256_storeu_pd(results, y_test);
        
        for (int i = 0; i < 4; i++) {
            double ref = log(test_vals[i]);
            double err = fabs(results[i] - ref);
            union { double d; int64_t i; } ref_u = {ref};
            union { double d; int64_t i; } res_u = {results[i]};
            int64_t ulp = llabs(ref_u.i - res_u.i);
            printf("log(%13.6e) = %20.15f (ref: %20.15f) ULP: %3" PRId64 "\n",
                   test_vals[i], results[i], ref, ulp);
        }
    }
    
    printf("\nSpecial cases:\n");
    double special[4] = {0.0, -1.0, INFINITY, NAN};
    __m256d x_special = _mm256_loadu_pd(special);
    __m256d y_special = log_pd_avx2(x_special);
    _mm256_storeu_pd(results, y_special);
    
    const char* names[4] = {"0.0", "-1.0", "+inf", "NaN"};
    for (int i = 0; i < 4; i++) {
        printf("log(%6s) = %20.15f\n", names[i], results[i]);
    }
}

int main() {
    test_log_avx2_double();
    return 0;
}