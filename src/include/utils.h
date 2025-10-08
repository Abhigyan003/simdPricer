#pragma once

#include <cmath>
#include <immintrin.h>

double normal_cdf(double value);

// SIMD functions
__m256d log_avx(__m256d x);
__m256d exp_avx(__m256d x);
__m256d normal_cdf_avx(__m256d x);
__m256d erfc_avx(__m256d x);
