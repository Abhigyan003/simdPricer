#include "include/vectorPricer.h"
#include "include/utils.h"
#include "include/pricer.h"
#include <vector>
#include <cstdint>
#include <immintrin.h>

//	double d1 = (std::log(S / K) + (r + 0.5 * v * v) * T) / (v * std::sqrt(T));
//	double d2 = d1 - v * sqrt(T);
// 	if(is_call)
// 		return S * normal_cdf(d1) - K * std::exp(-r * T) * normal_cdf(d2);
// 	return K * std::exp(-r * T) * normal_cdf(-d2) - S * normal_cdf(-d1);


__m256d VectorPricer::call(__m256d S,__m256d K,  __m256d r, __m256d T, __m256d d1, __m256d d2){
	d1 = normal_cdf_avx(d1);
	d2 = normal_cdf_avx(d2);
	__m256d rt = _mm256_mul_pd(r, T);
	__m256d term1 = _mm256_mul_pd(S, d1);
	__m256d term2 = _mm256_mul_pd(-K, _mm256_mul_pd(exp_avx(-rt), d2));
	return _mm256_add_pd(term1, term2);
}

__m256d VectorPricer::put(__m256d S,__m256d K,  __m256d r, __m256d T, __m256d d1, __m256d d2){
	d1 = normal_cdf_avx(-d1);
	d2 = normal_cdf_avx(-d2);
	__m256d rt = _mm256_mul_pd(r, T);
	__m256d term1 = _mm256_mul_pd(K, _mm256_mul_pd(exp_avx(-rt), d2));
	__m256d term2 = _mm256_mul_pd(-S, d1);
	return _mm256_add_pd(term1, term2);
}

std :: vector<double> VectorPricer :: price_all(
	const std :: vector<double>& spotPrices,
	const std :: vector<double>& strikePrices,
	const std :: vector<double>& expirations,
	const std :: vector<double>& riskFreeRates,
	const std :: vector<double>& volatilities,
	const std :: vector<long long>& is_call
	) 
{
	__m256i zero = _mm256_set1_epi64x(0LL);
	int numberOptions = spotPrices.size();
	std :: vector<double>results(numberOptions);
	for(int i=0;i+3<numberOptions;i+=4){
		__m256d spotPriceV = _mm256_loadu_pd(&spotPrices[i]);
		__m256d strikePriceV = _mm256_loadu_pd(&strikePrices[i]);
		__m256d expirationV = _mm256_loadu_pd(&expirations[i]);
		__m256d riskFreeRateV = _mm256_loadu_pd(&riskFreeRates[i]);
		__m256d volatilityV = _mm256_loadu_pd(&volatilities[i]);
		__m256i callV = _mm256_loadu_si256((__m256i*)&is_call[i]);
		
		// v * std :: sqrt(T)
		__m256d sqrt_T = _mm256_sqrt_pd(expirationV);
		__m256d denominator_d1 = _mm256_mul_pd(volatilityV, sqrt_T);
		
		// log(S / K)
		__m256d num_term1_d1 = log_avx(_mm256_div_pd(spotPriceV, strikePriceV));

		// (r + 0.5 * v * v) * T
		__m256d squared_v = _mm256_mul_pd(volatilityV, volatilityV);
		__m256d half = _mm256_set1_pd(0.5);
		__m256d half_squared_v = _mm256_mul_pd(half, squared_v);
		__m256d num_term2_d1 = _mm256_mul_pd(_mm256_add_pd(riskFreeRateV, half_squared_v), expirationV);

		// numerator
		__m256d numerator_d1 = _mm256_add_pd(num_term1_d1, num_term2_d1);

		// double d1 = (std :: log (S / K) + (r + 0.5 * v * v ) * T) / (v * std :: sqrt(T))
		__m256d d1 = _mm256_div_pd(numerator_d1, denominator_d1);

		// double d2 = d1 - v * sdt::sqrt(T)
		__m256d d2 = _mm256_sub_pd(d1, denominator_d1);

		__m256i mask = _mm256_cmpeq_epi64(zero, callV);
		__m256d double_mask = _mm256_castsi256_pd(mask);
		__m256d res = _mm256_blendv_pd(call(spotPriceV, strikePriceV, riskFreeRateV, expirationV, d1, d2), put(spotPriceV, strikePriceV, riskFreeRateV, expirationV, d1, d2), double_mask);
		_mm256_storeu_pd(&results[i], res);
	}

	for(int i=(numberOptions>>2<<2);i<numberOptions;++i){
		results[i] = black_scholes_price(spotPrices[i], strikePrices[i], expirations[i], riskFreeRates[i], volatilities[i], is_call[i]);
	}

	return results;
}
