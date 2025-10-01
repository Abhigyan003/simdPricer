#include "VectorPricer.h"
#include <vector>
#include <cstdint>
#include <immintrin.h>

//	double d1 = (std::log(S / K) + (r + 0.5 * v * v) * T) / (v * std::sqrt(T));
//	double d2 = d1 - v * sqrt(T);

std :: vector<double> VectorPricer :: price_all(
	const std :: vector<double>& spotPrices,
	const std :: vector<double>& strikePrices,
	const std :: vector<double>& expirations,
	const std :: vector<double>& riskFreeRates,
	const std :: vector<double>& volatilities,
	const std :: vector<uint8_t>& is_call,
	) 
{
	vector<double>results;
	size_t numberOptions = spotPrices.size();
	for(int i=0;i+3<numberOptions;i+=4){
		__m256d spotPriceV = _mm256_loadu_pd(&spotPrices[i]);
		__m256d strikePriceV = _mm256_loadu_pd(&strikePrices[i]);
		__m256d expirationV = _mm256_loadu_pd(&expirations[i]);
		__m256d riskFreeRateV = _mm256_loadu_pd(&riskFreeRates[i]);
		__m256d volatilityV = _mm256_loadu_pd(&volatilities[i]);
		
		// v * std :: sqrt(T)
		__m256d sqrt_T = _mm256_sqrt_pd(expirationV);
		__m256d denominator_d1 = _mm256_mul_pd(volatilityV, sqrt_T);
		
		// log(S / K)
		__m256d num_term1_d1 = _mm256_setzero_pd();

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

	}

	for(int i=(numberOptions>>2<<2);i<numberOptions;++i){
		
	}

	return results;

}
