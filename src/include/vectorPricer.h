#pragma once

#include "IPricer.h"
#include <vector>
#include <cstdint>
#include <immintrin.h>
class VectorPricer : public IPricer{
public:
	std :: vector<double> price_all(
	const std :: vector<double>& spotPrices,
	const std :: vector<double>& strikePrices,
	const std :: vector<double>& expirations,
	const std :: vector<double>& riskFreeRates,
	const std :: vector<double>& volatilities,
	const std :: vector<long long>& is_call
	)override;

private:
	__m256d put(__m256d S,__m256d K,  __m256d r, __m256d T, __m256d d1, __m256d d2);
	__m256d call(__m256d S,__m256d K,  __m256d r, __m256d T, __m256d d1, __m256d d2);
};
