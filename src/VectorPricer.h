#pragma once

#include "IPricer.h";
#include <vector>
#include <cstdint>
class VectorPricer : public IPricer{
public:
	std :: vector<double> price_all(
	const std :: vector<double>& spotPrices,
	const std :: vector<double>& strikePrices,
	const std :: vector<double>& expirations,
	const std :: vector<double>& volatility,
	const std :: vector<uint8_t>& is_call,
	) override;
};
