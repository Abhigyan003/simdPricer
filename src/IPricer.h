#pragma once

#include<vector>
#include<cstdint>

class IPricer{
	public:
		virtual vector<double> price_all(
			const std :: vector<double>& spotPrices, 
			const std :: vector<double>& strikePrices,
			const std :: vector<double>& expirations,
			const std :: vector<double>& volatilities,
			const std :: vector<uint8_t>& is_call
			) = 0;
};
