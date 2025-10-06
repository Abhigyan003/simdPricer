#pragma once

#include<vector>
#include<cstdint>

struct DataManger{
	std :: vector<double> spot_prices;
	std :: vector<double> strike_prices;
	std :: vector<double> expirations;
	std :: vector<double> riskFreeRates;
	std :: vector<double> volatilities;
	std :: vector<uint8_t> is_call;
};
