#include "include/IPricer.h"
class ScalarPricer : public IPricer{
    public:
    std :: vector<double> price_all(
    const std :: vector<double>& spotPrices, 
    const std :: vector<double>& strikePrices,
    const std :: vector<double>& expirations,
    const std :: vector<double>& riskFreeRates,
    const std :: vector<double>& volatilities,
    const std :: vector<long long>& is_call
    ) override;
};