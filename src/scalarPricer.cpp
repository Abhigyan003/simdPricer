#include "include/pricer.h"
#include "include/scalarPricer.h"

std::vector<double> ScalarPricer::price_all(
const std :: vector<double>& spotPrices, 
const std :: vector<double>& strikePrices,
const std :: vector<double>& expirations,
const std :: vector<double>& riskFreeRates,
const std :: vector<double>& volatilities,
const std :: vector<long long>& is_call
)
{
    size_t numberOptions = strikePrices.size();
    std::vector<double>results(numberOptions);
    for(int i=0;i<numberOptions;i++){
        results[i] = black_scholes_price(spotPrices[i], strikePrices[i], expirations[i], riskFreeRates[i], volatilities[i], is_call[i]);
    }
    return results;
}