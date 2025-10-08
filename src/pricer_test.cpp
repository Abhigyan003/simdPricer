#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include "include/scalarPricer.h"
#include "include/vectorPricer.h"
#include "include/utils.h"

// Function to generate random data for testing
void generate_random_data(
    std::vector<double>& spotPrices,
    std::vector<double>& strikePrices,
    std::vector<double>& expirations,
    std::vector<double>& riskFreeRates,
    std::vector<double>& volatilities,
    std::vector<long long>& is_call,
    size_t size
) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    spotPrices.reserve(size);
    strikePrices.reserve(size);
    expirations.reserve(size);
    riskFreeRates.reserve(size);
    volatilities.reserve(size);
    is_call.reserve(size);

    for (size_t i = 0; i < size; ++i) {
        spotPrices.push_back(dis(gen) * 100 + 50);
        strikePrices.push_back(dis(gen) * 100 + 50);
        expirations.push_back(dis(gen) * 2 + 0.1);
        riskFreeRates.push_back(dis(gen) * 0.05);
        volatilities.push_back(dis(gen) * 0.4 + 0.1);
        is_call.push_back(dis(gen) > 0.5);
    }
}

int main() {
    size_t data_size = 100000000;
    std::vector<double> spotPrices, strikePrices, expirations, riskFreeRates, volatilities;
    std::vector<long long> is_call;

    generate_random_data(spotPrices, strikePrices, expirations, riskFreeRates, volatilities, is_call, data_size);

    ScalarPricer scalar_pricer;
    VectorPricer vector_pricer;

    auto start_scalar = std::chrono::high_resolution_clock::now();
    std::vector<double> scalar_prices = scalar_pricer.price_all(spotPrices, strikePrices, expirations, riskFreeRates, volatilities, is_call);
    auto end_scalar = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> scalar_time = end_scalar - start_scalar;

    auto start_vector = std::chrono::high_resolution_clock::now();
    std::vector<double> vector_prices = vector_pricer.price_all(spotPrices, strikePrices, expirations, riskFreeRates, volatilities, is_call);
    auto end_vector = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> vector_time = end_vector - start_vector;

    double max_diff = 0.0;
    for (size_t i = 0; i < data_size; ++i) {
        double diff = std::abs(scalar_prices[i] - vector_prices[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    std::cout << "Data size: " << data_size << std::endl;
    std::cout << "Max difference between scalar and vector pricers: " << max_diff << std::endl;
    std::cout << "Scalar pricer execution time: " << scalar_time.count() << " seconds" << std::endl;
    std::cout << "Vector pricer execution time: " << vector_time.count() << " seconds" << std::endl;
    std::cout << "Time difference (scalar - vector): " << scalar_time.count() - vector_time.count() << " seconds" << std::endl;

    return 0;
}