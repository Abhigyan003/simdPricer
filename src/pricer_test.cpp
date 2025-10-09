#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include "include/scalarPricer.h"
#include "include/vectorPricer.h"
#include "include/utils.h"
#include "include/benchmark.h"

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
    size_t data_size = 10000000;
    std::vector<double> spotPrices, strikePrices, expirations, riskFreeRates, volatilities;
    std::vector<long long> is_call;

    generate_random_data(spotPrices, strikePrices, expirations, riskFreeRates, volatilities, is_call, data_size);

    Benchmark benchmark = Benchmark();
    ScalarPricer scalar_pricer;
    VectorPricer vector_pricer;
    
    std::vector<double> scalar_prices, vector_prices;
    BenchmarkResult scalar = benchmark.run([&]() -> void {
            scalar_prices = scalar_pricer.price_all(spotPrices, strikePrices, expirations, riskFreeRates, volatilities, is_call);
        }
    );

    BenchmarkResult vector = benchmark.run([&]() -> void {
            vector_prices = vector_pricer.price_all(spotPrices, strikePrices, expirations, riskFreeRates, volatilities, is_call);
        }
    );

    double max_diff = 0.0;
    double max_relative_diff = 0.0;
    for (size_t i = 0; i < data_size; ++i) {
        double diff = std::abs(scalar_prices[i] - vector_prices[i]);
        max_relative_diff = std::max(max_relative_diff, diff / scalar_prices[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }

    std::cout << "Data size: " << data_size << std::endl;
    std::cout << "Max difference between scalar and vector pricers: " << max_diff << std::endl;
    std::cout << "Max relative difference between scalar and vector pricers: " << max_relative_diff << std::endl;
    std::cout << "Scalar pricer execution time: " << scalar.duration_ms << " ms" << std::endl;
    std::cout << "Vector pricer execution time: " << vector.duration_ms << " ms" << std::endl;
    std::cout << "Improvement : " << scalar.duration_ms  / vector.duration_ms << " times improvement" << std::endl;

    return 0;
}