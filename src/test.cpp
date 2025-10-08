#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "include/utils.h" // Your header with AVX functions

// Helper function to print results in a formatted table
void print_results(const std::string& func_name, const std::vector<double>& inputs, const std::vector<double>& scalar_results, const std::vector<double>& avx_results) {
    std::cout << "\n--- Testing " << func_name << " ---" << std::endl;
    std::cout << std::left << std::setw(20) << "Input"
              << std::setw(25) << "Scalar Result"
              << std::setw(25) << "AVX Result"
              << "Difference" << std::endl;
    std::cout << "----------------------------------------------------------------------------------" << std::endl;
    
    for (size_t i = 0; i < inputs.size(); ++i) {
        std::cout << std::fixed << std::setprecision(12)
                  << std::left << std::setw(20) << inputs[i]
                  << std::setw(25) << scalar_results[i]
                  << std::setw(25) << avx_results[i]
                  << std::scientific << (scalar_results[i] - avx_results[i]) << std::fixed
                  << std::endl;
    }
}

// === TEST HARNESS FOR EACH FUNCTION ===

void test_exp_avx() {
    std::vector<double> inputs = {0.0, 1.0, -2.5, 4.0, 0.5, -10.0, 20.0, -0.01};
    std::vector<double> scalar_results(inputs.size());
    std::vector<double> avx_results(inputs.size());

    for (size_t i = 0; i < inputs.size(); ++i) {
        scalar_results[i] = std::exp(inputs[i]);
    }

    for (size_t i = 0; i < inputs.size(); i += 4) {
        __m256d input_vec = _mm256_loadu_pd(&inputs[i]);
        __m256d result_vec = exp_avx(input_vec);
        _mm256_storeu_pd(&avx_results[i], result_vec);
    }

    print_results("exp_avx", inputs, scalar_results, avx_results);
}

void test_log_avx() {
    std::vector<double> inputs = {1.0, 2.718, 10.0, 0.5, 1000.0, 0.001, 42.123, 1e6};
    std::vector<double> scalar_results(inputs.size());
    std::vector<double> avx_results(inputs.size());

    for (size_t i = 0; i < inputs.size(); ++i) {
        scalar_results[i] = std::log(inputs[i]);
    }
    
    for (size_t i = 0; i < inputs.size(); i += 4) {
        __m256d input_vec = _mm256_loadu_pd(&inputs[i]);
        __m256d result_vec = log_avx(input_vec);
        _mm256_storeu_pd(&avx_results[i], result_vec);
    }
    
    print_results("log_avx", inputs, scalar_results, avx_results);
}

void test_erfc_avx() {
    std::vector<double> inputs = {0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -3.0, 0.1};
    std::vector<double> scalar_results(inputs.size());
    std::vector<double> avx_results(inputs.size());

    for (size_t i = 0; i < inputs.size(); ++i) {
        scalar_results[i] = std::erfc(inputs[i]);
    }
    
    for (size_t i = 0; i < inputs.size(); i += 4) {
        __m256d input_vec = _mm256_loadu_pd(&inputs[i]);
        __m256d result_vec = erfc_avx(input_vec);
        _mm256_storeu_pd(&avx_results[i], result_vec);
    }
    
    print_results("erfc_avx", inputs, scalar_results, avx_results);
}

// Scalar implementation for Normal CDF to check against
double scalar_normal_cdf(double x) {
    return 0.5 * std::erfc(-x / M_SQRT2);
}

void test_normal_cdf_avx() {
    std::vector<double> inputs = {0.0, 1.96, -1.96, 1.0, -1.0, 3.0, -3.0, 0.0};
    std::vector<double> scalar_results(inputs.size());
    std::vector<double> avx_results(inputs.size());

    for (size_t i = 0; i < inputs.size(); ++i) {
        scalar_results[i] = scalar_normal_cdf(inputs[i]);
    }
    
    for (size_t i = 0; i < inputs.size(); i += 4) {
        __m256d input_vec = _mm256_loadu_pd(&inputs[i]);
        __m256d result_vec = normal_cdf_avx(input_vec);
        _mm256_storeu_pd(&avx_results[i], result_vec);
    }
    
    print_results("normal_cdf_avx", inputs, scalar_results, avx_results);
}

// Main runner
int main() {
    // test_exp_avx();
    // test_log_avx();
    test_erfc_avx();
    // test_normal_cdf_avx();
    return 0;
}