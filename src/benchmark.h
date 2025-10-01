#pragma once

#include <chrono>
#include <functional>

struct BenchmarkResult {
    double duration_s;
    double duration_ms;
    double duration_us;
};

class Benchmark {
public:
    BenchmarkResult run(const std::function<void()>& test) {
        auto start = std::chrono::high_resolution_clock::now();

        test();

        auto end = std::chrono::high_resolution_clock::now();

        auto duration = end - start;

        BenchmarkResult result;

        // Convert the duration object to a double representing seconds
        result.duration_s = std::chrono::duration<double>(duration).count();

        // Convert the duration object to a double representing milliseconds
        result.duration_ms = std::chrono::duration<double, std::milli>(duration).count();

        // Convert the duration object to a double representing microseconds
        result.duration_us = std::chrono::duration<double, std::micro>(duration).count();

        return result;
    }
};
