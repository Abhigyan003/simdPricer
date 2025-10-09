# SIMD Pricer

## Overview

This project is a high-performance option pricer written in C++. It uses SIMD (Single Instruction, Multiple Data) instructions to accelerate the pricing of a large number of options. The project includes both a scalar and a vector implementation of the Black-Scholes model, and a benchmark to compare their performance.

## Features

- **Black-Scholes Model:** Implements the Black-Scholes model for pricing European call and put options.
- **Scalar and Vector Pricers:** Includes a scalar implementation for baseline performance and a SIMD-based vector implementation for high-performance pricing.
- **Performance Benchmark:** A benchmark is included to compare the performance of the scalar and vector pricers.
- **CMake Build System:** Uses CMake for easy cross-platform building.

## Building the Project

The project uses CMake to generate a build system. The following commands will build the project:

```bash
mkdir build
cd build
cmake ..
make
```

This will create two executables in the `build` directory: `simd_pricer` and `pricer_test`.

## Running the Pricer

The `simd_pricer` executable calculates the price of a single European call and put option with hardcoded values. To run it:

```bash
./build/simd_pricer
```

## Running the Benchmark

The `pricer_test` executable runs a benchmark that compares the performance of the scalar and vector pricers. It generates a large number of random options and prices them with both pricers, measuring the execution time and comparing the results.

To run the benchmark:

```bash
./build/pricer_test
```

The output will show the execution time of both pricers and the performance improvement of the vector pricer over the scalar pricer.
