cmake_minimum_required(VERSION 3.10)

project(SIMD_Pricer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(simd_pricer
	src/main.cpp
	src/pricer.cpp
	src/utils.cpp
)

add_executable(pricer_test
	src/pricer_test.cpp
	src/scalarPricer.cpp
	src/vectorPricer.cpp
	src/pricer.cpp
	src/utils.cpp
)

enable_testing()
add_test(NAME PricerTest COMMAND pricer_test)


target_include_directories(simd_pricer PUBLIC src)

target_compile_options(simd_pricer PRIVATE -O3 -g -march=native -mavx2 -mfma -Wall)

target_include_directories(pricer_test PUBLIC src)

target_compile_options(pricer_test PRIVATE -O3 -g -march=native -mavx2 -mfma -Wall)

