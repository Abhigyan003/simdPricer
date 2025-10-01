cmake_minimum_required(VERSION 3.10)

project(SIMD_Pricer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(simd_pricer
	src/main.cpp
	src/pricer.cpp
	src/utils.cpp
)

target_include_directories(simd_pricer PUBLIC src)
target_compile_options(simd_pricer PRIVATE -O3 -g -mavx2 -Wall)
