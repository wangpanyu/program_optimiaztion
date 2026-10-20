cmake_minimum_required(VERSION 3.16)
project(image_filter_opt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(FILTER_SOURCES
    src/main.cpp
    src/convolution_naive.cpp
    src/kernels.cpp
    src/image_io.cpp
    src/benchmark.cpp)

foreach(target IN ITEMS filter_bench filter_bench_scalar)
    add_executable(${target} ${FILTER_SOURCES})
    target_include_directories(${target} PRIVATE include)
    target_include_directories(${target} SYSTEM PRIVATE third_party)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
endforeach()
target_compile_options(filter_bench PRIVATE -O3 -march=native)
target_compile_options(filter_bench_scalar PRIVATE -O2 -fno-tree-vectorize)
target_compile_definitions(filter_bench_scalar PRIVATE FILTER_SCALAR=1)

enable_testing()
add_executable(test_convolution tests/test_convolution.cpp src/convolution_naive.cpp src/kernels.cpp src/benchmark.cpp)
target_include_directories(test_convolution PRIVATE include)
target_compile_options(test_convolution PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME convolution_correctness COMMAND test_convolution)

