#pragma once

#include "convolution.h"

#include <cstddef>
#include <vector>

struct ErrorMetrics {
    double max_abs_error = 0.0;
    double mean_abs_error = 0.0;
};

struct TimingStats {
    std::vector<double> samples_ms;
    double min_ms = 0.0;
    double mean_ms = 0.0;
    double median_ms = 0.0;
    double checksum = 0.0;
};

ErrorMetrics compare_outputs(const float* reference, const float* actual, std::size_t count);
TimingStats benchmark_convolution(ConvolutionFunction function,
                                  const float* input, float* output,
                                  int width, int height, const float* kernel,
                                  int kernel_size, int warmup, int repeat);

