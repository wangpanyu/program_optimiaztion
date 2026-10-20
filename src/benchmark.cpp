#include "benchmark.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

ErrorMetrics compare_outputs(const float* reference, const float* actual, size_t count) {
    if (count == 0) throw std::invalid_argument("cannot compare empty outputs");
    ErrorMetrics result;
    double total = 0.0;
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(reference[i]) || !std::isfinite(actual[i])) {
            return {std::numeric_limits<double>::infinity(),
                    std::numeric_limits<double>::infinity()};
        }
        const double error = std::abs(static_cast<double>(reference[i]) - actual[i]);
        result.max_abs_error = std::max(result.max_abs_error, error);
        total += error;
    }
    result.mean_abs_error = total / static_cast<double>(count);
    return result;
}

TimingStats benchmark_convolution(ConvolutionFunction function,
                                  const float* input, float* output,
                                  int width, int height, const float* kernel,
                                  int kernel_size, int warmup, int repeat) {
    if (warmup < 0 || repeat <= 0) throw std::invalid_argument("invalid warmup or repeat");
    for (int i = 0; i < warmup; ++i)
        function(input, output, width, height, kernel, kernel_size);
    TimingStats stats;
    stats.samples_ms.reserve(static_cast<size_t>(repeat));
    for (int i = 0; i < repeat; ++i) {
        const auto start = std::chrono::steady_clock::now();
        function(input, output, width, height, kernel, kernel_size);
        const auto stop = std::chrono::steady_clock::now();
        stats.samples_ms.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }
    // Reading output outside the timer keeps the computed result observable.
    const size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    for (size_t i = 0; i < pixels; ++i) {
        if (!std::isfinite(output[i])) throw std::runtime_error("convolution produced NaN or Inf");
        stats.checksum += output[i];
    }
    auto sorted = stats.samples_ms;
    std::sort(sorted.begin(), sorted.end());
    stats.min_ms = sorted.front();
    stats.mean_ms = std::accumulate(sorted.begin(), sorted.end(), 0.0) / repeat;
    stats.median_ms = (sorted[(repeat - 1) / 2] + sorted[repeat / 2]) / 2.0;
    return stats;
}

