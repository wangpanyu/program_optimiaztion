#include "benchmark.h"
#include "convolution.h"
#include "kernels.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    try {
        const float input[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        float output[9] = {};
        const float identity[] = {0, 0, 0, 0, 1, 0, 0, 0, 0};
        convolve2d_naive(input, output, 3, 3, identity, 3);
        const auto identity_error = compare_outputs(input, output, 9);
        check(identity_error.max_abs_error == 0.0, "identity kernel failed");

        const float ones[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
        const float expected[] = {12, 21, 16, 27, 45, 33, 24, 39, 28};
        convolve2d_naive(input, output, 3, 3, ones, 3);
        check(compare_outputs(expected, output, 9).max_abs_error == 0.0,
              "manual 3x3 convolution / zero padding failed");

        const float single = 5;
        float single_out = 0;
        convolve2d_naive(&single, &single_out, 1, 1, ones, 3);
        check(single_out == 5, "single-pixel zero padding failed");

        for (int size : {3, 5, 7}) {
            const auto gaussian = make_gaussian_kernel(size);
            const double sum = std::accumulate(gaussian.begin(), gaussian.end(), 0.0);
            check(std::abs(sum - 1.0) < 1e-6, "Gaussian normalization failed");
            check(std::isfinite(sum), "Gaussian kernel is non-finite");
        }
        output[0] = std::numeric_limits<float>::quiet_NaN();
        check(!std::isfinite(compare_outputs(input, output, 9).max_abs_error),
              "NaN comparison failed");

        std::cout << "PASS: identity, manual matrix, zero padding, Gaussian normalization, NaN check\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}

