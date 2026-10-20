#include "kernels.h"

#include <cmath>
#include <stdexcept>

std::vector<float> make_gaussian_kernel(int size, double sigma) {
    if (size != 3 && size != 5 && size != 7) {
        throw std::invalid_argument("Gaussian kernel size must be 3, 5, or 7");
    }
    if (!std::isfinite(sigma) || sigma < 0.0) {
        throw std::invalid_argument("sigma must be finite and nonnegative");
    }
    if (sigma == 0.0) sigma = static_cast<double>(size) / 3.0;
    std::vector<float> kernel(static_cast<size_t>(size) * size);
    double total = 0.0;
    const int radius = size / 2;
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const int dx = x - radius;
            const int dy = y - radius;
            const double value = std::exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma));
            kernel[static_cast<size_t>(y) * size + x] = static_cast<float>(value);
            total += value;
        }
    }
    for (float& value : kernel) value = static_cast<float>(value / total);
    return kernel;
}

std::vector<float> make_kernel(const std::string& filter, int size, double sigma) {
    if (filter == "gaussian") return make_gaussian_kernel(size, sigma);
    if (size != 3) throw std::invalid_argument(filter + " requires a 3x3 kernel");
    if (filter == "sharpen") return {0, -1, 0, -1, 5, -1, 0, -1, 0};
    if (filter == "sobel" || filter == "sobel-x") return {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    if (filter == "sobel-y") return {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    throw std::invalid_argument("unknown filter: " + filter);
}

