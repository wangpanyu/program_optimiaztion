#include "convolution.h"

void convolve2d_naive(const float* input, float* output, int width, int height,
                      const float* kernel, int kernel_size) {
    const int radius = kernel_size / 2;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {
                    const int iy = y + ky - radius;
                    const int ix = x + kx - radius;
                    if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                        sum += input[iy * width + ix] * kernel[ky * kernel_size + kx];
                    }
                }
            }
            output[y * width + x] = sum;
        }
    }
}

