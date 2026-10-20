#pragma once

using ConvolutionFunction = void(*)(
    const float*, float*, int, int, const float*, int);

// Cross-correlation convention: the kernel is not flipped. Zero padding.
void convolve2d_naive(const float* input, float* output, int width, int height,
                      const float* kernel, int kernel_size);

