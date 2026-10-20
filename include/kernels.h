#pragma once

#include <string>
#include <vector>

std::vector<float> make_gaussian_kernel(int size, double sigma = 0.0);
std::vector<float> make_kernel(const std::string& filter, int size, double sigma = 0.0);

