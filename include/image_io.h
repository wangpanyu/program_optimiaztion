#pragma once

#include <string>
#include <vector>

struct GrayImage {
    int width = 0;
    int height = 0;
    std::vector<float> pixels;
};

GrayImage load_gray_image(const std::string& filename);
void save_gray_image(const std::string& filename, const GrayImage& image,
                     bool absolute_value = false);

