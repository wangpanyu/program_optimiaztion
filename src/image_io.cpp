#include "image_io.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <stdexcept>

GrayImage load_gray_image(const std::string& filename) {
    int width = 0, height = 0, channels = 0;
    stbi_uc* rgb = stbi_load(filename.c_str(), &width, &height, &channels, 3);
    if (!rgb) {
        throw std::runtime_error("cannot decode image '" + filename + "': " +
                                 stbi_failure_reason());
    }
    if (width <= 0 || height <= 0 ||
        static_cast<uint64_t>(width) * height > std::numeric_limits<size_t>::max() / sizeof(float)) {
        stbi_image_free(rgb);
        throw std::runtime_error("image dimensions are invalid or too large");
    }
    GrayImage image;
    image.width = width;
    image.height = height;
    const size_t count = static_cast<size_t>(width) * height;
    image.pixels.resize(count);
    for (size_t i = 0; i < count; ++i) {
        const size_t base = i * 3;
        image.pixels[i] = static_cast<float>(
            (0.2126 * rgb[base] + 0.7152 * rgb[base + 1] + 0.0722 * rgb[base + 2]) / 255.0);
    }
    stbi_image_free(rgb);
    return image;
}

void save_gray_image(const std::string& filename, const GrayImage& image,
                     bool absolute_value) {
    if (image.width <= 0 || image.height <= 0 ||
        image.pixels.size() != static_cast<size_t>(image.width) * image.height) {
        throw std::invalid_argument("invalid output image");
    }
    std::vector<uint8_t> bytes(image.pixels.size());
    for (size_t i = 0; i < bytes.size(); ++i) {
        const float value = absolute_value ? std::abs(image.pixels[i]) : image.pixels[i];
        bytes[i] = static_cast<uint8_t>(std::lround(std::clamp(value, 0.0f, 1.0f) * 255.0f));
    }
    const std::filesystem::path path(filename);
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    if (!stbi_write_png(filename.c_str(), image.width, image.height, 1,
                        bytes.data(), image.width)) {
        throw std::runtime_error("cannot save image: " + filename);
    }
}

