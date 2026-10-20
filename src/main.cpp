#include "benchmark.h"
#include "convolution.h"
#include "image_io.h"
#include "kernels.h"

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Options {
    std::string dataset;
    std::string input;
    std::string output;
    std::string csv;
    std::string summary;
    std::string filter = "gaussian";
    int width = 0;
    int height = 0;
    int kernel = 0;
    int warmup = -1;
    int repeat = -1;
    int max_images = 0;
    double sigma = 0.0;
    uint32_t seed = 2026;
};

int parse_int(const std::string& value, const std::string& name) {
    size_t used = 0;
    const long parsed = std::stol(value, &used);
    if (used != value.size() || parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max())
        throw std::invalid_argument("invalid " + name + ": " + value);
    return static_cast<int>(parsed);
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help" || key == "-h") {
            std::cout << "Usage: filter_bench [--dataset DIR | --input FILE [--output FILE] | "
                         "--width W --height H] [--filter gaussian|sharpen|sobel|sobel-y] "
                         "[--kernel 3|5|7] [--sigma S] [--warmup N] [--repeat N] "
                         "[--max-images N] [--csv FILE] [--summary FILE] [--seed N]\n";
            std::exit(0);
        }
        if (i + 1 >= argc) throw std::invalid_argument("missing value for " + key);
        const std::string value = argv[++i];
        if (key == "--dataset") options.dataset = value;
        else if (key == "--input") options.input = value;
        else if (key == "--output") options.output = value;
        else if (key == "--csv") options.csv = value;
        else if (key == "--summary") options.summary = value;
        else if (key == "--filter") options.filter = value;
        else if (key == "--width") options.width = parse_int(value, key);
        else if (key == "--height") options.height = parse_int(value, key);
        else if (key == "--kernel") options.kernel = parse_int(value, key);
        else if (key == "--warmup") options.warmup = parse_int(value, key);
        else if (key == "--repeat") options.repeat = parse_int(value, key);
        else if (key == "--max-images") options.max_images = parse_int(value, key);
        else if (key == "--seed") {
            const int seed = parse_int(value, key);
            if (seed < 0) throw std::invalid_argument("seed must be nonnegative");
            options.seed = static_cast<uint32_t>(seed);
        } else if (key == "--sigma") {
            size_t used = 0;
            options.sigma = std::stod(value, &used);
            if (used != value.size()) throw std::invalid_argument("invalid sigma");
        } else throw std::invalid_argument("unknown option: " + key);
    }
    const bool dataset = !options.dataset.empty();
    const bool image = !options.input.empty();
    if (dataset && image) throw std::invalid_argument("choose one of --dataset or --input");
    if ((dataset || image) && (options.width || options.height))
        throw std::invalid_argument("--width/--height are only for synthetic input");
    if (!dataset && !image && (options.width <= 0 || options.height <= 0))
        throw std::invalid_argument("synthetic input requires positive --width and --height");
    if (options.max_images < 0) throw std::invalid_argument("max-images must be nonnegative");
    if (options.kernel == 0) options.kernel = options.filter == "gaussian" ? 5 : 3;
    if (options.warmup < 0) options.warmup = dataset ? 1 : 3;
    if (options.repeat < 0) options.repeat = dataset ? 5 : 10;
    if (options.repeat <= 0) throw std::invalid_argument("repeat must be positive");
    if (dataset && options.filter != "gaussian")
        throw std::invalid_argument("DIV2K baseline benchmark uses Gaussian filter only");
    if (dataset && !options.output.empty())
        throw std::invalid_argument("--output is for single-image demo only");
    if (!image && !options.output.empty())
        throw std::invalid_argument("--output requires --input");
    return options;
}

std::string implementation_name() {
#ifdef FILTER_SCALAR
    return "naive_scalar";
#else
    return "naive_native";
#endif
}

size_t checked_pixels(int width, int height) {
    if (width <= 0 || height <= 0 ||
        static_cast<int64_t>(width) * height > std::numeric_limits<int>::max())
        throw std::invalid_argument("image dimensions exceed supported range");
    return static_cast<size_t>(width) * height;
}

void ensure_parent(const std::string& name) {
    const fs::path parent = fs::path(name).parent_path();
    if (!parent.empty()) fs::create_directories(parent);
}

std::string sample_list(const std::vector<double>& samples) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < samples.size(); ++i) {
        if (i) out << ';';
        out << samples[i];
    }
    return out.str();
}

double throughput(size_t pixels, double time_ms) {
    return static_cast<double>(pixels) / (time_ms * 1000.0);
}

void print_stats(const std::string& label, int width, int height,
                 int kernel_size, const TimingStats& stats) {
    const size_t pixels = checked_pixels(width, height);
    std::cout << std::fixed << std::setprecision(6)
              << "RESULT filename=" << label << " width=" << width << " height=" << height
              << " pixels=" << pixels << " kernel_size=" << kernel_size
              << " median_time_ms=" << stats.median_ms
              << " mean_time_ms=" << stats.mean_ms
              << " min_time_ms=" << stats.min_ms
              << " mpixels_s=" << throughput(pixels, stats.median_ms)
              << " checksum=" << std::setprecision(9) << stats.checksum
              << " sample_times_ms=" << sample_list(stats.samples_ms) << '\n';
}

void write_detail_header(std::ostream& out) {
    out << "implementation,filter,filename,width,height,pixels,kernel_size,"
           "median_time_ms,mean_time_ms,min_time_ms,mpixels_s,checksum,sample_times_ms\n";
}

void write_detail(std::ostream& out, const std::string& filter, const std::string& filename,
                  int width, int height, int kernel_size, const TimingStats& stats) {
    const size_t pixels = checked_pixels(width, height);
    out << implementation_name() << ',' << filter << ',' << filename << ','
        << width << ',' << height << ',' << pixels << ',' << kernel_size << ','
        << std::fixed << std::setprecision(6) << stats.median_ms << ','
        << stats.mean_ms << ',' << stats.min_ms << ','
        << throughput(pixels, stats.median_ms) << ','
        << std::setprecision(9) << stats.checksum << ',' << sample_list(stats.samples_ms) << '\n';
}

void run_dataset(const Options& options, const std::vector<float>& kernel) {
    const fs::path directory(options.dataset);
    if (!fs::is_directory(directory)) {
        throw std::runtime_error("DIV2K dataset not found.\nExpected path:\n" +
                                 options.dataset + "\nPlease see datasets/README.md for download instructions.");
    }
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".png")
            files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) throw std::runtime_error("no PNG images found in " + options.dataset);
    if (options.max_images > 0 && files.size() > static_cast<size_t>(options.max_images))
        files.resize(static_cast<size_t>(options.max_images));
    const std::string csv = options.csv.empty() ? "results/div2k_baseline.csv" : options.csv;
    const std::string summary = options.summary.empty() ? "results/div2k_summary.csv" : options.summary;
    ensure_parent(csv);
    ensure_parent(summary);
    std::ofstream detail(csv);
    if (!detail) throw std::runtime_error("cannot write " + csv);
    write_detail_header(detail);
    size_t total_pixels = 0;
    double total_ms = 0.0;
    std::vector<double> medians;
    for (const fs::path& path : files) {
        GrayImage input = load_gray_image(path.string());
        const size_t pixels = checked_pixels(input.width, input.height);
        std::vector<float> output(pixels);
        const TimingStats stats = benchmark_convolution(
            convolve2d_naive, input.pixels.data(), output.data(), input.width,
            input.height, kernel.data(), options.kernel, options.warmup, options.repeat);
        print_stats(path.filename().string(), input.width, input.height, options.kernel, stats);
        write_detail(detail, options.filter, path.filename().string(),
                     input.width, input.height, options.kernel, stats);
        total_pixels += pixels;
        total_ms += stats.median_ms;
        medians.push_back(stats.median_ms);
    }
    std::sort(medians.begin(), medians.end());
    const size_t n = medians.size();
    const double median = (medians[(n - 1) / 2] + medians[n / 2]) / 2.0;
    std::ofstream totals(summary);
    if (!totals) throw std::runtime_error("cannot write " + summary);
    totals << "implementation,filter,kernel_size,number_of_images,total_pixels,total_time_ms,"
              "mean_time_per_image_ms,median_time_per_image_ms,overall_mpixel_s\n"
           << implementation_name() << ',' << options.filter << ',' << options.kernel << ','
           << n << ',' << total_pixels << ',' << std::fixed << std::setprecision(6)
           << total_ms << ',' << total_ms / n << ',' << median << ','
           << throughput(total_pixels, total_ms) << '\n';
    std::cout << "Wrote " << csv << " and " << summary << '\n';
}

void run_single(const Options& options, const std::vector<float>& kernel) {
    GrayImage input = load_gray_image(options.input);
    GrayImage output;
    output.width = input.width;
    output.height = input.height;
    output.pixels.resize(checked_pixels(input.width, input.height));
    const TimingStats stats = benchmark_convolution(
        convolve2d_naive, input.pixels.data(), output.pixels.data(), input.width,
        input.height, kernel.data(), options.kernel, options.warmup, options.repeat);
    print_stats(fs::path(options.input).filename().string(), input.width, input.height,
                options.kernel, stats);
    if (!options.output.empty()) {
        save_gray_image(options.output, output,
                        options.filter == "sobel" || options.filter == "sobel-x" ||
                        options.filter == "sobel-y");
        std::cout << "Wrote " << options.output << '\n';
    }
}

void run_synthetic(const Options& options, const std::vector<float>& kernel) {
    const size_t pixels = checked_pixels(options.width, options.height);
    std::mt19937 engine(options.seed);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    std::vector<float> input(pixels);
    for (float& value : input) value = distribution(engine);
    std::vector<float> output(pixels);
    const TimingStats stats = benchmark_convolution(
        convolve2d_naive, input.data(), output.data(), options.width, options.height,
        kernel.data(), options.kernel, options.warmup, options.repeat);
    print_stats("synthetic", options.width, options.height, options.kernel, stats);
    if (!options.csv.empty()) {
        ensure_parent(options.csv);
        std::ofstream detail(options.csv);
        if (!detail) throw std::runtime_error("cannot write " + options.csv);
        write_detail_header(detail);
        write_detail(detail, options.filter, "synthetic", options.width, options.height,
                     options.kernel, stats);
        std::cout << "Wrote " << options.csv << '\n';
    }
}

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const auto kernel = make_kernel(options.filter, options.kernel, options.sigma);
        if (!options.dataset.empty()) run_dataset(options, kernel);
        else if (!options.input.empty()) run_single(options, kernel);
        else run_synthetic(options, kernel);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}

