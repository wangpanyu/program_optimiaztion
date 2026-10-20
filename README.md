# Image Filter Optimization

一个用于学习和评估 **二维图像卷积性能优化** 的 C++17 项目。项目当前提供清晰、可验证的单线程 naive baseline，并围绕合成输入和 [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) 高分辨率图片建立统一的正确性测试、性能测量和绘图流程。

当前版本是 **V0 baseline**：核心卷积保持最直接的 `y → x → ky → kx` 四层循环，尚未加入可分离卷积、分块、SIMD 或 OpenMP。这样后续合作者可以在相同接口、测试和 benchmark 口径下逐步加入优化实现。

详细的已有实验结果和瓶颈分析见 [RESULTS_ANALYSIS.md](RESULTS_ANALYSIS.md)，数据集说明见 [datasets/README.md](datasets/README.md)。

## 功能概览

- 单线程 naive 2D convolution，输入为连续 row-major `float` 灰度图。
- 支持 Gaussian 3×3、5×5、7×7，以及 Sharpen、Sobel X/Y。
- 支持单张 PNG/JPEG、确定性合成输入和 DIV2K 目录三种运行模式。
- 计时区域只包含卷积函数，不包含图片读取、颜色转换、内存分配和结果保存。
- 输出每次运行时间、中位数、均值、最小值、吞吐率和 checksum。
- 提供 CTest 正确性测试、批量 benchmark 和结果绘图脚本。
- 同时构建 native 与禁止自动向量化的 scalar 版本，便于比较编译器优化影响。

## 环境要求

核心程序：

- Linux x86-64
- CMake 3.16+
- 支持 C++17 的 GCC（当前 scalar target 使用 GCC 的 `-fno-tree-vectorize`）

运行 Python 辅助脚本还需要 Python 3、NumPy、Pillow 和 Matplotlib：

```bash
python3 -m pip install numpy pillow matplotlib
```

图像读写使用仓库内自带的 `stb_image.h` 和 `stb_image_write.h`，构建核心程序不需要 OpenCV。

## 快速开始

克隆仓库后，在项目根目录执行：

```bash
git clone <repository-url>
cd image-filter-opt

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

无需下载数据集即可运行确定性合成 benchmark：

```bash
./build/filter_bench \
  --width 2048 --height 2048 \
  --filter gaussian --kernel 5 \
  --warmup 3 --repeat 10
```

程序会输出类似下面的单行结果，方便脚本解析：

```text
RESULT filename=synthetic ... median_time_ms=... mpixels_s=... checksum=...
```

## 构建目标

| 目标 | 编译选项 | 用途 |
| --- | --- | --- |
| `filter_bench` | `-O3 -march=native` | 本机优化 benchmark 和图片处理 |
| `filter_bench_scalar` | `-O2 -fno-tree-vectorize` | 禁止 GCC 自动向量化的参考版本 |
| `test_convolution` | 常规警告选项 | 正确性单元测试 |

`-march=native` 会生成依赖当前 CPU 的指令，因此 `filter_bench` 应在构建它的机器上运行。

## 运行单张图片

Gaussian 滤波：

```bash
./build/filter_bench \
  --input path/to/input.png \
  --output images/output.png \
  --filter gaussian --kernel 5 \
  --warmup 3 --repeat 10
```

其他滤波器：

```bash
./build/filter_bench --input path/to/input.png --output images/sharpen.png --filter sharpen --kernel 3
./build/filter_bench --input path/to/input.png --output images/sobel_x.png --filter sobel --kernel 3
./build/filter_bench --input path/to/input.png --output images/sobel_y.png --filter sobel-y --kernel 3
```

输入 RGB 图片会在计时前转换为 `[0,1]` 范围的单通道灰度 `float`。输出 PNG 会截断到 `[0,1]`；Sobel 输出在保存前取绝对值。核心计算采用 zero padding，kernel 按原方向使用，因此语义是图像处理中常见的 cross-correlation。

## 运行合成性能实验

以下脚本使用固定随机种子 2026，测试图片尺寸扩展和卷积核尺寸扩展：

```bash
python3 scripts/run_scaling_bench.py --exe build/filter_bench
python3 scripts/plot_results.py
```

默认实验包括：

- 图片尺寸：512²、1024²、2048²、4096²，统一使用 Gaussian 5×5。
- 卷积核尺寸：3×3、5×5、7×7，统一使用 2048² 输入。
- 每个配置预热 3 次并计时 10 次。

CSV 和图表写入 `results/`。

## 运行 DIV2K benchmark

本项目使用 DIV2K validation HR 的 100 张图片（`0801.png`–`0900.png`）。数据集体积较大，不提交到 Git。

下载并解压：

```bash
mkdir -p datasets/DIV2K
curl -L https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip \
  -o datasets/DIV2K/DIV2K_valid_HR.zip
unzip datasets/DIV2K/DIV2K_valid_HR.zip -d datasets/DIV2K
```

预期目录结构：

```text
datasets/DIV2K/DIV2K_valid_HR/
├── 0801.png
├── 0802.png
├── ...
└── 0900.png
```

先进行三张图片的 smoke test，再运行全部 100 张：

```bash
python3 scripts/run_div2k_bench.py \
  --exe build/filter_bench \
  --dataset datasets/DIV2K/DIV2K_valid_HR \
  --kernel 5 --max-images 3 \
  --output results/div2k_smoke.csv \
  --summary results/div2k_smoke_summary.csv

python3 scripts/run_div2k_bench.py \
  --exe build/filter_bench \
  --dataset datasets/DIV2K/DIV2K_valid_HR \
  --kernel 5
```

默认每张图片预热 1 次并计时 5 次。详细结果写入 `results/div2k_baseline.csv`，汇总结果写入 `results/div2k_summary.csv`。

生成结果图：

```bash
python3 scripts/plot_results.py
python3 scripts/plot_div2k_per_image.py
```

第二个脚本会生成适合 PPT 的 16:9 逐图耗时柱状图，同时导出 PNG 和可编辑 SVG。

## 演示图片

生成添加强高斯噪声后的企鹅图片及 Gaussian 去噪对照：

```bash
python3 scripts/make_penguin_denoise_demo.py --exe build/filter_bench
```

默认使用灰度噪声标准差 100、Gaussian 7×7、滤波 σ=2.5。该设置用于强化课堂展示效果，不属于正式 DIV2K 性能数据。

生成高频合成图的滤波对照：

```bash
python3 scripts/make_high_frequency_demo.py --exe build/filter_bench
```

对椒盐噪声图片运行可复现的中值滤波：

```bash
python3 scripts/denoise_dog.py \
  --input images/dog.png \
  --output images/dog_denoised_median.png
```

## 常用命令行参数

程序支持三种互斥输入方式：`--input FILE`、`--dataset DIR`，或同时指定 `--width W --height H` 的合成输入。

| 参数 | 含义 |
| --- | --- |
| `--filter` | `gaussian`、`sharpen`、`sobel` 或 `sobel-y` |
| `--kernel` | kernel 尺寸；Gaussian 支持 3、5、7，其他滤波器固定为 3 |
| `--sigma` | 覆盖 Gaussian 默认 σ；默认值为 `kernel_size / 3` |
| `--warmup` | 正式计时前的预热次数 |
| `--repeat` | 独立计时次数 |
| `--seed` | 合成输入随机种子 |
| `--max-images` | 数据集模式最多处理多少张；0 表示全部 |
| `--csv` / `--summary` | 数据集详细结果和汇总结果路径 |

查看完整命令帮助：

```bash
./build/filter_bench --help
```

## 计时与正确性口径

- 使用 `std::chrono::steady_clock` 单独记录每次卷积调用。
- 主要耗时指标为多次测量的中位数，CSV 同时保留所有样本。
- 图片读取、PNG/JPEG 解码、RGB 转灰度、输出分配、checksum 和文件保存均在计时区间之外。
- checksum 在计时后读取输出，防止计算结果不可观察；它不能替代逐像素正确性比较。
- 测试覆盖 identity kernel、手算小矩阵、zero-padding 边界、Gaussian 归一化和 NaN/Inf 检测。
- 新优化实现应与 naive 结果进行逐像素比较，并记录最大误差与平均误差。

## 项目结构

```text
image-filter-opt/
├── CMakeLists.txt          # CMake 构建入口
├── include/                # 卷积、kernel、图像 IO 和 benchmark 接口
├── src/                    # naive 实现、命令行程序和计时逻辑
├── tests/                  # C++ 正确性测试
├── scripts/                # 批量实验、去噪演示和绘图脚本
├── datasets/README.md      # DIV2K 下载与校验说明
├── images/                 # 小型输入样例；生成图片默认写到这里
├── results/                # 本地生成的 CSV、图表和 profiling 输出
├── third_party/            # stb 图像 IO 头文件
└── RESULTS_ANALYSIS.md     # 当前 baseline 的实验结果与分析
```

`build/`、DIV2K 数据和默认生成结果均由 `.gitignore` 排除。需要长期保存的结果应在提交前明确选择，并在文档中记录机器、编译器、编译参数和实验配置。

## 后续优化与协作约定

建议沿同一接口逐步推进：

1. V1：利用 Gaussian kernel 可分离性，将复杂度从 `O(HWK²)` 降为 `O(HWK)`。
2. V2：拆分边界与内部区域，减少内层分支，并改善内存访问局部性。
3. V3：加入 SIMD 向量化。
4. V4：使用 OpenMP 并行独立输出行或分块。
5. V5：组合 SIMD 与 OpenMP，并评估扩展性。

提交新实现时请保持 benchmark 输入、预热次数、重复次数和编译配置一致，并至少完成：

```bash
cmake --build build -j
ctest --test-dir build --output-on-failure
python3 scripts/run_scaling_bench.py --exe build/filter_bench
```

性能提升应报告为同一机器、同一输入和同一配置下的 `baseline_time / optimized_time`。不要把缺失数据记录为零，也不要把图片 IO 时间与卷积核心时间混在一起比较。

## 第三方组件

`third_party/` 中的 stb 头文件来自 [nothings/stb](https://github.com/nothings/stb)，并保留其原始许可文本。
