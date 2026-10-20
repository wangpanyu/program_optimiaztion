# 二维图像卷积基线：实验结果与性能分析

实验日期：2026-10-09。本文汇总当前 V0 naive 2D convolution 的实现、正确性检查、合成 workload、DIV2K validation workload、GCC 向量化报告与 perf 采样结果。所有时间均来自本机实际运行；没有用估算值补齐缺失数据。

## 1. 实验目标与实现范围

本阶段建立可复现的 CPU baseline，为后续算法、单核和多核优化提供比较对象。核心函数是 `convolve2d_naive`，按 `y → x → ky → kx` 四层循环计算，每个输出像素执行 `K × K` 邻域乘加，边界采用 zero padding。核按原方向使用，不翻转；这对应图像滤波常见的 cross-correlation 定义。

输入图像在计时前读取、解码为灰度，并转换为 `[0,1]` 的连续 row-major `float` 数组。核心计算复杂度为 `O(H × W × K²)`。计时区间只包围一次卷积函数调用；读取图像、灰度转换、输出分配、checksum 和保存图像均不计入 convolution time。

实现包含 Gaussian 3×3、5×5、7×7 核，核系数归一化为和约等于 1；默认 `sigma = K/3`。Sharpen 与 Sobel X/Y 只用于单张图片展示，性能实验统一使用 Gaussian。当前没有 OpenMP、手写 SIMD、分块或 separable convolution。

## 2. 环境与测量设置

| 项目 | 实际环境 |
|---|---|
| CPU | AMD Ryzen 7 8845H，8 核 16 线程 |
| 系统 | Linux x86-64，WSL2，内核 `6.18.40.1-microsoft-standard-WSL2` |
| 编译器 | GCC 11.4.0 |
| 构建系统 | CMake 3.22.1，Release |
| native 目标 | `-O3 -march=native` |
| scalar 目标 | `-O2 -fno-tree-vectorize` |
| benchmark | 单线程；`steady_clock`；合成实验 warmup 3、repeat 10；DIV2K warmup 1、repeat 5 |

`filter_bench` 和 `filter_bench_scalar` 使用同一份 naive 算法源码，区别在编译选项。合成输入由固定 seed `2026` 生成。正式指标采用每项 workload 的 repeat 中位数；逐次耗时保存在 CSV 的 `sample_times_ms` 字段中。

## 3. 正确性与图像 IO

`ctest --test-dir build --output-on-failure` 实际结果为 **1/1 passed**。测试覆盖：identity kernel、小矩阵手工计算、zero-padding 边界、Gaussian 核归一化，以及 NaN 检测。512×512、Gaussian 3×3 的独立小测试也成功运行，checksum 为 `130830.766717240`，中位卷积时间为 `1.764563 ms`。

图像 IO 使用 stb 读取 PNG/JPEG 并写 PNG。真实图像的灰度转换使用权重 `0.2126, 0.7152, 0.0722`。`0801.png` 的单图 demo 成功输出 2040×1356 灰度 PNG，见 `images/output_0801.png`。checksum 和无 NaN/Inf 检查可确认输出被实际读用，但 checksum 不是逐像素正确性证明；目前还没有第二种优化实现可供像素级差分比较。

## 4. 合成图像实验

### 4.1 图像尺寸扩展

固定 Gaussian 5×5，每次同时将宽高翻倍，像素数增加 4 倍。表中的吞吐率按每个 workload 的中位耗时计算。

| 尺寸 | 像素数 | 中位时间 (ms) | 平均时间 (ms) | MPixels/s |
|---:|---:|---:|---:|---:|
| 512×512 | 262,144 | 3.821566 | 3.918253 | 68.596 |
| 1024×1024 | 1,048,576 | 16.394438 | 16.739895 | 63.959 |
| 2048×2048 | 4,194,304 | 65.714674 | 66.231512 | 63.826 |
| 4096×4096 | 16,777,216 | 287.762244 | 287.831750 | 58.302 |

宽高翻倍时，实测时间增长比分别为 **4.290×、4.008×、4.379×**，与像素数增长 4 倍大体一致，支持该实现随 `H×W` 扩展的预期。4096² 相比 512² 的吞吐率低约 15.0%，说明处理规模扩大时单位像素耗时有所上升；单凭这一现象无法判断是缓存容量、内存行为、频率变化还是系统调度造成。

数据与图片：`results/image_scaling.csv`、`results/image_scaling.png`。

### 4.2 卷积核尺寸扩展

固定 2048×2048，改变 Gaussian 核尺寸。单个输出像素的理论 tap 数为 9、25、49。

| 核尺寸 | tap 数 | 中位时间 (ms) | 平均时间 (ms) | MPixels/s |
|---:|---:|---:|---:|---:|
| 3×3 | 9 | 25.940509 | 26.544598 | 161.689 |
| 5×5 | 25 | 66.238448 | 67.025432 | 63.321 |
| 7×7 | 49 | 139.166332 | 140.541090 | 30.139 |

3×3→5×5 的耗时增长为 **2.553×**，tap 数增长为 2.778×；5×5→7×7 的耗时增长为 **2.101×**，tap 数增长为 1.960×；3×3→7×7 总体耗时增长为 **5.365×**，理论 tap 数增长为 5.444×。整体接近 `K²` 计算趋势。每个 tap 折算的中位耗时约为 0.63–0.69 ns/像素/tap，结果相对稳定。

注意：2048² Gaussian 5×5 在图像扩展批次为 65.714674 ms，在核尺寸批次为 66.238448 ms。两者来自不同批次和运行时间，存在正常的测量波动；因此不要把不同批次的最后几位差异解释为算法变化。

数据与图片：`results/kernel_scaling.csv`、`results/kernel_scaling.png`。

## 5. DIV2K 真实图像实验

从 [DIV2K 官方页面](https://data.vision.ee.ethz.ch/cvl/DIV2K/)下载了 `DIV2K_valid_HR.zip`，归档文件大小为 448,993,893 字节，SHA-256 为 `20dd31fd84d777bc1cf5d6b7654a3f569c0aec74458ae094122ad1d0489900fc`。ZIP 完整性检查通过，解压得到 100 张 validation PNG。原始图片不加入 Git，保存在 `datasets/DIV2K/`。

完整数据集运行配置是 Gaussian 5×5、每张图 warmup 1 次、计时 5 次。结果摘要：

| 指标 | 实测值 |
|---|---:|
| 图片数 | 100 |
| 总像素数 | 283,502,880 |
| 每张图像素范围 | 1,664,640–4,161,600 |
| 每张图中位卷积时间的范围 | 26.343910–63.606810 ms |
| 总时间（逐图中位时间求和） | 4,451.997135 ms |
| 每图平均时间 | 44.519971 ms |
| 每图时间的中位数 | 43.712187 ms |
| 总体吞吐率 | 63.679933 MPixels/s |
| 逐图吞吐率中位数 | 63.677432 MPixels/s |
| 逐图吞吐率 Q1–Q3 | 62.362–65.527 MPixels/s |
| 逐图吞吐率范围 | 57.717–68.243 MPixels/s |

逐图像素数与中位耗时的 Pearson 相关系数为 **0.954**。简单线性拟合约为 `time_ms = 0.787 + 15.426 × pixels_in_millions`，`R² = 0.909`。这表明在固定核和当前数据范围内，像素数能够解释大部分逐图时间差异，但数据不是严格等尺寸、相同内容的控制实验，拟合不能代替硬件级瓶颈分析。

100 张图片每张的 repeat 数均为 5。按每张图内部计算的样本变异系数，平均约 4.80%，中位数约 3.56%，最大值约 18.16%；少数 repeat 存在明显慢样本，这也是报告以中位数为主而不只看均值的原因。

结果文件：`results/div2k_baseline.csv`、`results/div2k_summary.csv`、`results/div2k_time_vs_pixels.png`、`results/div2k_throughput.png`。三张图片的批处理 smoke test 保存在 `results/div2k_smoke.csv` 和 `results/div2k_smoke_summary.csv`。

## 6. scalar 与 native 编译目标比较

使用相同 synthetic 输入、2048² Gaussian 5×5、warmup 3、repeat 10，直接比较两个可执行文件：

| 构建目标 | 中位时间 (ms) | 平均时间 (ms) | 最小值 (ms) | MPixels/s |
|---|---:|---:|---:|---:|
| scalar (`-O2 -fno-tree-vectorize`) | 71.496005 | 73.130064 | 66.890142 | 58.665 |
| native (`-O3 -march=native`) | 66.822139 | 68.935509 | 62.693330 | 62.768 |

本次中位时间对应 native 约 **1.070×** speedup，约快 6.5%。两组各自的样本存在离散值（scalar 有 89.31 ms 的慢样本，native 有 82.61 ms 的慢样本），因此这组差异应视为本机当前环境下的初步结果。重复多轮、交替运行顺序、固定 CPU 频率并在非虚拟化 Linux 上复测，可提高结论稳定性。

两种构建的 checksum 分别为 `2095115.357143916` 和 `2095115.357189193`，数值非常接近，但总和接近不能保证每个像素都满足误差阈值。若后续需要宣称数值等价，应保存两份输出并报告 `max_abs_error`、`mean_abs_error`。

样本与指标保存在 `results/build_comparison.csv`。

## 7. perf 与 GCC 向量化分析

### 7.1 perf stat

本机原先没有 `perf`。为完成 profiling，下载 Ubuntu `linux-tools-5.15.0-185` 并解压到 `/tmp`，执行的 perf 版本为 5.15.200；它在 WSL2 内核 6.18.40.1 上能够读取计数器。对 scalar 和 native 分别执行 `perf stat -r 5`，工作负载均为 2048² Gaussian 5×5。下面是 perf 给出的五次运行平均计数：

| 事件 | scalar | native |
|---|---:|---:|
| cycles | 4,326,512,201 | 3,990,663,572 |
| instructions | 26,896,305,775 | 22,084,926,663 |
| IPC | 6.15 | 5.46 |
| cache references | 37,632,499 | 38,149,202 |
| cache misses | 584,667 (1.545%) | 596,072 (1.566%) |
| branches | 5,866,292,082 | 5,035,468,747 |
| branch misses | 2,273,612 (约 0.039%) | 170,050 (约 0.0034%) |
| 整个进程墙钟时间 | 1.01146 s | 0.93558 s |

native 构建在该进程级计数中少约 7.8% cycles、少约 17.9% instructions；cache miss 比例相近，约 1.5%；分支失败率较低。perf 进程墙钟时间对应约 1.081×，但其计时覆盖整个程序一次启动中的随机数据初始化、3 次 warmup、10 次卷积、checksum 和退出流程，不能与前一节单次卷积中位时间混为同一个指标。

计数器来自 WSL2 环境，且 `cache-references` 不等价于所有内存访问。低 cache miss 比例与低 branch miss 比例可以说明本次计数没有显示大量对应事件，但不能单独证明访存或分支开销不重要，也不能直接推出“memory bound”或“compute bound”。原始记录见 `results/perf_scalar.txt`、`results/perf_native.txt`。

### 7.2 perf record 热点

native 2048² Gaussian 5×5 运行使用 99 Hz 采样，收集 96 个 cycles 样本，丢失样本为 0。`convolve2d_naive` 占 **98.31%**，`benchmark_convolution` 占 1.13%，其余为 synthetic 包装和程序开销。这确认热点集中在卷积函数本身；采样不能区分函数内部哪些指令或数据依赖最耗时。

文本报告为 `results/perf_native_report.txt`，原始采样文件为 `results/perf_native.data`。

### 7.3 GCC vectorization report

GCC 11.4.0 使用 `-O3 -march=native -fopt-info-vec-all` 检查 `src/convolution_naive.cpp`，报告该函数 **0 个循环成功向量化**。`kx` 内层循环被报告为存在 control flow，外层循环则因连续嵌套循环结构未向量化。baseline 保持朴素写法，没有为改变报告而额外修改代码。

这也解释了为什么 `-O3 -march=native` 与 scalar 构建的差距有限：当前测量中两者的核心计算都没有通过 GCC 循环向量化报告。native 仍可能从其他编译优化中获益；本次不能把全部 6.5% 时间差归因于某个特定编译变换。

## 8. 当前证据支持的结论与待验证假设

1. **工作量符合复杂度预期。** 图像像素数增至 4 倍时，时间大体增至 4 倍；固定图像而提高核尺寸时，时间总体接近 `K²` 增长。
2. **主热点已定位到核心函数。** perf record 有 98.31% 样本位于 `convolve2d_naive`，优先优化该函数是合理的。
3. **自动向量化没有覆盖核心循环。** GCC 报告给出了明确结果，后续 SIMD 阶段需要探索循环结构或显式向量化方式，并以 naive 输出验证误差。
4. **目前没有证据宣布单一硬件瓶颈。** perf 计数显示 cache miss 和 branch miss 比例不高，但其范围是进程级且运行于 WSL2。地址计算、内层边界检查、重叠邻域重复计算、缓存复用和单线程执行仍是待检验假设。
5. **Gaussian 的算法级优化有较大空间。** Gaussian 核可分离；后续 separable convolution 可把每像素工作从约 `K²` 个 tap 降为约 `2K` 个 tap。不过这属于下一阶段，当前数据只建立 baseline，不包含该优化的实测 speedup。

## 9. 局限与可复现性

- 所有结果来自单台 Ryzen 7 8845H 的 WSL2 环境，CPU 频率、温度和宿主调度没有锁定；不同机器之间不应直接比较绝对时间。
- 合成输入是固定随机灰度数组，不代表自然图像空间分布；DIV2K 提供实际 RGB 图片 workload，但灰度转换和 PNG 解码不计入卷积时间。
- DIV2K summary 的 total time 是逐图中位时间相加，不包含读文件、解码、转换和写结果所需时间。
- scalar/native 直接比较只运行一组各 10 repeats；两组都有慢样本，结论仍应看作初步观察。
- `perf stat` 覆盖整个程序进程，`perf record` 只有 96 个样本，且 WSL2 下 PMU 结果可能与原生 Linux 不同。
- 当前 correctness tests 检查 baseline 的规定行为；未来每个优化实现都应和 naive 输出逐像素比较，并报告最大与平均绝对误差。

复现构建与主要实验：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
ctest --test-dir build --output-on-failure
python3 scripts/run_scaling_bench.py --exe build/filter_bench
python3 scripts/run_div2k_bench.py --exe build/filter_bench --dataset datasets/DIV2K/DIV2K_valid_HR --kernel 5
python3 scripts/plot_results.py
```

PPT 用的 16:9 前后对比图为 `images/0801_before_after.png`，由 `scripts/make_comparison_figure.py` 从 0801 原图与 Gaussian 输出生成；左右图使用相同灰度输入，并标出同一局部放大区域。

100 张 DIV2K 验证图片的逐图耗时柱状图为 `results/div2k_per_image_time.png`，可编辑版本为同名 SVG，生成脚本为 `scripts/plot_div2k_per_image.py`。每个柱子来自 `results/div2k_baseline.csv` 中对应图片 5 次卷积的中位时间；平均值为 44.52 ms，最大值为 63.61 ms（0855.png）。该图只统计 Gaussian 5×5 卷积，不包含图片解码和结果保存时间。

用于课堂展示的黑白企鹅去噪对照图为 `images/penguin_denoise_bw_comparison.png`，另有独立的噪声图 `images/penguin_noisy_bw.png` 和滤波结果 `images/penguin_denoised_bw.png`。脚本先将照片转换为灰度，再以固定 seed 加入标准差 100（uint8 灰度级）的高斯噪声，最后由 C++ 程序执行 Gaussian 7×7（σ=2.5）滤波。对照图保持相同尺度并排展示，且不含文字；噪声强度专门为远距离投影展示而显著加强，该人工噪声 demo 不属于 DIV2K 性能测量数据。

本机 perf 命令的完整使用方式、原始 CSV/图片、数据集 archive 校验值及后续优化路线分别见 `README.md`、`results/`、`datasets/README.md`。数据集本身在 `datasets/DIV2K/` 并被 Git 忽略。

