#!/usr/bin/env python3
"""Plot available baseline CSV files with matplotlib only."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_rows(path):
    if not path.exists():
        print(f"Skipping {path}: file does not exist")
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save(path):
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    print("Wrote", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    root = Path(args.results)

    rows = read_rows(root / "image_scaling.csv")
    if rows:
        sizes = [int(row["width"]) for row in rows]
        times = [float(row["median_time_ms"]) for row in rows]
        plt.figure(figsize=(6, 4))
        plt.plot(range(len(sizes)), times, marker="o")
        plt.xticks(range(len(sizes)), [f"{size}×{size}" for size in sizes])
        plt.xlabel("Image Size")
        plt.ylabel("Execution Time (ms)")
        plt.grid(alpha=0.3)
        save(root / "image_scaling.png")

    rows = read_rows(root / "kernel_scaling.csv")
    if rows:
        sizes = [int(row["kernel_size"]) for row in rows]
        times = [float(row["median_time_ms"]) for row in rows]
        plt.figure(figsize=(6, 4))
        plt.plot(sizes, times, marker="o")
        plt.xticks(sizes, [f"{size}×{size}" for size in sizes])
        plt.xlabel("Kernel Size")
        plt.ylabel("Execution Time (ms)")
        plt.grid(alpha=0.3)
        save(root / "kernel_scaling.png")

    rows = read_rows(root / "div2k_baseline.csv")
    if rows:
        pixels = [int(row["pixels"]) for row in rows]
        times = [float(row["median_time_ms"]) for row in rows]
        rates = [float(row["mpixels_s"]) for row in rows]
        plt.figure(figsize=(6, 4))
        plt.scatter(pixels, times, alpha=0.7)
        plt.xlabel("Number of Pixels")
        plt.ylabel("Execution Time (ms)")
        plt.grid(alpha=0.3)
        save(root / "div2k_time_vs_pixels.png")

        plt.figure(figsize=(6, 4))
        plt.hist(rates, bins=min(12, max(1, len(rates))))
        plt.xlabel("Throughput (MPixels/s)")
        plt.ylabel("Number of Images")
        plt.grid(alpha=0.3)
        save(root / "div2k_throughput.png")


if __name__ == "__main__":
    main()

