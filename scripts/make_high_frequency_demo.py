#!/usr/bin/env python3
"""Generate a high-frequency image and process it with the C++ filter demo."""

import argparse
from pathlib import Path
import subprocess
import sys

from PIL import Image
import numpy as np


def make_pattern(width=1600, height=900):
    y, x = np.indices((height, width))
    image = np.zeros((height, width), dtype=np.uint8)
    split_x, split_y = width // 2, height // 2

    # Four regions stress different high-frequency orientations.
    image[:split_y, :split_x] = (((x[:split_y, :split_x] // 4 +
                                    y[:split_y, :split_x] // 4) % 2) * 255).astype(np.uint8)
    image[:split_y, split_x:] = (((x[:split_y, split_x:] // 2) % 2) * 255).astype(np.uint8)
    image[split_y:, :split_x] = (((y[split_y:, :split_x] // 2) % 2) * 255).astype(np.uint8)
    image[split_y:, split_x:] = ((((x[split_y:, split_x:] +
                                    y[split_y:, split_x:]) // 3) % 2) * 255).astype(np.uint8)
    return np.repeat(image[:, :, None], 3, axis=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", default="build/filter_bench")
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--input", default="images/high_frequency_input.png")
    parser.add_argument("--filtered", default="images/high_frequency_filtered.png")
    parser.add_argument("--figure", default="images/high_frequency_before_after.png")
    args = parser.parse_args()
    if args.sigma <= 0:
        parser.error("sigma must be positive")

    input_path = Path(args.input)
    input_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(make_pattern()).save(input_path)

    command = [args.exe, "--input", str(input_path), "--output", args.filtered,
               "--filter", "gaussian", "--kernel", "7", "--sigma", str(args.sigma),
               "--warmup", "1", "--repeat", "1"]
    subprocess.run(command, check=True)

    figure_command = [sys.executable, "scripts/make_comparison_figure.py",
                      "--input", str(input_path), "--filtered", args.filtered,
                      "--output", args.figure,
                      "--filter-label", f"Gaussian 7×7 (σ={args.sigma:g})",
                      "--caption", "Synthetic high-frequency pattern · 1600 × 900 · same grayscale input"]
    subprocess.run(figure_command, check=True)


if __name__ == "__main__":
    main()

