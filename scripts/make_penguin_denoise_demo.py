#!/usr/bin/env python3
"""Create a text-free penguin denoising comparison using the C++ convolution."""

import argparse
from pathlib import Path
import subprocess

from PIL import Image
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", default="build/filter_bench")
    parser.add_argument("--input", default="datasets/DIV2K/DIV2K_valid_HR/0801.png")
    parser.add_argument("--noisy", default="images/penguin_noisy_bw.png")
    parser.add_argument("--filtered", default="images/penguin_denoised_bw.png")
    parser.add_argument("--comparison", default="images/penguin_denoise_bw_comparison.png")
    parser.add_argument("--noise-sigma", type=float, default=100.0,
                        help="additive Gaussian noise standard deviation in uint8 levels")
    parser.add_argument("--kernel", type=int, default=7, choices=(3, 5, 7))
    parser.add_argument("--sigma", type=float, default=2.5,
                        help="Gaussian filter sigma")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.noise_sigma <= 0 or args.sigma <= 0 or args.seed < 0:
        parser.error("noise-sigma and sigma must be positive; seed must be nonnegative")

    source = Image.open(args.input).convert("L")
    gray = np.asarray(source, dtype=np.float32)
    rng = np.random.default_rng(args.seed)
    noisy = np.clip(gray + rng.normal(0.0, args.noise_sigma, gray.shape), 0, 255)
    noisy = np.floor(noisy + 0.5).astype(np.uint8)

    noisy_path = Path(args.noisy)
    noisy_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(noisy, mode="L").save(noisy_path)

    filtered_path = Path(args.filtered)
    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    command = [args.exe, "--input", str(noisy_path), "--output",
               str(filtered_path), "--filter", "gaussian", "--kernel",
               str(args.kernel), "--sigma", str(args.sigma),
               "--warmup", "1", "--repeat", "1"]
    subprocess.run(command, check=True)

    before = Image.open(noisy_path).convert("L")
    after = Image.open(filtered_path).convert("L")
    if before.size != after.size:
        raise ValueError(f"input/output dimensions differ: {before.size} vs {after.size}")

    gutter = 16
    canvas = Image.new("L", (before.width * 2 + gutter, before.height), 38)
    canvas.paste(before, (0, 0))
    canvas.paste(after, (before.width + gutter, 0))
    comparison_path = Path(args.comparison)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(comparison_path, optimize=True)
    print(f"Wrote text-free comparison: {comparison_path} ({canvas.width} × {canvas.height})")


if __name__ == "__main__":
    main()

