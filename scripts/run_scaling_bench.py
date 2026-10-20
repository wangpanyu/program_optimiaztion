#!/usr/bin/env python3
"""Run controlled image-size and kernel-size experiments."""

import argparse
import csv
from pathlib import Path
import subprocess
import sys


FIELDS = ("implementation", "filter", "filename", "width", "height", "pixels",
          "kernel_size", "median_time_ms", "mean_time_ms", "min_time_ms",
          "mpixels_s", "checksum", "sample_times_ms")


def run_case(exe, width, height, kernel, warmup, repeat, seed):
    command = [exe, "--width", str(width), "--height", str(height),
               "--filter", "gaussian", "--kernel", str(kernel),
               "--warmup", str(warmup), "--repeat", str(repeat),
               "--seed", str(seed)]
    print("Running:", " ".join(command), flush=True)
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    print(completed.stdout, end="", flush=True)
    line = next((line for line in completed.stdout.splitlines()
                 if line.startswith("RESULT ")), None)
    if line is None:
        raise RuntimeError("benchmark did not print a RESULT line")
    row = dict(item.split("=", 1) for item in line.split()[1:])
    row["implementation"] = "naive_scalar" if Path(exe).name.endswith("_scalar") else "naive_native"
    row["filter"] = "gaussian"
    return row


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", default="build/filter_bench")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat <= 0 or args.seed < 0:
        parser.error("warmup and seed must be nonnegative; repeat must be positive")
    output_dir = Path(args.output_dir)
    image_rows = [run_case(args.exe, size, size, 5, args.warmup, args.repeat, args.seed)
                  for size in (512, 1024, 2048, 4096)]
    write_csv(output_dir / "image_scaling.csv", image_rows)
    kernel_rows = [run_case(args.exe, 2048, 2048, size,
                            args.warmup, args.repeat, args.seed)
                   for size in (3, 5, 7)]
    write_csv(output_dir / "kernel_scaling.csv", kernel_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())

