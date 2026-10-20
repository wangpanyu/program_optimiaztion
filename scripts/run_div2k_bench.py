#!/usr/bin/env python3
"""Run the C++ DIV2K benchmark without including image IO in timed regions."""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", default="build/filter_bench")
    parser.add_argument("--dataset", default="datasets/DIV2K/DIV2K_valid_HR")
    parser.add_argument("--kernel", type=int, default=5, choices=(3, 5, 7))
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--output", default="results/div2k_baseline.csv")
    parser.add_argument("--summary", default="results/div2k_summary.csv")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    command = [args.exe, "--dataset", args.dataset, "--filter", "gaussian",
               "--kernel", str(args.kernel), "--max-images", str(args.max_images),
               "--warmup", str(args.warmup), "--repeat", str(args.repeat),
               "--csv", args.output, "--summary", args.summary]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())

