#!/usr/bin/env python3
"""Remove heavy salt-and-pepper noise from dog.png with median filtering."""

import argparse
from pathlib import Path

from PIL import Image, ImageFilter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="images/dog.png")
    parser.add_argument("--output", default="images/dog_denoised_median.png")
    parser.add_argument("--size", type=int, default=9)
    parser.add_argument("--passes", type=int, default=2)
    args = parser.parse_args()

    if args.size < 3 or args.size % 2 == 0:
        parser.error("--size must be an odd integer of at least 3")
    if args.passes < 1:
        parser.error("--passes must be positive")

    source = Image.open(args.input).convert("L")
    result = source
    for _ in range(args.passes):
        result = result.filter(ImageFilter.MedianFilter(size=args.size))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path, optimize=True)
    print(f"Wrote {output_path} ({result.width} x {result.height})")


if __name__ == "__main__":
    main()
