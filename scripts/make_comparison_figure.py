#!/usr/bin/env python3
"""Create a PPT-ready before/after figure with matching grayscale panels."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import numpy as np


def read_gray_before(path):
    rgb = mpimg.imread(path)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"expected an RGB image: {path}")
    rgb = rgb[:, :, :3].astype(np.float32)
    if rgb.max() <= 1.0:
        rgb *= 255.0
    # Match the C++ RGB-to-gray luminance weights and round to display bytes.
    gray = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return np.floor(gray + 0.5).clip(0, 255).astype(np.uint8)


def read_gray_after(path):
    image = mpimg.imread(path)
    if image.ndim == 3:
        image = image[:, :, :3] @ np.array([0.2126, 0.7152, 0.0722])
    image = image.astype(np.float32)
    if image.max() <= 1.0:
        image *= 255.0
    return image.clip(0, 255)


def add_panel(fig, bounds, image, title, roi):
    ax = fig.add_axes(bounds, facecolor="white")
    ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_axis_off()
    ax.set_title(title, fontsize=17, fontweight="semibold", color="#1D3557", pad=14)

    x0, y0, x1, y1 = roi
    rect = Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                     edgecolor="#F5A623", linewidth=2.2)
    ax.add_patch(rect)

    inset = ax.inset_axes([0.60, 0.045, 0.36, 0.36])
    inset.imshow(image[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255,
                 interpolation="nearest")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_color("white")
        spine.set_linewidth(2.5)
        spine.set_path_effects([
            path_effects.Stroke(linewidth=5, foreground="#18212F"),
            path_effects.Normal(),
        ])
    inset.text(0.04, 0.94, "DETAIL CROP", transform=inset.transAxes,
               ha="left", va="top", color="white", fontsize=9,
               fontweight="bold",
               bbox={"facecolor": "#18212F", "alpha": 0.78,
                     "edgecolor": "none", "pad": 3})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="datasets/DIV2K/DIV2K_valid_HR/0801.png")
    parser.add_argument("--filtered", default="images/output_0801.png")
    parser.add_argument("--output", default="images/0801_before_after.png")
    parser.add_argument("--filter-label", default="Gaussian 5×5")
    parser.add_argument(
        "--caption",
        default="DIV2K validation · 0801.png · 2040 × 1356 · same grayscale input",
    )
    args = parser.parse_args()

    before = read_gray_before(args.input)
    after = read_gray_after(args.filtered)
    if before.shape != after.shape:
        raise ValueError(f"image sizes differ: {before.shape} vs {after.shape}")
    height, width = before.shape

    # A high-texture area around the bird's head makes the subtle blur easier to see.
    crop_w = min(430, width)
    crop_h = min(320, height)
    center_x, center_y = min(880, width // 2), min(270, height // 2)
    x0 = max(0, min(width - crop_w, center_x - crop_w // 2))
    y0 = max(0, min(height - crop_h, center_y - crop_h // 2))
    roi = (x0, y0, x0 + crop_w, y0 + crop_h)

    fig = plt.figure(figsize=(16, 9), dpi=150, facecolor="#F3F6FA")
    fig.text(0.5, 0.955, f"{args.filter_label} Image Filtering",
             ha="center", va="center", fontsize=25,
             fontweight="bold", color="#14263D")
    fig.text(0.5, 0.915, args.caption,
             ha="center", va="center", fontsize=12, color="#59697D")

    add_panel(fig, [0.035, 0.285, 0.455, 0.565], before,
              "Before · grayscale", roi)
    add_panel(fig, [0.510, 0.285, 0.455, 0.565], after,
              f"After · {args.filter_label}", roi)

    fig.text(0.5, 0.12,
             "The orange box marks the same region in both panels; insets show the matched crop.",
             ha="center", va="center", fontsize=11, color="#59697D")
    fig.text(0.5, 0.075,
             "Naive 2D convolution · zero padding · output displayed in grayscale",
             ha="center", va="center", fontsize=10, color="#7A8796")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Wrote {output} ({width} × {height} source images, 2400 × 1350 figure)")


if __name__ == "__main__":
    main()

