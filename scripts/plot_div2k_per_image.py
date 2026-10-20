#!/usr/bin/env python3
"""Plot median processing time for each image in the DIV2K validation set."""

import argparse
import csv
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/div2k_baseline.csv")
    parser.add_argument("--output", default="results/div2k_per_image_time.png")
    parser.add_argument("--svg", default="results/div2k_per_image_time.svg")
    args = parser.parse_args()

    with Path(args.input).open(newline="", encoding="utf-8") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: row["filename"])
    if len(rows) != 100:
        raise ValueError(f"expected 100 benchmark rows, found {len(rows)}")

    labels = [Path(row["filename"]).stem for row in rows]
    times = [float(row["median_time_ms"]) for row in rows]
    positions = list(range(1, len(rows) + 1))
    average = mean(times)
    maximum_index = max(range(len(times)), key=times.__getitem__)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 17,
        "xtick.labelsize": 12,
        "ytick.labelsize": 13,
    })
    fig, ax = plt.subplots(figsize=(12.8, 7.2), facecolor="white")
    colors = ["#4C78A8"] * len(times)
    colors[maximum_index] = "#E45756"
    ax.bar(positions, times, width=0.82, color=colors, edgecolor="none")

    ax.axhline(
        average,
        color="#2F3B4C",
        linestyle=(0, (6, 4)),
        linewidth=2.0,
        label=f"Mean: {average:.2f} ms",
    )
    ax.scatter(
        positions[maximum_index],
        times[maximum_index],
        color="#E45756",
        s=45,
        zorder=3,
        label=(
            f"Maximum: {times[maximum_index]:.2f} ms "
            f"(image {labels[maximum_index]})"
        ),
    )

    tick_indices = [0] + list(range(9, len(rows), 10))
    ax.set_xticks([positions[index] for index in tick_indices])
    ax.set_xticklabels([labels[index] for index in tick_indices])
    ax.set_xlim(0.25, len(rows) + 0.75)
    ax.set_ylim(0, max(times) * 1.17)
    ax.set_xlabel("DIV2K validation image ID")
    ax.set_ylabel("Median convolution time (ms)")
    ax.set_title(
        "Per-image Convolution Time for 100 DIV2K Images\n"
        "Naive CPU baseline · Gaussian 5×5 · median of 5 runs",
        pad=18,
    )
    ax.grid(axis="y", color="#D9DEE7", linewidth=0.9, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#687386")
    ax.spines["bottom"].set_color("#687386")
    ax.legend(loc="upper left", frameon=False, fontsize=13, ncol=2)

    fig.tight_layout(pad=1.5)
    output_path = Path(args.output)
    svg_path = Path(args.svg)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, facecolor="white")
    fig.savefig(svg_path, facecolor="white")
    plt.close(fig)
    print(f"Wrote {output_path} (2560 x 1440)")
    print(f"Wrote {svg_path}")


if __name__ == "__main__":
    main()
