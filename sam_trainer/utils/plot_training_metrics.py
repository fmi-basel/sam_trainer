#!/usr/bin/env python
"""Plot training metrics from .err log files."""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


def parse_err_file(err_file):
    """Extract epoch and current metric from .err file."""
    epochs = []
    metrics = []

    pattern = r"Epoch (\d+):.*current metric: ([\d.]+)"

    with open(err_file, "r", encoding="utf-8") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                metric = float(match.group(2))

                # Only add if this is a new epoch (validation happens once per epoch)
                if not epochs or epoch > epochs[-1]:
                    epochs.append(epoch)
                    metrics.append(metric)

    return epochs, metrics


def plot_metrics(err_file):
    """Plot training metrics from .err file."""
    epochs, metrics = parse_err_file(err_file)

    if not epochs:
        print(f"No metrics found in {err_file}")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(
        epochs,
        metrics,
        marker=".",
        linewidth=2,
        markersize=2,
        color="blue",
        label="Original Data",
    )

    # 300 represents number of points to make between T.min and T.max
    xnew = np.linspace(min(epochs), max(epochs), len(epochs) * 40)

    spl = make_interp_spline(epochs, metrics, k=3)  # type: BSpline
    power_smooth = spl(xnew)
    # plt.plot(xnew, power_smooth, linewidth=1, color="red", label="Smoothed Curve")
    modes = ["full", "same", "valid"]
    plt.plot(
        epochs,
        np.convolve(metrics, np.ones(3) / 3, mode="same"),
        linewidth=1,
        color="red",
        label="Smoothed Curve (Moving Average)",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Current Metric")
    plt.title(f"Training Progress - {Path(err_file).stem}")
    plt.grid(True, alpha=0.3)

    # Save plot
    output_file = Path(err_file).with_suffix(".png")
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_file}")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_training_metrics.py <err_file>")
        sys.exit(1)

    err_file = sys.argv[1]
    plot_metrics(err_file)
