#!/usr/bin/env python3
"""Generate the unified simulation-results figure used by the project page.

The values mirror the paper's verified aggregate results and analysis plots.
Run with the bundled Codex Python environment or any Python installation with
matplotlib and numpy available.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "figures"

BLUE = "#2B6299"
GRAY = "#8A8F96"
LIGHT_GRAY = "#B7BBC0"
GRID = "#DDE1E6"
INK = "#25282C"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.2,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)


def finish_axis(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.7)
    ax.tick_params(length=3, width=0.7, color=INK)


def add_value_labels(ax, bars, fmt="{:.1f}", dy=1.2):
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + dy,
            fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=7.8,
            color=INK,
        )


def main():
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.15))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.105, top=0.94,
                        wspace=0.27, hspace=0.42)

    # (a) Aggregate simulation benchmarks.
    ax = axes[0, 0]
    datasets = ["LIBERO", "RoboTwin\nClean", "LIBERO-Plus", "RoboTwin\nRandomized"]
    joint = [98.5, 80.3, 68.8, 3.9]
    ours = [98.7, 84.2, 79.4, 5.1]
    x = np.arange(len(datasets))
    width = 0.34
    bars_joint = ax.bar(x - width / 2, joint, width, color=GRAY, label="FastWAM-Joint")
    bars_ours = ax.bar(x + width / 2, ours, width, color=BLUE, label="StreamingWAM")
    add_value_labels(ax, bars_joint, dy=1.5)
    add_value_labels(ax, bars_ours, dy=1.5)
    ax.set_title("(a) Simulation benchmark success", loc="left", fontweight="bold")
    ax.set_ylabel("Success rate (%)")
    ax.set_xticks(x, datasets)
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(loc="upper right", frameon=False, ncol=1)
    finish_axis(ax)

    # (b) Latency-success trade-off on LIBERO-Plus.
    ax = axes[0, 1]
    points = [
        ("FastWAM-Joint", 381.6, 68.8, "o", GRAY),
        ("Fast-WAM", 182.6, 51.5, "s", LIGHT_GRAY),
        ("Faster-WAM", 139.4, 75.0, "^", "#686D73"),
        ("StreamingWAM", 72.9, 79.4, "D", BLUE),
    ]
    label_offsets = {
        "FastWAM-Joint": (-8, -15, "right"),
        "Fast-WAM": (0, -16, "center"),
        "Faster-WAM": (9, -2, "left"),
        "StreamingWAM": (9, 2, "left"),
    }
    for name, latency, success, marker, color in points:
        size = 68 if name == "StreamingWAM" else 55
        ax.scatter(latency, success, s=size, marker=marker, color=color,
                   edgecolor="white", linewidth=0.7, zorder=3)
        dx, dy, ha = label_offsets[name]
        ax.annotate(name, (latency, success), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va="center",
                    fontsize=8.2, color=color,
                    fontweight="bold" if name == "StreamingWAM" else "normal")
    ax.set_title("(b) Latency–success trade-off", loc="left", fontweight="bold")
    ax.set_xlabel("Steady-state latency (ms)")
    ax.set_ylabel("LIBERO-Plus success (%)")
    ax.set_xlim(20, 420)
    ax.set_ylim(45, 84)
    ax.set_xticks([50, 150, 250, 350])
    finish_axis(ax)

    # (c) Sensitivity to the execution horizon K. K=10 is intentionally omitted.
    ax = axes[1, 0]
    k = [1, 2, 4, 8, 16]
    base = [37.01, 50.77, 61.03, 72.76, 69.21]
    streaming = [75.45, 77.22, 79.36, 78.73, 72.86]
    ax.plot(k, base, color=GRAY, marker="o", markersize=5.5,
            linewidth=1.8, label="FastWAM-Joint")
    ax.plot(k, streaming, color=BLUE, marker="D", markersize=5.2,
            linewidth=2.2, label="StreamingWAM")
    ax.fill_between(k, streaming, 30, color=BLUE, alpha=0.055)
    ax.set_title("(c) Robustness across execution horizons", loc="left", fontweight="bold")
    ax.set_xlabel(r"Execution steps $K$")
    ax.set_ylabel("Success rate (%)")
    ax.set_xticks(k)
    ax.set_ylim(30, 84)
    ax.legend(loc="lower right", frameon=False)
    finish_axis(ax)

    # (d) Change around action-chunk boundaries (offset 0 is the replan step).
    ax = axes[1, 1]
    offsets = np.arange(-3, 4)
    boundary_joint = [0.07449, 0.08155, 0.07195, 0.13794, 0.07298, 0.08154, 0.07181]
    boundary_ours = [0.06074, 0.06402, 0.06258, 0.06838, 0.06142, 0.06396, 0.06207]
    ax.plot(offsets, boundary_joint, color=GRAY, marker="o", markersize=5.5,
            linewidth=1.8, label="FastWAM-Joint")
    ax.plot(offsets, boundary_ours, color=BLUE, marker="D", markersize=5.2,
            linewidth=2.2, label="StreamingWAM")
    ax.axvline(0, color="#B6BAC0", linestyle="--", linewidth=0.9)
    ax.annotate("replan boundary", xy=(0, 0.13794), xytext=(22, -2),
                textcoords="offset points", fontsize=7.8, color="#666A70",
                va="center")
    ax.set_title("(d) Action continuity at chunk boundaries", loc="left", fontweight="bold")
    ax.set_xlabel("Control step relative to replan")
    ax.set_ylabel("Action change (lower is smoother)")
    ax.set_xticks(offsets)
    ax.set_ylim(0.05, 0.148)
    ax.legend(loc="upper left", frameon=False)
    finish_axis(ax)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "simulation-results-analysis.pdf")
    fig.savefig(OUT / "simulation-results-analysis.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
