#!/usr/bin/env python3
"""Generate the two real-robot analysis charts used by the project page.

The latency values and measured trajectory traces are copied from the verified
Figure 4 sources.  The browser uses the SVG exports so labels and curves stay
sharp at any page width; PDF and PNG exports are kept for reuse and review.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "scripts" / "data"
OUT = ROOT / "docs" / "assets" / "figures"

LATENCY_DATA = DATA / "real_latency_success.csv"
DISPLACEMENT_DATA = DATA / "real_joint_displacement.tsv"
ACCELERATION_DATA = DATA / "real_acceleration_proxy.tsv"

BLUE = "#2B5797"
ORANGE = "#D97706"
RED = "#C65D4B"
TEAL = "#5AA6A6"
GRAY = "#8F8F8F"
GRID = "#DDE1E6"
INK = "#25282C"


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 10.5,
        "axes.titlesize": 12.2,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.2,
        "ytick.labelsize": 9.2,
        "legend.fontsize": 8.5,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def finish_axis(ax: plt.Axes) -> None:
    ax.grid(axis="both", color=GRID, linewidth=0.7)
    ax.tick_params(length=3, width=0.7, color=INK)


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.svg")
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=300)
    plt.close(fig)


def read_latency() -> list[dict[str, str]]:
    with LATENCY_DATA.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_trace(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return {
        name: np.asarray(
            [np.nan if not row[name].strip() else float(row[name]) for row in rows]
        )
        for name in rows[0]
    }


def draw_latency() -> None:
    display = {
        "fastwam_joint": "FastWAM-Joint",
        "fastwam": "Fast-WAM",
        "pi05": "π0.5",
        "streamingwam": "StreamingWAM",
    }
    colors = {
        "fastwam_joint": ORANGE,
        "fastwam": TEAL,
        "pi05": GRAY,
        "streamingwam": BLUE,
    }
    markers = {
        "fastwam_joint": "o",
        "fastwam": "^",
        "pi05": "s",
        "streamingwam": "D",
    }
    offsets = {
        "fastwam_joint": (-8, 8, "right", "bottom"),
        "fastwam": (8, -2, "left", "center"),
        "pi05": (9, -1, "left", "center"),
        "streamingwam": (9, 3, "left", "bottom"),
    }

    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    fig.subplots_adjust(left=0.15, right=0.965, bottom=0.19, top=0.88)
    for row in read_latency():
        if row["execution_mode"].lower() != "async":
            continue
        method = row["method"]
        latency = float(row["latency_ms"])
        success = float(row["overall_success_percent"])
        color = colors[method]
        ax.scatter(
            latency,
            success,
            s=82 if method == "streamingwam" else 66,
            marker=markers[method],
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        dx, dy, ha, va = offsets[method]
        ax.annotate(
            display[method],
            (latency, success),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            color=color,
            fontsize=9.2,
            fontweight="bold" if method == "streamingwam" else "normal",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 0.25},
        )

    ax.set_title("(a) Response latency and task success", loc="left", fontweight="bold")
    ax.set_xlabel("Steady-state latency (ms)")
    ax.set_ylabel("Overall success across five tasks (%)")
    ax.set_xlim(25, 405)
    ax.set_ylim(22, 72)
    ax.set_xticks([50, 150, 250, 350])
    ax.set_yticks([25, 40, 55, 70])
    finish_axis(ax)
    save_figure(fig, "real-robot-latency-analysis")


def draw_continuity() -> None:
    displacement = read_trace(DISPLACEMENT_DATA)
    acceleration = read_trace(ACCELERATION_DATA)
    series = [
        ("FastWAM-Joint-asycn", "Joint Async", ORANGE, "-", 1.55),
        ("FastWAM-Joint-rtc", "Joint RTC", RED, (0, (4.0, 2.0)), 1.7),
        ("StreamingWAM", "StreamingWAM", BLUE, "-", 2.05),
    ]
    lengths = {len(values) for values in (*displacement.values(), *acceleration.values())}
    if len(lengths) != 1:
        raise ValueError(f"Expected aligned traces, found lengths {sorted(lengths)}")
    x = np.arange(lengths.pop())

    fig, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(5.0, 3.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1.02, 1.0], "hspace": 0.12},
    )
    fig.subplots_adjust(left=0.17, right=0.965, bottom=0.19, top=0.88, hspace=0.12)
    for source, label, color, style, width in series:
        zorder = 4 if source == "StreamingWAM" else 3
        top.plot(
            x,
            displacement[source],
            color=color,
            linestyle=style,
            linewidth=width,
            label=label,
            zorder=zorder,
        )
        bottom.plot(
            x,
            acceleration[source],
            color=color,
            linestyle=style,
            linewidth=width,
            zorder=zorder,
        )

    top.set_title("(b) Measured action continuity", loc="left", fontweight="bold")
    top.set_ylabel("Joint displacement\n(rad)")
    bottom.set_ylabel("Acceleration proxy\n(mrad)")
    bottom.set_xlabel("Sample index")
    top.set_ylim(0, 0.92)
    bottom.set_ylim(0, 13.2)
    bottom.set_xlim(0, 260)
    bottom.set_xticks([0, 50, 100, 150, 200, 250])
    top.set_yticks([0.0, 0.4, 0.8])
    bottom.yaxis.set_major_locator(MaxNLocator(4))
    top.legend(
        loc="upper center",
        bbox_to_anchor=(0.54, 0.98),
        ncol=3,
        frameon=False,
        handlelength=1.5,
        columnspacing=0.8,
        handletextpad=0.4,
    )
    top.tick_params(axis="x", labelbottom=False)
    for ax in (top, bottom):
        finish_axis(ax)
        ax.margins(x=0)
    save_figure(fig, "real-robot-continuity-analysis")


def main() -> None:
    draw_latency()
    draw_continuity()


if __name__ == "__main__":
    main()
