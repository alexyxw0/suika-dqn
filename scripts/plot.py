#!/usr/bin/env python3
"""Plot training logs. Reads the logs, so it works on finished and crashed runs.

Run: python scripts/plot.py runs/train.log runs/train-features.log -o runs/training.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Categorical slots 1 and 2 from the reference palette, validated as a pair
# (CVD dE 24.7, normal-vision 33.6, both >= 3:1 on the light surface).
SERIES = ["#2a78d6", "#eb6834"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8880"

EPISODE = re.compile(
    r"^Episode\s+(\d+)\s+reward\s+([\d.]+)\s+steps\s+(\d+)\s+eps\s+([\d.]+)"
    r"\s+buffer\s+(\d+)(?:\s+\([\d.]+ GB\))?\s+grads\s+(\d+)", re.M)
EVAL = re.compile(
    r"eval mean\s+([\d.]+)\s+over\s+(\d+)\s+episodes\s+\[([\d.]+)\.\.([\d.]+)\]")


def read(path: Path) -> dict:
    text = path.read_text()
    rows = EPISODE.findall(text)
    ev = EVAL.findall(text)
    n = len(rows)
    # Evaluations are spaced evenly through the run; place them at the episode
    # they were measured after rather than at an index, so two runs of very
    # different length are comparable on one x-axis.
    spacing = n / max(len(ev), 1)
    return {
        "name": path.stem,
        "episode": np.array([int(r[0]) for r in rows]),
        "reward": np.array([float(r[1]) for r in rows]),
        "steps": np.array([int(r[2]) for r in rows]),
        "grads": np.array([int(r[5]) for r in rows]),
        "eval_x": np.array([(i + 1) * spacing for i in range(len(ev))]),
        "eval_mean": np.array([float(e[0]) for e in ev]),
        "eval_lo": np.array([float(e[2]) for e in ev]),
        "eval_hi": np.array([float(e[3]) for e in ev]),
        "crashes": text.count("browser lost"),
    }


def smooth(y: np.ndarray, window: int = 15) -> np.ndarray:
    """Centred moving average, shrinking at the edges so no points are dropped."""
    if len(y) < 3:
        return y
    w = min(window, max(3, len(y) // 4))
    pad = w // 2
    padded = np.pad(y, pad, mode="edge")
    kernel = np.ones(w) / w
    return np.convolve(padded, kernel, mode="same")[pad:pad + len(y)]


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, fontweight="600", loc="left", pad=8)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=3)
    # Recessive grid and axes: the data is the ink that matters.
    ax.grid(True, color="#e6e5e0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d5d4cf")
        ax.spines[side].set_linewidth(0.8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("runs/training.png"))
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--note", default="",
                    help="caption for the evaluation panel; empty "
                         "by default, because what the runs show is "
                         "for the reader to judge")
    args = ap.parse_args()

    runs = [read(p) for p in args.logs if p.exists()]
    if not runs:
        raise SystemExit("no readable logs")
    labels = args.labels or [r["name"] for r in runs]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), facecolor=SURFACE)
    fig.subplots_adjust(hspace=0.34, wspace=0.22, top=0.86, bottom=0.09,
                        left=0.07, right=0.98)

    # ── per-episode reward: raw faint, smoothed on top ──────────────────────
    ax = axes[0][0]
    style(ax, "Training reward per episode", "episode", "score")
    for run, label, colour in zip(runs, labels, SERIES):
        ax.plot(run["episode"], run["reward"], color=colour, alpha=0.16,
                linewidth=1.0, zorder=2)
        ax.plot(run["episode"], smooth(run["reward"]), color=colour,
                linewidth=2.0, label=label, zorder=3)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper left")
    ax.text(0.99, 0.03, "faint = per episode · solid = 15-episode mean",
            transform=ax.transAxes, ha="right", color=MUTED, fontsize=7.5)

    # ── evaluations, with the min..max of each 5-episode point ──────────────
    ax = axes[0][1]
    style(ax, "Greedy evaluation (5 episodes each, band = worst..best)",
          "episode", "score")
    for run, label, colour in zip(runs, labels, SERIES):
        if not len(run["eval_mean"]):
            continue
        ax.fill_between(run["eval_x"], run["eval_lo"], run["eval_hi"],
                        color=colour, alpha=0.13, linewidth=0, zorder=2)
        ax.plot(run["eval_x"], run["eval_mean"], color=colour, linewidth=2.0,
                marker="o", markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=label, zorder=3)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper left")
    # Whatever the evaluations show is for the reader to see; a caption baked
    # into the plotting code would assert it for every future pair of logs.
    if args.note:
        ax.text(0.99, 0.03, args.note, transform=ax.transAxes, ha="right",
                color=MUTED, fontsize=7.5)

    # ── episode length ──────────────────────────────────────────────────────
    ax = axes[1][0]
    style(ax, "Episode length (drops survived)", "episode", "steps")
    for run, label, colour in zip(runs, labels, SERIES):
        ax.plot(run["episode"], run["steps"], color=colour, alpha=0.16,
                linewidth=1.0, zorder=2)
        ax.plot(run["episode"], smooth(run["steps"]), color=colour,
                linewidth=2.0, label=label, zorder=3)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper left")

    # ── gradient steps: the replay-ratio difference, which did work ─────────
    ax = axes[1][1]
    style(ax, "Gradient steps taken", "episode", "updates")
    peak = 0
    for run, label, colour in zip(runs, labels, SERIES):
        ax.plot(run["episode"], run["grads"], color=colour, linewidth=2.0,
                label=label, zorder=3)
        if len(run["grads"]):
            peak = max(peak, run["grads"][-1])
            rate = run["grads"][-1] / max(run["episode"][-1], 1)
            # Label to the right of the line end, not above it — above collides
            # with the legend on the steeper series.
            ax.annotate(f"{rate:.0f}/episode",
                        xy=(run["episode"][-1], run["grads"][-1]),
                        xytext=(8, -2), textcoords="offset points",
                        color=colour, fontsize=8.5, fontweight="600",
                        ha="left", va="center", zorder=4)
    # Headroom on both axes so the end-of-line labels sit inside the panel, and
    # the legend moves out of the way of two lines that both climb rightward.
    ax.set_ylim(0, peak * 1.12)
    ax.margins(x=0.30)   # room for the end-of-line rate labels
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="lower right")

    fig.suptitle("Suika DQN — two runs", x=0.07, y=0.965, ha="left",
                 color=INK, fontsize=15, fontweight="700")
    sub = "  ·  ".join(
        f"{lab}: {len(r['episode'])} episodes, {r['grads'][-1] if len(r['grads']) else 0:,} updates, "
        f"{r['crashes']} browser restarts" for r, lab in zip(runs, labels))
    fig.text(0.07, 0.925, sub, color=INK_2, fontsize=9)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"wrote {args.out}")

    # A table view, because the chart's claim is "these overlap" and a reader
    # should be able to check the numbers rather than trust the bands.
    print("\n  evaluations")
    print("    " + "  ".join(f"{lab:>28}" for lab in labels))
    depth = max(len(r["eval_mean"]) for r in runs)
    for i in range(depth):
        cells = []
        for r in runs:
            if i < len(r["eval_mean"]):
                cells.append(f"{r['eval_mean'][i]:7.0f} [{r['eval_lo'][i]:.0f}..{r['eval_hi'][i]:.0f}]".rjust(28))
            else:
                cells.append(" " * 28)
        print(f"    " + "  ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
