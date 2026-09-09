#!/usr/bin/env python3
"""Three panels: where the agent ended up, how it got there, and why it matters.

Every bar carries its standard error and its episode count, because most of the
wrong turns in this project came from reading a mean without them.

Run: python scripts/plot_results.py -o runs/results.png
"""

from __future__ import annotations

import argparse
import re
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import numpy as np                                                # noqa: E402

# Categorical slots 1 and 2 from the reference palette, validated as a pair
# (CVD dE 24.7, normal-vision 33.6, both >= 3:1 on the light surface). MUTED is
# the de-emphasis ink, used here for reference rows rather than as a series.
SERIES = ["#2a78d6", "#eb6834"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8880"
GRID = "#e6e5e0"

# name, mean, standard error, episodes, kind
POLICIES = [
    ("Random play",                    1424, 118, 10, "ref"),
    ("DQN, from scratch",              1455,  79, 16, "learned"),
    ("Cloned into the original head",  1530,  68, 25, "learned"),
    ("Board-value net, first attempt", 1794,  50, 25, "learned"),
    ("Cloned into a column head",      2449,  72, 45, "learned"),
    ("Hand-written policy",            2582,  82, 30, "written"),
    ("Board-value net + simulation",   2666, 145, 12, "learned"),
    ("Hand-written + simulation",      2826,  80, 30, "written"),
]

# the four rungs of the board-value ablation
LADDER = [
    ("estimated\nboards",   1794,  50),
    ("+ real\nphysics",     2190, 205),
    ("+ shortlist\nof 5",   2509, 124),
    ("+ uncertainty\npenalty", 2666, 145),
]


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, fontweight="600", loc="left",
                 pad=10)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=3)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d5d4cf")
        ax.spines[side].set_linewidth(0.8)


def read_variance(path):
    rows = re.findall(r"seed (\d+): ((?:\s+\d+)+)\s+mean", Path(path).read_text())
    return [(int(s), [float(x) for x in v.split()]) for s, v in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", type=Path, default=Path("runs/results.png"))
    ap.add_argument("--variance-log", type=Path, default=Path("/tmp/vsplit.log"))
    args = ap.parse_args()

    fig = plt.figure(figsize=(15, 5.8), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 0.92, 0.88], wspace=0.42,
                          left=0.165, right=0.985, top=0.775, bottom=0.16)

    # ── where everything landed ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    style(ax, "Where every policy landed", "score", "")
    names = [p[0] for p in POLICIES]
    y = np.arange(len(POLICIES))
    for i, (_n, m, se, n, kind) in enumerate(POLICIES):
        colour = {"ref": MUTED, "learned": SERIES[0], "written": SERIES[1]}[kind]
        ax.barh(i, m, height=0.62, color=colour, zorder=3,
                edgecolor=SURFACE, linewidth=1.5)
        ax.errorbar(m, i, xerr=se, color=INK, elinewidth=1.1, capsize=3,
                    capthick=1.1, zorder=4, fmt="none")
        ax.text(m + se + 45, i, f"{m:,}", va="center", ha="left",
                color=INK_2, fontsize=8.5)
        ax.text(60, i, f"n={n}", va="center", ha="left", color=SURFACE,
                fontsize=7.5, zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5, color=INK_2)
    ax.invert_yaxis()
    ax.set_xlim(0, 3480)
    ax.axvline(3000, color=INK_2, linewidth=1.0, linestyle=(0, (4, 3)),
               zorder=2)
    ax.text(3000, -0.95, "3000 target", ha="center", va="bottom",
            color=INK_2, fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES[0]),
               plt.Rectangle((0, 0), 1, 1, color=SERIES[1]),
               plt.Rectangle((0, 0), 1, 1, color=MUTED)]
    fig.legend(handles, ["learned", "hand-written", "reference"],
               frameon=False, fontsize=8.5, labelcolor=INK_2, ncol=3,
               loc="upper left", bbox_to_anchor=(0.165, 0.872))

    # ── the ablation ladder ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    style(ax, "One change at a time", "", "score")
    xs = np.arange(len(LADDER))
    vals = [v for _l, v, _s in LADDER]
    errs = [s for _l, _v, s in LADDER]
    ax.bar(xs, vals, width=0.55, color=SERIES[0], zorder=3,
           edgecolor=SURFACE, linewidth=1.5)
    ax.errorbar(xs, vals, yerr=errs, fmt="none", color=INK, elinewidth=1.1,
                capsize=3, capthick=1.1, zorder=4)
    for i in range(1, len(LADDER)):
        gain = vals[i] - vals[i - 1]
        ax.annotate(f"+{gain}", xy=(i, vals[i] + errs[i]), xytext=(0, 8),
                    textcoords="offset points", ha="center",
                    color=SERIES[1], fontsize=9, fontweight="600")
    ax.set_xticks(xs)
    ax.set_xticklabels([l for l, _v, _s in LADDER], fontsize=8, color=INK_2)
    ax.set_ylim(0, 3250)
    ax.text(0.02, 0.98, "+872 in total (5.7σ)", transform=ax.transAxes,
            color=INK_2, fontsize=8.5, va="top")

    # ── decisions vs luck ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    style(ax, "Decisions, not luck", "fruit sequence", "score")
    data = read_variance(args.variance_log)
    if data:
        for i, (_seed, scores) in enumerate(data):
            ax.scatter([i] * len(scores), scores, s=34, color=SERIES[0],
                       zorder=4, edgecolor=SURFACE, linewidth=1.0)
            ax.plot([i - 0.28, i + 0.28], [st.mean(scores)] * 2,
                    color=SERIES[1], linewidth=2.0, zorder=5)
        within = st.mean([st.pvariance(s) for _q, s in data])
        between = st.pvariance([st.mean(s) for _q, s in data])
        share = 100 * within / (within + between)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels([str(i + 1) for i in range(len(data))], fontsize=8.5,
                           color=INK_2)
        ax.text(0.02, 0.96,
                f"spread within a sequence is\ndecisions, not luck: "
                f"{share:.0f}% of variance",
                transform=ax.transAxes, color=INK_2, fontsize=8.5, va="top")
        ax.text(0.98, 0.03, "orange = sequence mean", transform=ax.transAxes,
                ha="right", color=MUTED, fontsize=7.5)

    fig.suptitle("Suika RL — measured results", x=0.165, y=0.975, ha="left",
                 color=INK, fontsize=14, fontweight="700")
    fig.text(0.165, 0.912,
             "means over whole episodes; error bars are one standard error",
             ha="left", color=MUTED, fontsize=8.5)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
