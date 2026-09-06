#!/usr/bin/env python3
"""Summarise a training log: learning curve, evaluations, and stability.

Reads the log rather than requiring instrumentation, so it works on runs that
have already finished — including the ones that crashed.

Run: python scripts/curve.py runs/train.log [runs/other.log ...]
"""

from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path

EPISODE = re.compile(
    r"^Episode\s+(\d+)\s+reward\s+([\d.]+)\s+steps\s+(\d+)\s+eps\s+([\d.]+)"
    r"\s+buffer\s+(\d+)(?:\s+\(([\d.]+) GB\))?\s+grads\s+(\d+)", re.M)
# the current format, mean over several episodes with a range
EVAL_MEAN = re.compile(r"eval mean\s+([\d.]+)\s+over\s+(\d+)\s+episodes\s+\[([\d.]+)\.\.([\d.]+)\]")
# the old single-episode format, kept so earlier runs still parse
EVAL_ONE = re.compile(r"eval \(greedy\) reward\s+([\d.]+)")


def summarise(path: Path, bands: int = 8) -> None:
    text = path.read_text()
    eps = EPISODE.findall(text)
    if not eps:
        print(f"{path}: no episodes")
        return

    rewards = [float(e[1]) for e in eps]
    steps = [int(e[2]) for e in eps]
    n = len(rewards)
    crashes = text.count("browser lost")
    died = "Traceback" in text

    print(f"\n{path}")
    print(f"  {n} episodes | epsilon {eps[-1][3]} | grads {eps[-1][6]}"
          f" | browser restarts {crashes}"
          f" | {'ENDED IN A TRACEBACK' if died else 'clean'}")

    width = max(1, n // bands)
    print(f"\n  training reward, blocks of {width}:")
    baseline = statistics.mean(rewards[:width])
    for i in range(0, n, width):
        block = rewards[i:i + width]
        if len(block) < max(2, width // 3):
            continue
        m = statistics.mean(block)
        delta = 100 * (m / baseline - 1)
        print(f"    ep {i:>4}-{min(i+width-1, n-1):<4} {m:7.0f}"
              f" {delta:+6.1f}%  {'#' * round(m / 60)}")

    print(f"\n  episode length: first block {statistics.mean(steps[:width]):.0f}"
          f" -> last block {statistics.mean(steps[-width:]):.0f} steps")

    means = EVAL_MEAN.findall(text)
    if means:
        print("\n  greedy evaluations (mean over N, with range):")
        for m, count, lo, hi in means:
            print(f"    {float(m):7.0f}  over {count}  [{float(lo):.0f}..{float(hi):.0f}]"
                  f"  {'#' * round(float(m) / 60)}")
        if len(means) >= 2:
            first, last = float(means[0][0]), float(means[-1][0])
            print(f"    first -> last: {first:.0f} -> {last:.0f}"
                  f"  ({100 * (last / first - 1):+.1f}%)")
    else:
        ones = EVAL_ONE.findall(text)
        if ones:
            vals = [float(v) for v in ones]
            print(f"\n  greedy evaluations (single episode each — too noisy to read):")
            print(f"    {' '.join(f'{v:.0f}' for v in vals)}")
            print(f"    spread {min(vals):.0f}..{max(vals):.0f}, which is why"
                  f" these are now averaged")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--bands", type=int, default=8)
    args = ap.parse_args()
    for log in args.logs:
        if log.exists():
            summarise(log, args.bands)
        else:
            print(f"{log}: not found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
