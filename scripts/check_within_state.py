#!/usr/bin/env python3
"""Is the clone's error small compared to what a single decision is worth?

A pooled fit statistic — correlation across every board, mean absolute error
across every board — is dominated by variance *between* states: full boards
score differently from empty ones, and predicting that is easy. The policy
never consumes it. What the policy consumes is the ordering of the forty
candidates *within* one state, and nothing pooled says whether that ordering
survives.

The two can come apart completely. A model can explain 73% of the variance in
board value and still rank candidates at chance, if all its error sits inside
states rather than across them. Worse, `argmax` does not average that error
away — it seeks it. Over forty candidates whose true values differ by less than
the model's own error, the winner is whichever candidate got the luckiest
positive error.

So this measures, per state:

- **within-state Spearman** between the model's forty outputs and the
  teacher's, which is the ranking the policy actually runs on
- **top-1 agreement**, how often it picks the teacher's column
- **regret in units of the within-state spread** — how much teacher-value is
  given up per decision, divided by the standard deviation of the teacher's
  forty scores, so it is comparable across boards of wildly different scale
- **error/spread**, the model's RMSE in those same units. Above about 1 the
  model cannot see the differences it is being asked to rank.

Run: python scripts/check_within_state.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def spearman_rows(a, b):
    """Rank correlation per row, without scipy."""
    ra = np.argsort(np.argsort(a, axis=1), axis=1).astype(np.float64)
    rb = np.argsort(np.argsort(b, axis=1), axis=1).astype(np.float64)
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    num = (ra * rb).sum(axis=1)
    den = np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
    return num / np.maximum(den, 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path,
                    default=Path("runs/demos-rollout.npz"))
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("runs/bc-rollout.h5"))
    ap.add_argument("--val-split", type=float, default=0.1,
                    help="must match the split pretrain.py used")
    args = ap.parse_args()

    from tensorflow.keras.models import load_model
    from agent import standardise_rows

    d = np.load(args.data)
    grid, vector, scores = d["grid"], d["vector"], d["scores"]
    cut = int(len(scores) * (1.0 - args.val_split))
    g, v, teacher = (grid[cut:].astype(np.float32),
                     vector[cut:].astype(np.float32), scores[cut:])
    print(f"  {len(teacher)} holdout boards, {teacher.shape[1]} candidates each")

    model = load_model(args.checkpoint)
    pred = model.predict([g, v], batch_size=512, verbose=0)

    # The teacher's own scale, per board. Everything below is in these units.
    spread = teacher.std(axis=1)
    target = standardise_rows(teacher)          # what the model was fitted to

    rho = spearman_rows(pred, teacher)
    top1 = (pred.argmax(axis=1) == teacher.argmax(axis=1)).mean()
    best = teacher.max(axis=1)
    taken = teacher[np.arange(len(teacher)), pred.argmax(axis=1)]
    regret = (best - taken) / np.maximum(spread, 1e-9)
    rmse = np.sqrt(((pred - target) ** 2).mean())

    print(f"\n  within-state Spearman   {rho.mean():.3f}"
          f"   (median {np.median(rho):.3f})")
    print(f"  top-1 agreement         {100 * top1:.1f}%   (chance 2.5%)")
    print(f"  regret per decision     {regret.mean():.3f} within-state sd")
    print(f"  model RMSE              {rmse:.3f} within-state sd")
    print(f"\n  ratio error/spread      {rmse:.2f}")
    if rmse > 0.7:
        print("    -> the model's error is comparable to the whole range it is\n"
              "       being asked to rank inside. argmax over 40 of those\n"
              "       selects for lucky error, not for value.")
    # How much of the teacher's advantage survives the model's choice?
    mean_spread = spread.mean()
    print(f"\n  teacher's forty scores have sd {mean_spread:.0f} in its own units")
    print(f"  the model gives up {regret.mean() * mean_spread:.0f} of those per drop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
