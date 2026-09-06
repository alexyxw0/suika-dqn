#!/usr/bin/env python3
"""Does the simulated drop match the real one?

`Game.rollout` runs the candidate drop in a scratch Matter.js world and returns
the board it settles into. That is only worth having if it agrees with the game
— a simulation that is subtly wrong is worse than the closed-form estimate it
replaces, because it is wrong with more authority.

So: roll a drop out, then actually perform it, and compare. Reported per drop —

- **score**: did the simulation predict the points the drop scored?
- **count**: does the board have the number of fruit it expected?
- **position**: matching each predicted fruit to the nearest real one, how far
  apart are they?

Perfect agreement is not the bar. Matter.js is deterministic given identical
state, but the live board is stepped 25 times per frame between observations
while the scratch world is stepped one drop at a time from a snapshot, so small
divergence is expected. What matters is whether it is small enough that the
board's *structure* — what is next to what, and what merged — is right.

Run: python scripts/check_rollout.py --drops 30
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BOARD_W = 640

READ = """
  return Composite.allBodies(engine.world)
    .filter(b => !b.isStatic && b.circleRadius && b.sizeIndex !== undefined)
    .map(b => [b.position.x, b.position.y, b.circleRadius, b.sizeIndex]);
"""


def match_error(predicted, actual):
    """Mean distance from each predicted fruit to the nearest real one of the
    same size. Size is part of the match: a fruit in the right place but the
    wrong size is a failed merge prediction, not a small position error."""
    if not predicted or not actual:
        return None
    errs = []
    for px, py, _pr, ps in predicted:
        same = [(ax, ay) for ax, ay, _ar, a_s in actual if a_s == ps]
        if not same:
            return None
        errs.append(min(np.hypot(px - ax, py - ay) for ax, ay in same))
    return float(np.mean(errs))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drops", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=25,
                    help="random drops first, so the board is not empty")
    ap.add_argument("--port", type=int, default=8952)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from suika_env.suika_browser_env import SuikaBrowserEnv

    np.random.seed(args.seed)
    env = SuikaBrowserEnv(headless=True, port=args.port, obs_mode="features")
    score_err, count_err, pos_err = [], [], []
    merges_predicted = merges_real = merges_agreed = 0
    checked = 0
    try:
        env.reset(seed=args.seed)
        for _ in range(args.warmup):
            env.step(np.array([np.random.rand()], dtype=np.float32))

        for d in range(args.drops):
            cur = env.driver.execute_script("return Game.currentFruitSize;")
            score_before = env.driver.execute_script("return Game.score;")
            # The env floors the action to an integer pixel, so predict the
            # same x the game will actually use.
            frac = float(np.random.rand())
            x = float(int(frac * BOARD_W))

            pred = env.driver.execute_script(
                "return Game.rollout(arguments[0], arguments[1])[0];", [x], cur)

            _obs, _r, done, trunc, info = env.step(
                np.array([frac], dtype=np.float32))
            real = env.driver.execute_script(READ)
            gained_real = info["score"] - score_before

            score_err.append(abs(pred["gained"] - gained_real))
            count_err.append(abs(len(pred["fruits"]) - len(real)))
            pe = match_error(pred["fruits"], real)
            if pe is not None:
                pos_err.append(pe)
            merges_predicted += pred["gained"] > 0
            merges_real += gained_real > 0
            merges_agreed += (pred["gained"] > 0) == (gained_real > 0)
            checked += 1

            if done or trunc:
                env.reset(seed=args.seed + d + 1)
                for _ in range(args.warmup):
                    env.step(np.array([np.random.rand()], dtype=np.float32))
    finally:
        try:
            env.close()
        except Exception:      # noqa: BLE001
            pass

    if not checked:
        print("  nothing checked")
        return 1

    print(f"\n  {checked} drops simulated then played")
    print(f"    score exactly right          "
          f"{100 * sum(1 for e in score_err if e == 0) / checked:5.1f}%")
    print(f"    mean score error             {st.mean(score_err):5.2f} points")
    print(f"    fruit count exactly right    "
          f"{100 * sum(1 for e in count_err if e == 0) / checked:5.1f}%")
    print(f"    agreed on whether it merged  "
          f"{100 * merges_agreed / checked:5.1f}%"
          f"   (predicted {merges_predicted}, real {merges_real})")
    if pos_err:
        print(f"    mean position error          {st.mean(pos_err):5.1f} px"
              f"   (median {st.median(pos_err):.1f}, "
              f"worst {max(pos_err):.1f})")
        print(f"    for scale: the smallest fruit has a 24 px radius")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
