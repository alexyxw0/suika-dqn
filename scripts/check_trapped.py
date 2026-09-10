#!/usr/bin/env python3
"""Is a fruit `trapped_small` calls unreachable really unreachable?

`trapped_small` decides, per fruit, whether any of the 40 columns the agent can
drop into would put an identical fruit in contact with it. It decides that with
the closed-form landing estimate — a fruit falls straight down and stops at the
first thing it overlaps — which ignores roll, bounce, and the support being
pushed aside. `check_rollout.py` measured that same estimate getting the merge
outcome wrong 27% of the time when it is used to *choose* a drop.

That error rate does not carry over directly, because this asks an easier
question: not "where exactly does it land" but "is there any column at all". A
fruit sitting in the open has forty chances to be right about.

So measure it. For every droppable fruit on a real board, actually simulate an
identical fruit dropped into all 40 columns and see whether any of them merges
with that fruit. That is the ground truth the estimate is scored against.

Two errors, and they are not equally bad:

- **missed** — called trapped, but the engine merges it. A penalty is charged
  for a position that was fine.
- **false free** — called reachable, but nothing merges it. A real trap goes
  unpenalised.

The estimate is deliberately biased towards the second: a fruit reachable only
by a fortunate roll reads as trapped, and that is the safer error for a term
that only ever subtracts.

Run: python scripts/check_trapped.py --boards 3
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import envpath                                                     # noqa: E402
envpath.ensure()

from heuristic import (BOARD_W, DROPPABLE, RADII, READ_STATE,      # noqa: E402
                       trapped_indices)


def merged_with(result, xi, yi, ri):
    """Did this rollout merge something into the fruit at (xi, yi)?

    A merge is reported at the midpoint of the pair, so a twin landing on this
    fruit fires one within a radius of it. A merge further away is some other
    pair reacting, which says nothing about whether this fruit was reachable.
    """
    for mx, my, _r, _tick, _pts in (result.get("merges") or []):
        if math.hypot(mx - xi, my - yi) <= ri * 1.6:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boards", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=40,
                    help="random drops per board, so it is not half empty")
    ap.add_argument("--actions", type=int, default=40)
    ap.add_argument("--port", type=int, default=8971)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--slack", type=float, nargs="+",
                    default=[0.0, 6.0, 12.0, 20.0, 32.0],
                    help="contact tolerances to score. The engine result is "
                         "the expensive half, so every value is scored "
                         "against the same 40 real drops per fruit")
    args = ap.parse_args()

    from suika_env.suika_browser_env import SuikaBrowserEnv

    xs = [min(max((a / (args.actions - 1)) * BOARD_W, 0.0), BOARD_W)
          for a in range(args.actions)]

    np.random.seed(args.seed)
    env = SuikaBrowserEnv(headless=True, port=args.port, obs_mode="features")
    # (board index, fruit index) -> did the engine actually merge it
    truth: dict = {}
    boards = []
    try:
        for b in range(args.boards):
            env.reset(seed=args.seed + b)
            for _ in range(args.warmup):
                env.step(np.array([np.random.rand()], dtype=np.float32))
            fruits = [tuple(f) for f in
                      env.driver.execute_script(READ_STATE)["fruits"]]
            boards.append(fruits)

            n_drop = 0
            for i, (xi, yi, ri, si) in enumerate(fruits):
                if si >= DROPPABLE:
                    continue
                n_drop += 1
                # 40 candidate drops of a twin, from this exact board.
                res = env.driver.execute_script(
                    "return Game.rollout(arguments[0], arguments[1], "
                    "arguments[2], arguments[3], arguments[4]);",
                    xs, int(si), 600, [list(f) for f in fruits], 1)
                truth[(b, i)] = any(merged_with(r, xi, yi, ri) for r in res)

            print(f"  board {b + 1}: {len(fruits):3d} fruit, "
                  f"{n_drop:2d} droppable, "
                  f"{sum(truth[(b, i)] for i in range(len(fruits)) if (b, i) in truth):2d}"
                  f" the engine can merge")
    finally:
        try:
            env.close()
        except Exception:                                  # noqa: BLE001
            pass

    if not truth:
        print("  nothing checked")
        return 1

    print(f"\n  {len(truth)} droppable fruit, each tested against 40 real drops")
    print(f"\n  {'slack':>6}  {'called trapped':>14}  {'missed':>17}"
          f"  {'false free':>17}  {'agrees':>7}")
    for slack in args.slack:
        called_trapped = missed = false_free = 0
        for b, fruits in enumerate(boards):
            trapped = set(trapped_indices(fruits, args.actions, slack))
            for i in range(len(fruits)):
                if (b, i) not in truth:
                    continue
                if i in trapped:
                    called_trapped += 1
                    missed += truth[(b, i)]
                else:
                    false_free += not truth[(b, i)]
        free = len(truth) - called_trapped
        agree = len(truth) - missed - false_free
        print(f"  {slack:6.0f}  {called_trapped:14d}"
              f"  {missed:6d} ({100 * missed / max(called_trapped, 1):3.0f}%)"
              f"  {false_free:6d} ({100 * false_free / max(free, 1):3.0f}%)"
              f"  {100 * agree / len(truth):6.1f}%")
    print("\n  missed     = called trapped, the engine merged it "
          "(a penalty charged on a fine position)")
    print("  false free = called reachable, nothing merged it "
          "(a real trap left unpenalised)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
