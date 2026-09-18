#!/usr/bin/env python3
"""Does replaying a seed reproduce the game?

It did not. Measured with a null control — the same policy in both arms of an
A/B, 20 seeds — two runs of one policy on one seed correlated at **-0.29**, and
the standard deviation of their paired difference was **1068**, as unrelated as
two different games. A seed fixed the fruit sequence and nothing else.

The cause was physics on a wall clock. `runFastPhysics` is paced by
requestAnimationFrame, so the number of ticks between a drop and the
observation depended on how busy the machine was; about 3% of drops were read
while the board was still moving, and from there the trajectories separated.

`?ticks=1` stops the free-running loop and makes `Game.advance()` the only
thing that steps the world, counted in ticks rather than milliseconds. This
checks whether that worked, by playing the same seeds twice with a
deterministic policy and comparing.

What it reports: how many of the repeated episodes matched exactly, and the
correlation and paired spread across them — the same statistics the null
control produced, so the two are directly comparable.

Run: python scripts/check_determinism.py --episodes 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import envpath                                                     # noqa: E402
envpath.ensure()

from heuristic import POLICIES, READ_STATE, score_all              # noqa: E402


def play(env, weights, seed, actions, max_steps):
    """One episode under a policy made deterministic for this test: ties go to
    the lowest index, so any difference between two runs is the environment's
    and not the tie-break's."""
    env.reset(seed=seed)
    bins = np.linspace(0.0, 1.0, actions)
    score, steps = 0.0, 0
    while steps < max_steps:
        state = env.driver.execute_script(READ_STATE)
        action = int(np.argmax(score_all(state, actions, weights)))
        _o, _r, done, trunc, info = env.step(
            np.array([bins[action]], dtype=np.float32))
        score = info["score"]
        steps += 1
        if done or trunc:
            break
    return score, steps


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--actions", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--seed-base", type=int, default=555000)
    ap.add_argument("--port", type=int, default=8969)
    args = ap.parse_args()

    from agent import browser_failures, restart_env
    from suika_env.suika_browser_env import SuikaBrowserEnv

    def make_env():
        return SuikaBrowserEnv(headless=True, port=args.port,
                               obs_mode="features")

    browser_dead = browser_failures()
    weights = dict(POLICIES["layered"], tiebreak=0.0)
    env = make_env()
    first, second = [], []
    try:
        print(f"  url: {env.game_url}")
        for i in range(args.episodes):
            seed = args.seed_base + i
            # Chrome drops the tab on a crowded board at a rate this project
            # has measured near 14%; a crash here is not evidence about
            # determinism, so the pair is replayed rather than recorded.
            for attempt in range(3):
                try:
                    a = play(env, weights, seed, args.actions, args.max_steps)
                    b = play(env, weights, seed, args.actions, args.max_steps)
                    break
                except browser_dead:
                    print(f"    seed {seed}  tab crashed, replaying the pair",
                          flush=True)
                    env = restart_env(env, make_env)
            else:
                continue
            first.append(a)
            second.append(b)
            same = "MATCH" if a == b else "differ"
            print(f"    seed {seed}  {a[0]:6.0f}/{a[1]:3d}   "
                  f"{b[0]:6.0f}/{b[1]:3d}   {same}", flush=True)
    finally:
        try:
            env.close()
        except Exception:                                      # noqa: BLE001
            pass

    a = np.array([x[0] for x in first])
    b = np.array([x[0] for x in second])
    matched = sum(1 for x, y in zip(first, second) if x == y)
    print(f"\n  {matched}/{len(first)} episodes reproduced exactly")
    if len(a) > 2:
        d = a - b
        print(f"    paired-difference sd {d.std(ddof=1):8.0f}"
              "      (null control, wall clock: 1068)")
        print(f"    same-seed correlation {np.corrcoef(a, b)[0, 1]:+8.2f}"
              "      (null control, wall clock: -0.29)")
    if matched == len(first):
        print("\n  Seeds reproduce. Pairing on them now removes the fruit\n"
              "  sequence, which is about half the variance in this game.")
        return 0
    print("\n  Seeds still do not reproduce; something else is on a clock.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
