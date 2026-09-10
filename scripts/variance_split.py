#!/usr/bin/env python3
"""How much of an episode's score is the policy's doing, and how much is luck?

The question this answers: is there anything to learn from the best episodes?

Filtering demonstrations down to the highest-scoring episodes only helps if
those episodes are good because of what the policy *did*. If they are good
because of the fruit that happened to arrive, the filter selects lucky
sequences and training on them fits noise — which is how the first weight-
fitting run in `runs/FINDINGS.md` inflated its own result by 463 points.

Both can be measured, because the fruit sequence is seedable and the policy's
randomness is a separate knob. Playing the same seed several times varies only
the exploration, so the spread within a seed is what the policy's choices are
worth. Playing different seeds varies the fruit, so the spread between seed
means is what luck is worth.

    within-seed variance   -> decisions
    between-seed variance  -> fruit

If between dominates, "learn from the top episodes" is mostly "learn from the
lucky ones", and the honest fix is to compare policies on shared seeds rather
than to filter.

Run: python scripts/variance_split.py --seeds 6 --repeats 3
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import envpath                                                     # noqa: E402
envpath.ensure()

from heuristic import (BOARD_W, BOARD_WEIGHTS, POLICIES,           # noqa: E402
                       READ_STATE, score_all, score_board)


def play(env, seed, args, weights, browser_dead, restart, make_env):
    score, steps = 0.0, 0
    try:
        env.reset(seed=seed)
        while steps < args.max_steps:
            state = env.driver.execute_script(READ_STATE)
            est = score_all(state, args.actions, weights)
            order = [int(i) for i in np.argsort(-est)[:args.rollout]]
            xs = [float(int((i / (args.actions - 1)) * BOARD_W)) for i in order]
            results = env.driver.execute_script(
                "return Game.rollout(arguments[0], arguments[1]);",
                xs, state["cur"])
            if np.random.rand() < args.epsilon:
                pick = int(np.random.randint(len(order)))
            else:
                pick = int(np.argmax([score_board(r, BOARD_WEIGHTS)
                                      for r in results]))
            _o, _r, done, trunc, info = env.step(
                np.array([order[pick] / (args.actions - 1)], dtype=np.float32))
            score = info["score"]
            steps += 1
            if done or trunc:
                break
    except browser_dead:
        return None, restart(env, make_env)
    return score, env


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--epsilon", type=float, default=0.25,
                    help="the exploration the demonstrations are collected "
                         "with; this is the decision variation being measured")
    ap.add_argument("--actions", type=int, default=40)
    ap.add_argument("--rollout", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--policy", choices=sorted(POLICIES), default="layered")
    ap.add_argument("--seed-base", type=int, default=20000)
    ap.add_argument("--port", type=int, default=8994)
    args = ap.parse_args()

    from agent import browser_failures, restart_env
    from suika_env.suika_browser_env import SuikaBrowserEnv

    weights = POLICIES[args.policy]
    browser_dead = browser_failures()

    def make_env():
        return SuikaBrowserEnv(headless=True, port=args.port,
                               obs_mode="features")

    env = make_env()
    by_seed: dict[int, list[float]] = {}
    began = time.time()
    try:
        for s in range(args.seeds):
            seed = args.seed_base + s
            by_seed[seed] = []
            for rep in range(args.repeats):
                score, env = play(env, seed, args, weights, browser_dead,
                                  restart_env, make_env)
                if score is None:
                    print(f"  seed {seed} rep {rep}: browser lost, retrying",
                          flush=True)
                    score, env = play(env, seed, args, weights, browser_dead,
                                      restart_env, make_env)
                if score is not None:
                    by_seed[seed].append(score)
                    print(f"  seed {seed}  rep {rep}  score {score:>7.0f}"
                          f"  [{(time.time()-began)/60:.0f}m]", flush=True)
    finally:
        try:
            env.close()
        except Exception:      # noqa: BLE001
            pass

    usable = {s: v for s, v in by_seed.items() if len(v) >= 2}
    if len(usable) < 2:
        print("  not enough data")
        return 1

    print(f"\n  {sum(len(v) for v in usable.values())} episodes over "
          f"{len(usable)} fruit sequences\n")
    for s, v in usable.items():
        print(f"    seed {s}: " + " ".join(f"{x:>6.0f}" for x in v)
              + f"   mean {st.mean(v):>6.0f}  spread {max(v)-min(v):>5.0f}")

    # within: exploration only. between: fruit only.
    within = st.mean([st.pvariance(v) for v in usable.values()])
    means = [st.mean(v) for v in usable.values()]
    between = st.pvariance(means)
    total = within + between
    print(f"\n    decisions (within a seed)   sd {within ** 0.5:6.0f}"
          f"   {100*within/total:5.1f}% of variance")
    print(f"    fruit     (between seeds)   sd {between ** 0.5:6.0f}"
          f"   {100*between/total:5.1f}% of variance")
    print(f"\n  If fruit dominates, the best episodes are mostly the lucky "
          f"ones and\n  filtering demonstrations by episode score selects "
          f"sequences, not skill.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
