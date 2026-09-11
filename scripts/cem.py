#!/usr/bin/env python3
"""Fit the placement weights with the cross-entropy method.

CEM keeps a Gaussian over weight vectors, samples a population from it, plays
each, keeps the best few, and refits the Gaussian to those. No gradients — it
only needs to be able to rank candidates, which is all a game score gives you.

This is the second attempt. The first is written up in runs/FINDINGS.md §5 and
returned +38 +/- 116 over hand-guessed weights across three designs. Its three
failures are what this file is shaped around:

1. **Maximising over noisy estimates selects for the largest error.** Picking
   the best of 12 candidates scored on 3 episodes each inflated the reported
   result by +463. Nothing here reports a training score: the number that comes
   out at the end is a re-measure on seeds the fit never saw.

2. **Pairing on a fixed seed set overfits those particular games.** A design
   that scored every candidate on the same five seeds reached about 3150 on
   them and 2427 held out. Seeds are redrawn every iteration here, so the only
   thing a weight vector can be good at is the game in general.

3. **A shared sigma across weights is meaningless when the weights differ by
   three orders of magnitude.** `merge` starts at 250 and `chain_vert` at 0.6.
   Sigma is per-weight and proportional, and it has a floor that decays rather
   than collapsing, which is what keeps CEM from converging onto the first
   lucky sample (Szita & Lorincz, 2006).

Run: python scripts/cem.py --iterations 6 --population 10 --episodes 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import envpath                                                     # noqa: E402
envpath.ensure()

from heuristic import (BOARD_WEIGHTS, POLICIES, WEIGHT_NAMES,      # noqa: E402
                       choose, choose_by_rollout, read_state)

# chain_vert is a mixing fraction, not a score weight; outside [0, 1] it either
# does nothing new or flips the sign of the chain bonus.
BOUNDS = {"chain_vert": (0.0, 1.0)}


def play(env, weights, seed, args):
    """One episode under one weight vector. Returns the final score."""
    env.reset(seed=seed)
    score, steps = 0.0, 0
    bins = np.linspace(0.0, 1.0, args.actions)
    while steps < args.max_steps:
        state = read_state(env, args.settle_wait)
        if args.rollout:
            action = choose_by_rollout(env, state, args.actions, weights,
                                       BOARD_WEIGHTS, args.rollout)
        else:
            action = choose(state, args.actions, weights)
        _obs, _r, done, trunc, info = env.step(
            np.array([bins[action]], dtype=np.float32))
        score = info["score"]
        steps += 1
        if done or trunc:
            break
    return score


def evaluate(env, weights, seeds, args, browser_dead, make_env):
    """Score per seed, as a dict. A crashed episode is replayed once and then
    given up on — Chrome loses the tab on crowded boards, which are the
    high-scoring ones, so a crash is not a random omission.

    Returning per-seed rather than a mean is what lets the caller keep the
    comparison paired. Candidates in an iteration share seeds precisely so the
    difference between them is measured on the same games; a candidate that
    lost one seed to a crash and is then compared on its mean of three against
    everyone else's mean of four has quietly become an unpaired comparison, on
    the seeds that happen to be missing.
    """
    out = {}
    for seed in seeds:
        for attempt in range(2):
            try:
                out[seed] = play(env, weights, seed, args)
                break
            except browser_dead:
                env = restart(env, make_env)
                if attempt:
                    print(f"      (seed {seed} crashed twice, dropped)",
                          flush=True)
    return out, env


def paired_means(results):
    """Mean per candidate over the seeds *every* candidate finished.

    If crashes took different seeds from different candidates, the common set
    is what can honestly be compared. Falls back to each candidate's own mean
    when nothing is shared, which only happens if the browser is failing badly
    enough that the iteration is worthless anyway.
    """
    common = set.intersection(*(set(r) for r in results)) if results else set()
    if not common:
        return [float(np.mean(list(r.values()))) if r else 0.0
                for r in results], 0
    return [float(np.mean([r[s] for s in common])) for r in results], len(common)


def restart(env, make_env):
    from agent import restart_env
    return restart_env(env, make_env)


def stderr(values):
    """Standard error of the mean. 0 for fewer than two values — not because
    the estimate is precise, but because there is nothing to estimate it from,
    and the caller prints the n beside it."""
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def vector(weights):
    return np.array([weights[n] for n in WEIGHT_NAMES], dtype=np.float64)


def unvector(v):
    w = {}
    for name, value in zip(WEIGHT_NAMES, v):
        lo, hi = BOUNDS.get(name, (-np.inf, np.inf))
        w[name] = float(np.clip(value, lo, hi))
    return w


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iterations", type=int, default=6)
    ap.add_argument("--population", type=int, default=10)
    ap.add_argument("--elite", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=4,
                    help="episodes per candidate per iteration")
    ap.add_argument("--holdout", type=int, default=12,
                    help="episodes for the final re-measure, on seeds no "
                         "iteration used")
    ap.add_argument("--sigma", type=float, default=0.35,
                    help="initial per-weight sd, as a fraction of |weight|")
    ap.add_argument("--noise", type=float, default=0.12,
                    help="sd floor added each iteration, as a fraction of "
                         "|weight|, decaying as 1/iteration. Without it CEM "
                         "collapses onto whichever early sample got lucky")
    ap.add_argument("--start", choices=sorted(POLICIES), default="layered")
    ap.add_argument("--actions", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--settle-wait", type=int, default=1000)
    ap.add_argument("--rollout", type=int, default=0)
    ap.add_argument("--port", type=int, default=8994)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=Path, default=Path("runs/cem.json"))
    args = ap.parse_args()

    from agent import browser_failures
    from suika_env.suika_browser_env import SuikaBrowserEnv

    def make_env():
        return SuikaBrowserEnv(headless=True, port=args.port,
                               obs_mode="features")

    browser_dead = browser_failures()
    rng = np.random.default_rng(args.seed)

    mu = vector(POLICIES[args.start])
    scale = np.abs(mu).copy()
    scale[scale < 1e-6] = 1.0            # a weight starting at 0 still moves
    sigma = args.sigma * scale

    # Held-out seeds are drawn first and never reused, so no iteration can
    # tune against them by accident.
    holdout_seeds = [int(s) for s in rng.integers(10**6, 2 * 10**6,
                                                  size=args.holdout)]

    env = make_env()
    history = []
    began = time.time()
    try:
        print(f"  start {args.start}: "
              + "  ".join(f"{n}={v:g}" for n, v in zip(WEIGHT_NAMES, mu)),
              flush=True)
        base_by_seed, env = evaluate(env, unvector(mu), holdout_seeds, args,
                                     browser_dead, make_env)
        base_scores = list(base_by_seed.values())
        base = float(np.mean(base_scores)) if base_scores else 0.0
        print(f"  baseline on the {args.holdout} held-out seeds: {base:.0f}"
              f" +/- {stderr(base_scores):.0f} (se, n={len(base_scores)})\n",
              flush=True)

        for it in range(args.iterations):
            # Fresh seeds every iteration: shared across the population so the
            # comparison within an iteration is paired, different from every
            # other iteration so nothing can overfit them.
            seeds = [int(s) for s in rng.integers(0, 10**6, size=args.episodes)]
            samples = [mu] + [mu + sigma * rng.standard_normal(len(mu))
                              for _ in range(args.population - 1)]
            per_seed = []
            for i, cand in enumerate(samples):
                res, env = evaluate(env, unvector(cand), seeds, args,
                                    browser_dead, make_env)
                per_seed.append(res)
                got = list(res.values())
                print(f"    iter {it}  cand {i:2d}  "
                      f"{np.mean(got) if got else 0:7.0f}"
                      f"  (n={len(got)})", flush=True)

            means, shared = paired_means(per_seed)
            if shared < len(seeds):
                print(f"    ranking on the {shared} seed(s) every candidate "
                      f"finished, of {len(seeds)}", flush=True)
            scored = list(zip(means, samples))
            scored.sort(key=lambda t: -t[0])
            elite = np.array([c for _s, c in scored[:args.elite]])
            mu = elite.mean(axis=0)
            floor = args.noise * scale / (it + 1)
            sigma = np.maximum(elite.std(axis=0), floor)

            held_by_seed, env = evaluate(env, unvector(mu), holdout_seeds,
                                         args, browser_dead, make_env)
            held_scores = list(held_by_seed.values())
            held = float(np.mean(held_scores)) if held_scores else 0.0
            history.append({"iteration": it,
                            "elite_mean_on_train": float(scored[0][0]),
                            "holdout": held,
                            "holdout_se": stderr(held_scores),
                            "holdout_n": len(held_scores),
                            "weights": unvector(mu)})
            print(f"  iter {it}: best-on-train {scored[0][0]:.0f}   "
                  f"HELD OUT {held:.0f} +/- {stderr(held_scores):.0f}   "
                  f"({(time.time() - began) / 60:.0f} min)\n", flush=True)
    finally:
        try:
            env.close()
        except Exception:                                  # noqa: BLE001
            pass

    final = unvector(mu)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"weights": final, "baseline_holdout": base, "history": history,
         "baseline_holdout_se": stderr(base_scores),
         "args": {k: str(v) for k, v in vars(args).items()}}, indent=2))

    print("  final weights: "
          + "  ".join(f"{n}={final[n]:.3g}" for n in WEIGHT_NAMES))
    if history:
        best = max(history, key=lambda h: h["holdout"])
        print(f"  best held-out {best['holdout']:.0f} +/- {best['holdout_se']:.0f}"
              f" at iteration {best['iteration']}, against a baseline of "
              f"{base:.0f} +/- {stderr(base_scores):.0f}")
        print("  Both are means over the same held-out seeds, so the gap is "
              "paired; treat it as real only if it clears about twice the "
              "larger of the two standard errors.")
    print(f"  written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
