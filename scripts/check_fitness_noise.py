#!/usr/bin/env python3
"""Is CEM selecting on signal, or on which candidate drew the better games?

The failure mode: Suika scores swing hundreds of points on the fruit sequence
alone, so with few episodes per candidate the top of a generation is whoever
got lucky. CEM then shrinks its variance around that point and converges,
confidently, on nothing.

Reasoning about it from one generation's spread does not settle it, because
common random numbers lower the null as well as the signal. The direct test is
test-retest: score the *same* candidates twice on disjoint seed sets and see
whether the ranking survives. A fitness that does not replicate cannot be
selected on, whatever its spread looks like.

What comes out:

- **Spearman** between the two replicates — the reliability of the ranking CEM
  actually uses. Near 0 means selection is noise.
- **Reliability** (Pearson r) decomposed into signal and noise variance, which
  says how much of the observed between-candidate spread is real.
- **Episodes needed** for a reliability of 0.8, by Spearman-Brown. This is the
  number that says whether CEM is affordable here at all.

Candidates are drawn from the same distribution the fit starts from, so this
measures the regime CEM's first generation actually operates in.

Run: python scripts/check_fitness_noise.py --candidates 8 --episodes 4
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

from cem import BOUNDS, play, unvector, vector                     # noqa: E402
from heuristic import POLICIES, WEIGHT_NAMES                       # noqa: E402


def spearman(a, b):
    """Rank correlation, without pulling in scipy."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=4,
                    help="episodes per candidate per replicate")
    ap.add_argument("--sigma", type=float, default=0.35,
                    help="must match the CEM run being diagnosed")
    ap.add_argument("--start", choices=sorted(POLICIES), default="layered")
    ap.add_argument("--actions", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--settle-wait", type=int, default=1000)
    ap.add_argument("--rollout", type=int, default=0)
    ap.add_argument("--port", type=int, default=8996)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    from agent import browser_failures, restart_env
    from suika_env.suika_browser_env import SuikaBrowserEnv

    def make_env():
        return SuikaBrowserEnv(headless=True, port=args.port,
                               obs_mode="features")

    browser_dead = browser_failures()
    rng = np.random.default_rng(args.seed)

    mu = vector(POLICIES[args.start])
    scale = np.abs(mu).copy()
    scale[scale < 1e-6] = 1.0
    cands = [mu] + [mu + args.sigma * scale * rng.standard_normal(len(mu))
                    for _ in range(args.candidates - 1)]

    # Two disjoint seed sets. Within a replicate every candidate sees the same
    # games, exactly as CEM scores a generation.
    seeds_a = [int(s) for s in rng.integers(0, 10**6, size=args.episodes)]
    seeds_b = [int(s) for s in rng.integers(10**6, 2 * 10**6,
                                            size=args.episodes)]

    env = make_env()
    per_ep = {"A": [], "B": []}
    means = {"A": [], "B": []}
    try:
        for label, seeds in (("A", seeds_a), ("B", seeds_b)):
            for i, cand in enumerate(cands):
                w = unvector(cand)
                got = []
                for seed in seeds:
                    for attempt in range(2):
                        try:
                            got.append(play(env, w, seed, args))
                            break
                        except browser_dead:
                            env = restart_env(env, make_env)
                means[label].append(float(np.mean(got)) if got else np.nan)
                per_ep[label].extend(got)
                print(f"    replicate {label}  cand {i:2d}  "
                      f"{means[label][-1]:7.0f}", flush=True)
            print("", flush=True)
    finally:
        try:
            env.close()
        except Exception:                                  # noqa: BLE001
            pass

    a = np.array(means["A"]), np.array(means["B"])
    a, b = a
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) < 3:
        print("  too few candidates survived to correlate")
        return 1

    rs, rp = spearman(a, b), float(np.corrcoef(a, b)[0, 1])
    sd_ep = float(np.std(per_ep["A"] + per_ep["B"], ddof=1))
    noise_var = sd_ep ** 2 / args.episodes          # variance of one mean
    obs_var = float(np.var(np.concatenate([a, b]), ddof=1))
    signal_var = max(obs_var - noise_var, 0.0)

    print(f"  {len(a)} candidates, {args.episodes} episodes per replicate")
    print(f"    per-episode sd                     {sd_ep:8.0f}")
    print(f"    sd of a {args.episodes}-episode mean (noise)     "
          f"{np.sqrt(noise_var):8.0f}")
    print(f"    sd between candidate means         {np.sqrt(obs_var):8.0f}")
    print(f"    implied true sd between candidates {np.sqrt(signal_var):8.0f}")
    print(f"\n    Spearman  A vs B                   {rs:8.2f}"
          "   <- the ranking CEM selects on")
    print(f"    Pearson   A vs B                   {rp:8.2f}")

    if rp > 0.02:
        # Spearman-Brown: episodes needed to raise reliability to 0.8
        need = args.episodes * (0.8 * (1 - rp)) / (rp * (1 - 0.8))
        print(f"\n    episodes per candidate for reliability 0.8: {need:.0f}")
    else:
        print("\n    reliability is indistinguishable from zero: no episode "
              "budget fixes this, the candidates do not differ")
    print("\n  Reliability under about 0.3 means most of a generation's "
          "ordering is\n  noise, and the elite are mostly the lucky.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
