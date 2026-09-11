#!/usr/bin/env python3
"""Compare two weight sets on the same games, with the arms interleaved.

Built because this repo has already been burned twice by the alternatives.

Running two arms one after the other reported a 200-action space ahead by +306;
alternating the order reversed the sign (runs/FINDINGS.md §6). Anything that
drifts over a session — thermal throttling, a browser slowly leaking, the
machine picking up other work — lands entirely on whichever arm ran second.
So the arms alternate seed by seed inside one process and one browser.

Pairing is the other half. Fruit sequence explains about half the variance in
this game, so scoring both arms on the *same* seeds removes it from the
difference. The statistic reported is the mean of the per-seed differences and
its standard error, not the difference of two independent means.

    python scripts/ab_weights.py --episodes 16 --b danger=0 --b order=0

`--a` and `--b` take `name=value` overrides on top of `--policy`. Arm A with
no overrides is the policy as it stands, so the example above measures what
the two new terms are worth.
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import envpath                                                     # noqa: E402
envpath.ensure()

from cem import play                                               # noqa: E402
from heuristic import POLICIES, WEIGHT_NAMES                       # noqa: E402


def overrides(base, pairs):
    w = dict(base)
    for item in pairs or []:
        name, _, value = item.partition("=")
        if name not in WEIGHT_NAMES:
            raise SystemExit(f"unknown weight {name!r}; "
                             f"expected one of {', '.join(WEIGHT_NAMES)}")
        w[name] = float(value)
    return w


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=16,
                    help="seeds; each is played by both arms")
    ap.add_argument("--policy", choices=sorted(POLICIES), default="layered")
    ap.add_argument("--a", action="append", metavar="NAME=VALUE")
    ap.add_argument("--b", action="append", metavar="NAME=VALUE")
    ap.add_argument("--actions", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=400,
                    help="higher than the usual 300: a cap that binds turns "
                         "a survival difference into no difference")
    ap.add_argument("--settle-wait", type=int, default=1000)
    ap.add_argument("--rollout", type=int, default=0)
    ap.add_argument("--port", type=int, default=8997)
    ap.add_argument("--seed-base", type=int, default=70000)
    args = ap.parse_args()

    from agent import browser_failures, restart_env
    from suika_env.suika_browser_env import SuikaBrowserEnv

    wa = overrides(POLICIES[args.policy], args.a)
    wb = overrides(POLICIES[args.policy], args.b)
    differing = [n for n in WEIGHT_NAMES if wa[n] != wb[n]]
    if not differing:
        raise SystemExit("A and B are the same weights; nothing to compare")
    print("  A: " + "  ".join(f"{n}={wa[n]:g}" for n in differing))
    print("  B: " + "  ".join(f"{n}={wb[n]:g}" for n in differing) + "\n",
          flush=True)

    def make_env():
        return SuikaBrowserEnv(headless=True, port=args.port,
                               obs_mode="features")

    browser_dead = browser_failures()
    env = make_env()
    pairs, steps = [], []
    try:
        for i in range(args.episodes):
            seed = args.seed_base + i
            # Alternate which arm goes first, so an effect of playing second
            # cancels across seeds instead of loading onto one arm.
            order = [("A", wa), ("B", wb)] if i % 2 == 0 else \
                    [("B", wb), ("A", wa)]
            got = {}
            for label, w in order:
                for attempt in range(2):
                    try:
                        got[label] = play(env, w, seed, args)
                        break
                    except browser_dead:
                        env = restart_env(env, make_env)
            if len(got) == 2:
                pairs.append((got["A"][0], got["B"][0]))
                steps.append((got["A"][1], got["B"][1]))
                print(f"    seed {seed}  A {got['A'][0]:6.0f}"
                      f"   B {got['B'][0]:6.0f}"
                      f"   diff {got['A'][0] - got['B'][0]:+6.0f}"
                      f"   (drops {got['A'][1]}/{got['B'][1]}"
                      f", first: {order[0][0]})", flush=True)
            else:
                print(f"    seed {seed}  incomplete, dropped", flush=True)
    finally:
        try:
            env.close()
        except Exception:                                  # noqa: BLE001
            pass

    if len(pairs) < 3:
        print("  too few complete pairs to conclude anything")
        return 1

    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    diffs = [x - y for x, y in pairs]
    se = st.stdev(diffs) / (len(diffs) ** 0.5)
    wins = sum(d > 0 for d in diffs)

    capped = sum(1 for pair in steps for st_ in pair if st_ >= args.max_steps)
    print(f"\n  {len(pairs)} paired seeds")
    print(f"    mean drops  A {st.mean([s[0] for s in steps]):.0f}"
          f"   B {st.mean([s[1] for s in steps]):.0f}"
          + (f"   WARNING: {capped} episode(s) hit the {args.max_steps}-drop "
             "cap, so the survival difference is truncated" if capped
             else f"   (no episode reached the {args.max_steps}-drop cap)"))
    print(f"    A  {st.mean(a):7.0f}  (sd {st.stdev(a):.0f})")
    print(f"    B  {st.mean(b):7.0f}  (sd {st.stdev(b):.0f})")
    print(f"    paired difference  {st.mean(diffs):+.0f} +/- {se:.0f} (se)")
    print(f"    A won {wins} of {len(pairs)}")
    print(f"\n    unpaired se would have been "
          f"{(st.stdev(a)**2 / len(a) + st.stdev(b)**2 / len(b)) ** 0.5:.0f}"
          " — pairing is worth having only when it is smaller than that")
    if abs(st.mean(diffs)) < 2 * se:
        print("    -> inside twice the standard error: no effect demonstrated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
