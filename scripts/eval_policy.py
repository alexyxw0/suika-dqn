#!/usr/bin/env python3
"""Play a saved checkpoint greedily and report what it scores.

The only number that settles anything. Holdout accuracy says how well the
network reproduces the heuristic's choices; this says whether the resulting
policy actually plays, which is a different question — a policy can agree with
the demonstrator on most boards and still lose the ones that matter.

Reports a standard error, because a five-episode mean here has one near 137 and
cannot detect a difference smaller than about 540 — which is how two earlier
non-results came to be read as signals.

Run: python scripts/eval_policy.py --checkpoint runs/bc.h5 --episodes 20
"""

from __future__ import annotations

import argparse
import math
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import envpath                                                     # noqa: E402
envpath.ensure()

REFERENCE = "random 1424 | DQN from scratch 1455 | cloned 2449 | greedy 2497 | layered 2696"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--actions", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--port", type=int, default=8961)
    ap.add_argument("--epsilon", type=float, default=0.0,
                    help="random-action rate; 0 is greedy")
    ap.add_argument("--sample-scale", type=float,
                    help="sample from softmax(logits * scale) instead of "
                         "taking the argmax. What a policy-gradient method "
                         "would actually play. The cloned network was fitted "
                         "by regression on standardised scores, so its outputs "
                         "are values rather than logits: at scale 1 they are "
                         "nearly uniform over the 40 columns")
    args = ap.parse_args()

    from tensorflow.keras.models import load_model

    from agent import (action_bins, browser_failures, restart_env,
                       to_continuous)
    from suika_env.suika_browser_env import SuikaBrowserEnv
    from train import observation, q_of

    model = load_model(args.checkpoint)
    bins = action_bins(args.actions)

    def make_env():
        return SuikaBrowserEnv(headless=True, port=args.port,
                               obs_mode="features")

    browser_dead = browser_failures()
    env = make_env()
    scores, lengths, partials = [], [], []
    crashes = 0
    ep, attempts = 0, 0
    try:
        while ep < args.episodes:
            score, steps = 0.0, 0
            try:
                obs, _ = env.reset()
                state = observation(obs)
                started = time.time()
                while steps < args.max_steps:
                    if args.epsilon and np.random.rand() < args.epsilon:
                        action = np.random.randint(args.actions)
                    else:
                        action = int(np.argmax(q_of(model, state)[0]))
                    obs, _r, done, trunc, info = env.step(
                        to_continuous(bins, action))
                    state = observation(obs)
                    score = info["score"]
                    steps += 1
                    if done or trunc:
                        break
            except browser_dead as error:
                # Crashes fall on long episodes, which are the high-scoring
                # ones, so dropping them would bias the mean down. Replay.
                crashes += 1
                partials.append(score)
                print(f"  episode {ep:>3}  CRASHED at step {steps} — "
                      f"{str(error).splitlines()[0]}", flush=True)
                env = restart_env(env, make_env)
                attempts += 1
                if attempts >= 3:
                    print(f"  episode {ep:>3}  abandoned", flush=True)
                    ep += 1
                    attempts = 0
                continue

            attempts = 0
            scores.append(score)
            lengths.append(steps)
            print(f"  episode {ep:>3}  score {score:>7.0f}  steps {steps:>4}"
                  f"  ({time.time() - started:.0f}s)", flush=True)
            ep += 1
    finally:
        try:
            env.close()
        except Exception:      # noqa: BLE001
            pass

    if not scores:
        print("  no episodes completed")
        return 1

    sd = st.stdev(scores) if len(scores) > 1 else 0.0
    se = sd / math.sqrt(len(scores))
    how = ("argmax" if not args.sample_scale
           else f"sampled@{args.sample_scale:g}")
    print(f"\n  {args.checkpoint.name} ({how}): n={len(scores)}"
          f"  mean {st.mean(scores):.0f} +/- {se:.0f} (se)  sd {sd:.0f}"
          f"  min {min(scores):.0f}  max {max(scores):.0f}"
          f"  mean length {st.mean(lengths):.0f}")
    if crashes:
        allof = scores + partials
        print(f"  {crashes} crash(es) replayed; including their partials "
              f"{st.mean(allof):.0f} (lower bound, n={len(allof)})")
    print(f"  reference: {REFERENCE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
