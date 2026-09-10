#!/usr/bin/env python3
"""Play by scoring simulated boards with the value ensemble.

The policy, and what is different from the attempt that scored 1794:

1. The heuristic ranks all forty columns and proposes a shortlist. The network
   is never asked to compare boards nobody would consider — which is where the
   previous version spent almost all of its judgement, and where it had no
   data.
2. `Game.rollout` turns each shortlisted drop into the board it actually
   settles into. The old version scored a geometric construction that got the
   merge outcome wrong 27% of the time.
3. The ensemble ranks by `mean - beta * std`. `argmax` over several estimates
   selects whichever the model overrates, so a board the members disagree about
   is penalised rather than rewarded for its optimism.

`--beta 0` and `--rollout 40` recover something close to the original, which
makes the difference measurable rather than assumed.

Run: python scripts/eval_value.py --episodes 20
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import envpath                                                     # noqa: E402
envpath.ensure()

from afterstate import encode                                     # noqa: E402
from heuristic import (BOARD_W, POLICIES, READ_STATE,             # noqa: E402
                       WEIGHT_NAMES, score_all)

REFERENCE = ("random 1424 | cloned DQN 2449 | estimate teacher 2582 | "
             "rollout teacher 2826")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, default=Path("runs/value"))
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--actions", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--rollout", type=int, default=5,
                    help="shortlist size the ensemble ranks")
    ap.add_argument("--beta", type=float, default=1.0,
                    help="penalty on member disagreement. 0 ranks by the mean "
                         "alone, which is what failed before")
    ap.add_argument("--policy", choices=sorted(POLICIES), default="layered")
    ap.add_argument("--port", type=int, default=8967)
    ap.add_argument("--restart-every", type=int, default=15)
    for name in WEIGHT_NAMES:
        ap.add_argument(f"--{name}", type=float)
    args = ap.parse_args()

    from tensorflow.keras.models import load_model

    from agent import browser_failures, restart_env
    from suika_env.suika_browser_env import SuikaBrowserEnv

    meta = json.loads((args.checkpoint / "scale.json").read_text())
    # The members were fitted on standardised labels, so they speak in units of
    # one label standard deviation. Ranking is unaffected — mean and spread are
    # both in those units — but anything printed has to be converted back, or a
    # disagreement of 0.1 gets reported as "0.1 points" when it is nearer 29.
    to_points = float(meta["scale"])
    members = [load_model(args.checkpoint / f"member{m}.h5")
               for m in range(meta["members"])]
    print(f"  {len(members)} ensemble members, beta {args.beta:g}, "
          f"shortlist {args.rollout}")

    weights = dict(POLICIES[args.policy])
    for name in WEIGHT_NAMES:
        if getattr(args, name) is not None:
            weights[name] = getattr(args, name)
    browser_dead = browser_failures()

    def make_env():
        return SuikaBrowserEnv(headless=True, port=args.port,
                               obs_mode="features")

    env = make_env()
    scores, lengths, spreads = [], [], []
    crashes, ep, attempts = 0, 0, 0
    try:
        while ep < args.episodes:
            score, steps = 0.0, 0
            try:
                env.reset()
                began = time.time()
                while steps < args.max_steps:
                    state = env.driver.execute_script(READ_STATE)
                    est = score_all(state, args.actions, weights)
                    order = [int(i) for i in np.argsort(-est)[:args.rollout]]
                    xs = [float(int((i / (args.actions - 1)) * BOARD_W))
                          for i in order]
                    results = env.driver.execute_script(
                        "return Game.rollout(arguments[0], arguments[1]);",
                        xs, state["cur"])

                    grids, vectors = [], []
                    for r in results:
                        g, v = encode(r["fruits"],
                                      (state["next"], state["next"]))
                        grids.append(g)
                        vectors.append(v)
                    batch = [np.asarray(grids, dtype=np.float32),
                             np.asarray(vectors, dtype=np.float32)]
                    preds = np.stack([
                        np.asarray(m(batch, training=False)).ravel()
                        for m in members])
                    value = preds.mean(axis=0) - args.beta * preds.std(axis=0)
                    # A drop that ends the game is worth nothing whatever the
                    # network says about the board it leaves behind.
                    for i, r in enumerate(results):
                        if r["lost"]:
                            value[i] = -1e9
                    spreads.append(float(preds.std(axis=0).mean())
                                   * to_points)
                    action = order[int(np.argmax(value))]

                    _o, _r, done, trunc, info = env.step(
                        np.array([action / (args.actions - 1)],
                                 dtype=np.float32))
                    score = info["score"]
                    steps += 1
                    if done or trunc:
                        break
            except browser_dead as error:
                crashes += 1
                print(f"  episode {ep:>3}  CRASHED at step {steps} — "
                      f"{str(error).splitlines()[0]}", flush=True)
                env = restart_env(env, make_env)
                attempts += 1
                if attempts >= 3:
                    ep += 1
                    attempts = 0
                continue

            attempts = 0
            scores.append(score)
            lengths.append(steps)
            print(f"  episode {ep:>3}  score {score:>7.0f}  steps {steps:>4}"
                  f"  ({time.time() - began:.0f}s)", flush=True)
            ep += 1
            if args.restart_every and ep % args.restart_every == 0:
                env = restart_env(env, make_env)
    finally:
        try:
            env.close()
        except Exception:      # noqa: BLE001
            pass

    if not scores:
        print("  no episodes completed")
        return 1
    sd = st.stdev(scores) if len(scores) > 1 else 0.0
    print(f"\n  value ensemble (beta {args.beta:g}, top {args.rollout}): "
          f"n={len(scores)}  mean {st.mean(scores):.0f} +/- "
          f"{sd / math.sqrt(len(scores)):.0f} (se)  sd {sd:.0f}"
          f"  min {min(scores):.0f}  max {max(scores):.0f}"
          f"  length {st.mean(lengths):.0f}")
    if spreads:
        print(f"  mean member disagreement while playing: "
              f"{st.mean(spreads):.1f} points")
    print(f"  reference: {REFERENCE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
