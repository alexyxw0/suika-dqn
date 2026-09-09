#!/usr/bin/env python3
"""Record simulated boards and what they turned out to be worth.

Training data for the afterstate value network, second attempt. The first one
fit well (holdout correlation 0.837) and produced a policy scoring 1794, and
two separate things were wrong with it.

*The boards were not physics.* They came from a closed-form construction — a
landing point computed as "straight down until first overlap", a merge resolved
by averaging two positions, no gravity afterwards. Measured against the game,
that construction got the merge outcome wrong 27% of the time. `Game.rollout`
gets it wrong 3%. The network was being asked to price boards that largely did
not happen.

*The evaluation distribution did not match the training one.* It learned from
the single board actually reached each step, then ranked forty, thirty-nine of
them somewhere it had never been — and argmax does not tolerate that error, it
seeks it. The fix is here rather than in the network: the policy only ever
ranks the shortlist the heuristic proposes, so collection explores *within that
shortlist*. Every board the value function will be asked about at inference is
a board this data covers.

Run: python scripts/collect_afterstates.py --episodes 60
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from afterstate import encode                                     # noqa: E402
from heuristic import (BOARD_W, BOARD_WEIGHTS, POLICIES,          # noqa: E402
                       READ_STATE, WEIGHT_NAMES, score_all, score_board)


def returns_to_go(rewards, gamma):
    out = np.empty(len(rewards), dtype=np.float32)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + gamma * running
        out[i] = running
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--actions", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--rollout", type=int, default=5,
                    help="shortlist size; the same K the policy will rank")
    ap.add_argument("--epsilon", type=float, default=0.25,
                    help="chance of taking a shortlist member other than the "
                         "best. Deliberately high: the point is to cover the "
                         "boards the value function will be asked to compare, "
                         "and every one of them is already a good candidate")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--policy", choices=sorted(POLICIES), default="layered")
    ap.add_argument("--port", type=int, default=8964)
    ap.add_argument("--restart-every", type=int, default=15)
    ap.add_argument("--out", type=Path, default=Path("runs/afterstates.npz"))
    for name in WEIGHT_NAMES:
        ap.add_argument(f"--{name}", type=float)
    args = ap.parse_args()

    from agent import browser_failures, restart_env
    from suika_env.suika_browser_env import SuikaBrowserEnv

    weights = dict(POLICIES[args.policy])
    for name in WEIGHT_NAMES:
        if getattr(args, name) is not None:
            weights[name] = getattr(args, name)
    browser_dead = browser_failures()

    def make_env():
        return SuikaBrowserEnv(headless=True, port=args.port,
                               obs_mode="features")

    env = make_env()
    grids, vectors, labels, ep_scores = [], [], [], []
    explored = total = crashes = 0
    ep, attempts = 0, 0
    try:
        while ep < args.episodes:
            g, v, rewards = [], [], []
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

                    values = [score_board(r, BOARD_WEIGHTS) for r in results]
                    if np.random.rand() < args.epsilon:
                        pick = int(np.random.randint(len(order)))
                        explored += 1
                    else:
                        pick = int(np.argmax(values))
                    total += 1
                    action, taken = order[pick], results[pick]

                    # The board the simulation says this drop produces, encoded
                    # as the network will see it when it is choosing.
                    gg, vv = encode(taken["fruits"],
                                    (state["next"], state["next"]))
                    g.append(gg.astype(np.float16))
                    v.append(vv.astype(np.float16))

                    _obs, _r, done, trunc, info = env.step(
                        np.array([action / (args.actions - 1)],
                                 dtype=np.float32))
                    rewards.append(info["score"] - score)
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
            grids.extend(g)
            vectors.extend(v)
            labels.extend(returns_to_go(rewards, args.gamma))
            ep_scores.append(score)
            print(f"  episode {ep:>3}  score {score:>7.0f}  steps {steps:>4}"
                  f"  boards {len(grids):>6}  ({time.time() - began:.0f}s)",
                  flush=True)
            ep += 1
            if args.restart_every and ep % args.restart_every == 0:
                env = restart_env(env, make_env)
    finally:
        try:
            env.close()
        except Exception:      # noqa: BLE001
            pass

    if not grids:
        print("  nothing collected")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        grid=np.asarray(grids, dtype=np.float16),
        vector=np.asarray(vectors, dtype=np.float16),
        value=np.asarray(labels, dtype=np.float32),
        episode_scores=np.asarray(ep_scores, dtype=np.float32))
    print(f"\n  {len(grids)} boards from {len(ep_scores)} episodes -> "
          f"{args.out} ({args.out.stat().st_size / 1e6:.0f} MB)")
    print(f"  behaviour policy scored {st.mean(ep_scores):.0f}"
          f"  ({crashes} crash(es) discarded)")
    print(f"  took a non-best shortlist member {100 * explored / total:.0f}% "
          f"of the time")
    print(f"  labels: mean {np.mean(labels):.0f}  max {np.max(labels):.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
