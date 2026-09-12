#!/usr/bin/env python3
"""Record the hand-written policy playing, as supervised training data.

Q-learning from scratch has to discover a good policy by trying things, and
this environment gives about 100k steps in an overnight run — tiny for 40
actions over 250-step episodes. But a good policy already exists: the heuristic
in `heuristic.py` scores 2742 against a random floor of 1424, and it decides
using only information the network already receives. So instead of making the
network find that policy by exploration, it can be shown it.

Each sample is the observation the network would see, paired with the column the
heuristic chose from the same board. `pretrain.py` fits the Q-network to those
choices; `train.py --resume` then continues with reinforcement learning from
those weights.

Stored as float16: every value in the grid is a quantised ratio, so the extra
precision is empty, and the file halves.

Run: python scripts/collect_demos.py --episodes 60 --out runs/demos.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import envpath                                                     # noqa: E402
envpath.ensure()

from heuristic import (BOARD_W, BOARD_WEIGHTS, POLICIES,          # noqa: E402
                       READ_STATE, WEIGHT_NAMES, score_all, score_board)


def rollout_targets(env, state, col_scores, weights, args):
    """Fold the rollout teacher's ordering into the closed-form score vector.

    The supervision target is the whole 40-column vector, so cloning the
    rollout teacher means the *vector* has to express its preferences, not
    just the action it took. Its own numbers cannot be used directly:
    `score_board` values a settled board and `score_candidate` values a
    placement, on unrelated scales, and only K of the 40 columns have a board
    value at all. Splicing them would hand the network a target with a
    discontinuity at the shortlist boundary.

    So the scale stays the closed-form one and only the *order* changes. Among
    the K shortlisted columns the closed-form scores are reassigned so the
    column the simulation liked best gets the largest of them, second-best the
    second largest, and so on; the other 35 keep their values. Same dense
    signal, saying which columns are comparably good, with the ordering
    corrected exactly where the physics disagreed with the estimate — which is
    the only place it ever does.
    """
    order = [int(i) for i in np.argsort(-col_scores)[:args.rollout]]
    xs = [float(int((i / (args.actions - 1)) * BOARD_W)) for i in order]
    results = env.driver.execute_script(
        "return Game.rollout(arguments[0], arguments[1]);", xs, state["cur"])
    values = [score_board(r, BOARD_WEIGHTS) for r in results]

    out = col_scores.copy()
    pool = sorted((col_scores[i] for i in order), reverse=True)
    by_sim = [order[i] for i in np.argsort(-np.asarray(values))]
    for column, score in zip(by_sim, pool):
        out[column] = score
    return out, by_sim[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--actions", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--port", type=int, default=8951)
    ap.add_argument("--restart-every", type=int, default=15,
                    help="rebuild the browser every N episodes. A page that "
                         "has been alive a long time is the one that hangs, "
                         "and a hang costs the client's full 120 s read "
                         "timeout before recovery can even start")
    ap.add_argument("--out", type=Path, default=Path("runs/demos.npz"))
    ap.add_argument("--rollout", type=int, default=0, metavar="K",
                    help="demonstrate with the rollout teacher: simulate the "
                         "top K closed-form candidates and choose on the "
                         "settled board. Without this the demonstrations come "
                         "from the closed-form policy, whatever the rollout "
                         "one scores")
    ap.add_argument("--policy", choices=sorted(POLICIES), default="layered",
                    help="layered scored 2771 against greedy's 2497 over 42 "
                         "episodes each (p=0.011); it is the better teacher")
    for name in WEIGHT_NAMES:
        ap.add_argument(f"--{name}", type=float,
                        help=f"override the {name} weight")
    args = ap.parse_args()

    from agent import browser_failures, restart_env
    from suika_env.suika_browser_env import SuikaBrowserEnv

    weights = dict(POLICIES[args.policy])
    for name in WEIGHT_NAMES:
        given = getattr(args, name)
        if given is not None:
            weights[name] = given
    print(f"  demonstrator: {args.policy}")

    def make_env():
        return SuikaBrowserEnv(headless=True, port=args.port,
                               obs_mode="features")

    browser_dead = browser_failures()
    env = make_env()
    grids, vectors, actions, scores, qscores = [], [], [], [], []
    crashes = 0
    ep = 0
    attempts = 0
    try:
        while ep < args.episodes:
            # Only kept if the episode finishes, so a crash cannot contribute a
            # truncated tail that looks like a game ending early.
            g, v, a, q = [], [], [], []
            score, steps = 0.0, 0
            try:
                obs, _ = env.reset()
                started = time.time()
                while steps < args.max_steps:
                    state = env.driver.execute_script(READ_STATE)
                    col_scores = score_all(state, args.actions, weights)
                    if args.rollout:
                        col_scores, action = rollout_targets(
                            env, state, col_scores, weights, args)
                    else:
                        action = int(np.argmax(col_scores))

                    g.append(np.asarray(obs["grid"], dtype=np.float16))
                    v.append(np.asarray(obs["vector"], dtype=np.float16))
                    a.append(action)
                    # The whole score vector, not just its argmax: forty
                    # supervised numbers a board instead of one, and it says
                    # which columns were nearly as good as the winner.
                    q.append(col_scores.astype(np.float32))

                    obs, _r, done, trunc, info = env.step(
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
                    print(f"  episode {ep:>3}  abandoned", flush=True)
                    ep += 1
                    attempts = 0
                continue

            attempts = 0
            grids.extend(g)
            vectors.extend(v)
            actions.extend(a)
            qscores.extend(q)
            scores.append(score)
            print(f"  episode {ep:>3}  score {score:>7.0f}  steps {steps:>4}"
                  f"  samples {len(grids):>6}  ({time.time() - started:.0f}s)",
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
        action=np.asarray(actions, dtype=np.int16),
        scores=np.asarray(qscores, dtype=np.float32),
        episode_scores=np.asarray(scores, dtype=np.float32))
    mb = args.out.stat().st_size / 1e6
    print(f"\n  {len(grids)} samples from {len(scores)} episodes -> "
          f"{args.out} ({mb:.0f} MB)")
    print(f"  demonstrator mean score {np.mean(scores):.0f}"
          f"  ({crashes} crash(es) discarded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
