#!/usr/bin/env python3
"""Does the Q function believe its own actions are better than they turn out?

The plateau in `runs/FINDINGS.md` is explained by inference rather than by
measurement: at this sample budget the bootstrapped targets are noisy, the
greedy policy takes an argmax over forty of them, and an argmax over forty
estimates with error sigma picks up roughly +2 sigma of bias — it selects
whichever action the network happens to overrate. That would make the policy
degrade faster than the value function improves, which is what plain TD
fine-tuning did (2449 down to ~1730).

This measures it directly. Play greedily, record Q(s, a_taken) at every step,
and afterwards compute the discounted return actually collected from that
state. Three numbers matter:

- **bias**: mean(Q - realised return). Positive is overestimation.
- **the argmax gap**: mean(max_a Q(s,a) - mean_a Q(s,a)). How far the chosen
  action sits above the field. If the bias is a similar size, then most of what
  the argmax is selecting for is error rather than genuine advantage.
- **correlation**: whether Q ranks states correctly even when miscalibrated.

Rewards are scaled exactly as training scaled them, so Q and the return are in
the same units.

Run: python scripts/check_value_calibration.py --checkpoint runs/dqn-finetune.h5
"""

from __future__ import annotations

import argparse
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import envpath                                                     # noqa: E402
envpath.ensure()


def discounted_returns(rewards, gamma):
    out = np.zeros(len(rewards), dtype=np.float64)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + gamma * running
        out[i] = running
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path,
                    default=Path("runs/dqn-finetune.h5"))
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--actions", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--reward-scale", type=float, default=0.01)
    ap.add_argument("--terminal-penalty", type=float, default=300.0)
    ap.add_argument("--port", type=int, default=8935)
    args = ap.parse_args()

    from tensorflow.keras.models import load_model

    from agent import (action_bins, browser_failures, restart_env,
                       terminal_reward, to_continuous)
    from suika_env.suika_browser_env import SuikaBrowserEnv
    from train import observation, q_of

    model = load_model(args.checkpoint)
    bins = action_bins(args.actions)
    browser_dead = browser_failures()

    def make_env():
        return SuikaBrowserEnv(headless=True, port=args.port,
                               obs_mode="features")

    env = make_env()
    q_taken, q_gap, realised, scores = [], [], [], []
    ep = 0
    try:
        while ep < args.episodes:
            try:
                obs, _ = env.reset()
                state = observation(obs)
                qs_ep, gap_ep, rew_ep = [], [], []
                score, steps = 0.0, 0
                while steps < args.max_steps:
                    q = q_of(model, state)[0]
                    action = int(np.argmax(q))
                    qs_ep.append(float(q[action]))
                    gap_ep.append(float(q.max() - q.mean()))

                    obs, _r, done, trunc, info = env.step(
                        to_continuous(bins, action))
                    shaped, _terminal = terminal_reward(
                        info["score"] - score, done, trunc,
                        args.terminal_penalty, args.reward_scale)
                    rew_ep.append(shaped)
                    score = info["score"]
                    state = observation(obs)
                    steps += 1
                    if done or trunc:
                        break
            except browser_dead:
                env = restart_env(env, make_env)
                continue

            ret = discounted_returns(rew_ep, args.gamma)
            q_taken.extend(qs_ep)
            q_gap.extend(gap_ep)
            realised.extend(ret)
            scores.append(score)
            print(f"  episode {ep:>3}  score {score:>7.0f}  steps {steps:>4}"
                  f"  mean Q {st.mean(qs_ep):6.2f}"
                  f"  mean return {ret.mean():6.2f}", flush=True)
            ep += 1
    finally:
        try:
            env.close()
        except Exception:      # noqa: BLE001
            pass

    if not q_taken:
        print("  nothing collected")
        return 1

    q_taken = np.asarray(q_taken)
    realised = np.asarray(realised)
    q_gap = np.asarray(q_gap)
    bias = float((q_taken - realised).mean())
    r = float(np.corrcoef(q_taken, realised)[0, 1])

    print(f"\n  {len(q_taken)} states over {len(scores)} episodes"
          f"  (mean score {st.mean(scores):.0f})")
    print(f"    mean Q of the chosen action      {q_taken.mean():8.3f}")
    print(f"    mean discounted return realised  {realised.mean():8.3f}")
    print(f"    bias (Q - return)                {bias:+8.3f}"
          f"   {'overestimates' if bias > 0 else 'underestimates'}")
    print(f"    correlation Q vs return          {r:8.3f}")
    print(f"    argmax gap (max Q - mean Q)      {q_gap.mean():8.3f}")
    if q_gap.mean() > 1e-9:
        print(f"\n    bias / argmax gap = {bias / q_gap.mean():.2f}")
        print("    Near or above 1 means the margin the greedy policy is "
              "selecting on\n    is about the size of its own error — the "
              "argmax is choosing noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
