#!/usr/bin/env python3
"""Check the Python board encoder against the page's own.

`afterstate.rasterise` is a transcription of JavaScript that runs in the
browser. A transcription that is subtly wrong would not throw — it would train
the value network on boards that do not exist, and every result after that
would be quietly untrustworthy.

So it is compared against the real thing: play a few steps, ask the page both
for its observation and for the raw fruit positions, encode the latter here,
and require the two to agree.

Run: python scripts/check_afterstate.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import afterstate as A                                            # noqa: E402
from heuristic import READ_STATE                                  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--port", type=int, default=8995)
    args = ap.parse_args()

    from suika_env.suika_browser_env import SuikaBrowserEnv

    env = SuikaBrowserEnv(headless=True, port=args.port, obs_mode="features")
    worst_grid = worst_vec = 0.0
    checked = 0
    try:
        obs, _ = env.reset()
        for step in range(args.steps):
            raw = env.driver.execute_script(READ_STATE)
            dg = float(np.abs(A.rasterise(raw["fruits"]) - obs["grid"]).max())
            dv = float(np.abs(A.summarise(raw["fruits"],
                                          (raw["cur"], raw["next"]))
                              - obs["vector"]).max())
            worst_grid = max(worst_grid, dg)
            worst_vec = max(worst_vec, dv)
            checked += 1
            print(f"  step {step:>2}  {len(raw['fruits']):>2} fruit"
                  f"  grid delta {dg:.2e}  vector delta {dv:.2e}", flush=True)
            obs, _r, done, trunc, _i = env.step(
                np.array([np.random.rand()], dtype=np.float32))
            if done or trunc:
                obs, _ = env.reset()
    finally:
        try:
            env.close()
        except Exception:      # noqa: BLE001
            pass

    ok = worst_grid < 1e-5 and worst_vec < 1e-5
    print(f"\n  {checked} boards compared")
    print(f"  worst grid disagreement   {worst_grid:.3e}")
    print(f"  worst vector disagreement {worst_vec:.3e}")
    print(f"  encoder matches the page: {'yes' if ok else 'NO'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
