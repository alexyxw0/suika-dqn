#!/usr/bin/env python3
"""Check that a seed fixes the fruit sequence.

The point of seeding is to compare policies on the same game so the luck
cancels instead of being averaged away. That needs the seed to fix the fruit
sequence — the game's only call into its PRNG.

What it does *not* buy is bit-identical episodes under a reactive policy. The
physics loop runs on requestAnimationFrame, so how much simulation happens
between two drops depends on wall-clock time; boards drift a little, and a
policy that reads the board then chooses differently amplifies that. So the
test fixes the actions, which isolates the RNG. Pairing therefore removes the
dominant variance (which fruit you were given) and not all of it.

An earlier version of this script used the layered policy and concluded seeding
did not work. It was measuring policy divergence, not the seed.

Run: python scripts/check_seeding.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import envpath                                                     # noqa: E402
envpath.ensure()


def fruit_sequence(env, seed, n=10):
    """The fruit handed out, under a fixed action so nothing else can differ."""
    env.reset(seed=seed)
    seq = []
    for _ in range(n):
        seq.append(env.driver.execute_script("return Game.currentFruitSize;"))
        env.step(np.array([0.5], dtype=np.float32))
    return seq


def main() -> int:
    from suika_env.suika_browser_env import SuikaBrowserEnv
    env = SuikaBrowserEnv(headless=True, port=8994, obs_mode="features")
    try:
        a = fruit_sequence(env, 7)
        b = fruit_sequence(env, 7)
        c = fruit_sequence(env, 8)
        d = fruit_sequence(env, None)
        e = fruit_sequence(env, None)
    finally:
        try:
            env.close()
        except Exception:      # noqa: BLE001
            pass

    print(f"  seed 7     {a}")
    print(f"  seed 7     {b}")
    print(f"  seed 8     {c}")
    print(f"  unseeded   {d}")
    print(f"  unseeded   {e}")
    same = a == b
    differs = a != c
    unseeded_varies = d != e
    print(f"\n  same seed reproduces the sequence: {'yes' if same else 'NO'}")
    print(f"  a different seed changes it:       {'yes' if differs else 'NO'}")
    print(f"  unseeded runs differ:              "
          f"{'yes' if unseeded_varies else 'no (suspicious)'}")
    ok = same and differs
    print(f"  seeding works: {'yes' if ok else 'NO'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
