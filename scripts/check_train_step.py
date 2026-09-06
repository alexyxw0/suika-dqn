#!/usr/bin/env python3
"""Check the compiled gradient step against the `model.fit` version it replaced.

The point of the rewrite was speed, so it has to be shown that it changed
nothing else. Two copies of the same network, identical weights, identical
batch, one update each — the resulting weights should agree.

`train.py` needs TensorFlow and the pytest suite runs in an environment without
it, which is why this is a script rather than a test.

Run: python scripts/check_train_step.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf                                          # noqa: E402

from agent import apply_targets, double_td_targets               # noqa: E402
from train import (build_column_model, make_joint_train_step,     # noqa: E402
                   make_train_step)

ACTIONS = 40
BATCH = 32
GRID = (30, 20, 2)
VEC = 15


def fake_batch(rng):
    grids = rng.random((BATCH,) + GRID).astype(np.float32)
    vectors = rng.random((BATCH, VEC)).astype(np.float32)
    actions = rng.integers(0, ACTIONS, BATCH).astype(np.int32)
    rewards = rng.random(BATCH).astype(np.float32) * 20
    dones = (rng.random(BATCH) < 0.1).astype(np.float32)
    steps = rng.integers(1, 21, BATCH)
    return [grids, vectors], actions, rewards, dones, steps


def main() -> int:
    rng = np.random.default_rng(0)
    tf.random.set_seed(0)

    old = build_column_model(num_actions=ACTIONS)
    new = build_column_model(num_actions=ACTIONS)
    new.set_weights(old.get_weights())
    step = make_train_step(new, ACTIONS)

    state_batch, actions, rewards, dones, steps = fake_batch(rng)
    next_batch, *_ = fake_batch(rng)

    q_next = np.asarray(old(next_batch, training=False))
    targets = double_td_targets(rewards, dones, q_next, q_next, 0.99, steps)

    # the path that was there before
    current = old.predict(state_batch, verbose=0)
    old.fit(state_batch, apply_targets(current, actions, targets),
            epochs=1, verbose=0)

    # the path that is there now
    step(state_batch, actions, targets.astype(np.float32))

    deltas = [np.max(np.abs(a - b))
              for a, b in zip(old.get_weights(), new.get_weights())]
    worst = max(deltas)
    moved = max(np.max(np.abs(w)) for w in new.get_weights())
    print(f"  largest weight disagreement after one update: {worst:.3e}")
    print(f"  (largest weight magnitude for scale: {moved:.3e})")

    ok = worst < 1e-6
    print(f"  equivalent: {'yes' if ok else 'NO'}")

    # and the reason for the change
    reps = 30
    t0 = time.time()
    for _ in range(reps):
        c = old.predict(state_batch, verbose=0)
        old.fit(state_batch, apply_targets(c, actions, targets),
                epochs=1, verbose=0)
    slow = (time.time() - t0) / reps

    step(state_batch, actions, targets.astype(np.float32))     # warm the trace
    t0 = time.time()
    for _ in range(reps):
        step(state_batch, actions, targets.astype(np.float32))
    fast = (time.time() - t0) / reps

    print(f"\n  predict + fit : {slow * 1000:6.1f} ms/update")
    print(f"  compiled step : {fast * 1000:6.1f} ms/update"
          f"   ({slow / fast:.1f}x faster)")

    ok = check_joint(rng) and ok
    return 0 if ok else 1


def check_joint(rng):
    """The demonstration term must not fight the TD term over the value level.

    Cloning standardises its targets per board, so matching raw Q-values would
    drag the value stream back to zero while TD pulls it towards the discounted
    return. Centring the output before comparing is what stops that, and the
    property to verify is that the demonstration loss is unchanged when every
    Q-value on a board shifts by the same constant.
    """
    print("\n  joint TD + demonstration step")
    model = build_column_model(num_actions=ACTIONS)
    step = make_joint_train_step(model, ACTIONS, demo_weight=1.0)

    state_batch, actions, rewards, dones, steps = fake_batch(rng)
    demo_batch, *_ = fake_batch(rng)
    demo_targets = rng.standard_normal((BATCH, ACTIONS)).astype(np.float32)
    demo_targets -= demo_targets.mean(axis=1, keepdims=True)
    targets = rewards.astype(np.float32)

    td0, bc0 = step(state_batch, actions, targets, demo_batch, demo_targets)
    print(f"    runs: td {float(td0):.4f}  bc {float(bc0):.4f}")

    # Shift the value head by a constant and the demonstration loss must not
    # move; the TD loss must.
    value = model.get_layer("value")
    w, b = value.get_weights()
    before_td, before_bc = step(state_batch, actions, targets,
                                demo_batch, demo_targets)
    value.set_weights([w, b + 50.0])
    after_td, after_bc = step(state_batch, actions, targets,
                              demo_batch, demo_targets)
    value.set_weights([w, b])

    bc_moved = abs(float(after_bc) - float(before_bc))
    td_moved = abs(float(after_td) - float(before_td))
    print(f"    value bias +50 -> demonstration loss moves {bc_moved:.5f}")
    print(f"                      TD loss moves            {td_moved:.2f}")
    good = bc_moved < 1e-2 and td_moved > 1.0
    print(f"    demonstration term is level-invariant: "
          f"{'yes' if good else 'NO'}")
    return good


if __name__ == "__main__":
    raise SystemExit(main())
