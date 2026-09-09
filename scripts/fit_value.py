#!/usr/bin/env python3
"""Fit an ensemble of value networks to what boards turned out to be worth.

An ensemble rather than one network, because of how the previous attempt
failed. A value function trained on the boards a policy reached, then used to
pick the best of several, is extrapolating on the ones it never saw — and
`argmax` selects whichever it overrates, so the error it makes is exactly the
error that decides the action. Members disagreeing about a board is the signal
that it is such a board, and `scripts/eval_value.py` ranks by
`mean - beta * std` so a board only one member likes cannot win.

Members differ only in initialisation and batch order. That is weaker than
bootstrapping the data, and it is what fits in the time available; it still
separates "all members agree" from "one member is guessing".

Labels are normalised by their own spread — the discounted remaining score runs
into the hundreds and an MSE loss on targets that size is badly conditioned.
The scale is written beside the checkpoint so values come back in points.

Split by episode, not by board: consecutive boards inside an episode are nearly
the same picture with nearly the same label, so a shuffled split leaks the
answer across it and the holdout number stops meaning anything.

Run: python scripts/fit_value.py --members 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf                                           # noqa: E402

from agent import explained_variance, mirror_boards               # noqa: E402
from train import build_afterstate_model                          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("runs/afterstates.npz"))
    ap.add_argument("--out", type=Path, default=Path("runs/value"))
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--sil-from", type=Path,
                    help="an existing ensemble, used to compute how much each "
                         "drop beat its own prediction. Turns on self-imitation")
    ap.add_argument("--sil-weight", type=float, default=2.0,
                    help="how much extra weight a drop gets per standard "
                        "deviation it beat the prediction by")
    args = ap.parse_args()

    d = np.load(args.data)
    grid = d["grid"].astype(np.float32)
    vector = d["vector"].astype(np.float32)
    value = d["value"].astype(np.float32)
    print(f"  {len(value)} boards from {len(d['episode_scores'])} episodes, "
          f"behaviour policy scored {np.mean(d['episode_scores']):.0f}")

    cut = int(len(value) * (1.0 - args.val_split))
    g_tr, v_tr, y_tr = grid[:cut], vector[:cut], value[:cut]
    g_va, v_va, y_va = grid[cut:], vector[cut:], value[cut:]

    # Mirroring is exact for this board and a reflection does not change what
    # it is worth, so the label carries over untouched.
    gm, _s, _a = mirror_boards(g_tr, np.zeros((len(g_tr), 2)),
                               np.zeros(len(g_tr), dtype=int), 2)
    g_tr = np.concatenate([g_tr, gm])
    v_tr = np.concatenate([v_tr, v_tr])
    y_tr = np.concatenate([y_tr, y_tr])

    centre, scale = float(y_tr.mean()), float(y_tr.std())

    # Self-imitation. `runs/FINDINGS.md` measures 51% of episode-to-episode
    # variance as fruit rather than decisions, so filtering whole episodes by
    # score would select lucky sequences half the time. Advantage is measured
    # against the value function's own prediction for *that board*, which is
    # what removes the confound: it asks "did this drop do better than this
    # position deserved", not "was this a good game".
    weights = None
    if args.sil_from:
        prior = [tf.keras.models.load_model(args.sil_from / f"member{m}.h5")
                 for m in range(json.loads(
                     (args.sil_from / "scale.json").read_text())["members"])]
        pmeta = json.loads((args.sil_from / "scale.json").read_text())
        # `predict` rather than calling the model directly. A direct call runs
        # one forward pass over everything given to it, and the first
        # convolution over 27k boards materialises about 2 GB of activations —
        # enough to send an 8 GB machine into swap for an hour. `predict`
        # chunks. (The opposite choice is right for single-state inference,
        # where predict's per-call overhead dominates; see train.q_of.)
        pred = np.mean([m.predict([g_tr, v_tr], batch_size=512,
                                  verbose=0).ravel()
                        for m in prior], axis=0)
        pred = pred * pmeta["scale"] + pmeta["centre"]
        advantage = (y_tr - pred) / scale
        # Only the upside. A drop that did worse than predicted is not evidence
        # about what to do more of, and down-weighting it would throw away the
        # very examples the value function most needs to stay calibrated.
        weights = 1.0 + args.sil_weight * np.maximum(advantage, 0.0)
        beat = float(np.mean(advantage > 0))
        print(f"  self-imitation from {args.sil_from}: "
              f"{100 * beat:.0f}% of drops beat their prediction, "
              f"weights {weights.min():.2f}-{weights.max():.2f} "
              f"(mean {weights.mean():.2f})")
    print(f"  {len(y_tr)} train (mirrored) / {len(y_va)} holdout"
          f"   labels mean {centre:.0f} sd {scale:.0f}")

    args.out.mkdir(parents=True, exist_ok=True)
    preds = []
    for m in range(args.members):
        tf.random.set_seed(m)
        model = build_afterstate_model(grid_shape=grid.shape[1:],
                                       vector_len=vector.shape[1], seed=m)
        stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience,
            restore_best_weights=True, verbose=0)
        model.fit([g_tr, v_tr], (y_tr - centre) / scale,
                  sample_weight=weights,
                  validation_data=([g_va, v_va], (y_va - centre) / scale),
                  epochs=args.epochs, batch_size=args.batch_size,
                  callbacks=[stop], verbose=0, shuffle=True)
        p = np.asarray(model([g_va, v_va], training=False)).ravel() * scale + centre
        preds.append(p)
        model.save(args.out / f"member{m}.h5")
        print(f"    member {m}: holdout mae {np.abs(p - y_va).mean():6.1f}"
              f"   r {np.corrcoef(p, y_va)[0, 1]:.3f}"
              f"   explained variance {explained_variance(p, y_va):6.3f}")

    preds = np.stack(preds)
    mean = preds.mean(axis=0)
    spread = preds.std(axis=0)
    print(f"\n  ensemble of {args.members}, {len(y_va)} holdout boards")
    print(f"    mean absolute error {np.abs(mean - y_va).mean():6.1f} points"
          f"   (labels sd {y_va.std():.0f})")
    print(f"    correlation         {np.corrcoef(mean, y_va)[0, 1]:6.3f}")
    print(f"    explained variance  {explained_variance(mean, y_va):6.3f}")
    print(f"    member disagreement {spread.mean():6.1f} points mean,"
          f" {np.percentile(spread, 95):.1f} at p95")
    print(f"    — a board the members disagree about is one the policy should "
          f"not\n      pick on one member's enthusiasm; that is what beta buys")

    (args.out / "scale.json").write_text(
        json.dumps({"centre": centre, "scale": scale,
                    "members": args.members}))
    print(f"\n  wrote {args.members} members to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
