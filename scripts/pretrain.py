#!/usr/bin/env python3
"""Fit the Q-network to the hand-written policy, then hand it to RL.

Q-learning has to find a good policy by trying things, and this environment
yields about 100k steps overnight — far too few for 40 actions over 250-step
episodes, which is what `runs/FINDINGS.md` measured. But a good policy already
exists: the heuristic scores ~2696 against a random floor of 1424, using only
information the network already receives. So it can be shown rather than
discovered.

Three things this gets right that the first attempt did not.

*The target is the whole score vector, not the argmax.* Among forty columns
several are usually near-equivalent, so which one wins is close to arbitrary;
a classifier trained on it is asked to reproduce a coin flip. Fitting all forty
scores is forty times the supervision per board and says which columns were
nearly as good. Targets are standardised per board — what a board is worth is
a Q-learning question that demonstrations cannot answer, and with a dueling
head the value stream is exactly the part imitation cannot reach anyway.

*The board is left-right symmetric.* Mirroring the grid, reversing the score
vector and mapping action a to (n-1-a) is an exact relabelling, not an
approximation, so it doubles the data for free and regularises hard.

*Early stopping.* The first run reached 94% train accuracy against 15% holdout
with validation loss rising from epoch five. Best-on-holdout weights are kept
rather than last-epoch ones.

Run: python scripts/pretrain.py --demos runs/demos.npz --out runs/bc.h5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tensorflow as tf                                           # noqa: E402

from agent import mirror_boards, standardise_rows                 # noqa: E402
from train import build_column_model                              # noqa: E402


def report(name, logits, target_scores, actions, rng, n_actions):
    predicted = np.argmax(logits, axis=1)
    gap = np.abs(predicted - actions)
    chance = np.abs(rng.integers(0, n_actions, len(actions)) - actions)
    # What the chosen column was actually worth, against the best and the mean
    # available on that board — the question that matters is not whether it
    # picked the heuristic's column but whether it picked a column as good.
    best = target_scores.max(axis=1)
    mean = target_scores.mean(axis=1)
    got = target_scores[np.arange(len(actions)), predicted]
    regret = np.mean((best - got) / np.maximum(best - mean, 1e-6))
    print(f"\n  {name}, {len(actions)} boards")
    print(f"    exact column        {np.mean(gap == 0) * 100:5.1f}%"
          f"   (chance {100 / n_actions:.1f}%)")
    for tol in (1, 2, 3):
        print(f"    within {tol} column{'s' if tol > 1 else ' '}     "
              f"{np.mean(gap <= tol) * 100:5.1f}%")
    print(f"    mean column error   {np.mean(gap):5.2f}"
          f"   (chance {np.mean(chance):.1f})")
    print(f"    normalised regret   {regret:5.3f}"
          f"   (0 = picked a best column, 1 = picked an average one)")
    return regret


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demos", type=Path, default=Path("runs/demos.npz"))
    ap.add_argument("--out", type=Path, default=Path("runs/bc.h5"))
    ap.add_argument("--actions", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--no-mirror", dest="mirror", action="store_false",
                    default=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = np.load(args.demos)
    if "scores" not in data:
        raise SystemExit(
            f"{args.demos} has no `scores` array — it was written by the older "
            "collect_demos.py. Re-collect with the current one.")
    grid = data["grid"].astype(np.float32)
    vector = data["vector"].astype(np.float32)
    action = data["action"].astype(np.int32)
    scores = data["scores"].astype(np.float32)
    print(f"  {len(action)} boards, demonstrator scored "
          f"{np.mean(data['episode_scores']):.0f} over "
          f"{len(data['episode_scores'])} episodes")

    rng = np.random.default_rng(args.seed)
    tf.random.set_seed(args.seed)

    # Split by episode-contiguous blocks would be better still, but consecutive
    # boards within an episode are highly correlated, so a shuffled split
    # flatters the holdout. Split first, then shuffle inside each side.
    cut = int(len(action) * (1.0 - args.val_split))
    idx_tr, idx_va = np.arange(cut), np.arange(cut, len(action))
    rng.shuffle(idx_tr)

    g_tr, v_tr, s_tr, a_tr = (grid[idx_tr], vector[idx_tr],
                              scores[idx_tr], action[idx_tr])
    g_va, v_va, s_va, a_va = (grid[idx_va], vector[idx_va],
                              scores[idx_va], action[idx_va])

    if args.mirror:
        gm, sm, am = mirror_boards(g_tr, s_tr, a_tr, args.actions)
        g_tr = np.concatenate([g_tr, gm])
        v_tr = np.concatenate([v_tr, v_tr])
        s_tr = np.concatenate([s_tr, sm])
        a_tr = np.concatenate([a_tr, am])
        print(f"  mirrored: {len(a_tr)} training boards")

    y_tr, y_va = standardise_rows(s_tr), standardise_rows(s_va)
    print(f"  {len(a_tr)} train / {len(a_va)} holdout"
          f"  (holdout is the last {args.val_split:.0%} of collection, "
          f"whole episodes)")

    model = build_column_model(num_actions=args.actions,
                               grid_shape=grid.shape[1:],
                               vector_len=vector.shape[1])
    print(f"  {model.count_params():,} parameters")
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss="mse")

    stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=args.patience, restore_best_weights=True,
        verbose=1)
    model.fit([g_tr, v_tr], y_tr,
              validation_data=([g_va, v_va], y_va),
              epochs=args.epochs, batch_size=args.batch_size,
              callbacks=[stop], verbose=2)

    report("holdout", np.asarray(model([g_va, v_va], training=False)),
           s_va, a_va, rng, args.actions)

    out = build_column_model(num_actions=args.actions,
                             grid_shape=grid.shape[1:],
                             vector_len=vector.shape[1])
    out.set_weights(model.get_weights())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.save(args.out)
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
