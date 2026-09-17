#!/usr/bin/env python3
"""Does the input representation explain the ~2600 ceiling?

Every learned method in this project lands near 2600 while the hand-written
policy scores 3100, and `scripts/check_within_state.py` located the failure
precisely: the clone ranks candidates well overall (within-state Spearman
0.849) but must resolve a median gap of 0.077 within-state standard deviations
between the best candidate and the second, with an error of 0.522. It is wrong
only on the knife-edge decisions, and over ~280 drops that is the whole gap.

One suspect is the observation. Forty actions were rasterised onto a grid 20
columns wide, so *two adjacent drop positions received identical spatial
input*. This fits the same learning problem — predict the teacher's forty
column scores, standardised per board — under three inputs on identical
boards, and reports the metrics the policy actually consumes rather than
pooled ones:

    coarse  20x30 raster, max-pooled down from the collected 40x30.
            What every previous network here saw.
    fine    40x30 raster. One grid column per action, nothing else changed.
    geom    eight exact geometric facts per column, computed from float fruit
            positions: where the drop lands, what it lands on, how far to the
            nearest same-size fruit. Precise perception, no scoring.

`geom` is not the heuristic in disguise — it carries no merge, chain, bury,
stack, trap, danger or order term, and the network has to discover what those
facts are worth. It does test the upper end of what precision alone can buy.

Run: python scripts/rep_test.py --input fine
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import envpath                                                     # noqa: E402
envpath.ensure()

from heuristic import BOARD_W, FLOOR_Y, N_SIZES, RADII             # noqa: E402


def coarsen(grid):
    """40 columns back down to 20, the way the env built them: the larger
    fruit wins an overlapping cell, so max is the faithful reduction."""
    n, h, w, c = grid.shape
    return grid.reshape(n, h, w // 2, 2, c).max(axis=3)


def geometry(fruits, cur, actions=40):
    """Exact per-column facts, from float positions rather than a raster.

    Deliberately descriptive, not evaluative: where a drop lands, what it lands
    on, what is nearby. The network still has to learn that landing on your own
    size is good.
    """
    live = fruits[fruits[:, 2] > 0]
    r = RADII[int(cur)]
    out = np.zeros((actions, 8), dtype=np.float32)
    for i in range(actions):
        x = min(max((i / (actions - 1)) * BOARD_W, r), BOARD_W - r)
        y = FLOOR_Y - r
        support = -1.0
        for fx, fy, fr, fs in live:
            dx = abs(fx - x)
            reach = fr + r
            if dx >= reach:
                continue
            cy = fy - math.sqrt(reach * reach - dx * dx)
            if cy < y:
                y, support = cy, fs
        same_dx = same_dy = 1.0
        best = 1e9
        for fx, fy, fr, fs in live:
            if int(fs) != int(cur):
                continue
            d = math.hypot(fx - x, fy - y)
            if d < best:
                best = d
                same_dx = (fx - x) / BOARD_W
                same_dy = (fy - y) / FLOOR_Y
        touching = sum(1 for fx, fy, fr, fs in live
                       if math.hypot(fx - x, fy - y) <= fr + r + 6.0)
        out[i] = (y / FLOOR_Y,
                  (y - r) / FLOOR_Y,
                  (support + 1) / N_SIZES,
                  (int(support) == int(cur)) * 1.0,
                  min(best, BOARD_W) / BOARD_W,
                  same_dx, same_dy,
                  min(touching, 6) / 6.0)
    return out


def build_geom_model(n_features, vector_len, num_actions=40, lr=1e-3):
    """The column head with the convolutional stack removed.

    Deliberately the same shape as `build_column_model` from the width-1
    convolution onward, so a difference between the two is a difference in what
    reached the head, not in the head.
    """
    from tensorflow.keras import layers, models
    import tensorflow as tf

    col_in = layers.Input(shape=(num_actions, n_features), name="columns")
    vec_in = layers.Input(shape=(vector_len,), name="vector")
    spread = layers.Dense(32, activation="relu")(vec_in)
    spread = layers.RepeatVector(num_actions)(spread)
    x = layers.Concatenate(axis=-1)([col_in, spread])
    x = layers.Conv1D(128, 1, activation="relu")(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.Conv1D(1, 1)(x)
    out = layers.Reshape((num_actions,))(x)
    model = models.Model(inputs=[col_in, vec_in], outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
    return model


def spearman_rows(a, b):
    ra = np.argsort(np.argsort(a, axis=1), axis=1).astype(np.float64)
    rb = np.argsort(np.argsort(b, axis=1), axis=1).astype(np.float64)
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    return ((ra * rb).sum(1)
            / np.maximum(np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1)), 1e-12))


def report(pred, teacher, target):
    spread = teacher.std(axis=1)
    rho = spearman_rows(pred, teacher)
    top1 = (pred.argmax(1) == teacher.argmax(1)).mean()
    taken = teacher[np.arange(len(teacher)), pred.argmax(1)]
    regret = ((teacher.max(1) - taken) / np.maximum(spread, 1e-9)).mean()
    rmse = float(np.sqrt(((pred - target) ** 2).mean()))
    rank = (teacher > taken[:, None]).sum(1).mean()
    print(f"    within-state Spearman   {rho.mean():.3f}")
    print(f"    top-1 agreement         {100 * top1:.1f}%")
    print(f"    mean rank of the pick   {rank:.2f} of {teacher.shape[1]}")
    print(f"    regret per decision     {regret:.3f} within-state sd")
    print(f"    RMSE                    {rmse:.3f} within-state sd")
    return dict(spearman=rho.mean(), top1=top1, regret=regret, rmse=rmse)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("runs/demos-fine.npz"))
    ap.add_argument("--input", choices=("coarse", "fine", "geom"),
                    default="fine")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    import tensorflow as tf
    from agent import standardise_rows
    from train import build_column_model

    d = np.load(args.data)
    teacher = d["scores"].astype(np.float32)
    vector = d["vector"].astype(np.float32)
    n, actions = teacher.shape
    cut = int(n * (1.0 - args.val_split))
    print(f"  {n} boards, {actions} candidates, input '{args.input}'")

    if args.input == "geom":
        fruits = d["fruits"].astype(np.float32)
        cur = np.rint(vector[:, N_SIZES] * N_SIZES - 1).astype(int)
        cur = np.clip(cur, 0, N_SIZES - 1)
        print("    building per-column geometry (a few minutes)", flush=True)
        x = np.stack([geometry(fruits[i], cur[i], actions) for i in range(n)])
        model = build_geom_model(x.shape[-1], vector.shape[1], actions)
        inputs_tr, inputs_va = [x[:cut], vector[:cut]], [x[cut:], vector[cut:]]
    else:
        grid = d["grid"].astype(np.float32)
        if args.input == "coarse":
            grid = coarsen(grid)
        print(f"    grid {grid.shape[1]}x{grid.shape[2]}"
              f"  ({actions // grid.shape[2]} action(s) per column)")
        model = build_column_model(num_actions=actions,
                                   grid_shape=grid.shape[1:],
                                   vector_len=vector.shape[1])
        inputs_tr = [grid[:cut], vector[:cut]]
        inputs_va = [grid[cut:], vector[cut:]]

    y = standardise_rows(teacher)
    print(f"    {model.count_params():,} parameters", flush=True)
    model.fit(inputs_tr, y[:cut], validation_data=(inputs_va, y[cut:]),
              epochs=args.epochs, batch_size=args.batch_size, verbose=2,
              callbacks=[tf.keras.callbacks.EarlyStopping(
                  monitor="val_loss", patience=args.patience,
                  restore_best_weights=True, verbose=1)])

    print(f"\n  holdout, {n - cut} boards, input '{args.input}'")
    report(model.predict(inputs_va, batch_size=512, verbose=0),
           teacher[cut:], y[cut:])
    if args.out:
        model.save(args.out)
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
