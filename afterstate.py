"""Encode a board the way the page does, so a candidate board can be scored.

Needed because `Game.rollout` hands back fruit positions and the value network
reads the same (grid, vector) pair the environment produces. The page builds
that in JavaScript for the *live* board only; this reproduces it for a
simulated one.

A transcription that is subtly wrong would not throw — it would train a network
on boards that do not exist — so `scripts/check_afterstate.py` compares it
against the live observation rather than trusting the transcription.

Kept free of TensorFlow so it can be tested where the suite runs.
"""

from __future__ import annotations

import numpy as np

BOARD_W = 640
BOARD_H = 960
N_SIZES = 11
MAX_RADIUS = 192.0
COUNT_SOFTENER = 4.0
LIVE_CAP = 40.0


def rasterise(fruits, grid_w=20, grid_h=30):
    """Fruit into the two-channel grid the network is fed.

    A literal transcription of the page's rasteriser: same cell-centre test,
    same "larger fruit wins an overlapping cell", same normalisations. Channel
    0 is (size + 1) / N_SIZES, channel 1 the radius over 192.
    """
    grid = np.zeros((grid_h, grid_w, 2), dtype=np.float32)
    for fx, fy, fr, size in fruits:
        cx = fx / BOARD_W * grid_w
        cy = fy / BOARD_H * grid_h
        rx = fr / BOARD_W * grid_w
        ry = fr / BOARD_H * grid_h
        y0 = max(0, int(np.floor(cy - ry)))
        y1 = min(grid_h - 1, int(np.ceil(cy + ry)))
        x0 = max(0, int(np.floor(cx - rx)))
        x1 = min(grid_w - 1, int(np.ceil(cx + rx)))
        v = (size + 1) / N_SIZES
        for gy in range(y0, y1 + 1):
            dy = (gy + 0.5 - cy) / max(ry, 1e-6)
            for gx in range(x0, x1 + 1):
                dx = (gx + 0.5 - cx) / max(rx, 1e-6)
                if dx * dx + dy * dy > 1.0:
                    continue
                if v >= grid[gy, gx, 0]:
                    grid[gy, gx, 0] = v
                    grid[gy, gx, 1] = fr / MAX_RADIUS
    return grid


def summarise(fruits, upcoming, grid_h=30):
    """The flat part of the observation: what is on the board and what is next.

    For a candidate board the "fruit in hand" is the one that will be dropped
    next, since the current one has already been spent producing this board.
    """
    counts = np.zeros(N_SIZES, dtype=np.float32)
    top = float(grid_h)
    for _fx, fy, _fr, size in fruits:
        counts[int(size)] += 1.0
        top = min(top, fy / BOARD_H * grid_h)
    counts = counts / (counts + COUNT_SOFTENER)
    cur, nxt = upcoming
    return np.concatenate([
        counts,
        [(cur + 1) / N_SIZES,
         (nxt + 1) / N_SIZES,
         1.0 - min(top, grid_h) / grid_h,
         min(len(fruits), LIVE_CAP) / LIVE_CAP],
    ]).astype(np.float32)


def encode(fruits, upcoming, grid_w=20, grid_h=30):
    """A candidate board as (grid, vector)."""
    return rasterise(fruits, grid_w, grid_h), summarise(fruits, upcoming, grid_h)
