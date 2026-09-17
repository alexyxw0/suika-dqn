#!/usr/bin/env python3
"""End-to-end invariants, in a few minutes rather than a few hours.

The 184 unit tests check that functions compute what they claim. Every real bug
in this project was something else: the pipeline quietly doing a different
experiment from the one intended. `collect_demos.py` demonstrating from the
closed-form teacher while the run was labelled as the rollout one. An A/B
handing a +376 advantage to whichever arm went first. A checkpoint loaded
against data whose observation width had changed underneath it.

None of those break anything. They produce plausible numbers that are answers
to the wrong question, and the only symptom is a disappointing result hours
later.

So this asserts the *invariants of the pipeline* instead of the outputs:

  env          a patched rollout returns a trace only when asked for one, and
               its last frame is the board it says it settled into
  demonstrator the recorded action is the argmax of the recorded target, and
               with --rollout the simulation actually changes the choice
               sometimes — if it never does, the teacher is not the one named
  training     a single epoch runs, the loss falls, and the model's output
               width matches the action count in the data
  checkpoints  every saved model's input widths match the datasets on disk,
               so a stale checkpoint cannot be silently evaluated

Run: python scripts/smoke.py            (about three minutes)
     python scripts/smoke.py --offline  (skips everything needing a browser)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import envpath                                                     # noqa: E402
envpath.ensure()

FAILURES: list = []


def check(name, condition, detail=""):
    mark = "ok  " if condition else "FAIL"
    print(f"    [{mark}] {name}" + (f"  — {detail}" if detail else ""),
          flush=True)
    if not condition:
        FAILURES.append(name)
    return condition


def check_env(args):
    """The patch is applied and `Game.rollout` behaves as the callers assume."""
    from suika_env.suika_browser_env import SuikaBrowserEnv
    from heuristic import READ_STATE

    print("  env")
    env = SuikaBrowserEnv(headless=True, port=args.port, obs_mode="features")
    try:
        env.reset(seed=1)
        for _ in range(12):
            env.step(np.array([np.random.rand()], dtype=np.float32))
        state = env.driver.execute_script(READ_STATE)
        check("board reads back", len(state["fruits"]) > 0,
              f"{len(state['fruits'])} fruit")

        plain = env.driver.execute_script(
            "return Game.rollout(arguments[0], arguments[1])[0];",
            [320.0], state["cur"])
        check("rollout without capture carries no trace",
              plain["trace"] is None and plain["merges"] is None)

        traced = env.driver.execute_script(
            "return Game.rollout(arguments[0], arguments[1], arguments[2], "
            "arguments[3], arguments[4])[0];", [320.0], state["cur"], 600,
            None, 2)
        ok = traced["trace"] is not None and len(traced["trace"]) > 1
        check("rollout with capture returns a trace", ok,
              f"{len(traced['trace']) if ok else 0} snapshots")
        if ok:
            tail = traced["trace"][-1]
            drift = max(abs(a[0] - b[0]) + abs(a[1] - b[1])
                        for a, b in zip(tail, traced["fruits"]))
            check("the trace ends on the board it settled into", drift < 1.0,
                  f"{drift:.2f}px")
            pts = sum(m[4] for m in (traced["merges"] or []))
            check("recorded merge values sum to the points gained",
                  pts == traced["gained"], f"{pts} vs {traced['gained']}")
        return env
    except Exception:
        env.close()
        raise


def check_demonstrator(env, args):
    """The teacher is the one the flags name, and the target agrees with it."""
    from collect_demos import rollout_targets
    from heuristic import POLICIES, READ_STATE, score_all

    print("  demonstrator")
    weights = dict(POLICIES["layered"])
    changed = agreed = 0
    bins = np.linspace(0.0, 1.0, args.actions)
    env.reset(seed=2)

    class A:
        rollout, actions = 5, args.actions

    for _ in range(args.drops):
        state = env.driver.execute_script(READ_STATE)
        closed = score_all(state, args.actions, weights)
        target, action = rollout_targets(env, state, closed, weights, A)
        agreed += int(action == int(np.argmax(target)))
        changed += int(action != int(np.argmax(closed)))
        _o, _r, done, trunc, _i = env.step(
            np.array([bins[action]], dtype=np.float32))
        if done or trunc:
            env.reset(seed=3)

    check("the recorded action is the argmax of the recorded target",
          agreed == args.drops, f"{agreed}/{args.drops}")
    # The check that would have caught the wrong-teacher bug: if simulating
    # never changes the choice, the rollout is not being consulted and the
    # demonstrations are the closed-form policy's under another name.
    check("simulating changes the choice sometimes", changed >= 2,
          f"{changed}/{args.drops} drops differ from the closed-form argmax")


def check_training(args):
    """One epoch on real data: it runs, it learns, the shapes line up."""
    from tensorflow.keras.callbacks import History               # noqa: F401
    from agent import standardise_rows
    from train import build_column_model

    print("  training")
    data = sorted(ROOT.glob("runs/demos*.npz"))
    if not check("a demonstration set exists", bool(data)):
        return
    d = np.load(data[-1])
    n = min(args.boards, len(d["scores"]))
    g = d["grid"][:n].astype(np.float32)
    v = d["vector"][:n].astype(np.float32)
    y = standardise_rows(d["scores"][:n].astype(np.float32))
    model = build_column_model(num_actions=y.shape[1], grid_shape=g.shape[1:],
                               vector_len=v.shape[1])
    check("model output width matches the action count",
          model.output_shape[-1] == y.shape[1],
          f"{model.output_shape[-1]} vs {y.shape[1]}")
    h = model.fit([g, v], y, epochs=2, batch_size=64, verbose=0)
    losses = h.history["loss"]
    check("loss falls over two epochs", losses[-1] < losses[0],
          f"{losses[0]:.3f} -> {losses[-1]:.3f}")


def check_checkpoints():
    """No saved model can be silently evaluated against data it does not fit.

    Three ways a model can legitimately not match a stored dataset, and they
    have to be told apart from the failure this exists to catch:

    - its input is *derived* rather than stored, like the per-column geometry
      `rep_test.py` computes from the fruit positions;
    - it is from an abandoned observation mode, like the pixel-era network
      whose run runs/FINDINGS.md records as void;
    - it is genuinely stale, which is the one that matters.
    """
    from tensorflow.keras.models import load_model

    print("  checkpoints")
    sets, derived = {}, {}
    for f in sorted(ROOT.glob("runs/*.npz")):
        d = np.load(f)
        if "grid" in d and "vector" in d:
            sets[f.name] = (d["grid"].shape[1:], d["vector"].shape[1])
        if "fruits" in d and "scores" in d:
            # what rep_test.geometry() builds from this set
            derived[f.name] = (d["scores"].shape[1], 8)
    if not check("datasets found", bool(sets)):
        return
    for f in sorted(ROOT.glob("runs/*.h5")) + sorted(ROOT.glob("runs/*/*.h5")):
        try:
            m = load_model(f, compile=False)
        except Exception as exc:                               # noqa: BLE001
            check(f"{f.name} loads", False, str(exc).splitlines()[0][:60])
            continue
        shapes = [tuple(i.shape[1:]) for i in m.inputs]
        if any(len(sh) == 3 and sh[-1] == 3 for sh in shapes):
            print(f"    [skip] {f.name} — pixel-era observation, abandoned; "
                  "FINDINGS.md records that run as void")
            continue
        fits = [n for n, (gs, vl) in sets.items()
                if gs in shapes and (vl,) in shapes]
        fits += [f"{n} (derived)" for n, sh in derived.items()
                 if sh in shapes]
        check(f"{f.name} matches data on disk", bool(fits),
              ", ".join(fits) if fits else f"inputs {shapes} match nothing")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offline", action="store_true",
                    help="skip the checks that need a browser")
    ap.add_argument("--drops", type=int, default=12)
    ap.add_argument("--boards", type=int, default=400)
    ap.add_argument("--actions", type=int, default=40)
    ap.add_argument("--port", type=int, default=8993)
    args = ap.parse_args()

    env = None
    try:
        if not args.offline:
            env = check_env(args)
            check_demonstrator(env, args)
        check_training(args)
        check_checkpoints()
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:                                  # noqa: BLE001
                pass

    print()
    if FAILURES:
        print(f"  {len(FAILURES)} FAILED: " + "; ".join(FAILURES))
        return 1
    print("  all invariants hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
