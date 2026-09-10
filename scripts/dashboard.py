#!/usr/bin/env python3
"""Watch any of the policies play, and switch between them.

The environment drives a real browser, so this has to run locally rather than
as a hosted page. It serves a small control panel on localhost: pick a policy,
press play, and watch the board it is actually reasoning about — the fruit
positions are read straight out of the physics engine, the same numbers the
agent sees, not a screenshot.

For the policies that simulate their candidates, the shortlist is drawn too, so
you can see which columns were considered and which one was taken.

Run: python scripts/dashboard.py           then open http://localhost:8500
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import envpath                                                     # noqa: E402
envpath.ensure()

from heuristic import (BOARD_W, BOARD_WEIGHTS, POLICIES,           # noqa: E402
                       READ_STATE, RADII, score_all, score_board)

FLOOR_Y = 912
BOARD_H = 960

# One shared snapshot, written by the runner thread and read by the server.
STATE = {
    "running": False, "status": "idle", "policy": None, "episode": 0, "step": 0,
    "score": 0, "best": 0, "history": [], "fruits": [], "cur": 0, "next": 0,
    "shortlist": [], "chosen": None, "error": None, "thinking_ms": 0,
}
LOCK = threading.Lock()
STOP = threading.Event()

# Every board of the current episode, so a position can be revisited. One
# episode runs to a few hundred drops; keeping the last 600 covers one with
# room to spare and bounds the memory at a few megabytes.
FRAMES = collections.deque(maxlen=600)

# Simulation requests from the page. Only the runner thread may touch the
# browser — Selenium is not safe to call from two threads — so a request is
# left here and picked up between drops, and the answer left in SIM_RESULT.
SIM_REQUEST = {"pending": None, "id": 0}
SIM_RESULT = {"id": -1, "candidates": None, "error": None}


def catalogue():
    """Everything that can be asked to play."""
    root = Path(__file__).resolve().parent.parent
    out = [
        {"id": "random", "name": "Random", "kind": "reference",
         "note": "uniform column, the floor everything is measured against"},
        {"id": "greedy", "name": "Hand-written (greedy)", "kind": "written",
         "note": "closed-form landing estimate, four scoring terms"},
        {"id": "layered", "name": "Hand-written (layered)", "kind": "written",
         "note": "adds stacking and trap terms"},
        {"id": "layered+rollout", "name": "Hand-written + simulation",
         "kind": "written",
         "note": "shortlists 5, simulates each in the physics engine"},
    ]
    for path, name in (("runs/bc-column.h5", "Cloned Q-network"),
                       ("runs/bc-layered.h5", "Cloned Q-network (layered demos)"),
                       ("runs/dqn-finetune.h5", "Q-network after RL fine-tuning")):
        if (root / path).exists():
            out.append({"id": f"net:{path}", "name": name, "kind": "learned",
                        "note": "one forward pass, argmax over 40 columns"})
    if (root / "runs/value/scale.json").exists():
        out.append({"id": "ensemble", "name": "Board-value ensemble",
                    "kind": "learned",
                    "note": "simulates 5 candidates, scores each board, "
                            "penalises member disagreement"})
    return out


def make_chooser(policy_id, actions):
    """Returns choose(env, state) -> (action, shortlist) for a catalogue id."""
    if policy_id == "random":
        return lambda env, st: (int(np.random.randint(actions)), [])

    if policy_id in POLICIES:
        weights = POLICIES[policy_id]

        def pick(env, st):
            scores = score_all(st, actions, weights)
            return int(np.argmax(scores)), []
        return pick

    if policy_id == "layered+rollout":
        weights = POLICIES["layered"]

        def pick(env, st):
            est = score_all(st, actions, weights)
            order = [int(i) for i in np.argsort(-est)[:5]]
            xs = [float(int((i / (actions - 1)) * BOARD_W)) for i in order]
            res = env.driver.execute_script(
                "return Game.rollout(arguments[0], arguments[1]);",
                xs, st["cur"])
            vals = [score_board(r, BOARD_WEIGHTS) for r in res]
            best = int(np.argmax(vals))
            shortlist = [{"column": c, "value": round(v, 1),
                          "gained": res[k]["gained"], "lost": res[k]["lost"]}
                         for k, (c, v) in enumerate(zip(order, vals))]
            return order[best], shortlist
        return pick

    if policy_id.startswith("net:"):
        from tensorflow.keras.models import load_model
        from train import observation, q_of
        root = Path(__file__).resolve().parent.parent
        model = load_model(root / policy_id[4:])

        # The network reads the environment's own observation rather than the
        # raw fruit list, so the runner hands it in as st["obs"].
        def pick_net(env, st):
            q = q_of(model, st["obs"])[0]
            order = [int(i) for i in np.argsort(-q)[:5]]
            shortlist = [{"column": c, "value": round(float(q[c]), 2)}
                         for c in order]
            return int(np.argmax(q)), shortlist
        return pick_net

    if policy_id == "ensemble":
        from tensorflow.keras.models import load_model
        from afterstate import encode
        root = Path(__file__).resolve().parent.parent
        meta = json.loads((root / "runs/value/scale.json").read_text())
        members = [load_model(root / f"runs/value/member{m}.h5")
                   for m in range(meta["members"])]
        weights = POLICIES["layered"]

        def pick(env, st):
            est = score_all(st, actions, weights)
            order = [int(i) for i in np.argsort(-est)[:5]]
            xs = [float(int((i / (actions - 1)) * BOARD_W)) for i in order]
            res = env.driver.execute_script(
                "return Game.rollout(arguments[0], arguments[1]);",
                xs, st["cur"])
            grids, vecs = [], []
            for r in res:
                g, v = encode(r["fruits"], (st["next"], st["next"]))
                grids.append(g)
                vecs.append(v)
            batch = [np.asarray(grids, np.float32), np.asarray(vecs, np.float32)]
            preds = np.stack([np.asarray(m(batch, training=False)).ravel()
                              for m in members])
            val = preds.mean(axis=0) - preds.std(axis=0)
            for i, r in enumerate(res):
                if r["lost"]:
                    val[i] = -1e9
            best = int(np.argmax(val))
            shortlist = [{"column": c,
                          "value": round(float(val[k] * meta["scale"]), 0),
                          "spread": round(float(preds.std(axis=0)[k]
                                                * meta["scale"]), 0),
                          "gained": res[k]["gained"]}
                         for k, c in enumerate(order)]
            return order[best], shortlist
        return pick

    raise ValueError(f"unknown policy {policy_id}")


def runner(policy_id, args):
    """Play episodes until asked to stop, publishing state as it goes."""
    from agent import browser_failures, restart_env
    from suika_env.suika_browser_env import SuikaBrowserEnv
    from train import observation

    browser_dead = browser_failures()

    def make_env():
        return SuikaBrowserEnv(headless=args.headless, port=args.env_port,
                               obs_mode="features")

    env = None
    try:
        with LOCK:
            STATE["status"] = "opening the browser"
        env = make_env()
        with LOCK:
            STATE["status"] = "loading the policy"
        choose = make_chooser(policy_id, args.actions)
        with LOCK:
            STATE["status"] = "playing"
        episode = 0
        best = 0
        history = []
        while not STOP.is_set():
            obs, _ = env.reset()
            score, step = 0.0, 0
            while step < args.max_steps and not STOP.is_set():
                raw = env.driver.execute_script(READ_STATE)
                raw["obs"] = observation(obs)
                began = time.time()
                try:
                    action, shortlist = choose(env, raw)
                except browser_dead:
                    raise
                think = (time.time() - began) * 1000

                with LOCK:
                    STATE.update(fruits=raw["fruits"], cur=raw["cur"],
                                 next=raw["next"], shortlist=shortlist,
                                 chosen=action, step=step, score=int(score),
                                 episode=episode, thinking_ms=round(think, 1))
                    FRAMES.append({
                        "step": step, "score": int(score),
                        "fruits": raw["fruits"], "cur": raw["cur"],
                        "next": raw["next"], "shortlist": shortlist,
                        "chosen": action, "episode": episode,
                    })
                    STATE["frames"] = len(FRAMES)
                serve_simulation(env, args)
                obs, _r, done, trunc, info = env.step(
                    np.array([action / (args.actions - 1)], dtype=np.float32))
                score = info["score"]
                step += 1
                if done or trunc:
                    break
            best = max(best, int(score))
            history.append(int(score))
            episode += 1
            with LOCK:
                STATE.update(score=int(score), best=best,
                             history=history[-40:], episode=episode)
                FRAMES.clear()
                STATE["frames"] = 0
    except Exception as exc:                       # noqa: BLE001
        with LOCK:
            STATE["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:                      # noqa: BLE001
                pass
        with LOCK:
            STATE["running"] = False
            STATE["status"] = "idle"


def serve_simulation(env, args):
    """Answer one pending "what would this column do here" request.

    Called between drops so the browser is only ever driven by this thread.
    The board comes from a recorded frame, so the answer is what the physics
    would really have done from that position, not a re-scoring of the board
    on screen now.
    """
    with LOCK:
        req = SIM_REQUEST["pending"]
        SIM_REQUEST["pending"] = None
    if not req:
        return
    try:
        res = env.driver.execute_script(
            "return Game.rollout(arguments[0], arguments[1], arguments[2], "
            "arguments[3]);",
            req["xs"], req["size"], 600, req["board"])
        out = [{"column": c, "fruits": r["fruits"], "gained": r["gained"],
                "lost": r["lost"], "ticks": r["ticks"]}
               for c, r in zip(req["columns"], res)]
        with LOCK:
            SIM_RESULT.update(id=req["id"], candidates=out, error=None)
    except Exception as exc:                       # noqa: BLE001
        with LOCK:
            SIM_RESULT.update(id=req["id"], candidates=None,
                              error=f"{type(exc).__name__}: {exc}")


PAGE = (Path(__file__).resolve().parent / "dashboard.html")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):                     # quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        raw = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, PAGE.read_text(), "text/html; charset=utf-8")
        if self.path == "/api/policies":
            return self._send(200, json.dumps(catalogue()))
        if self.path.startswith("/api/frames"):
            with LOCK:
                frames = [{"step": f["step"], "score": f["score"],
                           "cur": f["cur"], "next": f["next"],
                           "chosen": f["chosen"], "fruit": len(f["fruits"])}
                          for f in FRAMES]
            return self._send(200, json.dumps(frames))
        if self.path.startswith("/api/frame/"):
            try:
                i = int(self.path.rsplit("/", 1)[1])
            except ValueError:
                return self._send(400, json.dumps({"error": "bad index"}))
            with LOCK:
                if not 0 <= i < len(FRAMES):
                    return self._send(404, json.dumps({"error": "no such frame"}))
                return self._send(200, json.dumps(FRAMES[i]))
        if self.path.startswith("/api/simulation/"):
            want = int(self.path.rsplit("/", 1)[1])
            with LOCK:
                if SIM_RESULT["id"] != want:
                    return self._send(202, json.dumps({"pending": True}))
                return self._send(200, json.dumps(
                    {"candidates": SIM_RESULT["candidates"],
                     "error": SIM_RESULT["error"]}))
        if self.path == "/api/state":
            with LOCK:
                snap = dict(STATE)
            snap["radii"] = RADII
            snap["board"] = {"w": BOARD_W, "h": BOARD_H, "floor": FLOOR_Y}
            return self._send(200, json.dumps(snap))
        return self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or "{}")
        if self.path == "/api/start":
            with LOCK:
                if STATE["running"]:
                    return self._send(409, json.dumps({"error": "already running"}))
                STATE.update(running=True, status="starting",
                             policy=payload.get("policy"),
                             error=None, history=[], best=0, episode=0,
                             score=0, step=0, fruits=[], shortlist=[])
            STOP.clear()
            threading.Thread(target=runner,
                             args=(payload["policy"], self.server.args),
                             daemon=True).start()
            return self._send(200, json.dumps({"ok": True}))
        if self.path == "/api/simulate":
            with LOCK:
                if not STATE["running"]:
                    return self._send(409, json.dumps(
                        {"error": "nothing is playing — simulation runs in the "
                                  "same browser the policy uses"}))
                i = int(payload["frame"])
                if not 0 <= i < len(FRAMES):
                    return self._send(404, json.dumps({"error": "no such frame"}))
                frame = FRAMES[i]
                columns = payload.get("columns") or list(range(0, 40, 4))
                SIM_REQUEST["id"] += 1
                rid = SIM_REQUEST["id"]
                SIM_REQUEST["pending"] = {
                    "id": rid, "board": frame["fruits"], "size": frame["cur"],
                    "columns": columns,
                    "xs": [float(int((c / 39) * BOARD_W)) for c in columns],
                }
            return self._send(200, json.dumps({"id": rid}))
        if self.path == "/api/stop":
            STOP.set()
            return self._send(200, json.dumps({"ok": True}))
        return self._send(404, json.dumps({"error": "not found"}))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8500, help="dashboard port")
    ap.add_argument("--env-port", type=int, default=8600,
                    help="port the game's own page server uses")
    ap.add_argument("--actions", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--show-browser", dest="headless", action="store_false",
                    default=True,
                    help="also open the real game window, not just the "
                         "dashboard's rendering of it")
    args = ap.parse_args()

    try:
        import suika_env.suika_browser_env  # noqa: F401
    except ImportError as exc:
        print("  " + envpath.diagnose(exc).replace("\n", "\n  "))
        return 1

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.args = args
    print(f"  dashboard on http://localhost:{args.port}")
    print(f"  {len(catalogue())} policies available")
    print("  ctrl-c to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        STOP.set()
        time.sleep(1)
        print("\n  stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
