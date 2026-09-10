#!/usr/bin/env python3
"""A hand-written policy, to find out what "good" looks like on this game.

Not a learned agent and not a tree search — it reads the exact fruit positions
out of the physics engine, estimates where each of the 40 candidate drops would
land, scores the outcome, and takes the best. No rollout, so it is one ply of
*estimation* rather than simulation.

Why it exists. Training gave a policy indistinguishable from random (+66 over a
random baseline of 1424, against a standard error of 108). That result was
uninterpretable on its own, because there was no ceiling to compare it to: a
game where hand-coded domain knowledge scores 3000 is one the agent is failing
at, and a game where it scores 1600 is mostly luck. This measured which — and
the answer, at 2497 for `greedy` and 2696 for `layered`, is that the game is
learnable and the agent was the problem. See runs/FINDINGS.md.

Run: python scripts/heuristic.py --episodes 15
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import envpath                                                     # noqa: E402
envpath.ensure()

FLOOR_Y = 912          # top of the floor body
BOARD_W = 640
N_SIZES = 11
# Game.fruitSizes radii and score values, read from the game.
RADII = [24, 32, 40, 56, 64, 72, 84, 96, 128, 160, 192]
# The dropper only ever hands out the five smallest (`Math.floor(rand() * 5)`
# in the game). Anything above size 4 can never be merged by dropping a twin
# on it — it has to be built out of what is already on the board.
DROPPABLE = 5
# How close a dropped twin has to land to count as reaching a fruit, and over
# how many columns to look. The columns are the agent's own action space, so
# "trapped" means trapped given the moves it actually has. See trapped_indices
# for the measurement behind the slack.
TRAP_SLACK = 32.0
TRAP_ACTIONS = 40
SCORES = [1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66]
# Named policies. `greedy` is the one measured in runs/FINDINGS.md;
# `layered` adds the stacking terms. Lookahead is a separate switch, so
# the four combinations are all reachable.
POLICIES = {
    "greedy": dict(merge=250.0, chain=60.0, low=200.0, bury=-8.0,
                   stack=0.0, trap=0.0),
    "layered": dict(merge=250.0, chain=60.0, low=200.0, bury=-8.0,
                    stack=25.0, trap=-20.0),
}

# Scoring a *settled* board rather than a placement. Only reachable with
# --rollout, which replaces the landing estimate with the real physics, so
# these terms can read the board that actually results instead of a guess at
# it. `gained` is in game points; the rest are shaped to sit alongside them.
BOARD_WEIGHTS = dict(gained=12.0, lost=-4000.0, top=-900.0, ready=25.0,
                     buried=-12.0)
WEIGHT_NAMES = ("merge", "chain", "low", "bury", "stack", "trap")

# Reading the board is one script call; doing it per candidate would be 40.
READ_STATE = """
  const fruits = Composite.allBodies(engine.world)
    .filter(b => !b.isStatic && b.circleRadius && b.sizeIndex !== undefined)
    .map(b => [b.position.x, b.position.y, b.circleRadius, b.sizeIndex]);
  return {fruits, cur: Game.currentFruitSize, next: Game.nextFruitSize,
          score: Game.score, status: Game.stateIndex};
"""

# Wait for the board to stop moving, then read it — in one round trip.
#
# The environment already waits for the same predicate before returning an
# observation, but its wait is capped and the cap fires about 3% of the time,
# so roughly nine drops an episode were decided on a board that was still
# settling. Every landing estimate in this file assumes the fruit it reasons
# about is at rest; a fruit still in motion is at a position it is about to
# leave, and the whole placement is computed against a board that will not
# exist by the time the drop lands.
#
# Reports whether it had to wait, so the cost of the guarantee is visible.
SETTLED_READ_STATE = """
  const maxMs = arguments[0], done = arguments[1];
  const started = Date.now();
  (function check() {
    const waited = Date.now() - started;
    const calm = window.Game.settled();
    if (calm || waited >= maxMs) {
      const fruits = Composite.allBodies(engine.world)
        .filter(b => !b.isStatic && b.circleRadius && b.sizeIndex !== undefined)
        .map(b => [b.position.x, b.position.y, b.circleRadius, b.sizeIndex]);
      return done({fruits, cur: Game.currentFruitSize,
                   next: Game.nextFruitSize, score: Game.score,
                   status: Game.stateIndex, waited: waited, settled: calm});
    }
    setTimeout(check, 1);
  })();
"""


def read_state(env, settle_ms=0):
    """The board, optionally after waiting for it to come to rest."""
    if not settle_ms:
        return env.driver.execute_script(READ_STATE)
    return env.driver.execute_async_script(SETTLED_READ_STATE, int(settle_ms))


def landing_y(x, r, fruits):
    """Where a fruit of radius r dropped at x comes to rest.

    It falls from the top, so it stops at the first thing it meets — the
    smallest y among the floor and every fruit whose horizontal span it
    overlaps.
    """
    best = FLOOR_Y - r
    for fx, fy, fr, _ in fruits:
        dx = abs(fx - x)
        reach = fr + r
        if dx >= reach:
            continue
        # Contact when the centres are `reach` apart.
        best = min(best, fy - math.sqrt(reach * reach - dx * dx))
    return best


def contacts(x, y, r, fruits, slack=6.0):
    """Fruits touching a fruit of radius r resting at (x, y)."""
    out = []
    for fx, fy, fr, size in fruits:
        d = math.hypot(fx - x, fy - y)
        if d <= fr + r + slack:
            out.append((fx, fy, fr, size))
    return out


def score_candidate(x, cur, fruits, weights):
    """How good dropping the current fruit at x looks."""
    r = RADII[cur]
    # A drop is clamped to the walls, so candidates outside are the same as the
    # edge — scoring them as if they landed mid-air would be wrong.
    x = min(max(x, r), BOARD_W - r)
    y = landing_y(x, r, fruits)
    touching = contacts(x, y, r, fruits)

    value = 0.0

    # A merge is the whole point of the game.
    same = [t for t in touching if t[3] == cur]
    if same:
        value += weights["merge"] * SCORES[cur]
        # And if the merge product would itself touch its own size, that is a
        # chain — worth far more than one merge.
        if cur + 1 < N_SIZES:
            grown = RADII[cur + 1]
            for fx, fy, fr, size in contacts(x, y, grown, fruits):
                if size == cur + 1:
                    value += weights["chain"] * SCORES[cur + 1]
                    break

    # Keeping the pile low is how you survive, and survival is how you score.
    value += weights["low"] * (y / FLOOR_Y)

    # Dropping a big fruit onto much smaller ones buries them where they can
    # never meet a partner.
    for _, _, _, size in touching:
        if size < cur - 1:
            value += weights["bury"] * (cur - size)

    # Small resting on big is the structure that works. The small fruit stays
    # on the surface where a partner can still reach it, and the big one is
    # already where it belongs — nothing has to move it later.
    w_stack = weights.get("stack", 0.0)
    if w_stack:
        for _, _, _, size in touching:
            if size > cur:
                value += w_stack * (size - cur)

    # The reverse is the position there is no recovering from: a small fruit
    # with something much bigger over it will never meet its partner, and it
    # holds space for the rest of the game. Broader than `bury`, which counts
    # only direct contact — a fruit one layer further down is just as
    # unreachable — but bounded to the band this drop actually covers, so it
    # prices *this* placement instead of re-charging for damage already done.
    w_trap = weights.get("trap", 0.0)
    if w_trap:
        floor_of_band = y + 2.0 * r
        for fx, fy, fr, size in fruits:
            # Two sizes clear, matching `bury`. One size down is ordinary play
            # — a 6 under a 7 is not trapped, it is the thing you were meant
            # to land on — and charging for it would make the term fire on
            # every normal stack.
            if size > cur - 2:
                continue
            if y < fy < floor_of_band and abs(fx - x) < r + fr:
                value += w_trap * (cur - size)

    return value, x


def ready_pairs(fruits, slack=1.45):
    """Same-size fruit close enough to be a merge waiting to happen."""
    n = 0
    for i in range(len(fruits)):
        xi, yi, ri, si = fruits[i]
        for j in range(i + 1, len(fruits)):
            xj, yj, rj, sj = fruits[j]
            if si != sj:
                continue
            if math.hypot(xi - xj, yi - yj) <= (ri + rj) * slack:
                n += 1
    return n


def buried_small(fruits, gap=2):
    """Fruit with something at least `gap` sizes larger sitting over it.

    The position there is no recovering from: it cannot reach a partner, and it
    holds space for the rest of the game.
    """
    n = 0
    for xi, yi, ri, si in fruits:
        for xj, yj, rj, sj in fruits:
            if sj - si < gap:
                continue
            if yj < yi and abs(xi - xj) < ri + rj:
                n += 1
                break
    return n


def trapped_indices(fruits, actions=40, slack=TRAP_SLACK):
    """Which fruits no droppable twin can reach.

    `buried_small` asks whether something much bigger sits above a fruit, which
    is a proxy. This asks the question directly, and in the agent's own terms:
    of the `actions` columns it is allowed to drop into, is there one where an
    identical fruit would come to rest touching this one? If there is not, the
    fruit cannot be merged by any move available this turn — it is holding
    space and returning nothing.

    Only sizes the dropper actually produces count. A size-7 fruit with no
    partner is not trapped, it is waiting on the board to build one, and
    penalising that would be penalising ordinary play.

    The reachability test is the same closed-form fall `score_candidate` uses,
    which ignores roll and bounce, so `slack` stands in for the roll: a twin
    that lands within it counts as having reached this fruit. Measured against
    the engine over 86 fruit on six real boards, each tested against all forty
    drops actually simulated (`scripts/check_trapped.py`):

        slack   called trapped   missed    false free   agrees
            6               56   11 (20%)     0 ( 0%)    87.2%
           32               53    8 (15%)     0 ( 0%)    90.7%
           48               49    5 (10%)     1 ( 3%)    93.0%
           96               39    3 ( 8%)     9 (19%)    86.0%

    32 is the default because it is the widest setting with no false frees.
    Agreement is a point or two higher at 48, but the two errors are not worth
    the same to a term that only ever subtracts: a *missed* charges a penalty
    on a position that was fine, which potential-based shaping largely cancels
    when it persists across a step, while a *false free* lets a real trap go
    unpenalised, which is the whole signal being lost.
    """
    out = []
    for i, (xi, yi, ri, si) in enumerate(fruits):
        if si >= DROPPABLE:
            continue
        for a in range(actions):
            x = (a / max(actions - 1, 1)) * BOARD_W
            # The walls push a fruit dropped past the edge back inside, so the
            # column the agent picks is not always the column it gets.
            x = min(max(x, ri), BOARD_W - ri)
            if abs(x - xi) >= ri + ri:      # cannot touch from any height
                continue
            y = landing_y(x, ri, fruits)
            if math.hypot(x - xi, y - yi) <= ri + ri + slack:
                break
        else:
            out.append(i)
    return out


def trapped_small(fruits, actions=TRAP_ACTIONS, slack=TRAP_SLACK):
    """How many fruits on this board no droppable twin can reach."""
    return len(trapped_indices(fruits, actions, slack))


def score_board(result, weights):
    """How good is the board this drop actually produced?

    Takes the output of `Game.rollout` — a settled board, the points the drop
    really scored, and whether it lost the game. Nothing here is estimated,
    which is the whole point: `score_candidate` has to guess where a fruit
    stops and then reason about a board that may not exist.
    """
    fruits = result["fruits"]
    value = weights["gained"] * result["gained"]
    if result["lost"]:
        value += weights["lost"]
    if fruits:
        highest = min(fy - fr for _fx, fy, fr, _s in fruits)
        value += weights["top"] * max(0.0, 1.0 - highest / FLOOR_Y)
    value += weights["ready"] * ready_pairs(fruits)
    value += weights["buried"] * buried_small(fruits)
    return value


def choose_by_rollout(env, state, actions, weights, board_weights, top_k):
    """Rank cheaply, then simulate the shortlist and pick on the truth.

    Rolling out all forty costs about 550 ms a move, which is affordable but
    wasteful — most columns are obviously poor and the closed-form estimate
    identifies them perfectly well. It is the ordering among the *good* ones
    that the estimate gets wrong, because that is where roll and displacement
    decide the outcome. So the estimate shortlists and the physics decides.
    """
    est = score_all(state, actions, weights)
    order = [int(i) for i in np.argsort(-est)[:top_k]]
    xs = [float(int((i / (actions - 1)) * BOARD_W)) for i in order]
    results = env.driver.execute_script(
        "return Game.rollout(arguments[0], arguments[1]);", xs, state["cur"])
    best_value, best_action = -1e18, order[0]
    for action, result in zip(order, results):
        value = score_board(result, board_weights)
        if value > best_value:
            best_value, best_action = value, action
    return best_action


def score_all(state, actions, weights):
    """The score of every candidate column, not just the winner.

    Used as a supervision target. The argmax alone is a poor thing to imitate:
    among forty columns several are usually near-equivalent, so which one comes
    out on top is close to arbitrary and a classifier trained on it is being
    asked to reproduce a coin flip. The whole vector says which columns are
    *comparably* good, which is both the useful information and forty times as
    much of it per board.
    """
    fruits = state["fruits"]
    cur = state["cur"]
    out = np.empty(actions, dtype=np.float32)
    for i in range(actions):
        x = (i / (actions - 1)) * BOARD_W
        out[i], _ = score_candidate(x, cur, fruits, weights)
    return out


def choose(state, actions, weights):
    return int(np.argmax(score_all(state, actions, weights)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=15)
    ap.add_argument("--actions", type=int, default=40)
    ap.add_argument("--settle-wait", type=int, default=1000, metavar="MS",
                    help="wait up to this long for the board to stop moving "
                         "before scoring candidates. Every landing estimate "
                         "assumes the fruit it reasons about is at rest. "
                         "0 reads immediately, as earlier runs did")
    ap.add_argument("--rollout", type=int, default=0, metavar="K",
                    help="simulate the top K candidates with the real physics "
                         "and choose on the settled board rather than on an "
                         "estimate of where the fruit stops. 0 disables it. "
                         "About 14 ms a candidate")
    ap.add_argument("--seed-base", type=int,
                    help="play seeded games: episode i uses seed SEED_BASE+i. "
                         "Two policies given the same base see identical fruit "
                         "sequences, which turns a comparison between them into "
                         "a paired one and removes the largest source of "
                         "variance in this game")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--port", type=int, default=8988)
    ap.add_argument("--random", action="store_true",
                    help="play randomly instead, to re-measure the floor")
    ap.add_argument("--policy", choices=sorted(POLICIES), default="greedy",
                    help="greedy is the policy measured in runs/FINDINGS.md; "
                         "layered adds the stack and trap terms")
    for name in WEIGHT_NAMES:
        ap.add_argument(f"--{name}", type=float,
                        help=f"override the {name} weight for this policy")
    args = ap.parse_args()

    from agent import browser_failures, restart_env
    from suika_env.suika_browser_env import SuikaBrowserEnv

    weights = dict(POLICIES[args.policy])
    for name in WEIGHT_NAMES:
        given = getattr(args, name)
        if given is not None:
            weights[name] = given

    def make_env():
        return SuikaBrowserEnv(headless=True, port=args.port,
                               obs_mode="features")

    browser_dead = browser_failures()
    env = make_env()
    bins = np.linspace(0.0, 1.0, args.actions)
    scores, lengths, partials = [], [], []
    waited_total = waited_n = drops = stale = 0
    crashes = 0
    ep = 0
    attempts = 0
    try:
        while ep < args.episodes:
            score, steps = 0.0, 0
            try:
                env.reset(seed=None if args.seed_base is None
                          else args.seed_base + ep)
                started = time.time()
                while steps < args.max_steps:
                    if args.random:
                        action = np.random.randint(args.actions)
                    else:
                        state = read_state(env, args.settle_wait)
                        if state.get("waited") is not None:
                            waited_total += state["waited"]
                            waited_n += state["waited"] > 0
                            stale += not state["settled"]
                        if args.rollout:
                            action = choose_by_rollout(
                                env, state, args.actions, weights,
                                BOARD_WEIGHTS, args.rollout)
                        else:
                            action = choose(state, args.actions, weights)
                    obs, _r, done, trunc, info = env.step(
                        np.array([bins[action]], dtype=np.float32))
                    score = info["score"]
                    steps += 1
                    drops += 1
                    if done or trunc:
                        break
            except browser_dead as exc:
                # Chrome drops the tab on a board with enough bodies on it.
                # Crashes therefore land on long episodes, which are the
                # high-scoring ones, so quietly dropping them would drag the
                # mean down. Replay the episode instead and report the count.
                crashes += 1
                partials.append(score)
                first = str(exc).splitlines()[0]
                print(f"  episode {ep:>3}  CRASHED at step {steps}"
                      f" (score so far {score:.0f}) — {first}", flush=True)
                env = restart_env(env, make_env)
                attempts += 1
                if attempts >= 3:
                    print(f"  episode {ep:>3}  abandoned after 3 crashes",
                          flush=True)
                    ep += 1
                    attempts = 0
                continue

            attempts = 0
            scores.append(score)
            lengths.append(steps)
            print(f"  episode {ep:>3}  score {score:>7.0f}  steps {steps:>4}"
                  f"  ({time.time() - started:.0f}s)", flush=True)
            ep += 1
    finally:
        try:
            env.close()
        except Exception:      # noqa: BLE001
            pass

    if not scores:
        print("  no episodes completed")
        return 1

    label = "random" if args.random else args.policy
    label += f" ({args.actions} actions)"
    if args.rollout:
        label += f" +rollout{args.rollout}"
    sd = statistics.stdev(scores) if len(scores) > 1 else 0.0
    se = sd / math.sqrt(len(scores)) if scores else 0.0
    print(f"\n  {label}: n={len(scores)}  mean {statistics.mean(scores):.0f}"
          f" +/- {se:.0f} (se)  sd {sd:.0f}"
          f"  min {min(scores):.0f}  max {max(scores):.0f}"
          f"  mean length {statistics.mean(lengths):.0f}")
    if crashes:
        # A lower bound that keeps the censored episodes in, in case the
        # replayed ones came back systematically shorter.
        withp = scores + partials
        print(f"  {crashes} tab crash(es) replayed;"
              f" mean including their partial scores"
              f" {statistics.mean(withp):.0f} (lower bound, n={len(withp)})")
    if args.settle_wait and drops:
        print(f"  board was still moving on {100*waited_n/drops:.1f}% of "
              f"drops ({waited_total/max(waited_n,1):.0f} ms of extra wait "
              f"each); {100*stale/drops:.1f}% never settled within the cap")
    print("  reference: random 1424 | DQN from scratch 1455 | cloned 2449 | greedy 2497 | layered 2696")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
