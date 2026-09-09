# Suika DQN

A reinforcement-learning agent for [Suika](https://suikagame.com/), the
fruit-merging puzzle game, trained against a browser-based Gymnasium
environment driven through Selenium.

The agent learns from the game's own state rather than pixels, is bootstrapped
by imitating a hand-written policy, and chooses its move by simulating each
candidate drop in the game's physics engine before committing to one.

`runs/FINDINGS.md` records every measurement, including the approaches that did
not work.

**This repo contains only my own work.** The environment belongs to
[`edwhu/suika_rl`](https://github.com/edwhu/suika_rl), which ships no licence,
so redistributing it is not mine to do — `env-fixes.patch` is applied to a
fresh clone instead of vendoring it.

| component | author |
|---|---|
| Gymnasium environment, Selenium harness | [edwhu/suika_rl](https://github.com/edwhu/suika_rl) |
| the Suika clone the env hosts | [TomboFry/suika-game](https://github.com/TomboFry/suika-game) |
| agent, environment fixes, speedup, everything in this repo | mine |

## Setup

Requires Chrome and chromedriver — the environment drives a real browser.

```bash
git clone https://github.com/edwhu/suika_rl.git
cd suika_rl && pip install -e . && cd ..
git -C suika_rl apply ../env-fixes.patch
export PYTHONPATH=$PWD/suika_rl
pip install -r requirements.txt
```

The environment is never edited in place. It is a clean clone plus a patch, so
the working copy is reproducible rather than precious — which matters, because
the one used while writing this lived in `/tmp` and was cleared twice.

### Checking the setup

```bash
PYTHON=$(which python) ./scripts/verify_patch.sh   # clone, apply, compile, import
python -m pytest tests/ -q                         # 139 tests, no browser needed
```

`verify_patch.sh` imports the patched module rather than stopping at
`git apply --check`, because `--check` only says the hunks fit. An earlier
version of the patch passed it and produced a file with two statements spliced
onto one line: BSD `diff` had emitted a `\ No newline at end of file` marker
after every hunk.

Four more checks. The first three drive a browser; the last does not:

```bash
python scripts/check_afterstate.py    # board encoder matches the page exactly
python scripts/check_rollout.py       # a simulated drop matches the real one
python scripts/check_seeding.py       # a seed fixes the fruit sequence
python scripts/check_train_step.py    # compiled gradient step is exact, and faster
```

They exist because each covers something that fails silently rather than
loudly — an encoder off by half a cell, a simulation that disagrees with the
game, a seed that is accepted and ignored.

## The pipeline

Each stage writes into `runs/` and the next one reads it.

**1. Measure the ends of the scale.** A hand-written policy and random play,
so any later number has something to mean.

```bash
python scripts/heuristic.py --episodes 30 --policy layered --rollout 5
python scripts/heuristic.py --episodes 30 --random
```

**2. Record demonstrations.** The hand-written policy plays; every board it
reaches is paired with the column it chose.

```bash
python scripts/collect_demos.py --episodes 90        # -> runs/demos.npz
```

**3. Clone it.** Supervised fitting of the Q-network to those choices, which is
how the agent gets off the floor — learning from scratch does not work here.

```bash
python scripts/pretrain.py                           # -> runs/bc.h5
python scripts/eval_policy.py --checkpoint runs/bc.h5 --episodes 25
```

**4. Reinforcement learning from there.**

```bash
python train.py --resume --checkpoint runs/bc.h5 \
  --demos runs/demos.npz --n-step 20 --terminal-penalty 300 \
  --reward-scale 0.01 --lr 5e-5 --value-bias 9 --epsilon 0.05
```

`--demos` is not optional in practice: without it, TD fine-tuning forgets the
cloned policy within fifty episodes.

**Or, the board-value route.** Instead of one network emitting a value per
column, simulate each candidate drop and score the board it produces.

```bash
python scripts/collect_afterstates.py --episodes 60  # -> runs/afterstates.npz
python scripts/fit_value.py --members 4              # -> runs/value/
python scripts/eval_value.py --episodes 20
```

## Watching it play

```bash
python scripts/dashboard.py            # then open http://localhost:8500
```

A local control panel: pick any policy — random, either hand-written variant,
the simulating one, or any trained checkpoint in `runs/` — and watch it play.
The board is drawn from the fruit positions read out of the physics engine, so
it is the board the policy is actually reasoning about rather than a
screenshot of one.

For the policies that simulate their candidates, the columns they weighed are
drawn on the board and listed with what each was worth, which is the part worth
watching: you can see the shortlist, the points each candidate would score, and
which one was taken.

`--show-browser` also opens the real game window alongside it.

## How it works

**Observation.** The board rasterised into a 30×20×2 grid (fruit size, radius)
plus a 15-vector: how many of each size are live, what is in hand, what is
next, how full the board is. Read straight out of the physics engine.

**Network.** A convolutional stack that never strides the board's width.
Height is reduced away, leaving one feature vector per board column; the global
vector is broadcast onto every column; a width-1 convolution scores every
column with the same weights, so a pattern learned at one position applies at
all of them. The action is *which column*, so the output stays indexed by board
position.

**Simulated drops.** `Game.rollout` builds a scratch Matter.js world from the
live board, drops a candidate into it, and steps until nothing moves — the real
merge rule included. The policy shortlists with a cheap closed-form landing
estimate, then decides on boards the physics actually produced.

**Environment.** The upstream step took 640 ms, of which 500 was a fixed
`time.sleep(0.5)` after every drop. The patch replaces that with a settle
predicate evaluated inside the page, raises the physics multiplier to 25×,
mutes audio, renders on demand, adds a feature observation mode, and makes the
fruit sequence seedable. The step is now **88 ms**, of which 81 is genuinely
waiting for the board to stop moving.

## Layout

```
train.py              RL loop: replay, n-step targets, compiled gradient step
agent.py              replay buffer, n-step returns, TD targets, crash recovery
afterstate.py         encodes a candidate board the way the page does
scripts/heuristic.py  the hand-written policy, and the random baseline
scripts/dashboard.py  local control panel for watching any policy play
scripts/              demonstrations, cloning, evaluation, plotting, checks
tests/                139 tests, pure numpy, no browser required
runs/FINDINGS.md      every measurement, including the failures
env-fixes.patch       against a clean upstream clone
```

MIT licensed.
