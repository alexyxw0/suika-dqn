# Suika DQN

A reinforcement-learning agent for [Suika](https://suikagame.com/), the
fruit-merging puzzle game, trained against a browser-based Gymnasium
environment driven through Selenium.

The short version: a Q-network trained from scratch never beat random play. It
took a hand-written baseline to discover why — the network's head could not
represent a per-column value function at all — and behaviour cloning from that
baseline to get an agent that plays.

| policy | mean score | se | n |
|---|---|---|---|
| random | 1424 | 118 | 10 |
| DQN from scratch, original head | 1455 | 79 | 16 |
| **DQN cloned into a column-aligned head** | **2449** | 72 | 45 |
| hand-written policy (the teacher) | 2696 | 68 | 62 |

`runs/FINDINGS.md` has the full record, including the things that did not work
and what they cost.

![anchored versus plain TD fine-tuning](runs/finetune.png)

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

```bash
git clone https://github.com/edwhu/suika_rl.git
cd suika_rl && pip install -e . && cd ..
git -C suika_rl apply ../env-fixes.patch
export PYTHONPATH=$PWD/suika_rl
pip install -r requirements.txt
```

Requires Chrome and chromedriver — the environment drives a real browser.

```bash
./scripts/verify_patch.sh          # clones upstream, applies, compiles, imports
python -m pytest tests/ -q         # 96 tests, no browser needed
```

`verify_patch.sh` imports the patched module rather than stopping at
`git apply --check`, because `--check` only says the hunks fit. An earlier
version of the patch passed it and produced a file with two statements spliced
onto one line: BSD `diff` had emitted a `\ No newline at end of file` marker
after every hunk.

## Running it

The pipeline that produced the 2449 agent:

```bash
python scripts/heuristic.py --episodes 30 --policy layered   # measure the teacher
python scripts/collect_demos.py --episodes 90                # ~24k labelled boards
python scripts/pretrain.py                                   # clone it, ~5 min
python scripts/eval_policy.py --checkpoint runs/bc.h5 --episodes 25
```

Reinforcement learning on top, starting from those weights:

```bash
python train.py --resume --checkpoint runs/bc.h5 \
  --demos runs/demos.npz --n-step 20 --terminal-penalty 300 \
  --reward-scale 0.01 --lr 5e-5 --value-bias 9 --epsilon 0.05
```

`--demos` is not optional in practice: without it, TD fine-tuning forgets the
cloned policy and falls to ~1730. It is also, on the evidence, not an
improvement — see the honest limits below.

## How it works

**Observation.** The board rasterised into a 30x20x2 grid (fruit size, radius)
plus a 15-vector: how many of each size are live, what is in hand, what is
next, and how full the board is. All of it read straight out of the physics
engine rather than from pixels.

**Network.** A convolutional stack that never strides the board's width. Height
is reduced away, leaving one feature vector per board column; the global vector
is broadcast onto every column; the head emits values that stay tied to the
column they refer to. This is the part that mattered — the original head
flattened the board and rebuilt 40 values from a dense layer, and scores
random-level even when handed perfect demonstrations.

**Environment.** The upstream step took 640 ms, of which 500 was a fixed
`time.sleep(0.5)` after every drop. The patch replaces that with a real settle
predicate evaluated inside the page, raises the physics multiplier to 25x,
mutes audio, renders only on demand, and adds a feature observation mode. The
step is now **88 ms**, of which 81 is genuinely waiting for the board to stop
moving.

## Honest limits

- **The agent does not beat its teacher.** 2449 against 2696. Behaviour cloning
  fits a teacher's opinion, so it is capped by that teacher by construction.
- **Reinforcement learning adds nothing measurable** on top of cloning: +61 ±
  128 over 98 near-greedy episodes, flat across every window. The environment
  yields roughly 100k steps overnight, which is enough to imitate a policy and
  not enough to improve on one.
- **Double DQN, duelling, n-step returns, the terminal penalty and the reward
  scale were never ablated individually.** They are standard and cheap, and the
  bugs fixed in them were real correctness bugs, but this repo does not claim a
  measured benefit for any of them.
- **The teacher never simulates.** It estimates where a fruit lands and never
  checks. Real physics rollouts on the top few candidates is the next thing
  worth building, and `runs/FINDINGS.md` explains why that is the constraint.

## Layout

```
train.py              RL loop: replay, n-step targets, compiled gradient step
agent.py              replay buffer, n-step returns, TD targets, crash recovery
scripts/heuristic.py  the hand-written policy, and the random baseline
scripts/collect_demos.py, pretrain.py, eval_policy.py
scripts/check_train_step.py, check_seeding.py, verify_patch.sh
tests/                96 tests, pure numpy, no browser required
runs/FINDINGS.md      every measurement, including the failures
env-fixes.patch       against a clean upstream clone
```

MIT licensed.
