# What was measured

Every number here is a mean over whole episodes with its standard error, and
every comparison names its sample size. That discipline is the point: a
five-episode evaluation of this game has a standard error near 140 and cannot
detect a difference smaller than about 540 points, which is how two earlier
non-results in this project were briefly mistaken for signals.

## Where it ended up

| policy | mean | se | n | episode length |
|---|---|---|---|---|
| random | 1424 | 118 | 10 | 174 |
| DQN from scratch, original architecture | 1455 | 79 | 16 | 173 |
| cloned into the original (dense) head | 1530 | 68 | 25 | 186 |
| **cloned into the column head** | **2449** | 72 | 45 | 251 |
| cloned + anchored RL fine-tuning | 2420 | 45 | 98 | 254 |
| `greedy` hand-written policy | 2497 | 64 | 42 | 268 |
| `layered`, before the `order` term | 2582 | 82 | 30 | 251 |
| `layered` hand-written policy | 2835 | 88 | 20 | 283 |
| **`layered` + rollout 5** | **3100** | **48** | **100** | 307 |

The best policy is hand-written: 3100 over 100 episodes, with a 95% interval of
3005 to 3195. Against a target of 3000 that is z = 2.06, one-sided p = 0.020 —
it clears the mark by five points at the lower bound, which is a pass and not a
comfortable one. 61 of the 100 games scored above 3000.

That measurement is deliberately the most expensive in this document. At 20
episodes the standard error is about 95, and ten equally good variants tried at
that sample size give roughly a one-in-five chance that one of them reads 3000+
on luck alone. 100 episodes buys a standard error of 48, which is the precision
the question needs: a policy truly at 3000 clears it half the time at any
sample size, and it takes a true mean near 3070 to clear it reliably at this
one. It was run once, on seeds nothing else had used, and reported whatever it
said.

The learned agent went from indistinguishable-from-random to 2449, roughly 1.7x
the random floor. It remains below the hand-written policy it learned from, and
the gap widened rather than closed when the teacher improved — the cloned
network has not been retrained since the `order` term existed.

## The findings, in the order they were established

### 1. A baseline was worth more than another training run

The agent scored 1455 against a random 1424 — a difference of 31 against a
standard error of 142. On its own that says nothing about whether the agent is
bad or the game is unwinnable, and without knowing which there is no way to
choose what to do next.

A hand-written policy settled it. It reads fruit positions out of the physics
engine, estimates where each of the 40 candidate drops would land — the
smallest `y` among the floor and every fruit whose horizontal span it overlaps,
contact at `y = fy - sqrt((fr+r)^2 - dx^2)` — and scores the resulting contact
set. No learning, no rollouts.

At 2497 against 1424, arithmetic on the game state is worth 1.75x random. So
the game is learnable and the agent was failing at it, which ruled out the
alternative reading.

The more useful half: points per drop barely moved (8.2 to 9.7) while episode
length went from 174 to 268. **Almost all of the gain is surviving longer**,
not scoring better per placement.

### 2. The architecture was the blocker, not the learning signal

Three attempts to fix the reward signal all produced a policy at the random
floor. They found real bugs on the way — an n-step return discounted by `gamma`
once instead of `gamma**k`, truncation treated as termination, a terminal state
carrying no penalty when score deltas are never negative — and fixing them
changed nothing measurable. Then the same network was given *perfect demonstrations* and cloned:

| head | cloned score | imitation regret | parameters |
|---|---|---|---|
| dense (flatten, then MLP) | 1530 | 0.603 | 739,593 |
| **column** | **2449** | **0.135** | **171,971** |

The original head took the board's width from 20 columns down to 5 through two
stride-2 convolutions and then flattened, so board position survived only as an
index into a 2560-vector that a dense layer treats symmetrically. Worse, each
of the 40 outputs carried its own parameters: whatever the network learned
about a column at one position did not transfer to the next, so the same
pattern had to be relearned forty times.

Not a hard limit on what those weights *could* express — a 740k-parameter
network can approximate a great deal. A limit on what this data could teach
them. The training losses say which: 0.64 for the dense head against 0.246 for
the column head, so it failed to fit examples it had already seen. Underfitting,
not overfitting, which is why regularising it changed nothing.

It scores random-level *given perfect examples*, and that is what makes this the
explanation for all three earlier non-results rather than one more hypothesis.

The column head never strides width. It reduces height away, leaves one feature
vector per board column, broadcasts the global context (fruit in hand, fruit
next, how full the board is) onto every column, and emits values that stay tied
to the board position they refer to. Better and 4.3x smaller.

The tell had been in the data and was misread once: a *small* train/holdout gap
next to a *bad* holdout score is underfitting, not overfitting. The first
response was regularisation, which closed the gap and left the score alone.

### 3. Fine-tuning forgets, and the fix has to be level-invariant

Cloning is capped by its teacher, so exceeding it needs reinforcement learning.
Plain TD fine-tuning from cloned weights destroyed the policy — twice, at two
learning rates, from 2449 down to about 1730 by episode 50.

Cloning standardises its targets per board, so the value stream comes out near
zero while Q-learning wants it near the discounted return. The dominant error
is a uniform offset, and a dueling head can only raise every Q through the
value stream, while the cheapest way to shrink the loss is to flatten the
advantage — the part that picks the column. Initialising the value head near
the Bellman fixed point slowed the collapse and did not stop it.

Keeping the demonstrations in the loss fixes it (Hester et al., 2018). The term
is applied to the **mean-centred** output, or the two losses fight over the
value level. `scripts/check_train_step.py` verifies the property: shifting the
value bias by 50 moves the TD loss by 37.27 and the demonstration loss by
0.00025.

![anchored versus plain TD fine-tuning](finetune.png)

Both panels on the left tell the same story: plain TD loses ~800 points and ~60
drops of survival within ten episodes, while the anchored run holds.

With the anchor the policy holds at 2420 ± 45 over 98 near-greedy episodes,
flat across every window (+61 ± 128 against cloning alone).

**That null is weaker than it looks, and the reason is a scaling mistake in
this repo.** The joint loss is `td + demo_weight * bc`, where the TD term is
divided by the action count — deliberately, so the pure-TD path keeps the
learning rate it was tuned with — and the demonstration term is not. The two
also live in different units: Q is in scaled-reward units of order 5-9, the
demonstration target is per-board standardised. Logging both terms during a
short run gives `td 0.030-0.041` against `bc 0.411-0.576`, so `demo_weight=1.0`
is really about **14:1 in favour of the demonstrations** and TD contributed
roughly 7% of the loss.

So the honest reading is not "reinforcement learning had its say and added
nothing". It is that reinforcement learning barely got a vote. What *is*
established: TD at full strength, with no anchor, destroys the policy (2449
down to ~1730, twice, at two learning rates). The region between — TD weighted
comparably to the demonstrations — is untested, because the knob did not mean
what its name said. Both terms are now printed every episode.

A separate hypothesis was tested and refuted. Every other failure in this
project traces to maximising over noisy estimates, so the plateau was expected
to be the same thing: an argmax over 40 Q-values selecting whichever action the
network overrates. `scripts/check_value_calibration.py` measures it over 2,704
states — bias `+0.095` on values of order 5, correlation with the realised
return `0.892`, and an argmax gap of `1.749`, so **bias / gap = 0.05**. The
margin the greedy policy selects on is eighteen times its systematic error. The
Q function is well calibrated and that explanation does not apply here.

### 4. Rewarding structure helped; looking ahead did not

Two heuristic terms were added: `stack`, rewarding a small fruit landing on a
bigger one, and `trap`, penalising a smaller fruit left inside the band a drop
covers. Together (`layered`) they are worth **+199 over `greedy`**, and a
permutation test on the first 42-episode pair gave p = 0.011.

A next-fruit lookahead — build each successor board, score the best reply with
the fruit that is known to be coming — scored 2498 against 2497, at 40x the
computation per move. Not a bug. The successor board is an estimate, so the
lookahead compounds two steps of model error and then maximises over 40 noisy
replies, which selects whichever successor the model got most optimistically
wrong.

### 5. Things that did not work, and are no longer in the tree

| removed | measured |
|---|---|
| dense Q-head | 1530 vs the column head's 2449 |
| pixel observations | the 353-episode run was void — the observation showed the *next* fruit, not the one being dropped |
| next-fruit lookahead | 2498 vs 2497 |
| afterstate value network | 1794, despite fitting well (holdout r = 0.837, MAE 115 against label sd 276) |
| CEM weight fitting | 2647, 2427, 2734 across three designs, against a 2696 baseline |
| surface roughness + pile height | +38 ± 116 |
| `neighbour` term | the condition could never fire; every run before its repair had it inert |
| prioritised replay, head resets | never exercised by any run that produced a result |

The afterstate network is the most interesting failure. It fits the true return
well, and the policy is still bad, because it was trained on the one afterstate
actually taken per step and then asked to rank forty — thirty-nine of them off
the distribution it saw. `argmax` does not tolerate that error, it seeks it,
selecting whichever board the network is most over-optimistic about.

That is the same shape as the lookahead failure and the first CEM run
(selecting the best of 12 candidates scored on 3 noisy episodes inflated the
result by +463). **Maximising over many noisy estimates selects for the largest
error, not the largest value** — it cost three separate approaches here.

Fixing the statistics did not rescue CEM either. A paired design that scored
every candidate on the same fruit sequences overfitted those particular games
(≈3150 on its five training seeds, 2427 held out); redrawing seeds each
iteration finally optimised the right thing and returned +38 ± 116. Three
optimiser designs all landing at or below the hand-guessed weights is the
evidence that the policy class, not the optimiser, is the constraint.

### 6. A finer action space does not help, and seeding pays less than expected

The drop position is discretised into 40 bins. Raising that to 200 was measured
against 40 on identical fruit sequences, with the order of the two arms
alternated across rounds:

| | mean | se | n |
|---|---|---|---|
| 200 actions | 2551 | 143 | 12 |
| 40 actions | 2660 | 162 | 12 |

Paired difference **-109 ± 261**, and 200 won 5 of the 12 games. No improvement,
at five times the candidate scoring per move. Forty bins already place a fruit
every 16.4 px, well inside the smallest fruit's 24 px radius, and the landing
estimate ignores roll and secondary settling — error larger than the 3.2 px that
200 bins buys. Resolution finer than the physics is noise.

A first attempt ran the two arms sequentially and reported 200 ahead by
+306. That was an ordering artifact: its 40-action arm came in 2.9σ below the
same policy's 62-episode value, and re-running with the order alternated
reversed the sign. Sequential arms confound the comparison with anything that
drifts over a session.

**Seeding helped less than claimed.** `reset(seed=)` fixes the fruit sequence,
and the expectation was that scoring two policies on the same sequences would
cancel the luck. It does not, for policies that differ much: they choose
different columns from the first drop, so the same fruits arrive into
completely different boards. Measured here, the paired difference had a
standard deviation of 905 against 748 for an unpaired one — pairing made it
*worse*. Common random numbers pay when the compared policies stay close to
each other, which two different action-space sizes do not.

Seeding is still worth having — it makes a single policy's episode
reproducible, which is what `scripts/check_seeding.py` verifies — but it is not
the variance cure it was introduced as.

### 7. Simulating the drop, instead of estimating it, is the one thing that worked

Everything above is built on `landing_y`, which works out where a fruit stops
in closed form: straight down, halting at the first thing it overlaps. That
ignores roll, ignores the support being pushed aside, and ignores merges
resolving mid-fall. Every failure in this document inherited it.

`Game.rollout` replaces it. A scratch Matter.js world is built from the live
board, the candidate is dropped into it, and it is stepped until nothing is
moving — the real merge rule included, cascades and the size-wrap and the lose
condition. A second engine rather than a snapshot of this one, because
Matter.js has no snapshot and a merge creates and destroys bodies, so an undo
would have to track composition changes; rebuilding twenty circles is cheaper
than getting that right. All candidates are scored in one call, since forty
WebDriver round trips would cost more than the physics.

**It agrees with the game.** `scripts/check_rollout.py` simulates a drop and
then plays it, thirty times: the score is predicted exactly 90% of the time,
whether a merge happened 96.7%, and the median fruit ends up **1.7 px** from
where it was predicted, against a smallest-fruit radius of 24 px.

The policy ranks all forty columns with the cheap estimate, simulates the top
five, and chooses on the settled board — points actually gained, how high the
pile ended up, same-size pairs left within reach, small fruit left buried.
Nothing estimated. The shortlist is a choice rather than a necessity: the
estimate is perfectly good at spotting bad columns, and it is the ordering
among the *good* ones that roll and displacement decide.

Six rounds of five episodes each, the order of the two arms alternating every
round:

| | mean | se | sd | episode length |
|---|---|---|---|---|
| **rollout, top 5** | **2826** | 80 | 436 | **286** |
| estimate only | 2582 | 82 | 448 | 266 |

`+244 ± 114`, 2.1σ, permutation `p = 0.037`. The rollout won **6 of 6 rounds**,
a sign test at `p = 0.031` — two tests agreeing, one on size and one on
consistency. Episode length rose 266 to 286, which is the survival channel the
baseline identified at the start as carrying the score, so the gain appears
where the theory says it should.

Eleven of thirty episodes reached 3000 or above, against seven of thirty.

**The cost was misjudged for a long time.** This document previously argued
rollouts were impractical at "~80 ms a candidate, about 15 minutes an episode".
Measured, a candidate costs **13.7 ms** — all forty would be 549 ms a move,
roughly three minutes an episode. The 80 ms figure was wall-clock settling in
the live game, which steps physics on `requestAnimationFrame` and is therefore
frame-paced; a scratch engine has no renderer and no frame pacing.

### 8. The TD reward never mentioned the board, only the score

The reward the DQN learns from is `Game.score` minus what it was last step,
optionally with a penalty at death (`agent.terminal_reward`). That is the whole
of it. A score delta is never negative and says nothing about the state the
board is left in, so sealing a size-0 fruit under a size-8 one cost the agent
exactly nothing at the moment it did it — and by the time it cost anything, an
n-step return no longer reached back far enough to connect the two.

The hand-written policies had always known about this: `layered` carries a
`trap` term and `BOARD_WEIGHTS` a `buried` one. It reached the network only
through behaviour cloning, as an imitation of a demonstrator that happened to
avoid traps, never as a signal TD could act on.

`heuristic.trapped_small` states it in the agent's own terms: of the forty
columns it may drop into, is there one where an identical fruit would come to
rest touching this one? If not, that fruit cannot be merged by any move
available, whatever the network believes. Only sizes 0–4 count — the dropper
never hands out anything larger (`Math.floor(rand() * 5)`), so a lone size-7 is
waiting, not trapped.

It is a closed-form estimate, so it was measured against the engine: 86
droppable fruit across six real boards, each tested against all forty drops
actually simulated (`scripts/check_trapped.py`).

| contact slack | called trapped | engine merged it | real trap missed | agrees |
|---|---|---|---|---|
| 6 | 56 | 11 (20%) | 0 (0%) | 87.2% |
| 32 | 53 | 8 (15%) | 0 (0%) | 90.7% |
| 48 | 49 | 5 (10%) | 1 (3%) | 93.0% |
| 96 | 39 | 3 (8%) | 9 (19%) | 86.0% |

Slack stands in for roll, which the closed-form fall ignores. 32 is the default
because it is the widest setting that never calls a genuinely trapped fruit
free: for a term that only ever subtracts, losing the signal is worse than
charging a penalty on a position that turned out fine.

The term enters as potential-based shaping — `gamma * PHI(s') - PHI(s)` with
`PHI(s) = -w * trapped(s)` — which telescopes over an episode and so provably
leaves the optimal policy where it was. It charges the drop that creates a trap
and refunds the one that clears it. The literal per-step reading is available
as `--trap-shaping flat`, and is not the default because it also taxes
survival, which the numbers above say is where the points are.

**Not yet measured against score.** The weight is calibrated, not fitted: at
`--trap-penalty 1.0` one trapped fruit costs one game point, which is the
exchange rate `BOARD_WEIGHTS` already used (`gained=12`, `buried=-12`). Whether
it moves the final number needs a training run that has not been done.

### 9. The weights were never the problem; one missing feature was

CEM was re-run on the six placement weights and found nothing, as it had three
times before. This time the reason was measured rather than inferred.

`scripts/check_fitness_noise.py` scores the same eight candidates twice, on
disjoint seed sets, and correlates the two rankings. Reasoning from a single
generation's spread cannot settle this, because common random numbers lower the
null as well as the signal; test-retest does not care.

    per-episode sd                          458
    sd of a 4-episode mean (noise)          229
    sd between candidate means              222
    implied true sd between candidates        0

    Spearman A vs B   -0.02      Pearson A vs B   -0.10

The spread between candidates is entirely accounted for by noise. The ranking
CEM selects on has no relationship to the ranking the same candidates get on
fresh games, so the elite of each generation were the lucky ones.

The decomposition says something more useful than "too noisy": **more episodes
would not have helped.** If real differences were merely hidden, the observed
spread would exceed the noise floor; it does not. Perturbing these six weights
by ±35% produces policies that are equally good. CEM was searching a plateau.
(A true sd of exactly zero is a clipped estimate — the subtraction went
negative — so the defensible claim is that any difference is under ~100 points,
against 400-point swings from fruit luck.)

That turns the question from weights to features. `low` — a linear reward for
landing low — was removed, and two terms added: `danger`, flat until the pile
enters a band below the lose line and then rising as a cube, because a board is
fine until it is nearly full and then fatal; and `order`, rewarding the *change*
a drop makes to the board's size/x correlation, the structure of big fruit
gathered at one end with sizes descending away from it.

Measured with `scripts/ab_weights.py` — both arms on the same seeds, arms
alternating seed by seed, reporting the mean of per-seed differences:

| comparison | paired difference | won | p |
|---|---|---|---|
| `danger`+`order` vs neither | **+454 ± 171** | 12/16 | 0.008 |
| `order` on top of `danger` | **+426 ± 120** | 13/16 | 0.0004 |
| `danger` on top of `order` | +330 ± 167 | 11/16 | 0.05 |

`order` is the effect, and it is the largest single gain measured here since the
column-aligned head. It survives both artifacts this document has recorded
before: each result holds in both halves of its run and under both arm
orderings, and the sign tests agree.

`danger` sits exactly on the line: +330 against a standard error of 167, which
is 1.98 of them. An earlier reading of this table subtracted across different
seed sets and made it look like nothing (+28); that was wrong, and the direct
comparison is what this row reports. Kept, unresolved — resolving it would take
the ~37 seeds the table below prices at 1.8 hours, for a term that costs
nothing to leave in.

Re-measured at the settings the rest of this document uses, over 20 seeds:
**2835 ± 88**, against **2582 ± 82** for the same policy before `order`. That
matches `hand-written + rollout` (2826 ± 80) using one closed-form pass per drop
instead of five physics simulations.

70% of those episodes reached the 300-drop cap, which looked like a truncated
measurement. It was not: re-run with the cap at 800 the same policy scores
**2775 ± 102**, indistinguishable, because games rarely want to run much past
300 (mean 282 drops, longest 355, only 6 of 20 over 300).

### 10. Replaying a seed does not reproduce the game

Found while trying to pair the capped and uncapped runs above. Same policy, same
seed, only the step cap different — so any episode that never reached the cap
should have been an identical run. None were:

| seed | capped at 300 | capped at 800 |
|---|---|---|
| 80002 | 2225 / 243 drops | 2066 / 232 |
| 80006 | 2056 / 214 | **3597 / 355** |
| 80009 | 2315 / 253 | 2916 / 284 |
| 80012 | 2504 / 258 | 2536 / 262 |

Zero of four matched, one by 1541 points.

The cause is the settle wait. `read_state` waits up to 1000 **milliseconds of
wall clock** for the board to stop moving, and about 3% of drops never settle
inside that — on those the policy acts on a board still in motion, which board
it sees depends on how busy the machine is, and the game diverges from there.
The seed fixes the fruit sequence and nothing else.

What this does *not* affect: every paired difference above. Those are computed
from observed per-seed differences with the standard error estimated from the
same differences, so this nondeterminism is already inside the ±.

What it does affect: the reason pairing helps so little. It had been attributed
to two *different* policies diverging; in fact **a policy diverges from itself**.
Measured correlation between arms on the same seed: **+0.32**, cutting the
difference sd from 710 to only 588.

The fix is to make the wait deterministic — settle on a physics tick count
rather than a wall-clock budget — which would make seeds reproduce and sharpen
every paired comparison in this document. Not yet done.

| effect to resolve | paired seeds | episodes | wall clock |
|---|---|---|---|
| 400 points | 9 | 18 | ~25 min |
| 200 points | 37 | 74 | ~1.8 h |
| 100 points | 147 | 294 | ~7 h |

Anything worth under ~200 points is effectively unmeasurable in an afternoon on
this machine. That, not the optimiser and not the policy class, is what limits
the rate this project can learn anything — and it is why the right move is to
chase changes large enough to see rather than to tune.

## What would plausibly move it

The heuristic never simulates — it estimates where a fruit lands and never
checks. Every failure above traces back to that estimate: the lookahead
compounded its error, the value network was trained on estimated afterstates,
and CEM could only re-weight estimates.

That was done — see above. It is the only change in this document that moved
the ceiling. What follows from it: the agent has tracked its teacher closely
throughout, so the next step is a fresh demonstration set from the rollout
teacher and a re-clone, which has never been run.

## Reproducing

    python scripts/heuristic.py --episodes 30 --policy layered   # the ceiling
    python scripts/heuristic.py --episodes 30 --random           # the floor
    python scripts/collect_demos.py --episodes 90                # ~24k boards
    python scripts/pretrain.py                                   # ~5 min
    python scripts/eval_policy.py --checkpoint runs/bc.h5 --episodes 25
    python scripts/check_trapped.py --boards 6                   # the table in 8
    python scripts/check_fitness_noise.py                        # why CEM finds nothing
    python scripts/ab_weights.py --episodes 16 --b order=0       # what `order` is worth
