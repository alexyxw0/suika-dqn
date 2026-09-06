"""The parts of the agent that are just arithmetic.

Split out from train.py deliberately: none of this imports TensorFlow, needs a
GPU, or drives a browser, so it can be tested in a plain Python process. What
is left in train.py is the network definition and the episode loop — the parts
that genuinely need the environment running.

Getting the temporal-difference target wrong is the classic silent DQN bug: the
agent still trains, the loss still falls, and it just never learns anything.
`td_targets` exists as a named function so it can be asserted on.
"""

from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass

import numpy as np


def action_bins(num_actions: int) -> np.ndarray:
    """Discretise the environment's continuous drop position into `num_actions`
    evenly spaced points across [0, 1], endpoints included."""
    if num_actions < 2:
        raise ValueError(f"need at least 2 actions, got {num_actions}")
    return np.linspace(0.0, 1.0, num_actions)


def to_continuous(bins: np.ndarray, index: int) -> np.ndarray:
    """The environment wants a 1-element float32 array, not a bare index."""
    return np.array([bins[index]], dtype=np.float32)


@dataclass
class EpsilonSchedule:
    """Exponential decay per episode, floored at `minimum`."""

    start: float = 1.0
    minimum: float = 0.05
    decay: float = 0.995

    def at(self, episode: int) -> float:
        return max(self.minimum, self.start * self.decay ** episode)

    def __iter__(self):
        episode = 0
        while True:
            yield self.at(episode)
            episode += 1


class ReplayBuffer:
    """Uniform experience replay over a fixed-size window. No prioritisation.

    Frames are stored in whatever dtype they arrive in and converted to float32
    only when a batch is sampled. That is not a detail: a 128x128x3 frame is
    49 KB as uint8 and 197 KB as float32, and with a state and a next_state per
    transition a 10,000-entry buffer is either 1.0 GB or 3.9 GB. On an 8 GB
    machine also running TensorFlow and a browser, the difference is whether
    the run survives.
    """

    def __init__(self, capacity: int = 10_000):
        if capacity < 1:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, steps=1) -> None:
        """Store a transition. Pass frames as uint8; see the class docstring.

        `steps` is how many environment steps the reward spans, so an n-step
        return can be discounted by gamma**steps rather than gamma.
        """
        self.buffer.append((state, action, reward, next_state, done, steps))

    def nbytes(self) -> int:
        """Bytes held in stored frames, for logging what the run is costing."""
        if not self.buffer:
            return 0
        state, _, _, next_state, _, _ = self.buffer[0]

        def size_of(s):
            # A feature state is a (grid, vector) tuple, not one array.
            if isinstance(s, tuple):
                return sum(np.asarray(part).nbytes for part in s)
            return np.asarray(s).nbytes

        return len(self.buffer) * (size_of(state) + size_of(next_state))

    def sample(self, batch_size: int, stack: bool = True):
        """Draw a batch.

        `stack=False` returns the states as the tuples they were stored as,
        which is what a multi-input model needs — a (grid, vector) pair cannot
        be turned into one array. The caller stacks per input instead.
        """
        if batch_size > len(self.buffer):
            raise ValueError(
                f"asked for {batch_size} samples from a buffer holding {len(self.buffer)}")
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, steps = zip(*batch)
        return (
            np.array(states, dtype=np.float32) if stack else list(states),
            np.array(actions, dtype=np.int32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32) if stack else list(next_states),
            np.array(dones, dtype=np.float32),
            np.array(steps, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class NStepBuffer:
    """Accumulate transitions and emit n-step returns.

    In Suika a merge often resolves several drops after the placement that made
    it possible, and one-step TD moves credit backwards one step per update.
    An n-step return carries it n steps at once, which is the cheapest way to
    speed up credit assignment in an environment this slow.

    Emits (state, action, n-step reward, state n ahead, done, steps) once n
    transitions have accumulated, and drains what is left at the end of an
    episode — otherwise the last n-1 transitions of every episode, which are
    the ones nearest the terminal reward, would be silently discarded.
    """

    def __init__(self, n: int, gamma: float):
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        self.n = n
        self.gamma = gamma
        self.items: deque = deque(maxlen=n)

    def push(self, state, action, reward, next_state, done):
        """Add a transition; returns a ready n-step transition or None."""
        self.items.append((state, action, reward, next_state, done))
        if done:
            return None                      # the caller drains via flush()
        if len(self.items) < self.n:
            return None
        return self._collapse(len(self.items))

    def flush(self):
        """Every remaining partial return, longest first. Call at episode end."""
        out = []
        while self.items:
            out.append(self._collapse(len(self.items)))
            self.items.popleft()
        return out

    def _collapse(self, steps):
        """Fold `steps` transitions into one n-step transition."""
        reward = 0.0
        for i in range(steps):
            reward += (self.gamma ** i) * self.items[i][2]
            if self.items[i][4]:             # terminal: the return stops here
                steps = i + 1
                break
        state, action = self.items[0][0], self.items[0][1]
        _, _, _, next_state, done = self.items[steps - 1]
        return (state, action, reward, next_state, done, steps)


def td_targets(rewards, dones, next_qs, gamma: float, steps=None):
    """One-step Q-learning targets: r + gamma * max_a' Q(s', a'), and just r
    at a terminal state.

    The `(1 - done)` factor is the whole point. Bootstrapping through a terminal
    state teaches the agent that the value of dying is whatever the network
    happens to predict for the frame after the game ended.

    `steps` is the per-sample n-step length. Without it an n-step return gets
    discounted by gamma once instead of gamma**n, which overvalues the bootstrap
    by gamma**(1-n): at gamma 0.99 that is 2% at n=3 and 21% at n=20, and it
    grows sharply as gamma falls (7.4x at n=20 with gamma 0.9). A systematic
    bias on every sample rather than a catastrophe, and wrong either way.
    `double_td_targets` always took this argument; this one did not, so the bug
    only showed up under `--no-double`.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    best_next = np.max(np.asarray(next_qs, dtype=np.float32), axis=1)
    discount = (gamma ** np.asarray(steps, dtype=np.float32)
                if steps is not None else gamma)
    return rewards + (1.0 - dones) * discount * best_next


def double_td_targets(rewards, dones, next_qs_target, next_qs_online, gamma,
                      steps=None):
    """Double-DQN targets: the online network chooses, the target evaluates.

    Plain DQN takes max over the target network's own estimates, so any action
    it happens to overestimate is both selected *and* used as the target — the
    error compounds instead of averaging out. Choosing with one network and
    valuing with the other breaks that coupling (van Hasselt et al., 2016).

    `steps` is the per-sample n-step length, so gamma is raised to the actual
    number of steps each return covers rather than assumed uniform.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    online = np.asarray(next_qs_online, dtype=np.float32)
    target = np.asarray(next_qs_target, dtype=np.float32)

    chosen = np.argmax(online, axis=1)
    valued = target[np.arange(len(chosen)), chosen]

    discount = (gamma ** np.asarray(steps, dtype=np.float32)
                if steps is not None else gamma)
    return rewards + (1.0 - dones) * discount * valued


def apply_targets(current_qs, actions, targets):
    """Write each target into the row's chosen-action column, leaving the rest
    of the row equal to the prediction so it contributes no gradient."""
    updated = np.array(current_qs, dtype=np.float32, copy=True)
    updated[np.arange(len(actions)), np.asarray(actions, dtype=np.int32)] = targets
    return updated


def mirror_boards(grid, scores, actions, n_actions):
    """The same boards seen from the other side.

    Suika's board is left-right symmetric, so mirroring is an exact relabelling
    rather than an approximation: flip the grid's width axis, reverse the
    per-column score vector, and map action a to (n-1-a). Free data, and a
    strong regulariser — the first imitation run hit 94% train accuracy against
    15% on holdout.

    The grid is (N, H, W, C), so width is axis 2.
    """
    return (grid[:, :, ::-1, :],
            scores[:, ::-1],
            n_actions - 1 - np.asarray(actions))


def standardise_rows(scores, eps: float = 1e-6):
    """Zero mean, unit spread per row.

    The heuristic's score scale is arbitrary and drifts with what is on the
    board, so absolute values would teach the head a bias it ought to learn
    from returns instead. Only the shape across columns is worth imitating —
    and with a dueling head the value stream is the part imitation cannot
    reach anyway, since a softmax cancels a per-board constant.
    """
    scores = np.asarray(scores, dtype=np.float32)
    mu = scores.mean(axis=1, keepdims=True)
    sd = scores.std(axis=1, keepdims=True)
    return (scores - mu) / np.maximum(sd, eps)


def terminal_reward(reward, done, truncated, penalty: float = 0.0,
                    scale: float = 1.0):
    """Shape the reward for the end of an episode, and keep truncation out of it.

    Two different things are easy to conflate. Losing the game is terminal: the
    state that follows is worth nothing, so the return stops there. Hitting the
    step cap is not terminal — the board was still playable, and treating it as
    an ending teaches the agent that the world stops at step N and is worth
    zero, which is false. Only the first is a loss, and only the first is
    penalised.

    The penalty exists because a score delta is never negative. Without it the
    only cost of dying is the future reward forgone, which an n-step return
    reaches back only n steps to see — about three drops' worth of score at
    n=3, against episodes that run past 250. The measured baseline says
    survival is where the points are (`runs/FINDINGS.md`: 63% of a hand-written
    policy's advantage is episode length, not points per drop), so the cost of
    dying has to be visible.

    `scale` multiplies the result. It matters once n is large: a 20-step return
    sums twenty score deltas, so targets grow from tens to hundreds, and an MSE
    loss on targets that size — under a schedule that ramps the learning rate
    up to 1e-3 — is a good way to diverge. The penalty is given in raw game
    points and scaled with everything else so the two stay comparable.

    Returns `(shaped_reward, is_terminal)`.
    """
    return (reward - (penalty if done else 0.0)) * scale, bool(done)


def gae_advantages(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """Generalised advantage estimation (Schulman et al., 2016).

    The advantage of an action is how much better the return was than the value
    function expected. Estimating it from the full return is unbiased and very
    noisy; estimating it one step at a time is the reverse. GAE interpolates
    with `lam`, and the exponentially-weighted sum is what makes a policy
    gradient usable at this sample budget.

    `dones` marks a *terminal* state, not the end of the rollout. The
    distinction matters: the value beyond a lost game is zero and must not be
    bootstrapped, whereas a rollout that simply ran out of steps has to
    bootstrap from `last_value` or the final transitions are told the game
    ended when it did not.
    """
    n = len(rewards)
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    adv = np.zeros(n, dtype=np.float32)
    running = 0.0
    for t in range(n - 1, -1, -1):
        next_value = last_value if t == n - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        running = delta + gamma * lam * nonterminal * running
        adv[t] = running
    return adv


def explained_variance(predicted, actual):
    """How much of the return's variance the value function accounts for.

    1.0 is perfect, 0.0 is no better than predicting the mean, and negative
    means worse than that. The number to watch during PPO: if it sits near zero
    the advantages are noise and the policy gradient is being driven by
    nothing.
    """
    predicted = np.asarray(predicted, dtype=np.float64).ravel()
    actual = np.asarray(actual, dtype=np.float64).ravel()
    var = actual.var()
    if var < 1e-12:
        return 0.0
    return float(1.0 - (actual - predicted).var() / var)


def normalise(x, eps=1e-8):
    """Zero mean, unit variance. Advantages are normalised per batch so the
    step size does not depend on how large the rewards happen to be."""
    x = np.asarray(x, dtype=np.float32)
    return (x - x.mean()) / (x.std() + eps)


def browser_failures():
    """Exception types meaning the browser is gone, not that the code is wrong.

    Two different failures both leave the environment unusable, and they arrive
    as unrelated classes. Chrome drops the tab, which selenium reports as
    `WebDriverException`. Or the page stops answering altogether and selenium's
    HTTP client gives up, which surfaces from urllib3 as a `ReadTimeoutError` —
    no relation to `WebDriverException`, so an `except WebDriverException` sails
    straight past it. A two-hour collection run died that way with recovery code
    sitting right there.

    Deliberately not `Exception`. A typo or a shadowed variable inside the
    episode loop would then be retried three times and reported as a browser
    problem, which is how a real bug hides for an afternoon.

    Imported lazily so this module stays usable — and testable — without
    selenium installed.
    """
    types = []
    try:
        from selenium.common.exceptions import WebDriverException
        types.append(WebDriverException)
    except ImportError:
        pass
    try:
        from urllib3.exceptions import HTTPError
        types.append(HTTPError)
    except ImportError:
        pass
    return tuple(types) or (OSError,)


def restart_env(env, make_env, pause: float = 2.0):
    """Discard a dead environment and build a fresh one.

    Extracted from the training loop so it can be tested. The first version of
    this lived inline, referenced `time` without importing it, and therefore
    crashed *inside the crash handler* — which only showed up the first time a
    browser actually died, an hour into a run. Recovery code that has never
    been executed is not recovery code.
    """
    try:
        env.close()
    except Exception:      # noqa: BLE001 — it is already broken; nothing to salvage
        pass
    if pause:
        time.sleep(pause)
    return make_env()
