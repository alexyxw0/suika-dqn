"""DQN agent for the Suika browser environment.

Trains from raw frames. The environment is edwhu/suika_rl, patched by
env-fixes.patch — see README.md for setup.

The network, hyper-parameters and update rule are unchanged from the run the
README reports. What changed in cleanup: the checkpoint path and the
hyper-parameters are arguments rather than edited constants, the run is
seedable, and the two Q-network forward passes per gradient step are batched
into one call instead of two.

Bootstrap targets come from a frozen target network, synced from the online
network every `--target-sync` gradient steps. Without one the network chases a
value estimate it is itself moving, which is the instability DQN's target
network exists to fix.
"""

from __future__ import annotations

import argparse
import atexit
import random
import signal
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers.schedules import PolynomialDecay

from agent import (EpsilonSchedule, NStepBuffer, ReplayBuffer, action_bins,
                   browser_failures, double_td_targets, restart_env,
                   standardise_rows, td_targets, terminal_reward,
                   to_continuous)
from suika_env.suika_browser_env import SuikaBrowserEnv

# Both a dropped tab and a hung page have to be caught here, and they
# arrive as unrelated exception classes. See agent.browser_failures.
BROWSER_DEAD = browser_failures()


def build_column_model(grid_shape=(30, 20, 2), vector_len=15,
                       num_actions=40, dueling=True):
    """Q-network that keeps the board's columns aligned with the actions.

    The action is "which column to drop into", so the output is indexed by
    board position. The head this replaced flattened the board and rebuilt the
    forty values from a dense layer, which threw that indexing away: two
    stride-2 convolutions took the width from 20 down to 5. Cloned from the
    same demonstrations it scored 1530 against this one's 2359 — random-level,
    given perfect examples. See runs/FINDINGS.md.

    Here width is never strided. Height is reduced away instead, and the last
    convolution collapses what is left of it, leaving one feature vector per
    board column. The global vector — fruit in hand, fruit next, how full the
    board is — is broadcast onto every column, since it changes what a column
    is worth without belonging to any of them. A 1-wide convolution then acts
    as a per-column network, a 3-wide one lets neighbouring columns talk, and
    the head emits `num_actions / width` values per column, so every output
    stays tied to the piece of board it refers to.

    The value stream keeps a pooled view of the whole board, which is what a
    board being good or bad actually depends on.
    """
    height, width, channels = grid_shape
    if num_actions % width:
        raise ValueError(
            f"num_actions ({num_actions}) must be a multiple of the grid "
            f"width ({width}) so each column owns a whole number of actions")
    per_column = num_actions // width

    grid_in = layers.Input(shape=grid_shape, name="grid")
    vector_in = layers.Input(shape=(vector_len,), name="vector")

    g = layers.Conv2D(32, 3, padding="same", activation="relu")(grid_in)
    g = layers.Conv2D(64, 3, strides=(2, 1), padding="same",
                      activation="relu")(g)
    g = layers.Conv2D(64, 3, strides=(2, 1), padding="same",
                      activation="relu")(g)
    remaining = -(-height // 4)                 # two stride-2 height steps
    g = layers.Conv2D(96, (remaining, 1), activation="relu")(g)
    columns = layers.Reshape((width, 96))(g)

    # Same global context on every column.
    spread = layers.Dense(32, activation="relu")(vector_in)
    spread = layers.RepeatVector(width)(spread)
    x = layers.Concatenate(axis=-1)([columns, spread])

    x = layers.Conv1D(128, 1, activation="relu")(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)

    advantage = layers.Conv1D(per_column, 1, name="per_column")(x)
    advantage = layers.Reshape((num_actions,))(advantage)

    if dueling:
        pooled = layers.GlobalAveragePooling1D()(x)
        value = layers.Dense(1, name="value")(pooled)
        centred = layers.Lambda(
            lambda a: a - tf.reduce_mean(a, axis=1, keepdims=True))(advantage)
        outputs = layers.Add()([value, centred])
    else:
        outputs = advantage

    model = models.Model(inputs=[grid_in, vector_in], outputs=outputs)
    lr_schedule = PolynomialDecay(
        initial_learning_rate=1e-4, decay_steps=100_000,
        end_learning_rate=1e-3, power=1.0, cycle=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                  loss="mse")
    return model


def build_afterstate_model(grid_shape=(30, 20, 2), vector_len=15, lr=3e-4,
                          seed=None):
    """How good is this board? One number.

    The shape every strong Tetris agent uses, and it failed here once already —
    1794 against the cloned network's 2449. Two things were wrong with that
    attempt and only one of them was the architecture.

    The boards it was trained on were geometric constructions, not physics:
    a landing point computed as "straight down until first overlap", a merge
    resolved by averaging two positions, and no gravity afterwards. Measured
    against the real game that construction got the *merge outcome itself*
    wrong 27% of the time. `Game.rollout` gets it wrong 3%.

    The other problem stands regardless and is handled at inference rather than
    here: a value function trained on the board actually reached, then asked to
    rank forty, is extrapolating on thirty-nine of them, and argmax selects
    whichever it overrates. See `scripts/eval_value.py`.

    Flatten rather than pool: how high the pile is and where the gaps sit is
    what decides a board, and average pooling discards the vertical structure.
    """
    init = (tf.keras.initializers.GlorotUniform(seed=seed)
            if seed is not None else "glorot_uniform")
    grid_in = layers.Input(shape=grid_shape, name="grid")
    vector_in = layers.Input(shape=(vector_len,), name="vector")

    g = layers.Conv2D(32, 3, padding="same", activation="relu",
                      kernel_initializer=init)(grid_in)
    g = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu",
                      kernel_initializer=init)(g)
    g = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu",
                      kernel_initializer=init)(g)
    g = layers.Flatten()(g)

    v = layers.Dense(64, activation="relu", kernel_initializer=init)(vector_in)
    x = layers.Concatenate()([g, v])
    x = layers.Dense(128, activation="relu", kernel_initializer=init)(x)
    out = layers.Dense(1, name="value", kernel_initializer=init)(x)

    model = models.Model(inputs=[grid_in, vector_in], outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
    return model


def make_joint_train_step(model, num_actions: int, demo_weight: float):
    """A gradient step that fits returns *and* keeps imitating.

    Fine-tuning from cloned weights with a plain TD loss destroys the policy.
    Measured twice, from 2359 down to about 1730 by episode 50, once with the
    value stream left at zero and once with it initialised to the right scale —
    the second only delayed it. With roughly ten thousand environment steps of
    reinforcement data, a bootstrapped target is far noisier than the
    supervised signal that produced the policy, so gradient descent trades a
    good policy for a slightly better Bellman fit. Keeping demonstrations in
    the loss is the standard answer (Hester et al., 2018).

    The demonstration term is applied to the *mean-centred* output. The two
    losses would otherwise fight: cloning standardises its targets per board,
    so a raw match pulls the value stream back to zero, while the TD term wants
    it near the discounted return. Centring makes the demonstrations constrain
    only the relative worth of actions — which is what they actually know — and
    leaves the level to Q-learning, which is the one thing a demonstration
    cannot teach.
    """
    @tf.function
    def step(state_batch, actions, targets, demo_batch, demo_targets):
        mask = tf.one_hot(actions, num_actions, dtype=tf.float32)
        with tf.GradientTape() as tape:
            qs = model(state_batch, training=True)
            taken = tf.reduce_sum(qs * mask, axis=1)
            td = tf.reduce_mean(tf.square(targets - taken)) / num_actions

            demo_qs = model(demo_batch, training=True)
            centred = demo_qs - tf.reduce_mean(demo_qs, axis=1, keepdims=True)
            bc = tf.reduce_mean(tf.square(demo_targets - centred))

            loss = td + demo_weight * bc
        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return td, bc

    return step


def make_train_step(model, num_actions: int):
    """One compiled gradient step, in place of `model.fit` per update.

    `model.fit` re-enters the whole Keras training stack on every call — a
    dataset, callbacks and metrics, for a single batch of 32 — and with three
    `predict` calls beside it the update cost dominated the environment step
    that the patched browser had already been cut to 88 ms. This traces once.

    The loss is divided by the action count to match the previous formulation,
    where the target row equalled the prediction in every column but the chosen
    one and Keras averaged squared error across all of them. Holding the scale
    fixed keeps the learning-rate schedule meaning what it meant before, so
    this stays a speed change rather than a hyperparameter change.
    """
    @tf.function
    def step(state_batch, actions, targets):
        mask = tf.one_hot(actions, num_actions, dtype=tf.float32)
        with tf.GradientTape() as tape:
            qs = model(state_batch, training=True)
            taken = tf.reduce_sum(qs * mask, axis=1)
            loss = tf.reduce_mean(tf.square(targets - taken)) / num_actions
        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    return step


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--actions", type=int, default=40,
                   help="bins the continuous drop position is split into. At "
                        "10 the spacing (71px) exceeds the smallest fruit "
                        "(48px), so some placements cannot be expressed")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--buffer", type=int, default=10_000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--epsilon", type=float, default=1.0)
    p.add_argument("--epsilon-min", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=float, default=0.995)
    p.add_argument("--checkpoint", type=Path, default=Path("suika_dqn_model.h5"),
                   help="where to save after each episode")
    p.add_argument("--resume", action="store_true",
                   help="load --checkpoint instead of starting from scratch")
    p.add_argument("--headless", action="store_true",
                   help="do not render the browser window")
    p.add_argument("--target-sync", type=int, default=500,
                   help="gradient steps between copying online weights to the "
                        "target network; 0 disables the target network")
    p.add_argument("--warmup", type=int, default=500,
                   help="transitions to collect before the first gradient step")
    p.add_argument("--train-every", type=int, default=4,
                   help="environment steps between gradient steps; the replay "
                        "ratio is 1/this. Lower means more updates per sample")
    p.add_argument("--updates-per-step", type=int, default=1,
                   help="gradient steps taken each time training fires. With "
                        "--train-every 1 this IS the replay ratio, and above "
                        "~2 it needs --reset-every to stay stable")
    p.add_argument("--eval-every", type=int, default=0,
                   help="run a greedy evaluation episode every N episodes (0 = never)")
    p.add_argument("--max-steps", type=int,
                   help="stop an episode after this many steps")
    p.add_argument("--n-step", type=int, default=3,
                   help="steps an n-step return spans; 1 is plain one-step TD")
    p.add_argument("--port", type=int, default=8923,
                   help="port for the page's http server. Give concurrent runs "
                        "different ports: the env reuses a server it finds "
                        "already listening, so two runs sharing a port share a "
                        "server, and whichever exits first takes it down")
    p.add_argument("--demos", type=Path,
                   help="keep imitating these demonstrations while doing RL. "
                        "Without it, fine-tuning from cloned weights forgets "
                        "the policy it started from — see runs/FINDINGS.md")
    p.add_argument("--demo-weight", type=float, default=1.0,
                   help="weight on the demonstration term. Not a ratio of "
                        "equals: the TD term is divided by the action count "
                        "and works on Q values of order 5-9, while this one is "
                        "per-board standardised, so 1.0 measures out around "
                        "14:1 in favour of the demonstrations. Both terms are "
                        "printed each episode; read them before trusting a "
                        "value here")
    p.add_argument("--value-bias", type=float,
                   help="set the dueling value head's bias before training. "
                        "Cloning standardises its targets per board, so the "
                        "value stream comes out near zero while Q-learning "
                        "wants roughly the discounted return; without this the "
                        "first updates have to move value a long way and drag "
                        "the shared trunk — and the advantage stream that "
                        "actually picks the column — along with it")
    p.add_argument("--lr", type=float,
                   help="override the network's learning rate. Fine-tuning "
                        "from cloned weights starts from a policy worth "
                        "keeping, and the built-in schedule ramps up to 1e-3, "
                        "which is enough to undo it before the value stream "
                        "has caught up")
    p.add_argument("--reward-scale", type=float, default=1.0,
                   help="multiplier on the training reward. Long n-step returns "
                        "sum many score deltas, and targets in the hundreds are "
                        "badly conditioned for an MSE loss; the reported score "
                        "stays unscaled")
    p.add_argument("--terminal-penalty", type=float, default=0.0,
                   help="reward subtracted when the game is actually lost. "
                        "Score deltas are never negative, so without this the "
                        "only cost of dying is the forgone future value, which "
                        "a short n-step return can barely see")
    p.add_argument("--double", dest="double", action="store_true", default=True,
                   help="Double DQN targets (default)")
    p.add_argument("--no-double", dest="double", action="store_false",
                   help="plain max-over-target targets, for comparison")
    p.add_argument("--dueling", dest="dueling", action="store_true", default=True,
                   help="split value/advantage head (default)")
    p.add_argument("--no-dueling", dest="dueling", action="store_false")
    p.add_argument("--eval-episodes", type=int, default=5,
                   help="episodes averaged per evaluation point")
    p.add_argument("--seed", type=int)
    return p.parse_args(argv)


def greedy_evaluation(env, model, bins, max_steps=None, episodes=5):
    """Mean greedy score over several episodes.

    One episode is not a measurement here. Suika is stochastic — the next fruit
    is random — and single greedy episodes on this agent have ranged from 889
    to 2326 while the underlying policy barely moved. Averaging is the
    difference between a curve you can read and noise.
    """
    scores, lengths = [], []
    for _ in range(episodes):
        score, steps = greedy_episode(env, model, bins, max_steps)
        scores.append(score)
        lengths.append(steps)
    return (sum(scores) / len(scores), sum(lengths) / len(lengths),
            min(scores), max(scores))


def greedy_episode(env, model, bins, max_steps=None):
    """One episode with exploration off. Training scores are inflated by the
    random actions epsilon is still taking, so a clean number needs its own run."""
    obs, _ = env.reset()
    state = observation(obs)
    total, steps, done = 0.0, 0, False
    while not done and (max_steps is None or steps < max_steps):
        q = q_of(model, state)
        obs, reward, done, truncated, _ = env.step(
            to_continuous(bins, int(np.argmax(q[0]))))
        state = observation(obs)
        total += reward
        steps += 1
        done = done or truncated
    return total, steps


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    bins = action_bins(args.actions)
    schedule = EpsilonSchedule(args.epsilon, args.epsilon_min, args.epsilon_decay)

    if args.resume:
        if not args.checkpoint.exists():
            raise SystemExit(f"--resume given but {args.checkpoint} does not exist")
        model = load_model(args.checkpoint)
        print(f"resumed from {args.checkpoint}")
    else:
        model = make_model(args)

    demo_data = None
    if args.demos is not None:
        d = np.load(args.demos)
        demo_data = (d["grid"].astype(np.float32),
                     d["vector"].astype(np.float32),
                     standardise_rows(d["scores"]))
        print(f"anchored to {len(demo_data[2])} demonstration boards "
              f"at weight {args.demo_weight:g}")

    if args.value_bias is not None:
        try:
            value = model.get_layer("value")
        except ValueError:
            raise SystemExit("--value-bias needs a dueling model with a "
                             "'value' layer")
        w, b = value.get_weights()
        value.set_weights([w, np.full_like(b, args.value_bias)])
        print(f"value head bias set to {args.value_bias:g}")

    if args.lr is not None:
        # Recompiled rather than poked, so the optimizer state is fresh and
        # cannot carry momentum from whatever produced the checkpoint.
        model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss="mse")
        print(f"learning rate overridden to {args.lr:g}")

    # Compiled once, not per update: a tf.function retraces only when the
    # input signature changes, and the batch shape here never does.
    train_step = (make_joint_train_step(model, args.actions, args.demo_weight)
                  if demo_data is not None
                  else make_train_step(model, args.actions))

    # The target network is a frozen copy: same architecture, weights synced
    # on a schedule. Bootstrapping off it keeps the regression target still
    # while the online network moves toward it.
    target = None
    if args.target_sync > 0:
        target = make_model(args)
        target.set_weights(model.get_weights())

    env = SuikaBrowserEnv(headless=args.headless, obs_mode="features",
                          port=args.port)
    buffer = ReplayBuffer(args.buffer)
    gradient_steps = 0

    # Killing this process otherwise orphans chromedriver and its Chrome
    # children — they reparent to launchd and sit there holding memory, which
    # on a machine this size makes the *next* run more likely to be OOM-killed.
    # Closing on the way out costs nothing and keeps repeated runs clean.
    def shutdown(*_):
        try:
            env.close()
        except Exception:      # noqa: BLE001 — best effort on the way out
            pass

    atexit.register(shutdown)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: raise_exit())

    try:
        for episode in range(args.episodes):
            try:
                total_reward, steps, gradient_steps = run_episode(
                    env, model, target, buffer, bins, schedule.at(episode),
                    gradient_steps, args, train_step, demo_data)
            except BROWSER_DEAD as error:
                # The browser died — usually memory pressure, since the page,
                # TensorFlow and the replay buffer share one machine. Rebuild
                # it and carry on: the weights and the buffer are intact, so
                # the cost is one episode rather than the whole run.
                # The message, not just the type: "tab crashed" (the renderer
                # was killed, usually memory), a script timeout (the page hung)
                # and "disconnected" (chromedriver died) all arrive as
                # the same class and want different responses. Logging only
                # the class name made this undiagnosable after the fact.
                detail = str(error).split("\n")[0][:160]
                print(f"Episode {episode:4d}  browser lost: {detail}"
                      f"  | free pages {_free_pages()}  restarting", flush=True)
                env = restart_env(env, lambda: SuikaBrowserEnv(
                    headless=args.headless, obs_mode="features",
                    port=args.port))
                continue

            epsilon = schedule.at(episode)
            print(f"Episode {episode:4d}  reward {total_reward:8.1f}  "
                  f"steps {steps:4d}  eps {epsilon:.3f}  "
                  f"buffer {len(buffer):6d} ({buffer.nbytes()/1e9:.2f} GB)  "
                  f"grads {gradient_steps}"
                  + (f"  td {getattr(run_episode, 'last_losses', (0, 0))[0]:.3f}"
                     f"  bc {getattr(run_episode, 'last_losses', (0, 0))[1]:.3f}"
                     if args.demos else ""), flush=True)
            model.save(args.checkpoint)

            if args.eval_every and (episode + 1) % args.eval_every == 0:
                # The training loop recovers from a dead browser; this path did
                # not, so one crashed evaluation killed a run that was 50
                # episodes in. An eval is a measurement — losing one costs a
                # data point, and taking the run down with it costs the lot.
                try:
                    mean, steps_avg, lo, hi = greedy_evaluation(
                        env, model, bins, args.max_steps,
                        args.eval_episodes)
                    print(f"           eval mean {mean:8.1f}  over "
                          f"{args.eval_episodes} episodes  "
                          f"[{lo:.0f}..{hi:.0f}]  steps {steps_avg:.0f}",
                          flush=True)
                except BROWSER_DEAD as error:
                    print(f"           eval abandoned, browser died: "
                          f"{str(error).splitlines()[0]}", flush=True)

                # Fresh browser for the next stretch either way. Chrome drops
                # the tab often enough on this page to cost episodes, and a
                # page that has been alive for fifty episodes is the one that
                # drops it, so the restart is cheap insurance rather than a
                # reaction to a specific failure.
                env = restart_env(env, lambda: SuikaBrowserEnv(
                    headless=args.headless, obs_mode="features",
                    port=args.port))
    finally:
        env.close()

    return 0


def observation(obs):
    """The network's input, out of an env observation.

    The grid is the board rasterised; the vector is what is in hand, what is
    next, and how full the board is. A pixel mode existed and was removed —
    see runs/FINDINGS.md.
    """
    return (obs["grid"], obs["vector"])


def as_batch(states):
    """Stack stored states into the model's two inputs."""
    return [np.asarray([s[0] for s in states], dtype=np.float32),
            np.asarray([s[1] for s in states], dtype=np.float32)]


def q_of(model, state):
    """Q-values for one state.

    `model.predict` is built for large batches: called once per environment
    step with a batch of one it re-traces and grows memory, and it dominated
    the step time here. Calling the model directly is the documented path for
    single small inputs.
    """
    return np.asarray(model(single(state), training=False))


def single(state):
    """One state as a batch of one."""
    return [state[0][np.newaxis, ...].astype(np.float32),
            state[1][np.newaxis, ...].astype(np.float32)]


def _free_pages():
    """Free physical pages, for the crash log. Cheap, and it turns "the browser
    died" into "the browser died with the machine out of memory"."""
    try:
        import subprocess
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=2).stdout
        for line in out.splitlines():
            if line.startswith("Pages free:"):
                return line.split(":")[1].strip().rstrip(".")
    except Exception:      # noqa: BLE001 — diagnostics must never break the run
        pass
    return "?"


def make_model(args):
    return build_column_model(num_actions=args.actions, dueling=args.dueling)


def raise_exit():
    """Turn a signal into a normal exit so atexit handlers run."""
    raise SystemExit(130)


def run_episode(env, model, target, buffer, bins, epsilon, gradient_steps,
                args, train_step, demo_data=None):
    """Play one episode, training as it goes. Returns reward, steps, grads."""
    obs, _ = env.reset()
    state = observation(obs)
    done = False
    total_reward = 0.0
    steps = 0
    nstep = NStepBuffer(args.n_step, args.gamma)
    loss_terms: list = []

    while not done and (args.max_steps is None or steps < args.max_steps):
        steps += 1
        if np.random.rand() < epsilon:
            action_idx = np.random.randint(args.actions)
        else:
            action_idx = int(np.argmax(q_of(model, state)[0]))

        next_obs, reward, done, truncated, _info = env.step(
            to_continuous(bins, action_idx))
        next_state = observation(next_obs)

        shaped, terminal = terminal_reward(reward, done, truncated,
                                           args.terminal_penalty,
                                           args.reward_scale)
        ready = nstep.push(state, action_idx, shaped, next_state, terminal)
        if ready is not None:
            buffer.push(*ready)
        state = next_state
        total_reward += reward          # reported score stays the raw one
        done = done or truncated

        if len(buffer) >= max(args.warmup, args.batch_size) \
                and steps % args.train_every == 0:
            for _ in range(args.updates_per_step):
                split = _train_step(model, target or model, buffer,
                                    args.batch_size, args.gamma, args.double,
                                    train_step, demo_data)
                if split is not None:
                    loss_terms.append(split)
                gradient_steps += 1
                if target is not None and gradient_steps % args.target_sync == 0:
                    target.set_weights(model.get_weights())

    # The last n-1 transitions are the ones nearest the terminal reward; they
    # have to be drained or they never enter the buffer at all.
    for tail in nstep.flush():
        buffer.push(*tail)

    if loss_terms:
        td = sum(t for t, _ in loss_terms) / len(loss_terms)
        bc = sum(b for _, b in loss_terms) / len(loss_terms)
        run_episode.last_losses = (td, bc)
    return total_reward, steps, gradient_steps


def _train_step(model, target, buffer: ReplayBuffer, batch_size: int,
                gamma: float, double: bool = True,
                train_step=None, demo_data=None) -> None:
    states, actions, rewards, next_states, dones, steps = buffer.sample(
        batch_size, stack=False)

    state_batch = as_batch(states)
    next_batch = as_batch(next_states)

    next_qs_target = np.asarray(target(next_batch, training=False))

    if double:
        # One extra forward pass, and it buys the decoupling: the online net
        # picks the next action, the target net prices it.
        next_qs_online = np.asarray(model(next_batch, training=False))
        targets = double_td_targets(rewards, dones, next_qs_target,
                                    next_qs_online, gamma, steps)
    else:
        targets = td_targets(rewards, dones, next_qs_target, gamma, steps)

    # The compiled step gathers the chosen action out of the prediction itself,
    # so the forward pass over `state_batch` that `apply_targets` needed is no
    # longer required — three passes and a fit become one pass and one step.
    if demo_data is None:
        train_step(state_batch,
                   np.asarray(actions, dtype=np.int32),
                   np.asarray(targets, dtype=np.float32))
        return None
    else:
        d_grid, d_vector, d_scores = demo_data
        pick = np.random.randint(0, len(d_scores), batch_size)
        # Returned rather than discarded: which of the two terms is actually
        # driving the update is the difference between "reinforcement learning
        # added nothing" and "reinforcement learning never got a say", and
        # runs/FINDINGS.md could not tell those apart without this.
        td, bc = train_step(state_batch,
                            np.asarray(actions, dtype=np.int32),
                            np.asarray(targets, dtype=np.float32),
                            [d_grid[pick], d_vector[pick]],
                            d_scores[pick])
        return float(td), float(bc)


if __name__ == "__main__":
    raise SystemExit(main())
