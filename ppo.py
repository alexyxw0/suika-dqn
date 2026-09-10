#!/usr/bin/env python3
"""PPO on top of the cloned policy.

Why this is worth running, given that DQN fine-tuning already measured flat:
every value-based approach in this project has died to the same thing —
`argmax` over many noisy estimates selects for the largest error rather than
the largest value. It killed the next-fruit lookahead, the afterstate value
network, and the first CEM run. A policy gradient never takes that argmax. It
samples from a distribution and shifts it toward what worked, so the failure
mode does not apply.

Why it is not obviously the right tool: PPO is on-policy and throws its data
away after each update, while `runs/FINDINGS.md` measures the sample budget as
the binding constraint (~100k environment steps overnight, at 88 ms a step).
On that axis this is the wrong direction, and the continuous action space PPO
is usually reached for is not a constraint here either — 200 bins scored
-109 +/- 261 against 40.

Three choices follow from what has already been measured.

*Start from the cloned policy.* Learning from scratch is a settled dead end
here; the cloned network reaches 2449 where from-scratch reaches the random
floor. Its 40 duelling outputs are read directly as logits, and the duelling
structure costs nothing in that reading: the value stream adds one constant to
every action, which a softmax cancels. The advantage stream *is* the policy.

*A separate value network.* Sharing a trunk means the untrained value head's
gradients flow through the features the policy depends on, and that is exactly
what destroyed DQN fine-tuning — 2449 down to 1730 twice. Keeping them apart
costs parameters and removes the failure.

*Warm the value function up first.* Advantages computed from an untrained
critic are noise, and a policy updated toward noise on the first rollout has
already lost what cloning bought. The policy is frozen until the critic
explains a reasonable share of the return's variance.

Run: python ppo.py --init runs/bc-column.h5 --rollouts 20
"""

from __future__ import annotations

import argparse
import atexit
import math
import signal
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model

sys.path.insert(0, str(Path(__file__).resolve().parent))

import envpath                                                     # noqa: E402
envpath.ensure()

from agent import (action_bins, browser_failures, explained_variance,
                   gae_advantages, normalise, restart_env, terminal_reward,
                   to_continuous)
from suika_env.suika_browser_env import SuikaBrowserEnv
from train import as_batch, observation, single

BROWSER_DEAD = browser_failures()


def build_value_model(grid_shape=(30, 20, 2), vector_len=15, lr=3e-4):
    """How good is this board? One number, and its own weights.

    Deliberately not a second head on the policy trunk. An untrained critic
    produces large gradients early, and through a shared trunk those land on
    the features the policy is relying on.
    """
    grid_in = layers.Input(shape=grid_shape, name="grid")
    vector_in = layers.Input(shape=(vector_len,), name="vector")

    g = layers.Conv2D(32, 3, padding="same", activation="relu")(grid_in)
    g = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(g)
    g = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(g)
    g = layers.GlobalAveragePooling2D()(g)

    v = layers.Dense(64, activation="relu")(vector_in)
    x = layers.Concatenate()([g, v])
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(1, name="value")(x)

    model = models.Model(inputs=[grid_in, vector_in], outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
    return model


def make_policy_step(policy, clip_eps, ent_coef, lr):
    """One clipped-surrogate update.

    The clip is the whole of PPO: the ratio between the new policy and the one
    that collected the data is not allowed to move the objective beyond
    1 +/- eps, so a single batch cannot take a step large enough to leave the
    region where the collected advantages still mean anything.
    """
    opt = tf.keras.optimizers.Adam(lr)

    @tf.function
    def step(batch, actions, old_logp, advantages):
        with tf.GradientTape() as tape:
            logits = policy(batch, training=True)
            logp_all = tf.nn.log_softmax(logits, axis=-1)
            mask = tf.one_hot(actions, tf.shape(logits)[-1], dtype=tf.float32)
            logp = tf.reduce_sum(logp_all * mask, axis=-1)

            ratio = tf.exp(logp - old_logp)
            unclipped = ratio * advantages
            clipped = tf.clip_by_value(ratio, 1.0 - clip_eps,
                                       1.0 + clip_eps) * advantages
            policy_loss = -tf.reduce_mean(tf.minimum(unclipped, clipped))

            # Entropy keeps the distribution from collapsing onto one column
            # before the advantages have said anything.
            entropy = -tf.reduce_mean(
                tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=-1))
            loss = policy_loss - ent_coef * entropy

        grads = tape.gradient(loss, policy.trainable_variables)
        grads, gnorm = tf.clip_by_global_norm(grads, 0.5)
        opt.apply_gradients(zip(grads, policy.trainable_variables))

        approx_kl = tf.reduce_mean(old_logp - logp)
        # Written as two comparisons rather than abs(ratio - 1) > eps: the
        # graph optimiser fuses `exp(x) - 1` into `expm1`, which has no Metal
        # GPU kernel, so the arithmetically identical form crashes on Apple
        # silicon and this one does not.
        clipfrac = tf.reduce_mean(tf.cast(
            tf.logical_or(ratio > 1.0 + clip_eps, ratio < 1.0 - clip_eps),
            tf.float32))
        return policy_loss, entropy, approx_kl, clipfrac, gnorm

    return step


def collect(env, policy, value, bins, n_steps, args, state, episode_scores):
    """One on-policy rollout, continuing whatever episode was in progress."""
    grids, vectors, actions, logps, values = [], [], [], [], []
    rewards, dones = [], []
    steps_done = 0

    while steps_done < n_steps:
        obs_pair = state["obs"]
        logits = np.asarray(policy(single(obs_pair), training=False))[0]
        # Sample, do not argmax: the whole reason for using a policy gradient
        # here is to stop selecting the largest error out of forty estimates.
        logp_all = logits - tf.reduce_logsumexp(logits).numpy()
        probs = np.exp(logp_all)
        probs = probs / probs.sum()
        action = int(np.random.choice(len(probs), p=probs))
        v = float(np.asarray(value(single(obs_pair), training=False))[0, 0])

        next_obs, reward, done, trunc, info = env.step(
            to_continuous(bins, action))
        shaped, terminal = terminal_reward(reward, done, trunc,
                                           args.terminal_penalty,
                                           args.reward_scale)

        grids.append(obs_pair[0])
        vectors.append(obs_pair[1])
        actions.append(action)
        logps.append(float(logp_all[action]))
        values.append(v)
        rewards.append(shaped)
        dones.append(float(terminal))

        state["score"] = info["score"]
        state["length"] += 1
        steps_done += 1

        if done or trunc:
            episode_scores.append(state["score"])
            state["episode_lengths"].append(state["length"])
            fresh, _ = env.reset()
            state["obs"] = observation(fresh)
            state["score"], state["length"] = 0.0, 0
        else:
            state["obs"] = observation(next_obs)

    last_value = float(np.asarray(
        value(single(state["obs"]), training=False))[0, 0])
    return {
        "grid": np.asarray(grids, dtype=np.float32),
        "vector": np.asarray(vectors, dtype=np.float32),
        "action": np.asarray(actions, dtype=np.int32),
        "logp": np.asarray(logps, dtype=np.float32),
        "value": np.asarray(values, dtype=np.float32),
        "reward": np.asarray(rewards, dtype=np.float32),
        "done": np.asarray(dones, dtype=np.float32),
        "last_value": last_value,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init", type=Path, default=Path("runs/bc-column.h5"),
                   help="cloned policy to start from. Training from scratch is "
                        "a settled dead end here")
    p.add_argument("--rollouts", type=int, default=20)
    p.add_argument("--steps", type=int, default=1024,
                   help="environment steps per rollout, ~4 episodes")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--actions", type=int, default=40)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-5,
                   help="low on purpose: the starting policy is worth keeping")
    p.add_argument("--value-lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--target-kl", type=float, default=0.02,
                   help="stop a rollout's epochs early if the policy has moved "
                        "this far; the clip bounds each step, not their sum")
    p.add_argument("--value-warmup", type=int, default=3,
                   help="rollouts spent training only the critic. Advantages "
                        "from an untrained critic are noise, and a policy "
                        "updated toward noise loses what cloning bought")
    p.add_argument("--reward-scale", type=float, default=0.01)
    p.add_argument("--terminal-penalty", type=float, default=300.0)
    p.add_argument("--port", type=int, default=8931)
    p.add_argument("--headless", action="store_true", default=True)
    p.add_argument("--out", type=Path, default=Path("runs/ppo.h5"))
    p.add_argument("--seed", type=int)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.seed is not None:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    policy = load_model(args.init)
    print(f"policy from {args.init}  ({policy.count_params():,} params)")
    value = build_value_model(lr=args.value_lr)
    print(f"critic: {value.count_params():,} params, separate weights")

    policy_step = make_policy_step(policy, args.clip, args.ent_coef, args.lr)
    bins = action_bins(args.actions)

    def make_env():
        return SuikaBrowserEnv(headless=args.headless, port=args.port,
                               obs_mode="features")

    env = make_env()

    def shutdown():
        try:
            env.close()
        except Exception:      # noqa: BLE001
            pass

    atexit.register(shutdown)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: sys.exit(130))

    obs, _ = env.reset()
    state = {"obs": observation(obs), "score": 0.0, "length": 0,
             "episode_lengths": []}
    episode_scores: list[float] = []
    began = time.time()

    try:
        for r in range(args.rollouts):
            try:
                roll = collect(env, policy, value, bins, args.steps, args,
                               state, episode_scores)
            except BROWSER_DEAD as error:
                print(f"rollout {r:3d}  browser lost: "
                      f"{str(error).splitlines()[0]} — restarting", flush=True)
                env = restart_env(env, make_env)
                obs, _ = env.reset()
                state["obs"] = observation(obs)
                state["score"], state["length"] = 0.0, 0
                continue

            adv = gae_advantages(roll["reward"], roll["value"], roll["done"],
                                 roll["last_value"], args.gamma, args.lam)
            returns = adv + roll["value"]
            batch = [roll["grid"], roll["vector"]]

            # critic first, so the advantages the policy sees were produced by
            # the best critic available rather than the previous one
            vhist = value.fit(batch, returns, epochs=args.epochs,
                              batch_size=args.batch_size, verbose=0)
            pred = np.asarray(value(batch, training=False)).ravel()
            ev = explained_variance(pred, returns)

            warming = r < args.value_warmup
            kl = clipfrac = ent = 0.0
            if not warming:
                nadv = normalise(adv)
                idx = np.arange(len(nadv))
                stop = False
                for _ in range(args.epochs):
                    np.random.shuffle(idx)
                    for lo in range(0, len(idx), args.batch_size):
                        mb = idx[lo:lo + args.batch_size]
                        _pl, ent_t, kl_t, cf_t, _gn = policy_step(
                            [roll["grid"][mb], roll["vector"][mb]],
                            roll["action"][mb], roll["logp"][mb], nadv[mb])
                        kl, clipfrac, ent = (float(kl_t), float(cf_t),
                                             float(ent_t))
                    if kl > args.target_kl:
                        stop = True
                        break
                if stop:
                    print(f"           early stop: KL {kl:.4f} > "
                          f"{args.target_kl}", flush=True)

            recent = episode_scores[-10:]
            mean_score = st.mean(recent) if recent else float("nan")
            tag = "warmup" if warming else "      "
            print(f"rollout {r:3d} {tag}  score {mean_score:7.0f}"
                  f"  ({len(episode_scores)} eps)"
                  f"  ev {ev:6.3f}  entropy {ent:5.3f}"
                  f"  KL {kl:.4f}  clip {clipfrac:.2f}"
                  f"  [{(time.time() - began) / 60:.0f}m]", flush=True)

            args.out.parent.mkdir(parents=True, exist_ok=True)
            policy.save(args.out)
    finally:
        shutdown()

    if episode_scores:
        first = episode_scores[:10]
        last = episode_scores[-10:]
        print(f"\n  {len(episode_scores)} episodes")
        print(f"  first 10: {st.mean(first):.0f}   last 10: {st.mean(last):.0f}")
        if len(episode_scores) > 1:
            sd = st.stdev(episode_scores)
            print(f"  overall {st.mean(episode_scores):.0f} +/- "
                  f"{sd / math.sqrt(len(episode_scores)):.0f} (se)")
        print(f"  reference: cloned policy 2449 +/- 72 | layered teacher 2696")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
