import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import (EpsilonSchedule, ReplayBuffer, action_bins, apply_targets,
                   browser_failures, double_td_targets,
                   explained_variance, gae_advantages, mirror_boards,
                   normalise, standardise_rows, td_targets,
                   terminal_reward, to_continuous, trap_shaping)


class TestActionBins:
    def test_spans_the_full_range(self):
        bins = action_bins(10)
        assert bins[0] == 0.0 and bins[-1] == 1.0 and len(bins) == 10

    def test_bins_are_evenly_spaced(self):
        gaps = np.diff(action_bins(10))
        assert np.allclose(gaps, gaps[0])

    def test_the_env_gets_a_one_element_float32_array(self):
        action = to_continuous(action_bins(10), 3)
        assert action.shape == (1,) and action.dtype == np.float32

    def test_too_few_actions_is_rejected(self):
        with pytest.raises(ValueError, match="at least 2"):
            action_bins(1)


class TestEpsilonSchedule:
    def test_starts_at_the_start_value(self):
        assert EpsilonSchedule().at(0) == 1.0

    def test_decays_monotonically(self):
        s = EpsilonSchedule()
        values = [s.at(i) for i in range(50)]
        assert all(a >= b for a, b in zip(values, values[1:]))

    def test_never_falls_below_the_floor(self):
        s = EpsilonSchedule(minimum=0.05)
        assert s.at(100_000) == 0.05

    def test_the_reported_run_length_still_explores(self):
        # 30 episodes at decay 0.995 leaves epsilon high — worth pinning,
        # because it is part of why the reported scores are what they are.
        assert EpsilonSchedule().at(30) > 0.8


class TestReplayBuffer:
    def test_length_tracks_pushes(self):
        b = ReplayBuffer(100)
        for i in range(5):
            b.push(np.zeros((2, 2)), i, float(i), np.zeros((2, 2)), False)
        assert len(b) == 5

    def test_oldest_entries_are_evicted_at_capacity(self):
        b = ReplayBuffer(3)
        for i in range(5):
            b.push(np.full((1,), i), i, 0.0, np.zeros(1), False)
        assert len(b) == 3
        _, actions, _, _, _, _ = b.sample(3)
        assert set(actions.tolist()) == {2, 3, 4}

    def test_sample_returns_correctly_shaped_arrays(self):
        b = ReplayBuffer(50)
        for i in range(10):
            b.push(np.zeros((4, 4, 3)), i % 3, 1.0, np.ones((4, 4, 3)), i == 9)
        states, actions, rewards, next_states, dones, steps = b.sample(4)
        assert states.shape == (4, 4, 4, 3) and next_states.shape == (4, 4, 4, 3)
        assert actions.shape == (4,) and rewards.shape == (4,) and dones.shape == (4,)
        assert states.dtype == np.float32 and actions.dtype == np.int32

    def test_oversampling_is_an_error_not_a_short_batch(self):
        b = ReplayBuffer(10)
        b.push(np.zeros(1), 0, 0.0, np.zeros(1), False)
        with pytest.raises(ValueError, match="holding 1"):
            b.sample(4)

    def test_zero_capacity_is_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            ReplayBuffer(0)


class TestTdTargets:
    def test_bootstraps_through_a_non_terminal_step(self):
        got = td_targets([1.0], [0.0], [[0.0, 10.0, 5.0]], gamma=0.9)
        assert got == pytest.approx([1.0 + 0.9 * 10.0])

    def test_a_terminal_step_is_the_reward_alone(self):
        """The single most consequential line in DQN: without (1 - done) the
        agent bootstraps through the end of the episode."""
        got = td_targets([1.0], [1.0], [[0.0, 10.0, 5.0]], gamma=0.9)
        assert got == pytest.approx([1.0])

    def test_takes_the_max_over_actions_not_the_mean(self):
        got = td_targets([0.0], [0.0], [[1.0, 2.0, 9.0]], gamma=1.0)
        assert got == pytest.approx([9.0])

    def test_gamma_zero_makes_it_myopic(self):
        got = td_targets([3.0], [0.0], [[100.0]], gamma=0.0)
        assert got == pytest.approx([3.0])

    def test_handles_a_mixed_batch(self):
        got = td_targets([1.0, 2.0], [0.0, 1.0], [[5.0], [5.0]], gamma=0.5)
        assert got == pytest.approx([1.0 + 0.5 * 5.0, 2.0])


class TestApplyTargets:
    def test_only_the_chosen_action_changes(self):
        current = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        updated = apply_targets(current, [0, 2], np.array([99.0, -99.0]))
        assert updated[0].tolist() == [99.0, 2.0, 3.0]
        assert updated[1].tolist() == [4.0, 5.0, -99.0]

    def test_the_input_is_not_mutated(self):
        current = np.array([[1.0, 2.0]], dtype=np.float32)
        apply_targets(current, [0], np.array([9.0]))
        assert current.tolist() == [[1.0, 2.0]]


class TestBufferMemory:
    """The buffer is the largest thing in the process, so what it stores is a
    correctness *and* a survivability question — a run that swaps dies."""

    def test_frames_are_stored_in_the_dtype_they_arrive_in(self):
        b = ReplayBuffer(10)
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        b.push(frame, 0, 0.0, frame, False)
        stored_state, _, _, stored_next, _, _ = b.buffer[0]
        assert stored_state.dtype == np.uint8
        assert stored_next.dtype == np.uint8

    def test_sampling_converts_to_float32_for_the_network(self):
        b = ReplayBuffer(10)
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        for _ in range(4):
            b.push(frame, 0, 0.0, frame, False)
        states, _, _, next_states, _, _ = b.sample(2)
        assert states.dtype == np.float32
        assert next_states.dtype == np.float32

    def test_uint8_storage_is_a_quarter_the_size_of_float32(self):
        shape = (128, 128, 3)
        small, large = ReplayBuffer(100), ReplayBuffer(100)
        for _ in range(20):
            small.push(np.zeros(shape, np.uint8), 0, 0.0, np.zeros(shape, np.uint8), False)
            large.push(np.zeros(shape, np.float32), 0, 0.0, np.zeros(shape, np.float32), False)
        assert large.nbytes() == 4 * small.nbytes()

    def test_nbytes_is_zero_when_empty(self):
        assert ReplayBuffer(10).nbytes() == 0


class TestRestartEnv:
    """The crash handler is the code least likely to have ever run, and the
    most costly to get wrong: a bug here turns a recoverable hiccup into a
    dead multi-hour run. These execute the path deliberately."""

    class Broken:
        def __init__(self): self.closed = False
        def close(self): self.closed = True

    class Unclosable:
        def close(self): raise RuntimeError("the browser is already gone")

    def test_returns_a_freshly_built_environment(self):
        from agent import restart_env
        sentinel = object()
        assert restart_env(self.Broken(), lambda: sentinel, pause=0) is sentinel

    def test_closes_the_dead_one(self):
        from agent import restart_env
        dead = self.Broken()
        restart_env(dead, lambda: None, pause=0)
        assert dead.closed

    def test_survives_a_close_that_itself_fails(self):
        # The usual case: the browser is already gone, so close() raises too.
        from agent import restart_env
        sentinel = object()
        assert restart_env(self.Unclosable(), lambda: sentinel, pause=0) is sentinel


class TestNStepBuffer:
    """n-step returns are easy to get subtly wrong — an off-by-one in the
    discount or a dropped tail is invisible in the loss curve."""

    def make(self, n=3, gamma=0.9):
        from agent import NStepBuffer
        return NStepBuffer(n, gamma)

    def test_nothing_is_emitted_before_n_transitions(self):
        b = self.make(n=3)
        assert b.push("s0", 0, 1.0, "s1", False) is None
        assert b.push("s1", 0, 1.0, "s2", False) is None
        assert b.push("s2", 0, 1.0, "s3", False) is not None

    def test_the_reward_is_discounted_over_n_steps(self):
        b = self.make(n=3, gamma=0.5)
        b.push("s0", 0, 1.0, "s1", False)
        b.push("s1", 0, 1.0, "s2", False)
        out = b.push("s2", 0, 1.0, "s3", False)
        state, action, reward, next_state, done, steps = out
        assert reward == pytest.approx(1.0 + 0.5 * 1.0 + 0.25 * 1.0)
        assert steps == 3

    def test_it_spans_from_the_first_state_to_the_nth_next_state(self):
        b = self.make(n=3)
        b.push("s0", 7, 0.0, "s1", False)
        b.push("s1", 0, 0.0, "s2", False)
        state, action, _, next_state, _, _ = b.push("s2", 0, 0.0, "s3", False)
        assert state == "s0", "the return is credited to the first state"
        assert action == 7, "and to the action taken there"
        assert next_state == "s3", "bootstrapping from n steps later"

    def test_a_terminal_truncates_the_return(self):
        b = self.make(n=5, gamma=0.5)
        b.push("s0", 0, 1.0, "s1", False)
        b.push("s1", 0, 1.0, "s2", True)      # episode ends here
        out = b.flush()
        first = out[0]
        assert first[2] == pytest.approx(1.5), "1.0 + 0.5*1.0, and no further"
        assert first[4] is True, "and it is marked terminal"
        assert first[5] == 2

    def test_flush_drains_the_tail_longest_first(self):
        # Without this the last n-1 transitions of every episode are lost —
        # exactly the ones closest to the terminal reward.
        b = self.make(n=4)
        for i in range(3):
            b.push(f"s{i}", 0, 1.0, f"s{i+1}", False)
        out = b.flush()
        assert [o[5] for o in out] == [3, 2, 1]
        assert [o[0] for o in out] == ["s0", "s1", "s2"]

    def test_n_of_one_reduces_to_a_plain_transition(self):
        b = self.make(n=1, gamma=0.9)
        out = b.push("s0", 2, 5.0, "s1", False)
        assert out == ("s0", 2, 5.0, "s1", False, 1)

    def test_n_below_one_is_rejected(self):
        from agent import NStepBuffer
        with pytest.raises(ValueError, match="at least 1"):
            NStepBuffer(0, 0.9)


class TestDoubleTdTargets:
    def test_the_online_net_chooses_and_the_target_values(self):
        from agent import double_td_targets
        # online prefers action 0; target rates action 1 far higher. Plain DQN
        # would take the target's max (10); Double DQN must take the target's
        # value for the online net's pick (1).
        got = double_td_targets([0.0], [0.0], next_qs_target=[[1.0, 10.0]],
                                next_qs_online=[[5.0, 0.0]], gamma=1.0)
        assert got == pytest.approx([1.0])

    def test_it_differs_from_plain_max_when_the_nets_disagree(self):
        from agent import double_td_targets, td_targets
        plain = td_targets([0.0], [0.0], [[1.0, 10.0]], gamma=1.0)
        double = double_td_targets([0.0], [0.0], [[1.0, 10.0]], [[5.0, 0.0]], 1.0)
        assert plain[0] > double[0], "this is the overestimation being removed"

    def test_a_terminal_step_is_still_the_reward_alone(self):
        from agent import double_td_targets
        got = double_td_targets([3.0], [1.0], [[100.0]], [[100.0]], gamma=0.9)
        assert got == pytest.approx([3.0])

    def test_gamma_is_raised_to_the_actual_step_count(self):
        from agent import double_td_targets
        got = double_td_targets([0.0], [0.0], [[2.0]], [[2.0]], gamma=0.5,
                                steps=[3])
        assert got == pytest.approx([0.125 * 2.0]), "gamma**3, not gamma"

    def test_mixed_step_counts_in_one_batch(self):
        from agent import double_td_targets
        got = double_td_targets([0.0, 0.0], [0.0, 0.0], [[2.0], [2.0]],
                                [[2.0], [2.0]], gamma=0.5, steps=[1, 3])
        assert got == pytest.approx([1.0, 0.25])


class TestMultiInputStates:
    """Feature-mode states are (grid, vector) tuples, which cannot be stacked
    into a single array — the buffer has to hand them back unchanged."""

    def make(self, n=6):
        from agent import ReplayBuffer
        b = ReplayBuffer(50)
        for i in range(n):
            s = (np.full((3, 2, 2), i, np.float32), np.full((5,), i, np.float32))
            b.push(s, 0, float(i), s, False)
        return b

    def test_stack_false_returns_the_tuples_as_stored(self):
        states, *_ = self.make().sample(3, stack=False)
        assert isinstance(states, list) and len(states) == 3
        assert isinstance(states[0], tuple) and len(states[0]) == 2
        assert states[0][0].shape == (3, 2, 2) and states[0][1].shape == (5,)

    def test_nbytes_counts_both_parts_of_a_tuple_state(self):
        b = self.make(4)
        grid = np.zeros((3, 2, 2), np.float32).nbytes
        vec = np.zeros((5,), np.float32).nbytes
        assert b.nbytes() == 4 * 2 * (grid + vec)

    def test_stacking_still_works_for_plain_array_states(self):
        from agent import ReplayBuffer
        b = ReplayBuffer(10)
        for i in range(4):
            f = np.full((2, 2, 3), i, np.uint8)
            b.push(f, 0, 0.0, f, False)
        states, *_ = b.sample(2)
        assert states.shape == (2, 2, 2, 3) and states.dtype == np.float32


class TestNStepDiscounting:
    """`td_targets` used to discount by gamma once regardless of how many steps
    the return actually covered. Harmless at n=3, a large error at n=20."""

    def test_the_bootstrap_is_discounted_by_the_number_of_steps(self):
        next_qs = np.array([[0.0, 10.0]])
        got = td_targets([1.0], [0.0], next_qs, gamma=0.9, steps=[3])
        assert got[0] == pytest.approx(1.0 + 0.9 ** 3 * 10.0)

    def test_omitting_steps_keeps_the_one_step_behaviour(self):
        next_qs = np.array([[0.0, 10.0]])
        got = td_targets([1.0], [0.0], next_qs, gamma=0.9)
        assert got[0] == pytest.approx(1.0 + 0.9 * 10.0)

    def test_a_terminal_sample_ignores_the_bootstrap_entirely(self):
        next_qs = np.array([[0.0, 10.0]])
        got = td_targets([1.0], [1.0], next_qs, gamma=0.9, steps=[7])
        assert got[0] == pytest.approx(1.0)

    def test_it_matches_the_double_variant_when_the_argmax_agrees(self):
        # Same discounting rule in both paths, so --no-double must not change
        # how far an n-step return is discounted.
        next_qs = np.array([[1.0, 9.0], [4.0, 2.0]])
        steps = [5, 5]
        plain = td_targets([0.0, 0.0], [0.0, 0.0], next_qs, 0.95, steps)
        double = double_td_targets([0.0, 0.0], [0.0, 0.0], next_qs, next_qs,
                                   0.95, steps)
        assert plain == pytest.approx(double)

    def test_longer_returns_discount_the_bootstrap_more(self):
        next_qs = np.array([[0.0, 10.0]])
        short = td_targets([0.0], [0.0], next_qs, 0.9, [1])[0]
        long_ = td_targets([0.0], [0.0], next_qs, 0.9, [20])[0]
        assert long_ < short


class TestTerminalReward:
    def test_losing_the_game_costs_the_penalty(self):
        shaped, terminal = terminal_reward(12.0, True, False, 300.0)
        assert shaped == pytest.approx(-288.0) and terminal is True

    def test_hitting_the_step_cap_is_not_a_loss(self):
        # The board was still playable; penalising this would teach the agent
        # to fear the clock, and marking it terminal would zero the value of a
        # state that still had value.
        shaped, terminal = terminal_reward(12.0, False, True, 300.0)
        assert shaped == pytest.approx(12.0) and terminal is False

    def test_an_ordinary_step_is_untouched(self):
        shaped, terminal = terminal_reward(12.0, False, False, 300.0)
        assert shaped == pytest.approx(12.0) and terminal is False

    def test_a_zero_penalty_reproduces_the_old_behaviour(self):
        assert terminal_reward(12.0, True, False, 0.0) == (12.0, True)

    def test_termination_wins_if_both_flags_are_set(self):
        # The env can report both on the final step; a lost game is a lost game.
        shaped, terminal = terminal_reward(0.0, True, True, 100.0)
        assert shaped == pytest.approx(-100.0) and terminal is True

    def test_the_scale_applies_to_reward_and_penalty_together(self):
        # Both are in game points before scaling, so their ratio must survive.
        shaped, _ = terminal_reward(100.0, True, False, 300.0, scale=0.01)
        assert shaped == pytest.approx(-2.0)

    def test_the_scale_defaults_to_leaving_the_reward_alone(self):
        assert terminal_reward(7.0, False, False, 300.0)[0] == pytest.approx(7.0)

    def test_scaling_does_not_change_the_sign_of_a_loss(self):
        for scale in (1.0, 0.1, 0.01):
            assert terminal_reward(5.0, True, False, 300.0, scale)[0] < 0


class TestMirrorBoards:
    """Mirroring is exact for this game, so it has to be exactly right — a
    flipped axis or an off-by-one in the action map would quietly teach the
    network the wrong half of every board."""

    def _board(self):
        grid = np.zeros((1, 4, 6, 2), dtype=np.float32)
        grid[0, 3, 0, 0] = 1.0                    # something on the far left
        scores = np.arange(6, dtype=np.float32)[None, :]
        return grid, scores, np.array([5])

    def test_the_width_axis_is_the_one_that_flips(self):
        grid, scores, actions = self._board()
        g, _, _ = mirror_boards(grid, scores, actions, 6)
        assert g[0, 3, 5, 0] == 1.0 and g[0, 3, 0, 0] == 0.0

    def test_the_height_axis_is_left_alone(self):
        grid, scores, actions = self._board()
        g, _, _ = mirror_boards(grid, scores, actions, 6)
        # Still on the bottom row; only its column moved.
        assert g[0, 3].sum() == 1.0 and g[0, :3].sum() == 0.0

    def test_the_score_vector_reverses(self):
        grid, scores, actions = self._board()
        _, s, _ = mirror_boards(grid, scores, actions, 6)
        assert list(s[0]) == [5, 4, 3, 2, 1, 0]

    def test_the_best_action_maps_to_the_mirrored_best_column(self):
        grid, scores, actions = self._board()
        _, s, a = mirror_boards(grid, scores, actions, 6)
        assert a[0] == 0 and int(s[0].argmax()) == 0

    def test_the_action_map_is_its_own_inverse(self):
        actions = np.arange(40)
        _, _, once = mirror_boards(np.zeros((40, 2, 2, 1)),
                                   np.zeros((40, 40)), actions, 40)
        _, _, twice = mirror_boards(np.zeros((40, 2, 2, 1)),
                                    np.zeros((40, 40)), once, 40)
        assert list(twice) == list(actions)

    def test_mirroring_twice_returns_the_original_board(self):
        grid, scores, actions = self._board()
        g1, s1, a1 = mirror_boards(grid, scores, actions, 6)
        g2, s2, a2 = mirror_boards(g1, s1, a1, 6)
        assert np.array_equal(g2, grid) and np.array_equal(s2, scores)
        assert list(a2) == list(actions)


class TestStandardiseRows:
    def test_each_row_ends_up_zero_mean_and_unit_spread(self):
        out = standardise_rows(np.array([[1.0, 2, 3, 4], [10.0, 20, 30, 40]]))
        assert out.mean(axis=1) == pytest.approx([0, 0], abs=1e-5)
        assert out.std(axis=1) == pytest.approx([1, 1], abs=1e-5)

    def test_the_ranking_within_a_row_survives(self):
        raw = np.array([[3.0, 1.0, 2.0]])
        assert list(np.argsort(standardise_rows(raw)[0])) == list(np.argsort(raw[0]))

    def test_rows_are_scaled_independently(self):
        # Two boards with the same shape but wildly different scale become the
        # same target, which is the point: absolute scale is not imitable.
        out = standardise_rows(np.array([[1.0, 2, 3], [100.0, 200, 300]]))
        assert out[0] == pytest.approx(out[1])

    def test_a_flat_row_does_not_divide_by_zero(self):
        out = standardise_rows(np.array([[5.0, 5.0, 5.0]]))
        assert np.all(np.isfinite(out))
        assert np.allclose(out, 0.0)


class TestBrowserFailures:
    """The recovery path is only as good as the exceptions it catches. A
    collection run died on a urllib3 read timeout with `except
    WebDriverException` written right above it."""

    def test_it_returns_a_tuple_usable_in_an_except_clause(self):
        types = browser_failures()
        assert isinstance(types, tuple) and types
        assert all(isinstance(t, type) and issubclass(t, BaseException)
                   for t in types)

    def test_a_urllib3_read_timeout_is_covered(self):
        urllib3 = pytest.importorskip("urllib3")
        err = urllib3.exceptions.ReadTimeoutError(None, "url", "timed out")
        assert isinstance(err, browser_failures())

    def test_a_selenium_webdriver_error_is_covered(self):
        selenium = pytest.importorskip("selenium")
        from selenium.common.exceptions import WebDriverException
        assert isinstance(WebDriverException("tab crashed"), browser_failures())

    def test_it_does_not_swallow_ordinary_programming_errors(self):
        # The reason this is a tuple and not `Exception`: a shadowed variable
        # inside an episode loop should crash loudly, not be retried three
        # times and blamed on the browser.
        for mistake in (NameError("x"), AttributeError("y"), TypeError("z"),
                        KeyError("k"), ValueError("v")):
            assert not isinstance(mistake, browser_failures())


class TestGAE:
    """The advantage estimator. Its terminal handling is the subtle part: a
    lost game has no future, a rollout that ran out of steps does."""

    def test_lambda_one_is_the_full_discounted_return_when_values_are_zero(self):
        got = gae_advantages([1.0, 1.0, 1.0], [0.0] * 3, [0.0] * 3, 0.0,
                             gamma=1.0, lam=1.0)
        assert list(got) == [3.0, 2.0, 1.0]

    def test_lambda_zero_is_one_step_td(self):
        got = gae_advantages([1.0, 1.0, 1.0], [0.0] * 3, [0.0] * 3, 0.0,
                             gamma=1.0, lam=0.0)
        assert list(got) == [1.0, 1.0, 1.0]

    def test_a_perfect_value_function_gives_zero_advantage(self):
        # V(s_t) = 1 + V(s_{t+1}) for a reward of 1 and no discount.
        values = [3.0, 2.0, 1.0]
        got = gae_advantages([1.0, 1.0, 1.0], values, [0.0] * 3, 0.0,
                             gamma=1.0, lam=1.0)
        assert got == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)

    def test_a_terminal_state_does_not_bootstrap(self):
        # Whatever the value function says about the state after a lost game,
        # it must not enter the advantage.
        ends = gae_advantages([1.0], [0.0], [1.0], 999.0, gamma=0.99, lam=0.95)
        assert ends[0] == pytest.approx(1.0)

    def test_a_truncated_rollout_does_bootstrap(self):
        # Same numbers, but the episode was still alive: last_value counts.
        alive = gae_advantages([1.0], [0.0], [0.0], 10.0, gamma=0.99, lam=0.95)
        assert alive[0] == pytest.approx(1.0 + 0.99 * 10.0)

    def test_a_terminal_stops_credit_flowing_backwards(self):
        # Reward after the terminal must not reach the step before it.
        got = gae_advantages([0.0, 0.0], [0.0, 0.0], [1.0, 0.0], 0.0,
                             gamma=1.0, lam=1.0)
        assert got[0] == pytest.approx(0.0)

    def test_returns_reconstruct_as_advantage_plus_value(self):
        rng = np.random.default_rng(0)
        r = rng.normal(size=8).astype(np.float32)
        v = rng.normal(size=8).astype(np.float32)
        d = np.zeros(8, dtype=np.float32)
        adv = gae_advantages(r, v, d, 0.5, gamma=0.99, lam=0.95)
        assert np.all(np.isfinite(adv + v))

    def test_the_length_matches_the_rollout(self):
        assert len(gae_advantages([0.0] * 7, [0.0] * 7, [0.0] * 7, 0.0)) == 7


class TestExplainedVariance:
    def test_a_perfect_prediction_is_one(self):
        assert explained_variance([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)

    def test_predicting_the_mean_is_zero(self):
        assert explained_variance([2, 2, 2], [1, 2, 3]) == pytest.approx(0.0)

    def test_worse_than_the_mean_is_negative(self):
        assert explained_variance([3, 2, 1], [1, 2, 3]) < 0.0

    def test_a_constant_target_does_not_divide_by_zero(self):
        assert explained_variance([1, 2, 3], [5, 5, 5]) == 0.0


class TestNormalise:
    def test_it_centres_and_scales(self):
        out = normalise([1.0, 2.0, 3.0, 4.0])
        assert out.mean() == pytest.approx(0.0, abs=1e-6)
        assert out.std() == pytest.approx(1.0, abs=1e-4)

    def test_a_constant_input_does_not_blow_up(self):
        assert np.all(np.isfinite(normalise([2.0, 2.0, 2.0])))

    def test_the_ordering_survives(self):
        raw = [3.0, 1.0, 2.0]
        assert list(np.argsort(normalise(raw))) == list(np.argsort(raw))


class TestTrapShaping:
    """The trap term, which is the only per-step signal the agent gets that is
    not a score delta. Its whole justification is that it leaves the optimal
    policy alone, so that is what these check."""

    def test_weight_zero_costs_nothing(self):
        assert trap_shaping(0, 9, 0.0, 0.99, False) == 0.0

    def test_creating_a_trap_is_charged(self):
        assert trap_shaping(0, 1, 1.0, 0.99, False) < 0

    def test_clearing_a_trap_is_refunded(self):
        assert trap_shaping(1, 0, 1.0, 0.99, False) > 0

    def test_making_and_clearing_nets_out(self):
        # Undiscounted the two do not quite cancel; the residue is the
        # discount, and it is small and in the right direction.
        made = trap_shaping(0, 1, 1.0, 0.99, False)
        cleared = trap_shaping(1, 0, 1.0, 0.99, False)
        assert made + cleared == pytest.approx(0.01, abs=1e-9)

    def test_a_terminal_state_has_no_potential(self):
        # Phi(terminal) must be 0 or the telescoping leaves a residue on every
        # death and the policy-invariance guarantee is void. What is left is
        # the refund of what was charged when the traps were made.
        assert trap_shaping(3, 3, 1.0, 0.99, True) == pytest.approx(3.0)

    def test_it_telescopes_over_an_episode(self):
        """The sum of the discounted shaping over any trajectory depends only
        on where it started, never on the route. That is the property that
        makes this safe to add to the reward."""
        gamma = 0.97
        for route in ([0, 1, 2, 1, 3, 0, 2], [0, 3, 3, 3, 0, 0, 2],
                      [0, 0, 0, 0, 0, 0, 2]):
            total = 0.0
            for t, (before, after) in enumerate(zip(route, route[1:])):
                total += gamma ** t * trap_shaping(before, after, 1.0, gamma,
                                                   False)
            # gamma**T * Phi(s_T) - Phi(s_0), with Phi(s) = -w * trapped
            expected = gamma ** (len(route) - 1) * -route[-1] - -route[0]
            assert total == pytest.approx(expected, abs=1e-9)

    def test_flat_form_is_a_tax_on_holding(self):
        # The literal per-step reading: it charges every step a trap exists,
        # so an episode twice as long pays twice as much for the same board.
        assert trap_shaping(3, 3, 2.0, 0.99, False,
                            potential=False) == pytest.approx(-6.0)

    def test_flat_form_does_not_telescope(self):
        """The reason it is not the default: two routes ending in the same
        place cost different amounts, so it reorders policies."""
        gamma = 0.97
        held = sum(gamma ** t * trap_shaping(1, 1, 1.0, gamma, False,
                                             potential=False)
                   for t in range(6))
        cleared = sum(gamma ** t * trap_shaping(1, 0, 1.0, gamma, False,
                                                potential=False)
                      for t in range(6))
        assert held < cleared
