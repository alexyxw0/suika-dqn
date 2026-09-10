"""Geometry tests for the hand-written baseline.

`landing_y` is the part worth testing: if it is wrong the policy still runs and
still returns a number, it just quietly plays badly, which is exactly the
failure mode that made the earlier training runs so hard to read.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import numpy as np

from heuristic import (BOARD_W, BOARD_WEIGHTS, DROPPABLE, FLOOR_Y,
                       POLICIES, RADII, buried_small, choose, contacts,
                       landing_y, ready_pairs, score_all, score_board,
                       score_candidate, trapped_indices, trapped_small)

WEIGHTS = {"merge": 30.0, "chain": 60.0, "low": 200.0, "bury": -8.0,
           "stack": 0.0, "trap": 0.0}


class TestLandingY:
    def test_an_empty_board_lands_on_the_floor(self):
        assert landing_y(320, 64, []) == FLOOR_Y - 64

    def test_stacking_squarely_on_one_fruit(self):
        # Centres end up exactly (r + r) apart, straight above.
        floor = FLOOR_Y - 64
        assert landing_y(320, 64, [(320, floor, 64, 4)]) == floor - 128

    def test_an_offset_drop_rolls_partway_down_the_side(self):
        floor = FLOOR_Y - 64
        y = landing_y(320 + 64, 64, [(320, floor, 64, 4)])
        assert y == pytest.approx(floor - math.sqrt(128 ** 2 - 64 ** 2))

    def test_a_fruit_out_of_horizontal_reach_offers_no_support(self):
        floor = FLOOR_Y - 64
        assert landing_y(320 + 200, 64, [(320, floor, 64, 4)]) == FLOOR_Y - 64

    def test_it_stops_at_the_first_thing_it_meets_not_the_last(self):
        # Two supporters in the same column; the higher one catches it.
        low = (320, FLOOR_Y - 24, 24, 0)
        high = (320, 400, 64, 4)
        assert landing_y(320, 24, [low, high]) == 400 - 88

    def test_order_of_the_fruit_list_does_not_matter(self):
        a = (300, 700, 40, 2)
        b = (330, 500, 56, 3)
        assert landing_y(320, 32, [a, b]) == landing_y(320, 32, [b, a])

    def test_a_bigger_fruit_rests_higher_on_the_same_support(self):
        support = [(320, FLOOR_Y - 24, 24, 0)]
        assert landing_y(320, 96, support) < landing_y(320, 32, support)


class TestContacts:
    def test_it_finds_the_fruit_being_rested_on(self):
        floor = FLOOR_Y - 64
        found = contacts(320, floor - 128, 64, [(320, floor, 64, 4)])
        assert len(found) == 1 and found[0][3] == 4

    def test_it_ignores_fruit_that_is_merely_nearby(self):
        assert contacts(320, 400, 64, [(320, FLOOR_Y - 64, 64, 4)]) == []

    def test_a_side_by_side_neighbour_counts_as_touching(self):
        floor = FLOOR_Y - 64
        assert len(contacts(320 + 128, floor, 64, [(320, floor, 64, 4)])) == 1


class TestScoreCandidate:
    def test_a_merge_beats_an_empty_column(self):
        floor = FLOOR_Y - 64
        board = [(100, floor, 64, 4)]
        onto, _ = score_candidate(100, 4, board, WEIGHTS)
        empty, _ = score_candidate(500, 4, board, WEIGHTS)
        assert onto > empty

    def test_a_drop_beyond_the_wall_is_scored_where_it_would_actually_land(self):
        _, x = score_candidate(0, 4, [], WEIGHTS)
        assert x == RADII[4]
        _, x = score_candidate(BOARD_W, 4, [], WEIGHTS)
        assert x == BOARD_W - RADII[4]

    def test_landing_lower_is_preferred_when_no_merge_is_on_offer(self):
        # Sizes chosen so nothing merges and nothing is buried, leaving height
        # as the only term in play.
        tower = [(100, FLOOR_Y - 72, 72, 5), (100, FLOOR_Y - 216, 72, 5)]
        onto_tower, _ = score_candidate(100, 0, tower, WEIGHTS)
        open_floor, _ = score_candidate(500, 0, tower, WEIGHTS)
        assert open_floor > onto_tower

    def test_burying_a_much_smaller_fruit_is_penalised(self):
        small = [(100, FLOOR_Y - 24, 24, 0)]
        onto_small, _ = score_candidate(100, 6, small, WEIGHTS)
        clear, _ = score_candidate(500, 6, small, WEIGHTS)
        assert onto_small < clear


class TestChoose:
    def test_it_goes_for_the_merge(self):
        # Not necessarily straight on top: resting against the side of a
        # same-size fruit merges just as well and keeps the pile lower, so the
        # policy is expected to prefer that. What matters is that the chosen
        # column ends up touching the target.
        floor = FLOOR_Y - 64
        target = (480, floor, 64, 4)
        action = choose({"fruits": [target], "cur": 4}, 40, WEIGHTS)
        x = (action / 39) * BOARD_W
        y = landing_y(x, 64, [target])
        assert any(t[3] == 4 for t in contacts(x, y, 64, [target]))

    def test_it_prefers_the_lower_of_two_equally_merging_columns(self):
        floor = FLOOR_Y - 64
        target = (480, floor, 64, 4)
        action = choose({"fruits": [target], "cur": 4}, 40, WEIGHTS)
        x = (action / 39) * BOARD_W
        assert landing_y(x, 64, [target]) > floor - 128

    def test_it_returns_a_valid_action_index_on_an_empty_board(self):
        action = choose({"fruits": [], "cur": 0}, 40, WEIGHTS)
        assert isinstance(action, int) and 0 <= action < 40

    def test_it_never_leaves_the_action_range_on_a_crowded_board(self):
        fruits = [(x, FLOOR_Y - 24, 24, i % N) for i, x in
                  enumerate(range(24, BOARD_W, 48)) for N in (5,)]
        for cur in range(len(RADII)):
            action = choose({"fruits": fruits, "cur": cur}, 40, WEIGHTS)
            assert 0 <= action < 40


class TestScoreAll:
    def test_it_scores_every_column(self):
        scores = score_all({"fruits": [], "cur": 3}, 40, WEIGHTS)
        assert scores.shape == (40,)

    def test_its_argmax_is_what_choose_returns(self):
        floor = FLOOR_Y - 64
        state = {"fruits": [(300, floor, 64, 4), (500, floor, 40, 2)], "cur": 4}
        scores = score_all(state, 40, WEIGHTS)
        assert int(scores.argmax()) == choose(state, 40, WEIGHTS)

    def test_near_equivalent_columns_score_near_equally(self):
        # The reason the whole vector is the better target: on an empty board
        # every interior column is the same move, and a classifier trained on
        # the argmax is being told to distinguish them.
        scores = score_all({"fruits": [], "cur": 0}, 40, WEIGHTS)
        interior = scores[5:35]
        assert float(interior.max() - interior.min()) < 1e-3

    def test_a_merge_column_stands_out_from_its_neighbours(self):
        floor = FLOOR_Y - 64
        state = {"fruits": [(320, floor, 64, 4)], "cur": 4}
        scores = score_all(state, 40, WEIGHTS)
        assert scores.max() > scores.min()


class TestStackAndTrap:
    def _w(self, **over):
        w = dict(WEIGHTS)
        w.update(stack=0.0, trap=0.0)
        w.update(over)
        return w

    def test_resting_a_small_fruit_on_a_big_one_is_rewarded(self):
        big = [(100, FLOOR_Y - RADII[6], RADII[6], 6)]
        on, _ = score_candidate(100, 1, big, self._w(stack=25.0))
        off, _ = score_candidate(100, 1, big, self._w())
        assert on > off

    def test_the_reward_grows_with_the_size_difference(self):
        small_gap = [(100, FLOOR_Y - RADII[3], RADII[3], 3)]
        big_gap = [(100, FLOOR_Y - RADII[8], RADII[8], 8)]
        a, _ = score_candidate(100, 1, small_gap, self._w(stack=25.0))
        b, _ = score_candidate(100, 1, big_gap, self._w(stack=25.0))
        base_a, _ = score_candidate(100, 1, small_gap, self._w())
        base_b, _ = score_candidate(100, 1, big_gap, self._w())
        assert (b - base_b) > (a - base_a)

    def test_covering_a_smaller_fruit_is_punished(self):
        small = [(100, FLOOR_Y - RADII[0], RADII[0], 0)]
        over, _ = score_candidate(100, 7, small, self._w(trap=-20.0))
        clear, _ = score_candidate(500, 7, small, self._w(trap=-20.0))
        assert over < clear

    def test_the_penalty_only_reaches_the_band_this_drop_covers(self):
        # A small fruit far below is already buried; this placement should not
        # be charged for it again.
        deep = [(100, FLOOR_Y - RADII[0], RADII[0], 0),
                (100, FLOOR_Y - 400, RADII[6], 6)]
        with_trap, _ = score_candidate(100, 7, deep, self._w(trap=-20.0))
        without, _ = score_candidate(100, 7, deep, self._w())
        assert with_trap == pytest.approx(without)

    def test_dropping_onto_bigger_fruit_traps_nothing(self):
        big = [(100, FLOOR_Y - RADII[8], RADII[8], 8)]
        a, _ = score_candidate(100, 2, big, self._w(trap=-20.0))
        b, _ = score_candidate(100, 2, big, self._w())
        assert a == pytest.approx(b)

    def test_both_terms_are_off_by_default_so_the_baseline_is_reproducible(self):
        # The numbers in runs/FINDINGS.md were measured with the greedy preset,
        # so it has to keep meaning what it meant: a weights dict carrying only
        # the four original terms must score identically to the preset.
        board = [(100, FLOOR_Y - RADII[6], RADII[6], 6)]
        with_new, _ = score_candidate(100, 1, board, POLICIES["greedy"])
        legacy = {k: POLICIES["greedy"][k]
                  for k in ("merge", "chain", "low", "bury")}
        old_way, _ = score_candidate(100, 1, board, legacy)
        assert with_new == pytest.approx(old_way)


class TestReadyPairs:
    def test_an_empty_board_has_none(self):
        assert ready_pairs([]) == 0

    def test_two_touching_same_size_fruit_count(self):
        f = [(100, 800, 24, 0), (140, 800, 24, 0)]
        assert ready_pairs(f) == 1

    def test_different_sizes_do_not_count(self):
        f = [(100, 800, 24, 0), (140, 800, 32, 1)]
        assert ready_pairs(f) == 0

    def test_the_same_size_far_apart_does_not_count(self):
        f = [(50, 800, 24, 0), (600, 800, 24, 0)]
        assert ready_pairs(f) == 0

    def test_three_in_a_row_counts_only_the_pairs_within_reach(self):
        # Neighbours are 40px apart and the threshold is (24+24)*1.45 = 70,
        # so the two adjacent pairs count and the outer pair, 80px apart,
        # does not. Counting it would make a long row look like a merge
        # bonanza when only its neighbours can actually meet.
        f = [(100, 800, 24, 0), (140, 800, 24, 0), (180, 800, 24, 0)]
        assert ready_pairs(f) == 2


class TestBuriedSmall:
    def test_an_empty_board_has_none(self):
        assert buried_small([]) == 0

    def test_a_big_fruit_over_a_small_one_buries_it(self):
        f = [(100, 800, 24, 0), (100, 700, 96, 7)]
        assert buried_small(f) == 1

    def test_a_big_fruit_below_buries_nothing(self):
        f = [(100, 800, 96, 7), (100, 600, 24, 0)]
        assert buried_small(f) == 0

    def test_one_size_apart_is_ordinary_play(self):
        f = [(100, 800, 84, 6), (100, 700, 96, 7)]
        assert buried_small(f) == 0

    def test_a_fruit_off_to_the_side_buries_nothing(self):
        f = [(100, 800, 24, 0), (600, 700, 96, 7)]
        assert buried_small(f) == 0

    def test_each_buried_fruit_is_counted_once(self):
        f = [(100, 800, 24, 0), (100, 700, 96, 7), (100, 650, 128, 8)]
        assert buried_small(f) == 1


class TestScoreBoard:
    def _r(self, **kw):
        base = {"fruits": [], "gained": 0, "lost": False}
        base.update(kw)
        return base

    def test_points_scored_raise_the_value(self):
        w = BOARD_WEIGHTS
        assert score_board(self._r(gained=10), w) > score_board(self._r(), w)

    def test_losing_is_heavily_penalised(self):
        w = BOARD_WEIGHTS
        assert score_board(self._r(lost=True, gained=66), w) < \
               score_board(self._r(), w)

    def test_a_higher_pile_scores_worse(self):
        w = BOARD_WEIGHTS
        low = self._r(fruits=[(320, FLOOR_Y - 24, 24, 0)])
        high = self._r(fruits=[(320, 100, 24, 0)])
        assert score_board(low, w) > score_board(high, w)

    def test_a_pair_waiting_to_merge_is_worth_something(self):
        w = BOARD_WEIGHTS
        apart = self._r(fruits=[(60, 800, 24, 0), (600, 800, 24, 0)])
        together = self._r(fruits=[(300, 800, 24, 0), (340, 800, 24, 0)])
        assert score_board(together, w) > score_board(apart, w)

    def test_burying_a_small_fruit_costs(self):
        w = BOARD_WEIGHTS
        clear = self._r(fruits=[(100, 800, 24, 0), (500, 700, 96, 7)])
        buried = self._r(fruits=[(100, 800, 24, 0), (100, 700, 96, 7)])
        assert score_board(clear, w) > score_board(buried, w)


class TestTrappedSmall:
    """"Trapped" means: of the forty columns the agent may drop into, none puts
    an identical fruit in contact with this one. Measured against the engine in
    scripts/check_trapped.py; these pin the geometry it rests on."""

    def test_an_empty_board_traps_nothing(self):
        assert trapped_small([]) == 0

    def test_a_fruit_alone_on_the_floor_is_reachable(self):
        assert trapped_small([(320.0, FLOOR_Y - 24, 24.0, 0)]) == 0

    def test_a_fruit_under_a_boulder_is_trapped(self):
        small = (320.0, FLOOR_Y - 24, 24.0, 0)
        boulder = (320.0, FLOOR_Y - 48 - 160, 160.0, 9)
        assert trapped_indices([small, boulder]) == [0]

    def test_sizes_the_dropper_never_hands_out_do_not_count(self):
        """A size-9 fruit with no partner is not trapped, it is waiting on the
        board to build one. Penalising it would penalise ordinary play."""
        buried_big = [(320.0, FLOOR_Y - 160, 160.0, 9),
                      (320.0, FLOOR_Y - 320 - 192, 192.0, 10)]
        assert trapped_small(buried_big) == 0

    def test_every_droppable_size_can_be_trapped(self):
        for size in range(DROPPABLE):
            r = RADII[size]
            roof = RADII[10]
            board = [(320.0, FLOOR_Y - r, float(r), size),
                     (320.0, FLOOR_Y - 2 * r - roof, float(roof), 10)]
            assert trapped_small(board) == 1, f"size {size} should be trapped"

    def test_a_fruit_in_a_corner_is_still_reachable(self):
        """The walls clamp a drop back inside, so the edge columns are usable
        and a fruit against the wall is not trapped by the wall alone."""
        assert trapped_small([(24.0, FLOOR_Y - 24, 24.0, 0)]) == 0

    def test_more_columns_can_only_help(self):
        """Reachability is a search over the action space, so a finer one can
        never trap more fruit than a coarser one."""
        board = [(300.0, FLOOR_Y - 24, 24.0, 0),
                 (300.0, FLOOR_Y - 48 - 84, 84.0, 6),
                 (150.0, FLOOR_Y - 32, 32.0, 1),
                 (480.0, FLOOR_Y - 40, 40.0, 2)]
        coarse = trapped_small(board, actions=10)
        fine = trapped_small(board, actions=80)
        assert fine <= coarse

    def test_more_slack_can_only_help(self):
        """Slack stands in for roll. Allowing more of it can free a fruit but
        must never trap one that was already reachable."""
        board = [(300.0, FLOOR_Y - 24, 24.0, 0),
                 (300.0, FLOOR_Y - 48 - 96, 96.0, 7),
                 (200.0, FLOOR_Y - 32, 32.0, 1)]
        tight = set(trapped_indices(board, slack=0.0))
        loose = set(trapped_indices(board, slack=64.0))
        assert loose <= tight

    def test_it_is_stricter_than_buried_small(self):
        """A fruit with a clear column above it is reachable even with a bigger
        fruit overlapping its span — which buried_small counts and this does
        not."""
        board = [(300.0, FLOOR_Y - 24, 24.0, 0),
                 (355.0, FLOOR_Y - 56, 56.0, 3)]
        assert buried_small(board) == 1
        assert trapped_small(board) == 0
