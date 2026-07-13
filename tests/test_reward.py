"""Unit tests for every reward.RewardStrategy.

Each strategy is a pure function of (observation, action, info), so these
tests never touch MetaDrive -- they build minimal info dicts / action
arrays by hand and check the documented per-step reward range and edge
cases directly.
"""

import pytest

from reward import (
    AvoidCollisionRewardStrategy,
    BrakingRewardStrategy,
    CompositeRewardStrategy,
    HeadingAlignmentRewardStrategy,
    LaneCenteringRewardStrategy,
    MinSpeedPenaltyStrategy,
    MotionGatedRewardStrategy,
    ProgressRewardStrategy,
    ProximityPenaltyStrategy,
    RewardStrategy,
    RouteCompletionBonusStrategy,
    SpeedRewardStrategy,
    SteeringSmoothRewardStrategy,
)


class ConstantRewardStrategy(RewardStrategy):
    """Test double: always returns a fixed value, tracks reset() calls."""

    def __init__(self, value=1.0):
        self.value = value
        self.reset_calls = 0

    def compute(self, observation, action, info) -> float:
        return self.value

    def reset(self):
        self.reset_calls += 1


class TestSpeedRewardStrategy:
    def test_zero_velocity_is_zero(self):
        strategy = SpeedRewardStrategy(weight=1.0, max_speed=40.0)
        assert strategy.compute(None, None, {"velocity": 0.0}) == 0.0

    def test_max_speed_yields_full_weight(self):
        strategy = SpeedRewardStrategy(weight=2.0, max_speed=40.0)
        assert strategy.compute(None, None, {"velocity": 40.0}) == 2.0

    def test_half_speed_yields_half_weight(self):
        strategy = SpeedRewardStrategy(weight=2.0, max_speed=40.0)
        assert strategy.compute(None, None, {"velocity": 20.0}) == 1.0

    @pytest.mark.parametrize("max_speed", [0.0, -10.0])
    def test_non_positive_max_speed_raises(self, max_speed):
        with pytest.raises(ValueError):
            SpeedRewardStrategy(max_speed=max_speed)


class TestProgressRewardStrategy:
    def test_first_call_absorbs_baseline_and_returns_zero(self):
        strategy = ProgressRewardStrategy(weight=200.0)
        reward = strategy.compute(None, None, {"route_completion": 0.4})
        assert reward == 0.0

    def test_forward_progress_is_positive(self):
        strategy = ProgressRewardStrategy(weight=200.0)
        strategy.compute(None, None, {"route_completion": 0.10})
        reward = strategy.compute(None, None, {"route_completion": 0.11})
        assert reward == pytest.approx(200.0 * 0.01)

    def test_backward_motion_is_negative(self):
        strategy = ProgressRewardStrategy(weight=200.0)
        strategy.compute(None, None, {"route_completion": 0.10})
        reward = strategy.compute(None, None, {"route_completion": 0.08})
        assert reward == pytest.approx(200.0 * -0.02)

    def test_reset_clears_baseline(self):
        strategy = ProgressRewardStrategy(weight=200.0)
        strategy.compute(None, None, {"route_completion": 0.5})
        strategy.compute(None, None, {"route_completion": 0.6})
        strategy.reset()
        reward = strategy.compute(None, None, {"route_completion": 0.9})
        assert reward == 0.0


class TestLaneCenteringRewardStrategy:
    def test_centre_yields_full_weight(self):
        strategy = LaneCenteringRewardStrategy(weight=3.0,
                                               half_lane_width=0.6)
        assert strategy.compute(None, None,
                                {"lateral_offset": 0.0}) == 3.0

    def test_at_half_lane_width_yields_zero(self):
        strategy = LaneCenteringRewardStrategy(weight=3.0,
                                               half_lane_width=0.6)
        reward = strategy.compute(None, None, {"lateral_offset": 0.6})
        assert reward == pytest.approx(0.0)

    def test_beyond_half_lane_width_clips_to_zero(self):
        strategy = LaneCenteringRewardStrategy(weight=3.0,
                                               half_lane_width=0.6)
        reward = strategy.compute(None, None, {"lateral_offset": 5.0})
        assert reward == 0.0

    def test_symmetric_for_negative_offset(self):
        strategy = LaneCenteringRewardStrategy(weight=3.0,
                                               half_lane_width=0.6)
        left = strategy.compute(None, None, {"lateral_offset": -0.3})
        right = strategy.compute(None, None, {"lateral_offset": 0.3})
        assert left == right == pytest.approx(1.5)

    @pytest.mark.parametrize("half_lane_width", [0.0, -0.6])
    def test_non_positive_half_lane_width_raises(self, half_lane_width):
        with pytest.raises(ValueError):
            LaneCenteringRewardStrategy(half_lane_width=half_lane_width)


class TestHeadingAlignmentRewardStrategy:
    def test_aligned_yields_full_weight(self):
        strategy = HeadingAlignmentRewardStrategy(weight=3.0, max_error=0.5)
        assert strategy.compute(None, None,
                                {"heading_error": 0.0}) == 3.0

    def test_at_max_error_yields_zero(self):
        strategy = HeadingAlignmentRewardStrategy(weight=3.0, max_error=0.5)
        reward = strategy.compute(None, None, {"heading_error": 0.5})
        assert reward == pytest.approx(0.0)

    def test_beyond_max_error_clips_to_zero(self):
        strategy = HeadingAlignmentRewardStrategy(weight=3.0, max_error=0.5)
        reward = strategy.compute(None, None, {"heading_error": 3.0})
        assert reward == 0.0

    def test_symmetric_for_negative_error(self):
        strategy = HeadingAlignmentRewardStrategy(weight=3.0, max_error=0.5)
        left = strategy.compute(None, None, {"heading_error": -0.25})
        right = strategy.compute(None, None, {"heading_error": 0.25})
        assert left == right == pytest.approx(1.5)

    @pytest.mark.parametrize("max_error", [0.0, -0.5])
    def test_non_positive_max_error_raises(self, max_error):
        with pytest.raises(ValueError):
            HeadingAlignmentRewardStrategy(max_error=max_error)


class TestAvoidCollisionRewardStrategy:
    def test_no_incident_is_zero(self):
        strategy = AvoidCollisionRewardStrategy(crash_penalty=-30.0)
        info = {"crash": False, "out_of_road": False}
        assert strategy.compute(None, None, info) == 0.0

    def test_crash_yields_penalty(self):
        strategy = AvoidCollisionRewardStrategy(crash_penalty=-30.0)
        info = {"crash": True, "out_of_road": False}
        assert strategy.compute(None, None, info) == -30.0

    def test_out_of_road_yields_penalty(self):
        strategy = AvoidCollisionRewardStrategy(crash_penalty=-30.0)
        info = {"crash": False, "out_of_road": True}
        assert strategy.compute(None, None, info) == -30.0

    def test_both_incidents_not_double_counted(self):
        strategy = AvoidCollisionRewardStrategy(crash_penalty=-30.0)
        info = {"crash": True, "out_of_road": True}
        assert strategy.compute(None, None, info) == -30.0


class TestRouteCompletionBonusStrategy:
    def test_arrival_yields_bonus(self):
        strategy = RouteCompletionBonusStrategy(bonus=100.0)
        info = {"arrive_dest": True}
        assert strategy.compute(None, None, info) == 100.0

    def test_no_arrival_is_zero(self):
        strategy = RouteCompletionBonusStrategy(bonus=100.0)
        info = {"arrive_dest": False}
        assert strategy.compute(None, None, info) == 0.0

    def test_missing_key_raises_key_error(self):
        # Consistent with every sibling strategy: MetaDrive always
        # populates 'arrive_dest', so a missing key is a real wiring
        # bug that should surface, not be silently masked to 0.
        strategy = RouteCompletionBonusStrategy(bonus=100.0)
        with pytest.raises(KeyError):
            strategy.compute(None, None, {})


class TestSteeringSmoothRewardStrategy:
    def test_zero_steering_from_rest_is_zero(self):
        strategy = SteeringSmoothRewardStrategy(weight_magnitude=0.3,
                                                weight_delta=0.5)
        reward = strategy.compute(None, [0.0, 0.0], {})
        assert reward == 0.0

    def test_full_lock_from_rest_is_expensive(self):
        strategy = SteeringSmoothRewardStrategy(weight_magnitude=0.3,
                                                weight_delta=0.5)
        reward = strategy.compute(None, [1.0, 0.0], {})
        assert reward == pytest.approx(-(0.3 * 1.0 + 0.5 * 1.0))

    def test_gentle_cornering_is_cheap(self):
        strategy = SteeringSmoothRewardStrategy(weight_magnitude=0.3,
                                                weight_delta=0.5)
        reward = strategy.compute(None, [0.1, 0.0], {})
        assert reward == pytest.approx(-(0.3 * 0.01 + 0.5 * 0.01))

    def test_delta_measured_against_previous_step(self):
        strategy = SteeringSmoothRewardStrategy(weight_magnitude=0.3,
                                                weight_delta=0.5)
        strategy.compute(None, [0.5, 0.0], {})
        reward = strategy.compute(None, [0.5, 0.0], {})
        # Same steering twice: magnitude penalty stays, delta penalty
        # vanishes since steer - prev_steering == 0.
        assert reward == pytest.approx(-(0.3 * 0.25))

    def test_reset_clears_previous_steering(self):
        strategy = SteeringSmoothRewardStrategy(weight_magnitude=0.3,
                                                weight_delta=0.5)
        strategy.compute(None, [1.0, 0.0], {})
        strategy.reset()
        reward = strategy.compute(None, [1.0, 0.0], {})
        # Same as the "full lock from rest" case above -- prev is 0 again.
        assert reward == pytest.approx(-(0.3 * 1.0 + 0.5 * 1.0))


class TestBrakingRewardStrategy:
    def test_far_obstacle_is_zero(self):
        strategy = BrakingRewardStrategy(weight=2.0, safe_distance=8.0)
        info = {"front_distance": 20.0}
        assert strategy.compute(None, [0.0, -1.0], info) == 0.0

    def test_close_obstacle_without_braking_is_zero(self):
        strategy = BrakingRewardStrategy(weight=2.0, safe_distance=8.0)
        info = {"front_distance": 4.0}
        assert strategy.compute(None, [0.0, 1.0], info) == 0.0

    def test_close_obstacle_with_braking_is_rewarded(self):
        strategy = BrakingRewardStrategy(weight=2.0, safe_distance=8.0)
        info = {"front_distance": 4.0}
        reward = strategy.compute(None, [0.0, -0.5], info)
        proximity = 1.0 - 4.0 / 8.0
        assert reward == pytest.approx(2.0 * proximity * 0.5)

    @pytest.mark.parametrize("safe_distance", [0.0, -8.0])
    def test_non_positive_safe_distance_raises(self, safe_distance):
        with pytest.raises(ValueError):
            BrakingRewardStrategy(safe_distance=safe_distance)


class TestMinSpeedPenaltyStrategy:
    def test_at_or_above_min_speed_is_zero(self):
        strategy = MinSpeedPenaltyStrategy(weight=2.0, min_speed=10.0)
        assert strategy.compute(None, None, {"velocity": 10.0}) == 0.0
        assert strategy.compute(None, None, {"velocity": 15.0}) == 0.0

    def test_standstill_is_full_penalty(self):
        strategy = MinSpeedPenaltyStrategy(weight=2.0, min_speed=10.0)
        assert strategy.compute(None, None, {"velocity": 0.0}) == -2.0

    def test_half_min_speed_is_half_penalty(self):
        strategy = MinSpeedPenaltyStrategy(weight=2.0, min_speed=10.0)
        reward = strategy.compute(None, None, {"velocity": 5.0})
        assert reward == pytest.approx(-1.0)

    @pytest.mark.parametrize("min_speed", [0.0, -10.0])
    def test_non_positive_min_speed_raises(self, min_speed):
        with pytest.raises(ValueError):
            MinSpeedPenaltyStrategy(min_speed=min_speed)


class TestProximityPenaltyStrategy:
    def test_clear_road_is_zero(self):
        strategy = ProximityPenaltyStrategy(weight=0.02, safe_distance=0.3)
        assert strategy.compute(None, None, {"front_distance": 1.0}) == 0.0

    def test_exactly_at_safe_distance_is_zero(self):
        strategy = ProximityPenaltyStrategy(weight=0.02, safe_distance=0.3)
        reward = strategy.compute(None, None, {"front_distance": 0.3})
        assert reward == pytest.approx(0.0)

    def test_contact_is_full_penalty(self):
        strategy = ProximityPenaltyStrategy(weight=0.02, safe_distance=0.3)
        reward = strategy.compute(None, None, {"front_distance": 0.0})
        assert reward == pytest.approx(-0.02)

    def test_halfway_into_safe_distance_is_half_penalty(self):
        strategy = ProximityPenaltyStrategy(weight=0.02, safe_distance=0.3)
        reward = strategy.compute(None, None, {"front_distance": 0.15})
        assert reward == pytest.approx(-0.01)

    @pytest.mark.parametrize("safe_distance", [0.0, -0.3])
    def test_non_positive_safe_distance_raises(self, safe_distance):
        with pytest.raises(ValueError):
            ProximityPenaltyStrategy(safe_distance=safe_distance)


class TestMotionGatedRewardStrategy:
    def test_standstill_gates_reward_to_zero(self):
        inner = ConstantRewardStrategy(value=3.0)
        gated = MotionGatedRewardStrategy(inner, ref_speed=10.0)
        reward = gated.compute(None, None, {"velocity": 0.0})
        assert reward == 0.0

    def test_at_or_above_ref_speed_passes_full_reward(self):
        inner = ConstantRewardStrategy(value=3.0)
        gated = MotionGatedRewardStrategy(inner, ref_speed=10.0)
        assert gated.compute(None, None, {"velocity": 10.0}) == 3.0
        assert gated.compute(None, None, {"velocity": 20.0}) == 3.0

    def test_partial_speed_scales_proportionally(self):
        inner = ConstantRewardStrategy(value=4.0)
        gated = MotionGatedRewardStrategy(inner, ref_speed=10.0)
        reward = gated.compute(None, None, {"velocity": 5.0})
        assert reward == pytest.approx(2.0)

    def test_negative_velocity_does_not_flip_sign(self):
        inner = ConstantRewardStrategy(value=4.0)
        gated = MotionGatedRewardStrategy(inner, ref_speed=10.0)
        reward = gated.compute(None, None, {"velocity": -5.0})
        assert reward == 0.0

    def test_reset_forwards_to_wrapped_strategy(self):
        inner = ConstantRewardStrategy(value=1.0)
        gated = MotionGatedRewardStrategy(inner, ref_speed=10.0)
        gated.reset()
        assert inner.reset_calls == 1

    @pytest.mark.parametrize("ref_speed", [0.0, -10.0])
    def test_non_positive_ref_speed_raises(self, ref_speed):
        with pytest.raises(ValueError):
            MotionGatedRewardStrategy(ConstantRewardStrategy(),
                                      ref_speed=ref_speed)


class TestCompositeRewardStrategy:
    def test_sums_all_children(self):
        composite = CompositeRewardStrategy(strategies=[
            ConstantRewardStrategy(1.0),
            ConstantRewardStrategy(2.5),
            ConstantRewardStrategy(-0.5),
        ])
        assert composite.compute(None, None, {}) == pytest.approx(3.0)

    def test_empty_list_sums_to_zero(self):
        composite = CompositeRewardStrategy(strategies=[])
        assert composite.compute(None, None, {}) == 0.0

    def test_reset_propagates_to_every_child(self):
        children = [ConstantRewardStrategy(1.0) for _ in range(3)]
        composite = CompositeRewardStrategy(strategies=children)
        composite.reset()
        assert all(child.reset_calls == 1 for child in children)

    def test_nan_child_raises_value_error(self):
        composite = CompositeRewardStrategy(strategies=[
            ConstantRewardStrategy(1.0),
            ConstantRewardStrategy(float("nan")),
        ])
        with pytest.raises(ValueError):
            composite.compute(None, None, {})

    def test_inf_child_raises_value_error(self):
        composite = CompositeRewardStrategy(strategies=[
            ConstantRewardStrategy(float("inf")),
        ])
        with pytest.raises(ValueError):
            composite.compute(None, None, {})

    def test_nan_error_names_offending_strategy(self):
        composite = CompositeRewardStrategy(strategies=[
            ConstantRewardStrategy(float("nan")),
        ])
        with pytest.raises(ValueError, match="ConstantRewardStrategy"):
            composite.compute(None, None, {})
