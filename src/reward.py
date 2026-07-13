"""Reward strategies for the driving agent (Strategy pattern).

Each strategy maps a transition (observation, action, info) to a reward;
CompositeRewardStrategy sums several into the full reward used in main.py.
"""

import math
from abc import ABC, abstractmethod


class RewardStrategy(ABC):
    """Abstract base for one reward term."""

    @abstractmethod
    def compute(self, observation, action, info) -> float:
        """Return this term's reward for one transition."""

    def reset(self):
        """No-op default; stateful strategies override this."""
        pass


class SpeedRewardStrategy(RewardStrategy):
    """Reward proportional to forward speed."""

    def __init__(self, weight: float = 1.0, max_speed: float = 40.0):
        if max_speed <= 0:
            raise ValueError(f"max_speed must be positive. "
                             f"Current value: {max_speed}")
        self.weight = weight        # max reward at max_speed
        self.max_speed = max_speed  # speed (m/s) for full reward

    def compute(self, observation, action, info) -> float:
        return self.weight * (info["velocity"] / self.max_speed)


class ProgressRewardStrategy(RewardStrategy):
    """Reward forward progress along the route (the primary objective)."""

    def __init__(self, weight: float = 200.0):
        self.weight = weight  # reward earned over a full route (0->1)
        self._prev = None     # last step's route_completion

    def compute(self, observation, action, info) -> float:
        completion = info["route_completion"]
        if self._prev is None:
            self._prev = completion
            return 0.0
        delta = completion - self._prev  # progress made this step
        self._prev = completion
        return self.weight * delta

    def reset(self):
        self._prev = None


class LaneCenteringRewardStrategy(RewardStrategy):
    """Reward for staying near the lane centre."""

    def __init__(self, weight: float = 2.0, half_lane_width: float = 0.6):
        if half_lane_width <= 0:
            raise ValueError(f"half_lane_width must be positive. "
                             f"Current value: {half_lane_width}")
        self.weight = weight                    # max reward at centre
        self.half_lane_width = half_lane_width  # metres to zero reward

    def compute(self, observation, action, info) -> float:
        center = max(
            0.0, 1.0 - abs(info["lateral_offset"]) / self.half_lane_width
        )  # 1.0 at centre, 0.0 at half_lane_width
        return self.weight * center


class HeadingAlignmentRewardStrategy(RewardStrategy):
    """Reward for pointing along the lane tangent."""

    def __init__(self, weight: float = 2.0, max_error: float = 0.5):
        if max_error <= 0:
            raise ValueError(f"max_error must be positive. "
                             f"Current value: {max_error}")
        self.weight = weight        # max reward when aligned
        self.max_error = max_error  # radians to zero reward

    def compute(self, observation, action, info) -> float:
        align = max(0.0, 1.0 - abs(info["heading_error"]) / self.max_error)
        return self.weight * align


class AvoidCollisionRewardStrategy(RewardStrategy):
    """Terminal penalty for crashing or leaving the road."""

    def __init__(self, crash_penalty: float = -10.0):
        self.crash_penalty = crash_penalty  # reward on crash/out_of_road

    def compute(self, observation, action, info) -> float:
        if info["crash"] or info["out_of_road"]:
            return self.crash_penalty
        return 0.0


class RouteCompletionBonusStrategy(RewardStrategy):
    """One-off bonus for reaching the destination."""

    def __init__(self, bonus: float = 100.0):
        self.bonus = bonus  # reward on arrive_dest

    def compute(self, observation, action, info) -> float:
        if info["arrive_dest"]:
            return self.bonus
        return 0.0


class SteeringSmoothRewardStrategy(RewardStrategy):
    """Penalty for jerky steering (anti-oscillation)."""

    def __init__(self, weight_magnitude: float = 0.3,
                 weight_delta: float = 0.5):
        self.weight_magnitude = weight_magnitude  # penalty on |steer|
        self.weight_delta = weight_delta          # penalty on steer change
        self._prev_steering = 0.0                 # last step's steer

    def compute(self, observation, action, info) -> float:
        steer = action[0]
        magnitude = steer ** 2                      # sharp-lock penalty
        delta = (steer - self._prev_steering) ** 2  # jerk penalty
        self._prev_steering = steer
        return -(self.weight_magnitude * magnitude
                 + self.weight_delta * delta)

    def reset(self):
        self._prev_steering = 0.0


class BrakingRewardStrategy(RewardStrategy):
    """Reward braking when an obstacle is close ahead. Unused currently."""

    def __init__(self, weight: float = 2.0, safe_distance: float = 8.0):
        if safe_distance <= 0:
            raise ValueError(f"safe_distance must be positive. "
                             f"Current value: {safe_distance}")
        self.weight = weight                # max reward at full brake
        self.safe_distance = safe_distance  # distance where braking starts

    def compute(self, observation, action, info) -> float:
        front_dist = info["front_distance"]
        if front_dist >= self.safe_distance:
            return 0.0
        proximity = 1.0 - (front_dist / self.safe_distance)  # 0..1 closeness
        if action[1] < 0:  # braking (negative throttle)
            return self.weight * proximity * abs(action[1])
        return 0.0


class MinSpeedPenaltyStrategy(RewardStrategy):
    """Penalty for creeping below a minimum speed."""

    def __init__(self, weight: float = 2.0, min_speed: float = 10.0):
        if min_speed <= 0:
            raise ValueError(f"min_speed must be positive. "
                             f"Current value: {min_speed}")
        self.weight = weight        # max penalty at standstill
        self.min_speed = min_speed  # speed (m/s) above which penalty is 0

    def compute(self, observation, action, info) -> float:
        velocity = info["velocity"]
        if velocity >= self.min_speed:
            return 0.0
        return -self.weight * (1 - velocity / self.min_speed)


class ProximityPenaltyStrategy(RewardStrategy):
    """Continuous penalty for a close obstacle ahead."""

    def __init__(self, weight: float = 0.02, safe_distance: float = 0.3):
        if safe_distance <= 0:
            raise ValueError(f"safe_distance must be positive. "
                             f"Current value: {safe_distance}")
        self.weight = weight                # max penalty at contact
        self.safe_distance = safe_distance  # lidar fraction where ramp starts

    def compute(self, observation, action, info) -> float:
        front = info["front_distance"]  # normalised lidar, 1.0 = clear
        if front >= self.safe_distance:
            return 0.0
        proximity = 1.0 - front / self.safe_distance  # 0..1 closeness
        return -self.weight * proximity


class MotionGatedRewardStrategy(RewardStrategy):
    """Scales another strategy's reward by how fast the car is moving."""

    def __init__(self, strategy: RewardStrategy, ref_speed: float = 10.0):
        if ref_speed <= 0:
            raise ValueError(f"ref_speed must be positive. "
                             f"Current value: {ref_speed}")
        self.strategy = strategy    # wrapped reward term
        self.ref_speed = ref_speed  # speed (m/s) for full reward

    def compute(self, observation, action, info) -> float:
        factor = min(1.0, max(0.0, info["velocity"] / self.ref_speed))
        return factor * self.strategy.compute(observation, action, info)

    def reset(self):
        self.strategy.reset()


class CompositeRewardStrategy(RewardStrategy):
    """Sum of several reward strategies (Composite pattern)."""

    def __init__(self, strategies: list[RewardStrategy]):
        self.strategies = strategies  # reward terms to sum

    def compute(self, observation, action, info) -> float:
        total = 0.0
        for strategy in self.strategies:
            value = strategy.compute(observation, action, info)
            if not math.isfinite(value):  # catch NaN/inf before PPO sees it
                raise ValueError(
                    f"{type(strategy).__name__}.compute() returned a "
                    f"non-finite reward ({value!r}) -- check for NaN/inf "
                    f"in the info dict this step."
                )
            total += value
        return total

    def reset(self):
        for strategy in self.strategies:
            strategy.reset()
