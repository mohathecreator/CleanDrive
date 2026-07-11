from abc import ABC, abstractmethod


class RewardStrategy(ABC):
    @abstractmethod
    def compute(self, observation, action, info) -> float:
        pass


class SpeedRewardStrategy(RewardStrategy):
    def __init__(self, weight: float = 1.0, max_speed: float = 40.0):
        self.weight = weight
        self.max_speed = max_speed

    def compute(self, observation, action, info) -> float:
        return self.weight * (info["velocity"] / self.max_speed)


class LaneCenteringRewardStrategy(RewardStrategy):
    def __init__(self, weight: float = 2.0, half_lane_width: float = 0.6):
        self.weight = weight
        self.half_lane_width = half_lane_width

    def compute(self, observation, action, info) -> float:
        center = max(0.0, 1.0 - abs(info["lateral_offset"]) / self.half_lane_width)
        return self.weight * center


class AvoidCollisionRewardStrategy(RewardStrategy):
    def __init__(self, crash_penalty: float = -10.0):
        self.crash_penalty = crash_penalty

    def compute(self, observation, action, info) -> float:
        if info["crash"] or info["out_of_road"]:
            return self.crash_penalty
        return 0.0


class SteeringSmoothRewardStrategy(RewardStrategy):
    def __init__(self, weight: float = 0.5):
        self.weight = weight
        self._prev_steering = 0.0

    def compute(self, observation, action, info) -> float:
        delta = abs(action[0] - self._prev_steering)
        self._prev_steering = action[0]
        return -self.weight * delta


class BrakingRewardStrategy(RewardStrategy):
    def __init__(self, weight: float = 2.0, safe_distance: float = 8.0):
        self.weight = weight
        self.safe_distance = safe_distance

    def compute(self, observation, action, info) -> float:
        front_dist = info["front_distance"]
        if front_dist >= self.safe_distance:
            return 0.0
        proximity = 1.0 - (front_dist / self.safe_distance)
        if action[1] < 0:
            return self.weight * proximity * abs(action[1])
        return 0.0


class MinSpeedPenaltyStrategy(RewardStrategy):
    def __init__(self, weight: float = 2.0, min_speed: float = 10.0):
        self.weight = weight
        self.min_speed = min_speed

    def compute(self, observation, action, info) -> float:
        velocity = info["velocity"]
        if velocity >= self.min_speed:
            return 0.0
        return -self.weight * (1 - velocity / self.min_speed)


class CompositeRewardStrategy(RewardStrategy):
    def __init__(self, strategies: list[RewardStrategy]):
        self.strategies = strategies

    def compute(self, observation, action, info) -> float:
        return sum(s.compute(observation, action, info)
                   for s in self.strategies)
