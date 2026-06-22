from abc import ABC, abstractmethod


class RewardStrategy(ABC):
    @abstractmethod
    def compute(self, observation, action, info) -> float:
        pass


class SpeedRewardStrategy(RewardStrategy):
    def __init__(self, speed_weigth: float = 10):
        self.speed_weight = speed_weigth

    def compute(self, observation, action, info) -> float:
        reward = self.speed_weight * info["velocity"]

        return reward


class LaneCenteringRewardStrategy(RewardStrategy):
    def __init__(self, centering_weight: float = 1.0):
        self.centering_weight = centering_weight

    def compute(self, observation, action, info) -> float:
        reward = -self.centering_weight * abs(info["lateral_offset"])

        return reward


class AvoidCollisionRewardStrategy(RewardStrategy):
    def __init__(self, crash_penalty=-10.0):
        self.crash_penalty = crash_penalty

    def compute(self, observation, action, info) -> float:
        if info["crash"] or info["out_of_road"]:
            return self.crash_penalty

        return 0.0


class BrakingRewardStrategy(RewardStrategy):
    def __init__(self, braking_weight=1):
        self.braking_weight = braking_weight

    def compute(self, observation, action, info) -> float:
        distance_threshold = 0.4
        if info["front_distance"] < distance_threshold:
            return -self.braking_weight * action[1]

        return 0.0
