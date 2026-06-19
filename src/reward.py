from abc import ABC, abstractmethod


class RewardStrategy(ABC):
    @abstractmethod
    def compute(self, observation, action, info) -> float:
        pass


class SpeedRewardStrategy(RewardStrategy):
    def __init__(self, speed_weigth: float = 1.0,
                 crash_penalty: float = -10.0):
        self.speed_weight = speed_weigth
        self.crash_penalty = crash_penalty

    def compute(self, observation, action, info) -> float:
        reward = self.speed_weight * info["velocity"]
        if info["crash"] or info["out_of_road"]:
            reward += self.crash_penalty
        return reward
