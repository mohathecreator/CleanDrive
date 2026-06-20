from gymnasium import Env
from metadrive.envs.metadrive_env import MetaDriveEnv
from config import TrainingConfig


class EnvironmentWrapper(Env):
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.reward_strategy = config.reward_strategy
        self._metadrive_env = MetaDriveEnv(config.map_config)
        self.action_space = self._metadrive_env.action_space
        self.observation_space = self._metadrive_env.observation_space

    def reset(self, *, seed=None, options=None):
        return self._metadrive_env.reset(seed=seed, options=options)

    def step(self, action):
        observation, metadrive_reward, terminated, truncated, info = (
            self._metadrive_env.step(action)
        )
        reward = self.reward_strategy.compute(observation, action, info)
        return observation, reward, terminated, truncated, info

    def close(self):
        return self._metadrive_env.close()
