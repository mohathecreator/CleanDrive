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
        return self._metadrive_env.reset(seed=seed)

    def step(self, action):
        observation, metadrive_reward, terminated, truncated, info = (
            self._metadrive_env.step(action)
        )

        vehicle = self._metadrive_env.agent
        _, lateral = (
            vehicle.navigation.current_lane.local_coordinates(vehicle.position)
        )
        info["lateral_offset"] = lateral

        lidar = self._metadrive_env.engine.get_sensor("lidar")
        cloud_points, _ = lidar.perceive(
            vehicle,
            physics_world=self._metadrive_env.engine.physics_world.dynamic_world,
            num_lasers=vehicle.config["lidar"]["num_lasers"],
            distance=vehicle.config["lidar"]["distance"],
        )
        front_window = 5
        num_lasers = len(cloud_points)
        front_rays = [
            cloud_points[i % num_lasers]
            for i in range(-front_window, front_window + 1)
        ]
        info["front_distance"] = min(front_rays)

        reward = self.reward_strategy.compute(observation, action, info)

        return observation, reward, terminated, truncated, info

    def close(self):
        return self._metadrive_env.close()
