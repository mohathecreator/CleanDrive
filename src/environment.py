"""Gymnasium wrapper around MetaDrive that adds lane-frame features.

Beyond the native MetaDrive observation the wrapper appends three lane-frame
channels (lateral offset, lateral velocity and pi-normalised heading error)
and delegates the reward to a :class:`reward.RewardStrategy`.
"""

import numpy as np
from gymnasium import Env, spaces
from metadrive.envs.metadrive_env import MetaDriveEnv
from config import TrainingConfig


class EnvironmentWrapper(Env):
    """MetaDrive env extended with lane-frame observations and a
    plug-in reward strategy."""

    def __init__(self, config: TrainingConfig):
        """Build the MetaDrive env and the extended observation space."""
        self.config = config
        self.reward_strategy = config.reward_strategy
        self._metadrive_env = MetaDriveEnv(config.map_config)
        self.action_space = self._metadrive_env.action_space

        base = self._metadrive_env.observation_space
        self.observation_space = spaces.Box(
            # extra bounds for [lateral, lateral_velocity, heading_error]
            low=np.append(base.low, [-5.0, -2.0, -1.0]),
            high=np.append(base.high, [5.0, 2.0, 1.0]),
            dtype=np.float32,
        )
        self._prev_lateral = 0.0  # last step's lateral offset

    def _lane_frame(self):
        """Lateral offset and heading error relative to the lane tangent."""
        vehicle = self._metadrive_env.agent
        lane = vehicle.navigation.current_lane
        long, lateral = lane.local_coordinates(vehicle.position)
        heading_fn = getattr(lane, "heading_theta_at", None) or lane.heading_at
        # positive = car rotated left of the lane tangent (steering left)
        heading_error = vehicle.heading_theta - heading_fn(long)
        heading_error = float(
            np.arctan2(np.sin(heading_error), np.cos(heading_error))
        )
        return float(lateral), heading_error

    def _front_distance(self):
        """Closest lidar return within the frontal ray window."""
        vehicle = self._metadrive_env.agent
        lidar = self._metadrive_env.engine.get_sensor("lidar")
        cloud_points, _ = lidar.perceive(
            vehicle,
            physics_world=(
                self._metadrive_env.engine.physics_world.dynamic_world
            ),
            num_lasers=vehicle.config["lidar"]["num_lasers"],
            distance=vehicle.config["lidar"]["distance"],
        )
        # Narrow frontal cone (+/-5 of 120 lasers ~= +/-15deg) so the
        # proximity signal reacts to cars *in the path* and not to cars in
        # the adjacent lane that the agent can simply pass.
        front_window = 5
        num_lasers = len(cloud_points)
        front_rays = [
            cloud_points[i % num_lasers]
            for i in range(-front_window, front_window + 1)
        ]
        return min(front_rays)

    def _extend_obs(self, obs, lateral, lateral_velocity, heading_error):
        """Append the three lane-frame channels to a base observation."""
        heading_norm = heading_error / np.pi  # normalise to [-1, 1]
        extended = np.append(obs, [lateral, lateral_velocity, heading_norm])
        return extended.astype(np.float32)

    def reset(self, *, seed=None, options=None):
        """Reset the episode; return the extended initial observation."""
        obs, info = self._metadrive_env.reset(seed=seed)
        self.reward_strategy.reset()
        lateral, heading_error = self._lane_frame()
        self._prev_lateral = lateral
        info["lateral_offset"] = lateral
        info["lateral_velocity"] = 0.0
        info["heading_error"] = heading_error
        return self._extend_obs(obs, lateral, 0.0, heading_error), info

    def step(self, action):
        """Advance one step; return the extended observation and reward."""
        observation, _, terminated, truncated, info = (
            self._metadrive_env.step(action)
        )

        lateral, heading_error = self._lane_frame()
        lateral_velocity = lateral - self._prev_lateral  # per-step delta
        self._prev_lateral = lateral
        info["lateral_offset"] = lateral
        info["lateral_velocity"] = lateral_velocity
        info["heading_error"] = heading_error
        info["front_distance"] = self._front_distance()

        reward = self.reward_strategy.compute(observation, action, info)
        extended_obs = self._extend_obs(
            observation, lateral, lateral_velocity, heading_error
        )
        return extended_obs, reward, terminated, truncated, info

    def close(self):
        """Close the underlying MetaDrive environment."""
        return self._metadrive_env.close()
