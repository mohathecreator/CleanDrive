"""Gymnasium wrapper around MetaDrive that adds lane-frame features.

Beyond the native MetaDrive observation the wrapper appends three lane-frame
channels (lateral offset, lateral velocity and (pi-normalised) heading
error) and delegates the reward to a :class:'reward.RewardStrategy'.
"""

import numpy as np
from gymnasium import Env, spaces
from metadrive.envs.metadrive_env import MetaDriveEnv
from config import TrainingConfig


class EnvironmentWrapper(Env):
    """MetaDrive env with extra lane-frame observations and a plug-in reward.

    The observation is the MetaDrive vector extended by three channels in the
    fixed order '[lateral_offset, lateral_velocity, heading_error]'; both
    :meth:'reset' and :meth:'step' emit that same order and length. Reward
    comes entirely from 'config.reward_strategy'. The native MetaDrive
    reward is discarded.

    Sign convention (validated empirically): a positive 'heading_error'
    means the car is rotated counter-clockwise relative to the lane, i.e.
    steering left produces a positive error.
    """

    def __init__(self, config: TrainingConfig):
        """Build the MetaDrive env and the extended observation space."""
        self.config = config
        self.reward_strategy = config.reward_strategy
        self._metadrive_env = MetaDriveEnv(config.map_config)
        self.action_space = self._metadrive_env.action_space

        base = self._metadrive_env.observation_space
        self.observation_space = spaces.Box(
            low=np.append(base.low, [-5.0, -2.0, -1.0]),
            high=np.append(base.high, [5.0, 2.0, 1.0]),
            dtype=np.float32,
        )
        self._prev_lateral = 0.0

    def _lane_frame(self):
        """Lateral offset and lane-tangent heading error of the agent.

        Returns '(lateral, heading_error)' where 'lateral' is metres
        from the centreline and 'heading_error' is the wrapped angle
        (radians in [-pi, pi]) between the car's heading and the lane
        tangent. A single projection feeds both values.
        """
        vehicle = self._metadrive_env.agent
        lane = vehicle.navigation.current_lane
        long, lateral = lane.local_coordinates(vehicle.position)
        heading_fn = getattr(lane, "heading_theta_at", None) or lane.heading_at
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
        """Append the three lane-frame channels to a base observation.

        The single place that assembles the extended vector, so 'reset'
        and 'step' cannot drift in channel order or normalisation. The
        heading channel is divided by pi to land in [-1, 1].
        """
        heading_norm = heading_error / np.pi
        extended = np.append(obs, [lateral, lateral_velocity, heading_norm])
        return extended.astype(np.float32)

    def reset(self, *, seed=None, options=None):
        """Reset the episode and return the extended initial observation.

        Also resets the reward strategy so stateful terms (e.g. steering
        smoothness) do not leak across episode boundaries.
        """
        obs, info = self._metadrive_env.reset(seed=seed)
        self.reward_strategy.reset()
        lateral, heading_error = self._lane_frame()
        self._prev_lateral = lateral
        info["lateral_offset"] = lateral
        info["lateral_velocity"] = 0.0
        info["heading_error"] = heading_error
        return self._extend_obs(obs, lateral, 0.0, heading_error), info

    def step(self, action):
        """Advance one step; return the extended observation and reward.

        Populates 'info' with the lane-frame features and
        'front_distance' that the reward strategies consume, then
        computes the composite reward from them.
        """
        observation, _, terminated, truncated, info = (
            self._metadrive_env.step(action)
        )

        lateral, heading_error = self._lane_frame()
        lateral_velocity = lateral - self._prev_lateral
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
