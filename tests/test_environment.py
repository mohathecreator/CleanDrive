"""Tests for environment.EnvironmentWrapper.

_extend_obs is pure (touches no instance state), so it's tested directly
without building a MetaDrive environment. The remaining tests build a
real, minimal MetaDriveEnv (fastest possible map/config) and are marked
``integration`` since they're slower and need MetaDrive's assets.
"""

import math

import numpy as np
import pytest

from config import TrainingConfig
from environment import EnvironmentWrapper
from reward import SpeedRewardStrategy


class TestExtendObs:
    """_extend_obs never reads self, so pass None and call it directly."""

    def test_appends_three_channels_in_order(self):
        obs = np.array([1.0, 2.0], dtype=np.float32)
        extended = EnvironmentWrapper._extend_obs(
            None, obs, lateral=0.5, lateral_velocity=-0.2,
            heading_error=0.0,
        )
        assert extended.shape == (5,)
        np.testing.assert_allclose(extended[:2], [1.0, 2.0])
        assert extended[2] == pytest.approx(0.5)
        assert extended[3] == pytest.approx(-0.2)
        assert extended[4] == pytest.approx(0.0)

    def test_heading_channel_is_pi_normalised(self):
        obs = np.array([], dtype=np.float32)
        extended = EnvironmentWrapper._extend_obs(
            None, obs, lateral=0.0, lateral_velocity=0.0,
            heading_error=math.pi / 2,
        )
        assert extended[-1] == pytest.approx(0.5)

    def test_output_dtype_is_float32(self):
        obs = np.array([1.0], dtype=np.float64)
        extended = EnvironmentWrapper._extend_obs(
            None, obs, lateral=0.0, lateral_velocity=0.0,
            heading_error=0.0,
        )
        assert extended.dtype == np.float32


@pytest.fixture
def minimal_env():
    """A real, fast EnvironmentWrapper: one straight-road scenario."""
    map_config = dict(
        use_render=False, manual_control=False,
        num_scenarios=1, map="S", horizon=100, traffic_density=0.0,
        vehicle_config=dict(lidar=dict(num_lasers=120, distance=50)),
    )
    config = TrainingConfig(
        map_config=map_config,
        reward_strategy=SpeedRewardStrategy(),
    )
    env = EnvironmentWrapper(config)
    yield env
    env.close()


@pytest.mark.integration
class TestEnvironmentWrapperIntegration:
    def test_observation_space_extends_base_by_three_channels(
        self, minimal_env
    ):
        base_dim = minimal_env._metadrive_env.observation_space.shape[0]
        assert minimal_env.observation_space.shape == (base_dim + 3,)

    def test_reset_returns_obs_matching_observation_space(self, minimal_env):
        obs, info = minimal_env.reset()
        assert obs.shape == minimal_env.observation_space.shape
        assert obs.dtype == np.float32

    def test_reset_populates_lane_frame_info_with_zero_velocity(
        self, minimal_env
    ):
        _, info = minimal_env.reset()
        assert "lateral_offset" in info
        assert "heading_error" in info
        assert info["lateral_velocity"] == 0.0

    def test_step_returns_obs_matching_observation_space(self, minimal_env):
        minimal_env.reset()
        action = minimal_env.action_space.sample() * 0.0
        obs, reward, terminated, truncated, info = minimal_env.step(action)
        assert obs.shape == minimal_env.observation_space.shape
        assert isinstance(reward, float)

    def test_step_populates_front_distance(self, minimal_env):
        minimal_env.reset()
        action = minimal_env.action_space.sample() * 0.0
        _, _, _, _, info = minimal_env.step(action)
        assert "front_distance" in info
        assert 0.0 <= info["front_distance"] <= 1.0

    def test_lateral_velocity_is_delta_of_consecutive_lateral_offsets(
        self, minimal_env
    ):
        minimal_env.reset()
        action = minimal_env.action_space.sample() * 0.0
        _, _, _, _, info_1 = minimal_env.step(action)
        lateral_1 = info_1["lateral_offset"]
        _, _, _, _, info_2 = minimal_env.step(action)
        expected_velocity = info_2["lateral_offset"] - lateral_1
        assert info_2["lateral_velocity"] == pytest.approx(
            expected_velocity, abs=1e-6
        )
