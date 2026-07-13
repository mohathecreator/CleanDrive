"""Unit tests for agent.AgentManager.

Uses a tiny synthetic Gymnasium env instead of MetaDrive so these tests
stay fast (no simulator, no physics, no rendering) -- AgentManager only
depends on the standard Gym API, not on MetaDrive specifics.
"""

import gymnasium as gym
import numpy as np
import pytest

from agent import AgentManager


class TinyEnv(gym.Env):
    """Minimal random-walk env: just enough surface for PPO to train on."""

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self._steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._steps = 0
        return self.observation_space.sample(), {}

    def step(self, action):
        self._steps += 1
        terminated = self._steps >= 20
        return self.observation_space.sample(), 0.0, terminated, False, {}


@pytest.fixture
def tiny_env():
    return TinyEnv()


class TestCreateModel:
    def test_builds_a_model(self, tiny_env):
        manager = AgentManager()
        manager.create_model(tiny_env)
        assert manager.model is not None

    def test_use_sde_flag_is_applied(self, tiny_env):
        manager = AgentManager()
        manager.create_model(tiny_env, use_sde=False)
        assert manager.model.use_sde is False

    def test_use_sde_defaults_to_true(self, tiny_env):
        manager = AgentManager()
        manager.create_model(tiny_env)
        assert manager.model.use_sde is True

    def test_learning_rate_is_applied(self, tiny_env):
        manager = AgentManager()
        manager.create_model(tiny_env, learning_rate=1e-4)
        assert manager.model.learning_rate == pytest.approx(1e-4)


class TestTrain:
    def test_train_before_create_or_load_raises(self):
        manager = AgentManager()
        with pytest.raises(RuntimeError):
            manager.train(total_timesteps=512)

    def test_train_runs_one_rollout(self, tiny_env):
        manager = AgentManager()
        manager.create_model(tiny_env, use_sde=False)
        result = manager.train(total_timesteps=512)
        assert result is manager.model


class TestSaveLoad:
    def test_save_before_create_or_load_raises(self, tmp_path):
        manager = AgentManager()
        with pytest.raises(RuntimeError):
            manager.save(tmp_path / "ppo_model")

    def test_round_trip_preserves_predictable_model(
        self, tiny_env, tmp_path
    ):
        manager = AgentManager()
        manager.create_model(tiny_env, use_sde=False)
        save_path = tmp_path / "ppo_model"
        manager.save(save_path)

        loaded = AgentManager()
        loaded.load(save_path, env=tiny_env)

        obs, _ = tiny_env.reset()
        action, _ = loaded.model.predict(obs, deterministic=True)
        assert action.shape == tiny_env.action_space.shape

    def test_load_applies_learning_rate_override(self, tiny_env, tmp_path):
        manager = AgentManager()
        manager.create_model(tiny_env, learning_rate=3e-4, use_sde=False)
        save_path = tmp_path / "ppo_model"
        manager.save(save_path)

        loaded = AgentManager()
        loaded.load(save_path, env=tiny_env, learning_rate=1e-5)
        assert loaded.model.learning_rate == pytest.approx(1e-5)
