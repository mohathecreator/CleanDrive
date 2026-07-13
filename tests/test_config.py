"""Unit tests for config.TrainingConfig.validate()."""

import dataclasses

import pytest

from config import TrainingConfig
from reward import SpeedRewardStrategy


def make_config(checkpoint_and_log_dirs, **overrides):
    """Build a TrainingConfig with sane defaults, pointed at real tmp dirs."""
    checkpoint_path, log_path = checkpoint_and_log_dirs
    defaults = dict(
        map_config={},
        reward_strategy=SpeedRewardStrategy(),
        checkpoint_path=checkpoint_path,
        log_path=log_path,
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)


class TestValidate:
    def test_valid_config_does_not_raise(self, checkpoint_and_log_dirs):
        config = make_config(checkpoint_and_log_dirs)
        config.validate()  # should not raise

    @pytest.mark.parametrize("learning_rate", [0.0, -0.1, 1.0, 1.5])
    def test_learning_rate_out_of_range_raises(
        self, checkpoint_and_log_dirs, learning_rate
    ):
        config = make_config(checkpoint_and_log_dirs,
                             learning_rate=learning_rate)
        with pytest.raises(ValueError):
            config.validate()

    @pytest.mark.parametrize("total_timesteps", [0, -100])
    def test_non_positive_total_timesteps_raises(
        self, checkpoint_and_log_dirs, total_timesteps
    ):
        config = make_config(checkpoint_and_log_dirs,
                             total_timesteps=total_timesteps)
        with pytest.raises(ValueError):
            config.validate()

    @pytest.mark.parametrize("batch_size", [0, -1])
    def test_non_positive_batch_size_raises(
        self, checkpoint_and_log_dirs, batch_size
    ):
        config = make_config(checkpoint_and_log_dirs, batch_size=batch_size)
        with pytest.raises(ValueError):
            config.validate()

    def test_missing_checkpoint_path_gets_created(
        self, checkpoint_and_log_dirs, tmp_path
    ):
        missing = tmp_path / "does_not_exist"
        config = make_config(checkpoint_and_log_dirs,
                             checkpoint_path=missing)
        assert not missing.exists()
        config.validate()
        assert missing.is_dir()

    def test_missing_log_path_gets_created(
        self, checkpoint_and_log_dirs, tmp_path
    ):
        missing = tmp_path / "does_not_exist"
        config = make_config(checkpoint_and_log_dirs, log_path=missing)
        assert not missing.exists()
        config.validate()
        assert missing.is_dir()

    def test_missing_nested_path_creates_parents_too(
        self, checkpoint_and_log_dirs, tmp_path
    ):
        nested = tmp_path / "a" / "b" / "checkpoints"
        config = make_config(checkpoint_and_log_dirs,
                             checkpoint_path=nested)
        config.validate()
        assert nested.is_dir()

    def test_existing_path_is_left_untouched(
        self, checkpoint_and_log_dirs
    ):
        checkpoint_path, _ = checkpoint_and_log_dirs
        marker = checkpoint_path / "already_here.txt"
        marker.write_text("keep me")
        config = make_config(checkpoint_and_log_dirs)
        config.validate()
        assert marker.read_text() == "keep me"


class TestTypeValidation:
    """validate() must raise a clear TypeError for every wrong-typed
    field, not fail later with an unrelated/cryptic error (or, worse,
    silently pass through -- e.g. map_config/reward_strategy previously
    had no validation at all)."""

    def test_map_config_wrong_type_raises_type_error(
        self, checkpoint_and_log_dirs
    ):
        config = make_config(checkpoint_and_log_dirs, map_config=None)
        with pytest.raises(TypeError):
            config.validate()

    def test_reward_strategy_wrong_type_raises_type_error(
        self, checkpoint_and_log_dirs
    ):
        config = make_config(checkpoint_and_log_dirs, reward_strategy=None)
        with pytest.raises(TypeError):
            config.validate()

    def test_reward_strategy_plain_object_raises_type_error(
        self, checkpoint_and_log_dirs
    ):
        config = make_config(checkpoint_and_log_dirs,
                             reward_strategy="not a strategy")
        with pytest.raises(TypeError):
            config.validate()

    def test_learning_rate_wrong_type_raises_type_error(
        self, checkpoint_and_log_dirs
    ):
        config = make_config(checkpoint_and_log_dirs,
                             learning_rate="0.001")
        with pytest.raises(TypeError):
            config.validate()

    def test_total_timesteps_wrong_type_raises_type_error(
        self, checkpoint_and_log_dirs
    ):
        config = make_config(checkpoint_and_log_dirs,
                             total_timesteps=1.5)
        with pytest.raises(TypeError):
            config.validate()

    def test_batch_size_wrong_type_raises_type_error(
        self, checkpoint_and_log_dirs
    ):
        config = make_config(checkpoint_and_log_dirs, batch_size="64")
        with pytest.raises(TypeError):
            config.validate()

    def test_checkpoint_path_plain_string_raises_type_error(
        self, checkpoint_and_log_dirs
    ):
        checkpoint_path, _ = checkpoint_and_log_dirs
        config = make_config(checkpoint_and_log_dirs,
                             checkpoint_path=str(checkpoint_path))
        with pytest.raises(TypeError):
            config.validate()

    def test_log_path_plain_string_raises_type_error(
        self, checkpoint_and_log_dirs
    ):
        _, log_path = checkpoint_and_log_dirs
        config = make_config(checkpoint_and_log_dirs,
                             log_path=str(log_path))
        with pytest.raises(TypeError):
            config.validate()


class TestImmutability:
    def test_fields_cannot_be_reassigned(self, checkpoint_and_log_dirs):
        config = make_config(checkpoint_and_log_dirs)
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.learning_rate = 1e-3
