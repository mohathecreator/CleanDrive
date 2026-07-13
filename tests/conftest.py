"""Shared pytest fixtures for the CleanDrive test suite."""

import pytest


@pytest.fixture
def checkpoint_and_log_dirs(tmp_path):
    """A pair of existing empty directories for checkpoint_path/log_path.

    TrainingConfig.validate() requires both to exist before a run starts,
    so tests that need a *valid* config use this instead of the real
    ./checkpoints and ./logs directories.
    """
    checkpoint_path = tmp_path / "checkpoints"
    log_path = tmp_path / "logs"
    checkpoint_path.mkdir()
    log_path.mkdir()
    return checkpoint_path, log_path
