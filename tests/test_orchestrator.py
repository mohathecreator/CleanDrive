"""Tests for orchestrator.TrainingOrchestrator.

Scope note: TrainingOrchestrator.__init__ builds 24 real parallel MetaDrive
subprocess environments as soon as validation passes -- that's the actual
training workload, not something to spin up in a unit test (it takes real
wall-clock time and spawns OS processes). What *is* cheap and worth
protecting here is that config validation happens first and fails fast on
a genuinely invalid setting, before any environment is built. (A missing
checkpoint/log directory is no longer such a case: TrainingConfig.validate
creates it automatically -- see test_config.py -- so it is not tested
here, since exercising that path through the orchestrator would require
actually building the environments.)
"""

import pytest

from config import TrainingConfig
from orchestrator import TrainingOrchestrator
from reward import SpeedRewardStrategy


def test_invalid_hyperparameter_fails_before_building_environments(
    checkpoint_and_log_dirs,
):
    checkpoint_path, log_path = checkpoint_and_log_dirs
    config = TrainingConfig(
        map_config={},
        reward_strategy=SpeedRewardStrategy(),
        learning_rate=-1.0,
        checkpoint_path=checkpoint_path,
        log_path=log_path,
    )
    with pytest.raises(ValueError):
        TrainingOrchestrator(config, num_envs=1)
