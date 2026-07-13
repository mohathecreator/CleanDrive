"""Immutable training configuration and its validation."""

from dataclasses import dataclass
from reward import RewardStrategy
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    """All settings for one training run (frozen, i.e. read-only).

    Attributes:
        map_config: MetaDrive environment configuration dict.
        reward_strategy: The composed reward function to optimise.
        learning_rate: PPO optimiser step size.
        total_timesteps: Number of environment steps to train for.
        batch_size: PPO minibatch size.
        checkpoint_path: Directory for model and normaliser checkpoints.
        log_path: Directory for TensorBoard logs.
    """

    map_config: dict
    reward_strategy: RewardStrategy
    learning_rate: float = 3e-4
    total_timesteps: int = 100_000
    batch_size: int = 64
    checkpoint_path: Path = Path("./checkpoints")
    log_path: Path = Path("./logs")

    def validate(self):
        """Fail fast on invalid settings or missing directories.

        Raises ValueError for out-of-range hyper-parameters and
        FileNotFoundError if the checkpoint or log directory is
        missing (they must exist before a run starts).
        """
        if not (0 < self.learning_rate < 1):
            raise ValueError(f"The learning_rate value should be between "
                             f"0 and 1. "
                             f"Current Value: {self.learning_rate}")

        if self.total_timesteps <= 0:
            raise ValueError(f"The total_timesteps value should be a positive "
                             f"integer bigger than 0. "
                             f"Current Value: {self.total_timesteps}")

        if self.batch_size <= 0:
            raise ValueError(f"The batch_size value should be a positive "
                             f"integer bigger than 0. "
                             f"Current Value: {self.batch_size}")

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"The current checkpoints path "
                                    f"doesn't exist. "
                                    f"Check if the path is correct "
                                    f"or has been created. "
                                    f"Current path: {self.checkpoint_path}")

        if not self.log_path.exists():
            raise FileNotFoundError(f"The current logs path doesn't exist. "
                                    f"Check if the path is correct "
                                    f"or has been created. "
                                    f"Current path: {self.log_path}")
