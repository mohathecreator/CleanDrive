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
        use_sde: Whether PPO uses gSDE (state-dependent exploration).
        checkpoint_path: Directory for model and normaliser checkpoints.
        log_path: Directory for TensorBoard logs.
    """

    map_config: dict
    reward_strategy: RewardStrategy
    learning_rate: float = 3e-4
    total_timesteps: int = 100_000
    batch_size: int = 64
    use_sde: bool = True
    checkpoint_path: Path = Path("./checkpoints")
    log_path: Path = Path("./logs")

    def validate(self):
        """Fail fast on invalid settings; create missing directories.

        Every field gets the same two-tier check: wrong *type* raises
        TypeError, wrong *value* raises ValueError. This also covers
        fields that previously had no validation at all (``map_config``,
        ``reward_strategy``), which used to fail much later with a
        confusing, unrelated error deep inside training.

        ``checkpoint_path``/``log_path`` are created (including any
        missing parent directories) if they don't already exist, so a
        run never fails just because someone forgot to ``mkdir`` first.
        """
        if not isinstance(self.map_config, dict):
            raise TypeError(f"map_config must be a dict. "
                            f"Current type: "
                            f"{type(self.map_config).__name__}")

        if not isinstance(self.reward_strategy, RewardStrategy):
            raise TypeError(f"reward_strategy must be a RewardStrategy. "
                            f"Current type: "
                            f"{type(self.reward_strategy).__name__}")

        if not isinstance(self.learning_rate, (int, float)):
            raise TypeError(f"learning_rate must be a number. "
                            f"Current type: "
                            f"{type(self.learning_rate).__name__}")
        if not (0 < self.learning_rate < 1):
            raise ValueError(f"The learning_rate value should be between "
                             f"0 and 1. "
                             f"Current Value: {self.learning_rate}")

        if not isinstance(self.total_timesteps, int):
            raise TypeError(f"total_timesteps must be an int. "
                            f"Current type: "
                            f"{type(self.total_timesteps).__name__}")
        if self.total_timesteps <= 0:
            raise ValueError(f"The total_timesteps value should be a positive "
                             f"integer bigger than 0. "
                             f"Current Value: {self.total_timesteps}")

        if not isinstance(self.batch_size, int):
            raise TypeError(f"batch_size must be an int. "
                            f"Current type: "
                            f"{type(self.batch_size).__name__}")
        if self.batch_size <= 0:
            raise ValueError(f"The batch_size value should be a positive "
                             f"integer bigger than 0. "
                             f"Current Value: {self.batch_size}")

        if not isinstance(self.checkpoint_path, Path):
            raise TypeError(f"checkpoint_path must be a pathlib.Path. "
                            f"Current type: "
                            f"{type(self.checkpoint_path).__name__}")
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        if not isinstance(self.log_path, Path):
            raise TypeError(f"log_path must be a pathlib.Path. "
                            f"Current type: "
                            f"{type(self.log_path).__name__}")
        self.log_path.mkdir(parents=True, exist_ok=True)
