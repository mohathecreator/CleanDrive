from dataclasses import dataclass
from reward import RewardStrategy
from pathlib import Path


@dataclass(frozen=True)
class TrainingConfig:
    map_config: dict = ...
    reward_strategy: RewardStrategy = ...
    learning_rate: float = 3e-4
    total_timesteps: int = 100_000
    batch_size: int = 64
    checkpoint_path: Path = Path("./checkpoints")
    log_path: Path = Path("./logs")
