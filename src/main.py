"""Entry point: assembles the config and launches a training run.

Defines the map, reward, and hyperparameters, then hands everything to
TrainingOrchestrator. Run with ``python src/main.py``. See HANDOFF.md for
the rationale behind the current settings.
"""

from pathlib import Path
from config import TrainingConfig
from reward import (ProgressRewardStrategy, HeadingAlignmentRewardStrategy,
                    LaneCenteringRewardStrategy, MotionGatedRewardStrategy,
                    SteeringSmoothRewardStrategy, ProximityPenaltyStrategy,
                    AvoidCollisionRewardStrategy, RouteCompletionBonusStrategy,
                    CompositeRewardStrategy)
from orchestrator import TrainingOrchestrator


map_config = dict(use_render=False, manual_control=False,
                  num_scenarios=20, map=3, horizon=1000,
                  traffic_density=0.05,
                  vehicle_config=dict(lidar=dict(num_lasers=120, distance=50)))

reward_strategy = CompositeRewardStrategy(strategies=[
    ProgressRewardStrategy(weight=200.0),
    MotionGatedRewardStrategy(HeadingAlignmentRewardStrategy(weight=3.0)),
    MotionGatedRewardStrategy(
        LaneCenteringRewardStrategy(weight=1.5, half_lane_width=1.5)),
    SteeringSmoothRewardStrategy(weight_magnitude=0.4, weight_delta=0.8),
    ProximityPenaltyStrategy(weight=0.02, safe_distance=0.3),  # anti-freeze
    AvoidCollisionRewardStrategy(crash_penalty=-30.0),
    RouteCompletionBonusStrategy(bonus=100.0),
])

ckpt_path = Path("./checkpoint")
config = TrainingConfig(map_config=map_config,
                        reward_strategy=reward_strategy,
                        learning_rate=1e-4,  # resuming, keep it gentle
                        total_timesteps=4_000_000,
                        use_sde=False,       # avoids gSDE bang-bang steering
                        checkpoint_path=ckpt_path)

if __name__ == "__main__":
    orchestrator = TrainingOrchestrator(config)
    orchestrator.run(resume_from=config.checkpoint_path / "ppo_model")
