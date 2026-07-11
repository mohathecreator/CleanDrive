from config import TrainingConfig
from reward import (SpeedRewardStrategy, LaneCenteringRewardStrategy,
                    AvoidCollisionRewardStrategy, MinSpeedPenaltyStrategy,
                    SteeringSmoothRewardStrategy, BrakingRewardStrategy,
                    CompositeRewardStrategy)
from orchestrator import TrainingOrchestrator


map_config = dict(use_render=False, manual_control=False,
                  num_scenarios=20, map=3,
                  vehicle_config=dict(lidar=dict(num_lasers=120, distance=50)))

reward_strategy = CompositeRewardStrategy(strategies=[
    SpeedRewardStrategy(weight=2.0),
    LaneCenteringRewardStrategy(weight=3.0),
    AvoidCollisionRewardStrategy(crash_penalty=-100.0),
    MinSpeedPenaltyStrategy(weight=2.0, min_speed=10.0),
    SteeringSmoothRewardStrategy(weight=0.15),
    BrakingRewardStrategy(weight=1.0, safe_distance=8.0),
])

config = TrainingConfig(map_config=map_config,
                        reward_strategy=reward_strategy,
                        learning_rate=1e-4,
                        total_timesteps=3_000_000)

if __name__ == "__main__":
    orchestrator = TrainingOrchestrator(config)
    orchestrator.run(resume_from=config.checkpoint_path / "ppo_model")
