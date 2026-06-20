from config import TrainingConfig
from reward import SpeedRewardStrategy
from orchestrator import TrainingOrchestrator


map_config = dict(use_render=False, manual_control=False,
                  num_scenarios=1, map="S",)

reward_strategy = SpeedRewardStrategy()

config = TrainingConfig(map_config=map_config,
                        reward_strategy=reward_strategy,
                        total_timesteps=15)

orchestrator = TrainingOrchestrator(config)
orchestrator.run()
