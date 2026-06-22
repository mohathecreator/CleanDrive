from agent import AgentManager
from config import TrainingConfig
from environment import EnvironmentWrapper
from reward import SpeedRewardStrategy


map_config = dict(use_render=True, manual_control=False,
                  num_scenarios=1, map="S",)

reward_strategy = SpeedRewardStrategy()

config = TrainingConfig(map_config=map_config,
                        reward_strategy=reward_strategy)

env = EnvironmentWrapper(config)

agent_manager = AgentManager()
agent_manager.load(config.checkpoint_path / "ppo_model")

observation, info = env.reset()
for _ in range(2000):
    action, _ = agent_manager.model.predict(observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()

env.close()
