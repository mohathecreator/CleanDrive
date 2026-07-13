"""Entry point: renders a trained checkpoint driving.

Loads the model and observation normaliser from ``checkpoint_path`` and
runs the deterministic policy in a rendered MetaDrive window. Run with
``python src/evaluate.py``.

Episode outcomes (arrive_dest / out_of_road / crash vehicle / max step)
are printed by MetaDrive itself as each episode ends.
"""

from pathlib import Path

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from agent import AgentManager
from config import TrainingConfig
from environment import EnvironmentWrapper
from reward import SpeedRewardStrategy


map_config = dict(use_render=True, manual_control=False,
                  num_scenarios=20, map="SC", horizon=1000,
                  traffic_density=0.05,
                  vehicle_config=dict(lidar=dict(num_lasers=120, distance=50)))

reward_strategy = SpeedRewardStrategy()
config = TrainingConfig(map_config=map_config,
                        reward_strategy=reward_strategy,
                        checkpoint_path=Path("./checkpoint"))

env = DummyVecEnv([lambda: EnvironmentWrapper(config)])
env = VecNormalize.load(config.checkpoint_path / "vec_normalize.pkl", env)
env.training = False
env.norm_reward = False

agent_manager = AgentManager()
agent_manager.load(config.checkpoint_path / "ppo_model", env=env)

obs = env.reset()
for _ in range(2000):
    action, _ = agent_manager.model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done[0]:
        obs = env.reset()

env.close()
