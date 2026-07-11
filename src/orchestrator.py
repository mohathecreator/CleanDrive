from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from agent import AgentManager
from config import TrainingConfig
from environment import EnvironmentWrapper


class TrainingOrchestrator:
    def __init__(self, config: TrainingConfig, num_envs: int = 20):
        self.config = config
        self.config.validate()
        env_fns = [self._make_env for _ in range(num_envs)]
        self.environment = VecNormalize(
            SubprocVecEnv(env_fns), norm_obs=True, norm_reward=False
        )
        self.agent_manager = AgentManager()

    def _make_env(self):
        return Monitor(EnvironmentWrapper(self.config))

    def run(self, resume_from=None):
        normalizer_path = self.config.checkpoint_path / "vec_normalize.pkl"
        if resume_from:
            self.environment = VecNormalize.load(normalizer_path, self.environment.venv)
            self.agent_manager.load(
                resume_from,
                env=self.environment,
                learning_rate=self.config.learning_rate,
            )
        else:
            self.agent_manager.create_model(
                self.environment,
                learning_rate=self.config.learning_rate,
                tensorboard_log=str(self.config.log_path),
            )
        checkpoint_callback = CheckpointCallback(
            save_freq=50_000,
            save_path=str(self.config.checkpoint_path),
            name_prefix="ppo_checkpoint",
        )
        self.agent_manager.train(self.config.total_timesteps, callback=checkpoint_callback)
        self.agent_manager.save(self.config.checkpoint_path / "ppo_model")
        self.environment.save(normalizer_path)