"""Wires the environment, agent and callbacks into a training run."""

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from agent import AgentManager
from config import TrainingConfig
from environment import EnvironmentWrapper


class TrainingOrchestrator:
    """Owns the vectorised environment and drives a training run."""

    def __init__(self, config: TrainingConfig, num_envs: int = 24):
        """Validate the config and build the normalised vector env."""
        self.config = config
        self.num_envs = num_envs  # match CPU cores: sim-bound, not GPU-bound
        self.config.validate()
        env_fns = [self._make_env for _ in range(num_envs)]
        self.environment = VecNormalize(
            SubprocVecEnv(env_fns), norm_obs=True, norm_reward=False
        )
        self.agent_manager = AgentManager()

    def _make_env(self):
        """Factory for a single monitored environment."""
        return Monitor(EnvironmentWrapper(self.config))

    def run(self, resume_from=None):
        """Train the agent, then save the model and the obs normaliser."""
        normalizer_path = self.config.checkpoint_path / "vec_normalize.pkl"
        if resume_from:
            # continue training: load the matching VecNormalize stats too
            self.environment = VecNormalize.load(
                normalizer_path, self.environment.venv
            )
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
                use_sde=self.config.use_sde,
            )
        # save_freq counts callback calls (one per vec step), so divide the
        # target timestep interval by num_envs to actually hit ca. 50k steps.
        checkpoint_callback = CheckpointCallback(
            save_freq=max(50_000 // self.num_envs, 1),
            save_path=str(self.config.checkpoint_path),
            name_prefix="ppo_checkpoint",
            save_vecnormalize=True,  # needed to evaluate mid-training
        )
        self.agent_manager.train(
            self.config.total_timesteps, callback=checkpoint_callback
        )
        self.agent_manager.save(self.config.checkpoint_path / "ppo_model")
        self.environment.save(normalizer_path)
