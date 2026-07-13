"""Wires the environment, agent and callbacks into a training run."""

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from agent import AgentManager
from config import TrainingConfig
from environment import EnvironmentWrapper


class TrainingOrchestrator:
    """Owns the vectorised environment and drives a training run.

    Builds 'num_envs' parallel environments ('SubprocVecEnv') wrapped in
    'VecNormalize' for running observation normalisation, and delegates the
    model lifecycle to :class:'agent.AgentManager'.
    """

    def __init__(self, config: TrainingConfig, num_envs: int = 24):
        """Validate the config and build the normalised vector env.

        'num_envs' should roughly match the number of CPU cores: this
        workload is simulation-bound (MetaDrive physics on CPU), so
        throughput scales with cores, not with GPU.
        """
        self.config = config
        self.num_envs = num_envs
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
        """Train the agent, then save the model and the obs normaliser.

        With 'resume_from' set, load that checkpoint together with its
        saved 'VecNormalize' statistics and keep training; otherwise start
        a fresh model. A 'CheckpointCallback' writes a checkpoint plus its
        matching 'VecNormalize' stats roughly every 50k timesteps -- both
        a safety net against policy collapse and what makes intermediate
        checkpoints evaluable.
        """
        normalizer_path = self.config.checkpoint_path / "vec_normalize.pkl"
        if resume_from:
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
            )
        # save_freq counts callback calls (one per vec step), so divide the
        # target timestep interval by num_envs to actually hit ca. 50k steps.
        checkpoint_callback = CheckpointCallback(
            save_freq=max(50_000 // self.num_envs, 1),
            save_path=str(self.config.checkpoint_path),
            name_prefix="ppo_checkpoint",
            save_vecnormalize=True,
        )
        self.agent_manager.train(
            self.config.total_timesteps, callback=checkpoint_callback
        )
        self.agent_manager.save(self.config.checkpoint_path / "ppo_model")
        self.environment.save(normalizer_path)
