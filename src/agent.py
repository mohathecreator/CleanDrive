"""Thin wrapper around the Stable-Baselines3 PPO model."""

from stable_baselines3 import PPO


class AgentManager:
    """Creates, trains, saves and loads the PPO agent."""

    def __init__(self):
        self.model = None  # set by create_model() or load()

    def create_model(self, env, learning_rate=3e-4, tensorboard_log=None,
                     use_sde=True):
        """Build a fresh PPO model on the given (vectorised) env."""
        self.model = PPO(
            "MlpPolicy", env,
            policy_kwargs=dict(net_arch=[256, 256]),  # 2-layer MLP
            learning_rate=learning_rate,
            n_steps=512,        # rollout length per update
            use_sde=use_sde,    # state-dependent exploration
            sde_sample_freq=4,  # only matters when use_sde=True
            tensorboard_log=tensorboard_log,
        )

    def train(self, total_timesteps, callback=None):
        """Run learning for total_timesteps. Requires a model to exist."""
        if self.model is None:
            raise RuntimeError("create_model() needs "
                               "to be called before train()")
        self.model = self.model.learn(total_timesteps, callback=callback)
        return self.model

    def save(self, path):
        """Persist the current model to path. Requires a model to exist."""
        if self.model is None:
            raise RuntimeError("create_model() or load() needs "
                               "to be called before save()")
        return self.model.save(path)

    def load(self, path, env=None, learning_rate=None):
        """Load a saved model, optionally overriding the learning rate."""
        self.model = PPO.load(path, env=env)
        if learning_rate is not None:
            self.model.learning_rate = learning_rate  # for logging
            self.model.lr_schedule = lambda _: learning_rate  # actual rate
        return self.model
