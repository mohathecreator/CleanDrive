"""Thin wrapper around the Stable-Baselines3 PPO model.

Isolates all SB3 specifics behind a small interface, so the orchestrator
depends only on this manager rather than on PPO directly.
"""

from stable_baselines3 import PPO


class AgentManager:
    """Creates, trains, saves and loads the PPO agent."""

    def __init__(self):
        """Start with no model; call :meth:'create_model' or :meth:'load'."""
        self.model = None

    def create_model(self, env, learning_rate=3e-4, tensorboard_log=None,
                     use_sde=True):
        """Build a fresh PPO model on the given (vectorised) env.

        Uses a two-layer MLP policy. ``use_sde`` toggles gSDE
        (state-dependent exploration): when on, the exploration noise is
        temporally correlated, which is meant to counter jittery per-step
        Gaussian steering. It is a config knob because gSDE can also leave a
        bang-bang steering limit cycle in the deterministic policy, so a run
        may want it off. ``sde_sample_freq`` only matters when it is on.
        """
        self.model = PPO("MlpPolicy", env,
                         policy_kwargs=dict(net_arch=[256, 256]),
                         learning_rate=learning_rate,
                         n_steps=512,
                         use_sde=use_sde,
                         sde_sample_freq=4,
                         tensorboard_log=tensorboard_log)

    def train(self, total_timesteps, callback=None):
        """Run learning for 'total_timesteps'.

        Raises RuntimeError if no model has been created or loaded yet.
        """
        if self.model is None:
            raise RuntimeError("create_model() needs "
                               "to be called before train()")
        self.model = self.model.learn(total_timesteps, callback=callback)
        return self.model

    def save(self, path):
        """Persist the current model to 'path'.

        Raises RuntimeError if no model has been created or loaded yet
        (mirrors the same guard on :meth:`train`).
        """
        if self.model is None:
            raise RuntimeError("create_model() or load() needs "
                               "to be called before save()")
        return self.model.save(path)

    def load(self, path, env=None, learning_rate=None):
        """Load a saved model, optionally overriding the learning rate.

        The learning-rate override (both the attribute and the schedule) is
        what lets a resumed run continue at a different rate than it was
        originally trained with.
        """
        self.model = PPO.load(path, env=env)
        if learning_rate is not None:
            self.model.learning_rate = learning_rate
            self.model.lr_schedule = lambda _: learning_rate
        return self.model
