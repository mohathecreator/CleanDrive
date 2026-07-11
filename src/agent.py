from stable_baselines3 import PPO


class AgentManager:
    def __init__(self):
        self.model = None

    def create_model(self, env, learning_rate=3e-4, tensorboard_log=None):
        self.model = PPO("MlpPolicy", env,
                         policy_kwargs=dict(net_arch=[256, 256]),
                         learning_rate=learning_rate,
                         n_steps=512,
                         tensorboard_log=tensorboard_log)

    def train(self, total_timesteps, callback=None):
        if self.model is None:
            raise RuntimeError("create_model() needs "
                               "to be called before train()")
        self.model = self.model.learn(total_timesteps, callback=callback)
        return self.model

    def save(self, path):
        return self.model.save(path)

    def load(self, path, env=None, learning_rate=None):
        self.model = PPO.load(path, env=env)
        if learning_rate is not None:
            self.model.learning_rate = learning_rate
            self.model.lr_schedule = lambda _: learning_rate
        return self.model
