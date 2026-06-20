from stable_baselines3 import PPO


class AgentManager:
    def __init__(self):
        self.model = None

    def create_model(self, env):
        self.model = PPO("MlpPolicy", env)

    def train(self, total_timesteps):
        if self.model is None:
            raise RuntimeError("create_model() needs "
                               "to be called before train()")
        self.model = self.model.learn(total_timesteps)
        return self.model

    def save(self, path):
        return self.model.save(path)

    def load(self, path):
        self.model = PPO.load(path)
        return self.model
