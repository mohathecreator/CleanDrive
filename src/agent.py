from stable_baselines3 import PPO


class AgentManager:
    def __init__(self):
        self.model = None

    def create_model(self, env):
        self.model = PPO("MlpPolicy", env)

    def train():
        

    def save():

    def load():
