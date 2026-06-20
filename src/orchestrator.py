from config import TrainingConfig
from environment import EnvironmentWrapper
from agent import AgentManager


class TrainingOrchestrator:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.config.validate()
        self.environment = EnvironmentWrapper(config)
        self.agent_manager = AgentManager()

    def run(self):
        self.agent_manager.create_model(self.environment)
        self.agent_manager.train(self.config.total_timesteps)
        self.agent_manager.save(self.config.checkpoint_path / "ppo_model")
