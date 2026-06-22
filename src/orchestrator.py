from agent import AgentManager
from config import TrainingConfig
from environment import EnvironmentWrapper


class TrainingOrchestrator:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.config.validate()
        self.environment = EnvironmentWrapper(config)
        self.agent_manager = AgentManager()

    def run(self):
        self.agent_manager.create_model(
            self.environment, tensorboard_log=str(self.config.log_path)
        )
        self.agent_manager.train(self.config.total_timesteps)
        self.agent_manager.save(self.config.checkpoint_path / "ppo_model")
