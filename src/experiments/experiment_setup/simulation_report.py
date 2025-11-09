from abc import ABC

from curiosity_gym.core.gridengine import GridEngine

import uuid

class SimulationReport(ABC):

    """
    SimulationReport holds all quantitive information of a given simulation run with a certain model in a certain environment

    :param model_name: The name of the model, as a string
    :param environment_name: The name of the environment, as a string
    :param environment_steps_taken: Amout of steps that were taken until the simulation ended, as an integer
    :param total_reward: The total amount of reward the agent retrived in the environment, as an integer
    """
    def __init__(self,
                 model_name: str,
                 environment: GridEngine,
                 environment_steps_taken: int = 0,
                 total_reward: int | float = 0) -> None:
        super().__init__()
        self._id = uuid.uuid4()
        self._model_name = model_name
        self._environment = environment
        self._environment_steps_taken = environment_steps_taken
        self._total_reward = total_reward
    
    @property
    def model_name(self) -> str:
        """Getter of the model name"""
        return self._model_name

    @property
    def environment_name(self) -> str:
        """Getter of the environment name"""
        return self._environment.name

    @property
    def environment(self) -> GridEngine:
        """Getter of the environment"""
        return self._environment

    @property
    def environment_steps_taken(self) -> int:
        """Getter of the steps the taken until simulation termination"""
        return self._environment.step_count

    @property
    def total_reward(self) -> int | float:
        """Getter of the total reward recived during simulation"""
        return self._total_reward

    @total_reward.setter
    def total_reward(self, total_reward: int | float):
        """Setter of the total reward recived during simulation"""
        self._total_reward = total_reward
    
    def __str__(self) -> str:
        return f"In environment {self.environment_name} model {self.model_name} achieved a reward of {self.total_reward} in {self.environment_steps_taken} steps"
