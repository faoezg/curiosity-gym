from abc import ABC

from stable_baselines3.common.base_class import BaseAlgorithm

class ExperimentModel(ABC):
    """
    ExperimentModel presents a wrapper around models to provide some additional information to the experiment and its harness

    :param model: The model to be wrapped, must be of type TODO
    :param model_name: The name of the model, as a string
    """
    def __init__(self, model: BaseAlgorithm, model_name: str) -> None:
        super().__init__()
        self._model = model
        self._model_name = model_name

    @property
    def model(self) -> BaseAlgorithm:
        """Getter of the model"""
        return self._model

    @property
    def model_name(self) -> str:
        """Getter of the model name"""
        return self._model_name
