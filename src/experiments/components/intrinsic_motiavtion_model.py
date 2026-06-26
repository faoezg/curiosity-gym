from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import torch

# marker interface
class IntrinsicMotivationModel(ABC):
    def __init__(self, name, seed = 42) -> None:
        self.name = name
        self.seed = seed
        torch.manual_seed(self.seed)
        pass

    @abstractmethod
    def calc_intrinsic_reward(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def _train_network(self, *args, **kwargs) -> float | tuple[float, float]:
        pass
