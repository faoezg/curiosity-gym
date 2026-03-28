import numpy as np
from dataclasses import dataclass
from experiments.components.buffers import PriorizedTransition


@dataclass
class SGDTransition(PriorizedTransition):
    goal_state: np.ndarray
