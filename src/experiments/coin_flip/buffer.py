from dataclasses import dataclass
from experiments.utils.buffers import PriorizedTransition

import numpy as np

@dataclass
class CFNTransition(PriorizedTransition):
    coin_flip_vector: np.ndarray
