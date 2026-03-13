from typing_extensions import override
from curiosity_gym.core.pov import AgentPOV

import numpy as np
from gymnasium import spaces
from curiosity_gym.core.objects import Agent

class GlobalView(AgentPOV):
    """Agent point-of-view observing the full state of the environment.

    Parameters
    ----------
    env_size : tuple[int, int]
        Number of (horizontal, vertical) cells in the grid environment where the pov class is used.
    """

    @override
    def __init__(self, env_size: tuple[int, int], simple_observation_space: bool = True) -> None:
        action_space = spaces.Discrete(4)
        number_of_nodes = env_size[0] * env_size[1]
        observation_space_shape = (number_of_nodes,) if simple_observation_space else (number_of_nodes,3)
        observation_space = spaces.Box(
            shape=observation_space_shape, high=10, low=0, dtype=np.int64
        )
        super().__init__(action_space, observation_space, env_size)

    @override
    def transform_obs(self, state: np.ndarray, agent: Agent) -> np.ndarray:
        return state
