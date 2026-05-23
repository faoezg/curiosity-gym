from typing_extensions import override
from curiosity_gym.core.pov import AgentPOV

import numpy as np
from gymnasium import spaces
from curiosity_gym.core.objects import Agent
from curiosity_gym.core.objects import Agent, GridObject
from curiosity_gym.utils.constants import IX_TO_COLOR
from curiosity_gym.utils.enums import Action, SimplerAction

class GlobalView(AgentPOV):
    """Agent point-of-view observing the full state of the environment.

    Parameters
    ----------
    env_size : tuple[int, int]
        Number of (horizontal, vertical) cells in the grid environment where the pov class is used.
    """

    @override
    def __init__(self,
                 env_size: tuple[int, int],
                 simple_observation_space: bool = True,
                 simple_action_space: bool = True,
                 individual_obj_ids: bool = True,
                 use_rgb_state: bool = False,
                 rgb_state_size: tuple[int, int, int] | None = None
                 ) -> None:
        action_space = spaces.Discrete(len(SimplerAction)) if simple_action_space else spaces.Discrete(len(Action)) 
        number_of_nodes = env_size[0] * env_size[1]
        if (not individual_obj_ids):
            obj_id_count = len(GridObject.id_map.keys())
        else:
            obj_id_count = GridObject._next_instance_id - 1
        self.total_label_count = obj_id_count + (len(IX_TO_COLOR.keys()) + 1) + obj_id_count * 3 # +1 for color zero indexed and for states TODO encode obj state as well?
        observation_space_shape = (number_of_nodes * self.total_label_count,) if simple_observation_space else (number_of_nodes,3)
        if (simple_observation_space):
            observation_space_shape = (number_of_nodes * self.total_label_count,)
            observation_space = spaces.Box(
                shape=observation_space_shape, high=1000, low=0, dtype=np.int64
            )
 
        elif (use_rgb_state and rgb_state_size is not None):
            observation_space = spaces.Box(
                shape=rgb_state_size, high=255, low=0, dtype=np.uint8
            )

        else:
            observation_space_shape = (number_of_nodes,3)
            observation_space = spaces.Box(
                shape=observation_space_shape, high=1000, low=0, dtype=np.int64
            )

        super().__init__(action_space, observation_space, env_size)

    @override
    def transform_obs(self, state: np.ndarray, agent: Agent) -> np.ndarray:
        return state

    def get_cell_label_count(self):
        return self.total_label_count
