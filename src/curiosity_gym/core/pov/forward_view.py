from typing_extensions import override
from curiosity_gym.core.pov import AgentPOV

import numpy as np
from gymnasium import spaces
from curiosity_gym.core.objects import Agent
from curiosity_gym.core.objects import Agent, GridObject
from curiosity_gym.utils.constants import IX_TO_COLOR

class ForwardView(AgentPOV):
    """Agent point-of-view observing grid cells in front of the agent.

    Parameters
    ----------
    pov_length : int
        Length of the visible field in front of the agent.
    pov_width : int
        Width of the visible field in front of the agent.
    env_size : tuple[int, int]
        Number of (horizontal, vertical) cells in the grid environment where the pov class is used.
    xray : bool
        Whether the agent can observe cells behind walls and closed doors.


    .. figure:: ../../source/images/ForwardView_2_3.gif
        :width: 500
        :align: center

        Example of ForwardView with a length of 2 and a width of 3 without xray.
    """

    @override
    def __init__(
        self,
        pov_length: int,
        pov_width: int,
        env_size: tuple[int, int],
        xray: bool = False,
        simple_observation_space: bool = True,
        individual_obj_ids: bool = True
    ) -> None:
        self.pov_width = pov_width
        self.pov_length = pov_length
        self.xray = xray
        action_space = spaces.Discrete(4)
        number_of_cells = (pov_length + 1) * pov_width
        if (not individual_obj_ids):
            obj_id_count = len(GridObject.id_map.keys())
        else:
            obj_id_count = GridObject._next_instance_id - 1
        total_label_count = obj_id_count  + (len(IX_TO_COLOR.keys()) + 1) + 1 # +1 for color zero indexed and for states TODO encode obj state as well?
        observation_space_shape = (number_of_cells * total_label_count,) if simple_observation_space else (number_of_cells,3)
        observation_space = spaces.Box(
            shape=observation_space_shape, high=10, low=0, dtype=np.int64
        )
        super().__init__(action_space, observation_space, env_size)

    @override
    def transform_obs(self, state: np.ndarray, agent: Agent) -> np.ndarray:
        self.visible_positions = []
        pos = agent.position.tolist()
        local = np.full(((self.pov_length + 1) * self.pov_width, 3), [0, 0, 0])

        if agent.state % 3 == 0:
            length_range = range(0, self.pov_length + 1)
            width_range = range(-int(self.pov_width / 2), int(self.pov_width / 2) + 1)

        else:
            length_range = range(0, -self.pov_length - 1, -1)
            width_range = range(
                int(self.pov_width / 2), -int(self.pov_width / 2) - 1, -1
            )

        ix_new = 0
        for l in length_range:
            for w in width_range:
                if agent.state % 2 == 0:
                    x, y = l, w
                else:
                    x, y = w, l

                ix = (pos[0] + x) + self.width * (pos[1] + y)
                cell = (pos[0] + x, pos[1] + y)

                if ix < 0 or not self.is_visible(state, pos, cell) and not self.xray:
                    continue

                self.visible_positions.append(cell)
                local[ix_new] = state[ix]
                ix_new += 1

        return local
