"""Definition of the curiosity-gym sparse reward environment."""

from typing import override

import numpy as np

from curiosity_gym.core.objects import Key, Door, Enemy, RandomBlock, Agent, Target 
from curiosity_gym.core.pov import AgentPOV
from curiosity_gym.core.gridengine import GridEngine
from curiosity_gym.utils.constants import MAP_SPARSE
from curiosity_gym.utils.dataclasses import (
    EnvironmentSettings,
    RenderSettings,
    EnvironmentObjects,
)


class SimpleSparseEnv(GridEngine):
    """Defines the structure of the curiosity-gym sparse reward environment.\n
    The environment consists of five rooms connected by four locked
    :class:`~curiosity_gym.core.objects.Door` objects. It also contains multiple
    :class:`~curiosity_gym.core.objects.Enemy` s and a
    :class:`~curiosity_gym.core.objects.RandomBlock`. The environment represents
    a classic navigation task, where the agent needs to reach the green target
    cell. It is designed to test the agent's ability to learn in sparse reward
    settings, where random exploration mechanisms are often insufficient.

    Parameters
    ----------
    agentPOV : :class:`~curiosity_gym.core.agentpov.AgentPOV` | str, optional
        Object or string defining the observations and action spaces of the RL agent.
        Valid string values are *'global'*, *'local_W'* and *'forward_L_W'*, where
        *W* and *L* are integers defining the width and length of the respective POV.
        By default :class:`~curiosity_gym.core.agentpov.GlobalView`.
    render_mode : str | None, optional
        Render mode in which the environment is run. If render mode is *human*, the
        environment will be rendered in PyGame. By default None.
    window_width : int, optional
        Horizontal size of the PyGame window in *human* render mode. By default 1200.


    .. figure:: ../../source/images/SparseEnv_optimal.gif
        :width: 500
        :align: center

        Example of a SparseEnv episode with an optimal policy.
    """

    @override
    def __init__(
        self,
        agentPOV: AgentPOV | str = "global",
        render_mode: str | None = None,
        window_width: int = 800,
        simple_actions: bool = False,
        simple_obs: bool = True,
        use_globaly_unique_id: bool = True,
        use_colours: bool = True,
        use_rgb_state: bool = False,
    ) -> None:
        env_settings = EnvironmentSettings(
            min_steps=66,
            max_steps=500,
            width=15,
            height=11,
            reward_range=(0, 1),
            simple_actions=simple_actions,
            simple_obs=simple_obs,
            use_globaly_unique_id=use_globaly_unique_id,
            use_rgb_state=use_rgb_state
        )

        render_settings = RenderSettings(
            render_mode=render_mode,
            window_width=window_width,
            window_height=int(
                window_width * (env_settings.height / env_settings.width)
            ),
        )

        other_objects = np.array(
            [
                # Room 1:
                Key((5, 2), color=3, use_colour=use_colours),
                Door((9, 2), state=2, color=3, use_colour=use_colours),
                # Room 2:
                Key((13, 1), color=4, use_colour=use_colours),
                Door((12, 4), state=2, color=4, use_colour=use_colours),
                # Room 3:
                Key((11, 8), color=5, use_colour=use_colours),
                Door((8, 6), state=2, color=5, use_colour=use_colours),
                # Room 4:
                Key((5, 6), color=6, use_colour=use_colours),
                Door((4, 8), state=2, color=6, use_colour=use_colours),
                RandomBlock((6, 6)),
            ]
        )

        env_objects = EnvironmentObjects(
            agent=Agent((1, 1)),
            target=Target((7, 4), color=2),
            walls=self.load_walls(MAP_SPARSE),
            other=other_objects,
        )

        super().__init__(
            env_name = "simple_sparse_environment",
            env_full_name="Simplen-Spärlichen-Umgebung",
            env_settings=env_settings,
            render_settings=render_settings,
            env_objects=env_objects,
            agent_pov=agentPOV,
        )

    @override
    def check_task(self) -> bool:
        """Check whether the agent has reached the green target cell.

        Returns
        -------
        bool
            True if the agent is at the target position, False otherwise.
        """
        return bool(np.all(self.objects.target.position == self.objects.agent.position))
