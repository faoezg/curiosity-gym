from typing import override

import numpy as np

from curiosity_gym.core.objects import Agent, Target
from curiosity_gym.core.pov import AgentPOV
from curiosity_gym.core.gridengine import GridEngine
from curiosity_gym.utils.layout_generator import generate_detachment_wall_layout
from curiosity_gym.utils.dataclasses import (
    EnvironmentSettings,
    RenderSettings,
    EnvironmentObjects,
)

class DetachmentEnv(GridEngine):

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
            min_steps=0,
            max_steps=500,
            width=69, # 15x15 on each side
            height=25,
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

        other_objects = np.array([])

        env_objects = EnvironmentObjects(
            agent=Agent((34, 12)),
            target=None,
            walls=self.load_walls(generate_detachment_wall_layout(env_settings)),
            other=other_objects,
        )

        super().__init__(
            env_name = "detachment_environment",
            env_full_name="Detachment-Umgebung",
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
        return False
