"""Definition of the curiosity-gym multitask environment."""

from typing import override

import numpy as np

from curiosity_gym.core.objects import Target, Ball, Door, Agent, Key 
from curiosity_gym.core.pov import AgentPOV
from curiosity_gym.core.gridengine import GridEngine
from curiosity_gym.utils.constants import MAP_MULTITASK
from curiosity_gym.utils.dataclasses import (
    EnvironmentSettings,
    RenderSettings,
    EnvironmentObjects,
)


class MultitaskEnv(GridEngine):
    """Defines the structure of the curiosity-gym multitask environment.\n
    The environment consists of three rooms with two distinct tasks. The agent
    always starts in the middle room. The left room is used for task no. 1, where
    the agent needs to collect a key, open a door and move to the green target cell.
    The right room represents task no. 2, where the agent needs to push a ball to the
    purple target cell. Only one task is active at a time. The agent will only gain a
    reward for completing the task that is currently active. The environment is designed
    to test the agent's capability of transitioning between different tasks within the
    same environment.

    Parameters
    ----------
    agentPOV : :class:`~curiosity_gym.core.agentpov.AgentPOV` | str, optional
        Object or string defining the observations and action spaces of the RL agent.
        Valid string values are *'global'*, *'local_W'* and *'forward_L_W'*, where
        *W* and *L* are integers defining the width and length of the respective POV.
        By default :class:`~curiosity_gym.core.agentpov.GlobalView`.
    task : int, optional
        Identifier for task. In task no. 1 the agent needs to reach the green target cell
        in the left room. In task no. 2 the agent needs to push the ball to the purple target
        cell in the right room. By default 1.
    random : bool, optional
        Whether the position of the target for both tasks should be randomly selected within
        their respective rooms. By default False.
    render_mode : str | None, optional
        Render mode in which the environment is run. If render mode is *human*, the
        environment will be rendered in PyGame. By default None.
    window_width : int, optional
        Horizontal size of the PyGame window in *human* render mode. By default 1200.


    .. figure:: ../../source/images/MultitaskEnv_optimal.gif
        :width: 500
        :align: center

        Example of a MultitaskEnv episode with an optimal policy for alternating tasks.
    """

    # pylint: disable=too-many-arguments
    # Additional arguments are required for the constructor, to cover the different task variants.

    @override
    def __init__(
        self,
        agentPOV: AgentPOV | str = "global",
        task: int = 1,
        random: bool = False,
        render_mode: str | None = None,
        window_width: int = 1200,
        simple_actions: bool = False,
        simple_obs: bool = True,
        use_globaly_unique_id: bool = True,
        use_colours: bool = True,
        use_rgb_state: bool = False,
    ) -> None:

        assert task <= 2, f"Invalid task id: {task}."
        self.task = task
        self.random = random

        env_settings = EnvironmentSettings(
            min_steps=15,
            max_steps=500,
            width=19,
            height=7,
            reward_range=(0, 10),
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

        self.ball_target = Target((15, 3), color=5)
        self.ball = Ball((12, 3), zone_low=(13, 1), zone_high=(17, 5), color=5)
        other_objects = np.array(
            [
                Door((6, 3), color=3, state=2),
                Key((7, 1), color=3),
                self.ball_target,
                self.ball,
            ]
        )

        env_objects = EnvironmentObjects(
            agent=Agent((9, 3), state=1),
            target=Target((3, 3), color=2),
            walls=self.load_walls(MAP_MULTITASK),
            other=other_objects,
        )

        super().__init__(
            env_name = "multitask_environment",
            env_full_name="Multitask-Umgebung",
            env_settings=env_settings,
            render_settings=render_settings,
            env_objects=env_objects,
            agent_pov=agentPOV,
        )

    @override
    def check_task(self) -> bool:
        """Checks whether the agent has completed the task that is currently active.

        Returns
        -------
        bool
            True if task has been completed, False otherwise.
        """
        if self.task == 1:
            return bool(
                np.all(self.objects.target.position == self.objects.agent.position)
            )
        return bool(np.all(self.ball.position == self.ball_target.position))

    @override
    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        super().reset(**kwargs)
        if self.random:
            self.objects.target.position = np.random.randint(1, 6, 2)
            self.ball_target.position = np.array(
                [np.random.randint(14, 18), np.random.randint(1, 6)]
            )

        if self.task == 1:
            pos = self.objects.target.position
            self.env_settings.min_steps = 18 - pos[0] + abs(3 - pos[1]) + (3 != pos[1])

        else:
            pos = self.ball_target.position
            self.env_settings.min_steps = (
                2 * pos[0] - 22 + (pos[1] != 3) * (2 * abs(pos[1] - 3) + 5)
            )

        return (self._get_obs(), self._get_info())
