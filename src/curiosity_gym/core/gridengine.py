"""Definitions for the curiosity-gym grid environment dynamics.

This module defines the GridEngine class, which is the abstract
base class for all curiosity-gym environments. It implements the
grid-based dynamics for all environments and implements the 
gymnasium api, which can be used to interact with RL algorithms. 
"""

from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pygame
import pandas as pd
import seaborn as sns

from curiosity_gym.core.objects import GridObject, Wall
from curiosity_gym.core.pov import AgentPOV, GlobalView, LocalView, ForwardView
from curiosity_gym.utils.enums import Action, SimplerAction
from curiosity_gym.utils.dataclasses import (
    EnvironmentSettings,
    RenderSettings,
    EnvironmentObjects,
)

# TODO validate all doc strings in project for :class:
class GridEngine(gym.Env, ABC):
    """Abstract grid-based environment class that implements the gymnasium api.

    Parameters
    ----------
    env_name : str
        Name of the environment
    env_settings : :class:`~curiosity_gym.utils.dataclasses.EnvironmentSettings`
        Object storing settings for the environment.
    render_settings : :class:`~curiosity_gym.utils.dataclasses.RenderSettings`
        Object storing render settings that should be apllied when the environment
        is used.
    env_objects : :class:`~curiosity_gym.utils.dataclasses.EnvironmentObjects`
        Object storing all grid objects that were placed in the environment.
    agent_pov : :class:`~curiosity_gym.core.agentpov.AgentPOV` | str
        Object or string defining the observations and action spaces of the RL agent.
        Valid string values are *'global'*, *'local_W'* and *'forward_L_W'*, where
        W and L are integers defining the width and length of the respective POV.
    """

    # pylint: disable=too-many-instance-attributes
    # Additional instance attributes are required for this class, to cover the gym.Env interface.

    def __init__(
        self,
        env_name: str,
        env_settings: EnvironmentSettings,
        render_settings: RenderSettings,
        env_objects: EnvironmentObjects,
        agent_pov: AgentPOV | str,
    ) -> None:
        self._name = env_name
        # Store settings
        self.env_settings = env_settings
        self.render_settings = render_settings
        self.reward_range = env_settings.reward_range
        """Range of rewards that can be obtained within one episode."""

        # Initialise agent pov
        self.agent_pov = self._init_pov(agent_pov)
        self.action_space = self.agent_pov.action_space
        """Space of possible actions a RL agent can choose from."""
        self.observation_space = self.agent_pov.observation_space
        """Space of possible observations returned by the environment."""

        # Current environment state
        self.objects = env_objects
        self.step_count = 0
        self.pos_count = {
            (x, y): 0
            for x in range(self.env_settings.width)
            for y in range(self.env_settings.height)
        }

        # Initialise render objects
        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
        """Metadata of the environment. Contains possible render modes and render fps."""
        self.render_mode = render_settings.render_mode
        """Render mode in which the environment is run."""
        assert self.render_mode in (
            self.metadata["render_modes"] + [None]
        ), f"Invalid render_mode: {self.render_mode}"
        if self.render_settings.render_mode == "human":
            self.init_render()

    @abstractmethod
    def check_task(self) -> bool:
        """Check whether the main task of an environment has been completed by the agent.\n
        A specific task needs to be implemented by all environments that inherit
        from :class:`GridEngine`. The completion of the task yields the maximum reward
        and ends the current episode.

        Returns
        -------
        bool
            :const:`True` if main task was completed by the agent, :const:`False` otherwise.
        """

    def step(
        self, action: int | Action | SimplerAction
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Run one timestep of the environment’s dynamics using the agent actions.\n
        When the end of an episode is reached (terminated or truncated), it is necessary
        to call :meth:`reset` to reset this environment’s state for the next episode.

        .. seealso::
            The method is part of the `Gymnasium
            <https://gymnasium.farama.org/api/env/#gymnasium.Env.step>`__
            environment api.

        Parameters
        ----------
        action : int | :type:`~curiosity_gym.utils.enums.Action`
            Action selected by the RL agent.

        Returns
        -------
        observation : np.ndarray
            An element of the :attr:`observation_space` that represents the agents observation.
        reward : float
            The reward as a result of taking the specified action.
        terminated : bool
            End of episode by reaching a terminal state. Can be achieved by completing the task
            of an environment or agent contact with a harmful grid object. If true, the user
            needs to call :meth:`reset`.
        truncated : bool
            Typically, this is a timelimit, but could also be used to indicate an agent
            physically going out of bounds. Can be used to end the episode prematurely before
            a terminal state is reached. If true, the user needs to call reset().
        info : dict
            Contains auxiliary diagnostic information for debugging, learning and logging.
        """
        self.step_count += 1
        self.pos_count[tuple(self.objects.agent.position)] += 1
        obs = self._get_obs()

        reward = self._calc_reward(action)

        if self.render_settings.render_mode == "human":
            pygame.display.set_caption(
                f"Curiosity Gym [Current Steps: {self.step_count}, "
                + f"Step reward: {reward}]"
            )
            self._render_frame()

        return (
            obs,
            reward,
            self._get_terminated(),
            self._get_truncated(),
            self._get_info(),
        )

    def _calc_reward(self, action: int | Action | SimplerAction):
        external_reward = self._calc_obj_reward(action) + self._calc_task_reward()
        return external_reward
   
    def _calc_obj_reward(self, action: int | Action | SimplerAction):
        reward = 0
        for ob in self.objects.get_non_wall():
            reward += ob.step(
                self.agent_pov.transform_action(action, self.env_settings.simple_actions),
                self.find_object(self.objects.agent.get_front()),
                self._check_walkable(self.objects.agent.get_front()),
            )
        return reward
    
    def _calc_task_reward(self,):
        return (
            (
                self.env_settings.min_steps
                / self.step_count
                * self.env_settings.reward_range[1]
            )
            if self.check_task()
            else 0
        )

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        """Reset the environment to an initial internal state.\n

        .. seealso::
            The method calls the parent method from the `Gymnasium
            <https://gymnasium.farama.org/api/env/#gymnasium.Env.reset>`__
            environment class.

        Returns
        -------
        observation : np.ndarray
            Observation of the initial state.
        info : dict
            Auxiliary information about the internal state.
        """
        super().reset(**kwargs)
        for ob in list(self.objects.other) + [self.objects.agent, self.objects.target]:
            ob.reset()
        self.step_count = 0
        self.pos_count[tuple(self.objects.agent.position)] += 1
        return (self._get_obs(), self._get_info())

    def render(self) -> np.ndarray | None:
        """Compute the render frames as specified by 
        :attr:`~curiosity_gym.utils.dataclasses.RenderSettings.render_mode`.

        By convention, if the :attr:`~curiosity_gym.utils.dataclasses.RenderSettings.render_mode` 
        is:

        * None (default): no render is computed.
        * “human”: The environment is continuously rendered in the current PyGame display. This \
            rendering occurs during step() and render() doesn’t need to be called. Returns None.
        * “rgb_array”: Return a single frame representing the current state of the environment.\
            A frame is a np.ndarray with shape (x, y, 3) representing RGB values for an x-by-y \
            pixel image.

        .. seealso:: 
            The method is part of the `Gymnasium 
            <https://gymnasium.farama.org/api/env/#gymnasium.Env.render>`__
            environment api.

        Returns
        -------
        np.ndarray | None
            A frame for render_mode = “rgb_array”, None otherwise.
        """
        if self.render_settings.render_mode == "rgb_array":
            return self._render_frame()
        return None

    def close(self) -> None:
        """Clean up the environment.\n
        Will close the rendering window. Calling close on an already closed
        environment has no effect and won’t raise an error.

        .. seealso::
            The method is part of the `Gymnasium
            <https://gymnasium.farama.org/api/env/#gymnasium.Env.close>`__
            environment api.
        """
        if self.render_settings.window is not None:
            pygame.display.quit()
            pygame.quit()

    def find_object(self, position: np.ndarray) -> GridObject | None:
        """Get non-wall grid object at given position.

        Parameters
        ----------
        position : np.ndarray
            Position where the object is located inside the environment.

        Returns
        -------
        objects.GridObject | None
            Gridobject if there is a non-wall gridobject at given position, None otherwise.
        """
        for ob in self.objects.get_non_wall():
            if np.all(ob.position == position):
                return ob
        return None

    def get_object_ids(self) -> dict[GridObject, int]:
        """Get ids for all grid object types.

        Returns
        -------
        dict[objects.GridObject, int]
            Dictionary containing all grid object types with their corresponding ids in the
            :attr:`observation_space`.
        """
        return GridObject.id_map

    def get_state(self) -> np.ndarray:
        """Get the current state of the environment.\n
        The returned state is independent of the agent's :attr:`observation_space`.

        Returns
        -------
        state : np.ndarray
            Current state of the environment.
        """
        state = np.zeros(
            [self.env_settings.width * self.env_settings.height, 3], dtype=int
        )
        for ob in self.objects.get_all():
            x, y = ob.position
            assert x + y * self.env_settings.width < len(
                state
            ), f"""Position [{x},{y}] of object with type {self.get_object_ids()[ob.identifier]}
            is invalid for grid with size ({self.env_settings.width}, {self.env_settings.height})"""
            state[x + y * self.env_settings.width] = ob.get_identity()
        return state
    
    def heatmap(self) -> Figure | None:
        """Display heatmap of position counts of the agent."""
        max_col = max(key[0] for key in self.pos_count.keys())
        max_row = max(key[1] for key in self.pos_count.keys())
        data = pd.DataFrame(0, index=range(max_row + 1), columns=range(max_col + 1))

        for (col, row), value in self.pos_count.items():
            if value == 0:
                value = None
            data.iat[row, col] = value

        plt.figure(figsize=(self.env_settings.width, self.env_settings.height))
        axes = sns.heatmap(data, cbar=True, cmap="Greens")
        return axes.get_figure()

    def init_render(self) -> None:
        """Initialise render objects."""
        pygame.init()
        pygame.display.init()
        window_size = (
            self.render_settings.window_width,
            self.render_settings.window_height,
        )
        self.render_settings.window = pygame.display.set_mode(window_size)
        self.render_settings.clock = pygame.time.Clock()

    def load_walls(self, positions: np.ndarray) -> np.ndarray:
        """Convert array of positions to wall objects for environment.

        Parameters
        ----------
        positions : np.ndarray
            Array of positions, where wall objects shall be placed. Wall positions of standard
            curiosity-gym environments are stored in :mod:`curiosity_gym.utils.constants`.

        Returns
        -------
        np.ndarray
            Array of wall objects.
        """
        walls = [Wall(position) for position in positions]
        return np.array(walls)

    def simulate(self, action: int | Action | SimplerAction) -> np.ndarray:
        """Simulate the state of the environment if a given action were taken.\n
        Does not change the actual state of the environment.

        .. warning::
            For most applications, it is not advisable to use this function for training RL agents,
            as it allows direct access to the dynamics of the environment.

        Parameters
        ----------
        action : int | :type:`~curiosity_gym.utils.enums.Action`
            Action to simulate.

        Returns
        -------
        np.ndarray
            State of the environment after simulated action.
        """
        state = np.zeros([self.env_settings.width * self.env_settings.height, 3])
        for ob in self.objects.get_all():
            ob_simulated = ob.simulate(
                SimplerAction(action) if self.env_settings.simple_actions else Action(action),
                self.find_object(self.objects.agent.get_front()),
                self._check_walkable(self.objects.agent.get_front()),
            )
            x, y = ob_simulated.position
            state[x + y * self.env_settings.width] = ob_simulated.get_identity()
        return state

    def _check_harmful(self, position: np.ndarray) -> bool:
        for ob in self.objects.other:
            if np.all(ob.position == position) and ob.is_harmful():
                return True
        return False

    def _check_walkable(self, position: np.ndarray) -> bool:
        inbounds_horizontal = 0 < position[0] < self.env_settings.width
        inbounds_vertical = 0 < position[1] < self.env_settings.height

        if not inbounds_horizontal or not inbounds_vertical:
            return False

        for ob in self.objects.get_all():
            if np.all(ob.position == position) and not ob.is_walkable():
                return False
        return True

    def _get_info(self) -> dict[str, Any]:
        return {"Current Steps": self.step_count}

    def _get_obs(self) -> np.ndarray:
        obs = self.agent_pov.transform_obs(self.get_state(), self.objects.agent)
        if (self.env_settings.simple_obs):
            obs = self._simplifiey_obs(obs)
        print("OBS WWW", obs.shape)
        return obs

    def _simplifiey_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        simplified_state = np.zeros(len(raw_obs), dtype=int)

        # This works as long as (id, colour) from a primary key regarding objects
        for idx, cell in enumerate(raw_obs):
            simplified_state[idx] = cell[0] + cell[1] + cell[2]

        return simplified_state





    def _get_terminated(self) -> bool:
        return self._check_harmful(self.objects.agent.position) or self.check_task()

    def _get_truncated(self) -> bool:
        return self.step_count >= self.env_settings.max_steps

    def _init_pov(self, agent_pov: AgentPOV | str) -> AgentPOV:
        assert isinstance(
            agent_pov, (AgentPOV, str)
        ), f"Invalid type of agent_pov: {type(agent_pov)}"

        if isinstance(agent_pov, AgentPOV):
            return agent_pov

        # Construct pov by string
        xray = False
        if agent_pov.lower() == "global":
            return GlobalView((self.env_settings.width, self.env_settings.height), self.env_settings.simple_obs)

        if agent_pov.lower().startswith("local_"):
            radius = agent_pov[6:]

            if radius.lower().startswith("xray_"):
                xray = True
                radius = radius[5:]

            assert (
                radius.isnumeric() and int(radius) >= 0
            ), f"Invalid radius for local pov: {radius}"
            return LocalView(
                int(radius), (self.env_settings.width, self.env_settings.height), xray, self.env_settings.simple_obs
            )

        if agent_pov.lower().startswith("forward_"):
            pov_width = 1
            pov_length = agent_pov[8:]

            if pov_length.lower().startswith("xray_"):
                xray = True
                pov_length = pov_length[5:]
            if "_" in pov_length:
                pov_length, pov_width = pov_length.split("_")
            assert (
                str(pov_length).isnumeric() and int(pov_length) >= 0
            ), f"Invalid length for forward pov: {pov_length}."
            assert (
                str(pov_width).isnumeric() and int(pov_width) >= 0
            ), f"Invalid width for forward pov: {pov_width}."
            assert (
                int(pov_width) % 2 == 1
            ), f"Invalid width {pov_width} for pov. Width must be odd."
            return ForwardView(
                int(pov_length),
                int(pov_width),
                (self.env_settings.width, self.env_settings.height),
                xray,
                self.env_settings.simple_obs
            )

        raise ValueError(f"Invalid agent pov: {agent_pov}.")

    def _render_frame(self) -> np.ndarray | None:
        # Define canvas for new Frame
        pygame.init()
        window_size = (
            self.render_settings.window_width,
            self.render_settings.window_height,
        )
        canvas = pygame.Surface(window_size)
        canvas.fill((255, 255, 255))

        # Draw objects
        tilesize = window_size[0] / self.env_settings.width
        for ob in self.objects.get_all():
            ob.render(canvas, tilesize)

        # Draw grid lines
        line_color = (210, 210, 210)
        for x in range(self.env_settings.height + 1):
            pygame.draw.line(
                canvas,
                line_color,
                (0, tilesize * x),
                (window_size[0], tilesize * x),
                width=3,
            )
        for y in range(self.env_settings.width + 1):
            pygame.draw.line(
                canvas,
                line_color,
                (tilesize * y, 0),
                (tilesize * y, window_size[1]),
                width=3,
            )

        # Add overlay for visible cells
        for pos in self.agent_pov.visible_positions:
            overlay = pygame.Surface((tilesize, tilesize))
            overlay.set_alpha(30)
            overlay.fill((255, 153, 20))
            canvas.blit(overlay, (pos[0] * tilesize, pos[1] * tilesize))

        # Display canvas in window
        if self.render_settings.render_mode == "human":
            assert self.render_settings.window is not None, "No window defined"
            assert self.render_settings.clock is not None, "No clock defined"
            self.render_settings.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.render_settings.clock.tick(self.render_settings.render_fps)

        # Return for rgb_array
        elif self.render_settings.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        return None
    
    @property
    def name(self) -> str:
        """Getter for the environemt name"""
        return self._name
