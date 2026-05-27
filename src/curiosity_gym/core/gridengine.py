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
import math
import random

from curiosity_gym.core.objects import GridObject, Wall, ObjectState, Agent
from curiosity_gym.core.pov import AgentPOV, GlobalView, LocalView, ForwardView
from curiosity_gym.utils.enums import Action, SimplerAction
from curiosity_gym.utils.dataclasses import (
    EnvironmentSettings,
    RenderSettings,
    EnvironmentObjects,
)
from curiosity_gym.utils.utils import one_hot_encode, one_hot_encode_zero_indexed
from curiosity_gym.utils.constants import IX_TO_COLOR

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

        # Initialise agent pov
        self.agent_pov = self._init_pov(agent_pov)
        self.label_count_per_cell = self.agent_pov.get_cell_label_count()
        self.action_space = self.agent_pov.action_space
        """Space of possible actions a RL agent can choose from."""
        self.observation_space = self.agent_pov.observation_space
        """Space of possible observations returned by the environment."""


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
        
        #self._change_object_colours()

        return (
            obs,
            reward,
            self._get_terminated(),
            self._get_truncated(),
            self._get_info(),
        )
    
    def step_with_global_state(self, action):
        obs, reward, terminated, truncated, info = self.step(action)
        raw_global_obs = self.get_raw_state()
        return obs, reward, terminated, truncated, info, raw_global_obs

    def step_with_rgb_state(self, action):
        _, reward, terminated, truncated, info = self.step(action)
        rgb_state = self._render_frame(with_view_overlay=False)
        return rgb_state, reward, terminated, truncated, info

    def _calc_reward(self, action: int | Action | SimplerAction):
        external_reward = self._calc_obj_reward(action) + self._calc_task_reward()
        return external_reward
   
    def _calc_obj_reward(self, action: int | Action | SimplerAction):
        reward = 0
        for ob in self.objects.get_non_wall():
            if (isinstance(ob, Agent) and self.env_settings.simple_actions):
                
                interactable_cells = self.agent_pov.get_interactable_cells(self.objects.agent.position)
                interactable_objs = []
                for obj in self.objects.other:
                    for cell in interactable_cells:
                        if cell[0] == obj.position[0] and cell[1] == obj.position[1]:
                            interactable_objs.append(obj)

                reward += ob.step(
                    action=self.agent_pov.transform_action(action, self.env_settings.simple_actions),
                    front_object=None,
                    walkable=self._check_walkable(self.objects.agent.get_front()),
                    walkable_behind=self._check_walkable(self.objects.agent.get_back()),
                    walkable_left=self._check_walkable(self.objects.agent.get_left()),
                    walkable_right=self._check_walkable(self.objects.agent.get_right()),
                    interactable_objs=interactable_objs
                )
            else:
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
    
    def _change_object_colours(self):
        for wall in self.objects.walls:
            random_color = random.choice(list(IX_TO_COLOR.keys()))
            wall.color = random_color

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
    
    def reset_with_rgb_state(self, **kwargs):
        _, info = self.reset(**kwargs)
        rbg_state = self._render_frame(with_view_overlay=False)
        return rbg_state, info
    
    def reset_to_specific_global_state(self, raw_state: list[ObjectState], **kwargs):
        _, inital_info = self.reset(**kwargs)
        obj_array: list[GridObject] = list(self.objects.other) + [self.objects.agent, self.objects.target]
        for grid_object in obj_array:
            for obj_state in raw_state:
                if id(grid_object) == obj_state.identifier:
                    grid_object.position[0] = obj_state.x_pos
                    grid_object.position[1] = obj_state.y_pos
                    grid_object.color = obj_state.color
                    grid_object.state = obj_state.state
        return self._get_obs(), inital_info


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
    
    def print_inital_env_state_as_pdf(self):
        self.reset()
        rgb = self.render()
        plt.figure(figsize=(rgb.shape[1] / 100, rgb.shape[0] / 100), dpi=100) # type: ignore
        plt.axis('off')
        plt.imshow(rgb) # type: ignore
        plt.tight_layout(pad=0)
        plt.savefig(f"{self.name}_init_state.pdf", format="pdf", bbox_inches='tight', pad_inches=0)
        plt.close()

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
            if (self.env_settings.use_globaly_unique_id):
                state[x + y * self.env_settings.width] = ob.get_unique_identity()
            else:
                state[x + y * self.env_settings.width] = ob.get_identity()
        return state

    def get_raw_state(self) -> list[ObjectState]:
        """Get the current state of the environment.\n
        The returned state is independent of the agent's :attr:`observation_space`.

        Returns
        -------
        state : np.ndarray
            Current state of the environment.
        """
        state = [] * self.env_settings.width * self.env_settings.height
        for ob in self.objects.get_all():
            x, y = ob.position
            assert x < self.env_settings.width and y < self.env_settings.height
            f"""Position [{x},{y}] of object with type {self.get_object_ids()[ob.identifier]}
            is invalid for grid with size ({self.env_settings.width}, {self.env_settings.height})"""
            state.append(ob.get_object_state()) 
        return state

    def overlay_heatmap_from_data(self, raw_data: dict[tuple[int,int], int]| dict[tuple[int,int], float]) -> Figure | None:
        max_col = max(key[0] for key in self.pos_count.keys())
        max_row = max(key[1] for key in self.pos_count.keys())
        data = pd.DataFrame(0, index=range(max_row + 1), columns=range(max_col + 1), dtype="float")

        for (col, row), value in raw_data.items():
            if value == 0:
                value = None
            data.iat[row, col] = value
        
        rgb_array = self._render_frame(False)
        _, axes = plt.subplots(figsize=(self.env_settings.width, self.env_settings.height))
        sns.heatmap(data, cbar=True, cmap="Greens", alpha=0.7, zorder=1, ax=axes, cbar_kws={"label": "Prozentverteilung intrinsischer Belohnung"})
        axes.imshow(rgb_array, zorder=0, extent=[0, data.shape[1], data.shape[0],0]) # type: ignore
        
        return axes.get_figure()
    
    def heatmap_from_data(self, raw_data: dict[tuple[int,int], int]| dict[tuple[int,int], float]) -> Figure | None:
        max_col = max(key[0] for key in self.pos_count.keys())
        max_row = max(key[1] for key in self.pos_count.keys())
        data = pd.DataFrame(0, index=range(max_row + 1), columns=range(max_col + 1), dtype="float")

        for (col, row), value in raw_data.items():
            if value == 0:
                value = None
            data.iat[row, col] = value

        plt.figure(figsize=(self.env_settings.width, self.env_settings.height))
        axes = sns.heatmap(data, cbar=True, cmap="Greens", cbar_kws={"label": "Prozentverteilung intrinsischer Belohnung"})
        return axes.get_figure()
    
    def heatmap(self) -> Figure | None:
        """Display heatmap of position counts of the agent."""

        max_col = max(key[0] for key in self.pos_count.keys())
        max_row = max(key[1] for key in self.pos_count.keys())
        data = pd.DataFrame(0, index=range(max_row + 1), columns=range(max_col + 1), dtype="float")

        for (col, row), value in self.pos_count.items():
            if value == 0:
                value = None
            else:
                value = math.log(value)
            data.iat[row, col] = value

        plt.figure(figsize=(self.env_settings.width, self.env_settings.height))
        axes = sns.heatmap(data, cbar=True, cmap="Blues", cbar_kws={"label": "Log-Anzahl Besichtigungen"})
        return axes.get_figure()

    def overlay_heatmap(self) -> Figure | None:
        """Display heatmap of position counts of the agent."""
        max_col = max(key[0] for key in self.pos_count.keys())
        max_row = max(key[1] for key in self.pos_count.keys())
        data = pd.DataFrame(0, index=range(max_row + 1), columns=range(max_col + 1), dtype="float")

        for (col, row), value in self.pos_count.items():
            if value == 0:
                value = None
            else:
                value = math.log(value)
            data.iat[row, col] = value
        
        rgb_array = self._render_frame(False)
        _, axes = plt.subplots(figsize=(self.env_settings.width, self.env_settings.height))
        sns.heatmap(data, cbar=True, cmap="Blues", alpha=0.7, zorder=1, ax=axes, cbar_kws={"label": "Log-Anzahl Besichtigungen"})
        axes.imshow(rgb_array, zorder=0, extent=[0, data.shape[1], data.shape[0],0]) # type: ignore
        
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

    def _check_walkable(self, position: np.ndarray | tuple[int, int]) -> bool:
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
        return obs
    
    def get_obs_by_state_and_agent_pos(self, agent_pos: tuple[int,int], colour: int, rotation: int) -> np.ndarray:
        actual_agent_pos = self.objects.agent.position
        actual_agent_colour = self.objects.agent.color
        actual_agent_rotation = self.objects.agent.state
        self.objects.agent.position = np.array(agent_pos)
        self.objects.agent.color = colour
        self.objects.agent.state = rotation

        if (not self.env_settings.use_rgb_state):
            obs = self.agent_pov.transform_obs(self.get_state(), self.objects.agent)
            if (self.env_settings.simple_obs):
                obs = self._simplifiey_obs(obs)
        else:
            obs = self._render_frame(with_view_overlay=False)

        self.objects.agent.position = actual_agent_pos
        self.objects.agent.color = actual_agent_colour
        self.objects.agent.state = actual_agent_rotation

        return obs

    def get_obs_by_state_and_agent_pos_as_rgb_state(self, agent_pos: tuple[int,int], colour: int, rotation: int) -> np.ndarray:
        actual_agent_pos = self.objects.agent.position
        actual_agent_colour = self.objects.agent.color
        actual_agent_rotation = self.objects.agent.state
        self.objects.agent.position = np.array(agent_pos)
        self.objects.agent.color = colour
        self.objects.agent.state = rotation
        obs = self._render_frame(with_view_overlay=False)

        self.objects.agent.position = actual_agent_pos
        self.objects.agent.color = actual_agent_colour
        self.objects.agent.state = actual_agent_rotation

        return obs
    
    
    def get_every_wakable_cor(self):
        walkable_cor = []
        for x in range(self.env_settings.width):
            for y in range(self.env_settings.height):
                cor = (x,y)
                if (self._check_walkable(cor)):
                    walkable_cor.append(cor)
        return walkable_cor

    def _simplifiey_obs(self, raw_obs: np.ndarray) -> np.ndarray:
        simplified_state = self._one_hot_encode_state(raw_obs)
        return simplified_state

    def _one_hot_encode_state(self, state: np.ndarray):
        if (not self.env_settings.use_globaly_unique_id):
            obj_class_count = len(GridObject.id_map.keys())
        else:
            obj_class_count = GridObject._next_instance_id - 1

        all_ids = state[:, 0]
        all_colours = state[:, 1]
        all_obj_states = state[:, 2]

        id_labels = range(1,obj_class_count+2)
        id_encoding = one_hot_encode(all_ids, id_labels)

        colour_labels = list(IX_TO_COLOR.keys())
        colour_encoding = one_hot_encode_zero_indexed(all_colours, colour_labels)

        obj_state_labels = range(1,obj_class_count * 3)
        obj_state_encoding = one_hot_encode_zero_indexed(all_obj_states, obj_state_labels)

        encoded_state = np.asarray([])
        for cell_idx in range(len(state)):
            cell_id_encoded = id_encoding[cell_idx]
            cell_colour_encoded = colour_encoding[cell_idx]
            cell_obj_state_encoded = obj_state_encoding[cell_idx]


            encoded_state = np.concatenate((encoded_state, cell_id_encoded, cell_colour_encoded, cell_obj_state_encoded), axis=None)
        
        return encoded_state

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
        
        if (self.env_settings.use_rgb_state):
            tmp = self._render_frame(with_view_overlay=False)
            if (tmp is not None):
                rgb_state_size = tmp.shape
            else:
                rgb_state_size = None
        else:
            rgb_state_size = None

        # Construct pov by string
        xray = False
        if agent_pov.lower() == "global":
            return GlobalView(
                (self.env_settings.width, self.env_settings.height),
                self.env_settings.simple_obs,
                self.env_settings.simple_actions,
                self.env_settings.use_globaly_unique_id,
                use_rgb_state=self.env_settings.use_rgb_state,
                rgb_state_size=rgb_state_size
            )

        if agent_pov.lower().startswith("local_"):
            radius = agent_pov[6:]

            if radius.lower().startswith("xray_"):
                xray = True
                radius = radius[5:]

            assert (
                radius.isnumeric() and int(radius) >= 0
            ), f"Invalid radius for local pov: {radius}"
            return LocalView(
                int(radius),
                (self.env_settings.width, self.env_settings.height),
                xray,
                self.env_settings.simple_obs,
                self.env_settings.simple_actions,
                self.env_settings.use_globaly_unique_id,
                use_rgb_state=self.env_settings.use_rgb_state,
                rgb_state_size=rgb_state_size
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
                self.env_settings.simple_obs,
                self.env_settings.simple_actions,
                self.env_settings.use_globaly_unique_id,
                use_rgb_state=self.env_settings.use_rgb_state,
                rgb_state_size=rgb_state_size
            )

        raise ValueError(f"Invalid agent pov: {agent_pov}.")

    def _render_frame(self, with_view_overlay: bool = True) -> np.ndarray | None:
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
        if (with_view_overlay):
            for pos in self.agent_pov.visible_positions:
                overlay = pygame.Surface((tilesize, tilesize))
                overlay.set_alpha(30)

                if pos in self.agent_pov.interactable_positions:
                    overlay.fill((155, 53, 120))
                else:
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
