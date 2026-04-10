"""Definitions for inputs and outputs of the curiosity-gym grid environments.

This module defines point-of-view classes that implement the processing of 
inputs going into a grid environment (actions) and outputs coming out of a
grid environment (observations). By default, there are three distinct pov
classes. The GlobalView always returns the full state of an environment.
The observation of LocalView only contains grid cells that are inside a
given radius of the agent. The ForwardView only contains grid cells that
are in front of the agent within a field of given width and length. 
LocalView and ForwardView only provide information about a cell if there 
is no wall or closed door between the agent and the particular cell.
"""

from abc import ABC, abstractmethod

import numpy as np
from gymnasium import spaces

from curiosity_gym.core.objects import Agent, Wall, Door
from curiosity_gym.utils.enums import Action, SimplerAction


class AgentPOV(ABC):
    """Abstract agent point-of-view class. \n
    Implements the processing of inputs going into a grid environment (actions)
    and outputs coming out of a grid environment (observations). The :meth:`transform_obs`
    method has to be implemented by inheriting pov classes.

    Parameters
    ----------
    action_space : gymnasium.spaces.Discrete
        The action space defining what actions can be taken by an RL agent within a grid
        environment.
    observation_space : gymnasium.spaces.MultiDiscrete
        The observation space defining the structure of the observations being returned
        by a grid environment.
    env_size : tuple[int, int]
        Number of (horizontal, vertical) cells in the grid environment where the pov class is used.
    """

    def __init__(
        self,
        action_space: spaces.Discrete,
        observation_space: spaces.Box,
        env_size: tuple[int, int],
    ) -> None:
        self.action_space = action_space
        self.observation_space = observation_space
        self.width = env_size[0]
        self.height = env_size[1]
        self.visible_positions = []

    @abstractmethod
    def transform_obs(self, state: np.ndarray, agent: Agent) -> np.ndarray:
        """Transform environment state to agent observation.

        Parameters
        ----------
        state : np.ndarray
            State provided by the grid environment.
        agent : objects.Agent
            Agent grid object within the grid environment. Is required to construct
            observations in relation to the agent objects position and rotation.

        Returns
        -------
        np.ndarray
            Observation within the :attr:`~observation_space`.
        """

    def transform_action(self, action: int | Action | SimplerAction, simple_actions: bool) -> Action | SimplerAction:
        """Transform given action so that it is compatible with the grid environment. \n
        Can be used to define alternative action spaces and map them to the grid dynamics.

        Parameters
        ----------
        action : int | :type:`~curiosity_gym.utils.enums.Action`
            Action selected by the RL agent.

        Returns
        -------
        :type:`~curiosity_gym.utils.enums.Action`
            Action within :attr:`action_space` to perform.
        """

        return SimplerAction(action) if simple_actions else Action(action)

    def is_visible(
        self,
        state: np.ndarray,
        pos_agent: tuple[int, int],
        pos_cell: tuple[int, int],
    ) -> bool:
        """Check if grid cell at given position is visible by the agent. \n
        A grid cell is visible if there is no wall or closed door between the agent and the cell.

        Parameters
        ----------
        state : np.ndarray
            Current environment state.
        pos_agent : tuple[int,int]
            Position of the agent grid object.
        pos_cell : tuple[int,int]
            Position of the cell to check for visibility.

        Returns
        -------
        bool
            True if there is no wall or closed door between agent and the given cell,
            False otherwise.
        """
        dx = np.sign(pos_cell[0] - pos_agent[0])
        dy = np.sign(pos_cell[1] - pos_agent[1])
        x, y = pos_agent[0] + dx, pos_agent[1] + dy

        while (x, y) != (pos_cell[0], pos_cell[1]):
            if (
                state[x + y * self.width][0] in Wall.instance_map
                or state[x + y * self.width][0] in Door.instance_map
                and state[x + y * self.width][2] == 2
            ):
                return False

            x += dx if x != pos_cell[0] else 0
            y += dy if y != pos_cell[1] else 0
        return True
