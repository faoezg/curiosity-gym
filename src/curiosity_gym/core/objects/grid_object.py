"""Object definitions for curiosity-gym grid environments.

This module defines a collection of classes representing various objects for the grid-based
RL environments. Each object can define its own properties, behaviors, and visual representation 
within the environment.
"""

import copy
from abc import ABC, abstractmethod
from typing_extensions import Self

import numpy as np
import pygame

from curiosity_gym.utils.enums import Action, SimplerAction

import random

class GridObject(ABC):
    """Abstract class representing elements that can be placed in a grid environment. \n
    It contains the attributes position, color, and state, which define the characteristics
    and behavior of the object. The class maintains a unique identifier for each subclass
    and provides default implementations of the :meth:`~reset`, :meth:`~simulate`,
    :meth:`~get_identity`, :meth:`~is_walkable` and :meth:`~is_harmful` methods. It enforces
    the implementation of a :meth:`~render` method by all subclasses.

    Parameters
    ----------
    position : tuple[int,int]
        Position in the grid where the object will be placed. Values must be in range
        (:attr:`~curiosity_gym.utils.dataclasses.EnvironmentSettings.width` - 1,
        :attr:`~curiosity_gym.utils.dataclasses.EnvironmentSettings.height` - 1).
    color : int
        Color of the grid object. Values must be in range(0,10).
        Color mappings are defined in :const:`~curiosity_gym.utils.constants.IX_TO_COLOR`.
    state : int
        State of the grid object. State characteristics vary by object type.
        Values must be in range(0,4).
    """

    # Class-level attributes
    identifier = None
    """Unique id number for each subclass."""
    id_map = {}
    """Dictionary for all ids and their corresponding subclasses."""
    _next_id = 1

    def __init__(
        self, position: tuple[int, int], color: int = 0, state: int = 0
    ) -> None:
        self.start_position = np.array(position)
        self.position = np.array(position)
        self.start_color = color
        self.color = color
        self.start_state = state
        self.state = state

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.identifier = GridObject._next_id
        GridObject.id_map[GridObject._next_id] = cls
        GridObject._next_id += 1

    @abstractmethod
    def render(self, canvas: pygame.Surface, pixelsquare: float) -> None:
        """Render grid object in PyGame. \n
        Must be implemented by each grid object type to allow visual recognition
        in *human* rendering mode.

        Parameters
        ----------
        canvas : pygame.Surface
            The PyGame window, in which the grid objects are rendered.
        pixelsquare : float
            The size of a single square in the grid environment.
        """

    def get_identity(self) -> tuple[int | None, int, int]:
        """Return a tuple that identifies the grid object and its state in the enviroment.

        Returns
        -------
        tuple[int | None, int, int]
            A tuple consisting of :attr:`~identifier`, :attr:`~color` and :attr:`~state`.
        """
        return (self.identifier, self.color, self.state)

    def interact(self, agent: Self) -> None:
        """Interact with agent. \n
        Concrete interaction depends on grid object type. The default interaction has no effect.

        Parameters
        ----------
        agent : :class:`~Agent`
            Agent object performing the interaction.
        """

    def reset(self) -> None:
        """Reset attributes of the grid object to their starting values. \n
        Sets :attr:`~position`, :attr:`~state` and :attr:`~color` to their respective values
        assigned when the object was initialised.
        """
        self.position = self.start_position
        self.state = self.start_state
        self.color = self.start_color

    def step(
        self,
        action: Action | SimplerAction,
        front_object: Self | None = None,
        walkable: bool = False,
    ) -> float:
        """Compute grid object changes after single timestep. \n
        The default step behaviour is idle and gives 0 reward.

        Parameters
        ----------
        action : :type:`~curiosity_gym.utils.enums.Action`
            Action taken in a grid environment by the RL agent.
        front_object : :type:`GridObject` | None, optional
            Grid object currently in front of the RL agent, by default None.
        walkable : bool, optional
            Wheter the agent can move over the cell in front, by default False.

        Returns
        -------
        float
            Optional reward, which is independent of the environment task and episode termination.
        """
        del action, front_object, walkable
        return 0

    def simulate(
        self,
        action: Action | SimplerAction,
        front_object: Self | None = None,
        walkable: bool = False,
    ) -> Self:
        """Simulate how grid object would change if a given action were taken. \n
        Is used in the :meth:`curiosity_gym.core.gridengine.simulate` method to get the state of
        the environment if a given action is performed by the RL agent.

        Parameters
        ----------
        action : :type:`~curiosity_gym.utils.enums.Action`
            Action of the RL agent to be simulated.
        front_object : :type:`GridObject` | None, optional
            Grid object currently in front of the RL agent, by default None.
        walkable : bool, optional
            Wheter the agent can move over the cell in front, by default False.

        Returns
        -------
        :type:`GridObject`
            Copy of the grid object, including changes caused by the given action.
        """
        simulated = copy.deepcopy(self)
        simulated.step(action, front_object, walkable)
        return simulated

    def is_walkable(self) -> bool:
        """Determine whether agent can move on grid object.\n
        Returns :const:`False` by default. Walkable grid objects must override this method.

        Returns
        -------
        bool
            :const:`True` if agent can move over grid object, :const:`False` otherwise.
        """
        return False

    def is_harmful(self) -> bool:
        """Determine whether grid object is harmful to the agent. \n
        Returns :const:`False` by default. Harmful grid objects must override this method.
        Value is used in :class:`curiosity_gym.core.gridengine` to determine if episode ended.

        Returns
        -------
        bool
            :const:`True` if grid object is harmful, :const:`False` otherwise.
        """
        return False
