"""Definition of dataclasses that are used for grouping related arguments of grid environments."""

from dataclasses import dataclass, field

import numpy as np
import pygame

from curiosity_gym.core.objects import GridObject, Wall, Agent, Target


@dataclass
class EnvironmentSettings:
    """Stores settings about the size and step count of an environment.

    Parameters
    ----------
    min_steps : int
        Minimum amount of steps needed to complete the environment task.
    max_steps : int
        Maximum number of steps allowed before the episode is truncated.
    width : int
        Number of horizontal cells in the grid environment.
    height : int
        Number of vertical cells in the grid environment.
    reward_range : tuple[int,int]
        Range of total rewards within an episode.
    """

    min_steps: int
    max_steps: int = 50
    width: int = 10
    height: int = 10
    reward_range: tuple[int, int] = (0, 1)
    simple_actions: bool = False
    simple_obs: bool = True


@dataclass
class RenderSettings:
    """Stores settings defining the rendering of an environment.

    Parameters
    ----------
    render_mode : str | None
        Render mode in which the environment is run. If render mode is *human*, the
        environment will be rendered in PyGame. By default None.
    render_fps : int
        Number of frames per second at which the PyGame rendering operates. By default 4.
    window_width : int
        Horizontal size of the PyGame window in *human* render mode. By default 512.
    window_height : int
        Vertical size of the PyGame window in *human* render mode. By default 512.
    window : pygame.surface.Surface | None
        PyGame window object used in *human* render mode. By default None.
    clock: pygame.time.Clock | None
        PyGame clock object used in *human* render mode. By default None.
    """

    render_mode: str | None = None
    render_fps: int = 4
    window_width: int = 512
    window_height: int = 512
    window: pygame.surface.Surface | None = None
    clock: pygame.time.Clock | None = None


@dataclass
class EnvironmentObjects:
    """Stores grid objects placed in an environment.

    Parameters
    ----------
    agent : :class:`~curiosity_gym.core.objects.Agent`
        Agent grid object of an environment.
    target : :class:`~curiosity_gym.core.objects.Target`
        Target grid object of an environment, used for navigation tasks.
    walls : np.ndarray
        List of wall grid objects of an environment.
    other : np.ndarray
        List of all other grid objects placed in an environment.
    """

    agent: Agent
    target: Target
    walls: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=Wall))
    other: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=GridObject)
    )

    def get_all(self) -> np.ndarray:
        """Returns all stored grid objects.

        Returns
        -------
        np.ndarray
            Numpy array containing all stored grid objects.
        """
        return np.concatenate((self.get_non_wall(), self.walls))

    def get_non_wall(self) -> np.ndarray:
        """Returns all stored grid objects, except wall objects.

        Returns
        -------
        np.ndarray
            Numpy array containing all stored grid objects, except wall objects.
        """
        return np.concatenate(
            (np.array([self.target]), self.other, np.array([self.agent]))
        )
