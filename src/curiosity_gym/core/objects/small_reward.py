from typing_extensions import override
from curiosity_gym.core.objects.grid_object import GridObject

import numpy as np
from curiosity_gym.utils.enums import Action, SimplerAction
from curiosity_gym.utils.constants import IX_TO_COLOR
import pygame

class SmallReward(GridObject):
    """Grid object yielding a small reward when the agent moves over it.

    Parameters
    ----------
    position : tuple[int,int]
        Position in the grid where the object will be placed. Values must be in range
        (:attr:`~curiosity_gym.core.gridengine.env_settings.width` - 1,
        :attr:`~curiosity_gym.core.gridengine.env_settings.height` - 1).
    reward : float
        Amount of reward to yield when agent walks over the grid object.
    """

    @override
    def __init__(
        self,
        position: tuple[int, int],
        reward: float,
    ) -> None:
        super().__init__(position, 9, 0)
        self.reward = reward

    @override
    def is_walkable(self) -> bool:
        """SmallReward grid object is always walkable.
        Allows for the agent to collect the reward by walking over it.

        Returns
        -------
        bool
            :const:`True`
        """
        return True

    @override
    def step(
        self,
        action: Action | SimplerAction,
        front_object: GridObject | None = None,
        walkable: bool = False,
    ) -> float:
        """Determine whether agent is walking over the grid object.

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
            Returns :attr:`SmallReward.reward` if agent is moving over small reward
            object, 0 otherwise.
        """
        if front_object == self:
            self.position = np.array([-1, -1])
            return self.reward
        return 0

    @override
    def render(self, canvas: pygame.Surface, pixelsquare: float) -> None:
        pygame.draw.polygon(
            canvas,
            IX_TO_COLOR[self.color],
            (
                pygame.Vector2(
                    *((self.position + np.array([0.5, 0.15])) * pixelsquare)
                ),
                pygame.Vector2(
                    *((self.position + np.array([0.65, 0.35])) * pixelsquare)
                ),
                pygame.Vector2(
                    *((self.position + np.array([0.65, 0.65])) * pixelsquare)
                ),
                pygame.Vector2(
                    *((self.position + np.array([0.5, 0.85])) * pixelsquare)
                ),
                pygame.Vector2(
                    *((self.position + np.array([0.35, 0.65])) * pixelsquare)
                ),
                pygame.Vector2(
                    *((self.position + np.array([0.35, 0.35])) * pixelsquare)
                ),
            ),
        )
