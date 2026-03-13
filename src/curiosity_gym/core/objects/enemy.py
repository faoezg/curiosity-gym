from typing_extensions import override
from curiosity_gym.core.objects.grid_object import GridObject

import numpy as np
from curiosity_gym.utils.enums import Action, SimplerAction
from curiosity_gym.utils.constants import IX_TO_COLOR, STATE_TO_ROTATION
import pygame

class Enemy(GridObject):
    """Moving grid object harmful to the agent. \n
    Can be used to introduce additional reward sparcity by early episode termination.

    Parameters
    ----------
    position : tuple[int,int]
        Position in the grid where the object will be placed. Values must be in range
        (:attr:`~curiosity_gym.core.gridengine.env_settings.width` - 1,
        :attr:`~curiosity_gym.core.gridengine.env_settings.height` - 1).
    state : int
        State of the enemy grid object. Determines current movement direction.
    reach : int
        Number of cells the enemy can move from its starting position.
    """

    @override
    def __init__(
        self, position: tuple[int, int], state: int = 0, reach: int = 2
    ) -> None:
        super().__init__(position, 9, state)
        self.reach = reach

    @override
    def is_harmful(self) -> bool:
        """The enemy grid object is always harmful to the agent.\n
        This will terminate the episode if agent and an enemy are
        on the same grid position.
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
        """Perform enemy movement.
        Behaviour is determined by starting position and reach of the enemy grid object.

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
            Returns 0.
        """
        if self.reach > 0:
            self.position = self.position + STATE_TO_ROTATION[self.state] * np.array(
                [1, -1]
            )
        if (
            self.position[self.state % 2]
            == self.start_position[self.state % 2] + self.reach
            or self.position[self.state % 2]
            == self.start_position[self.state % 2] - self.reach
            or self.position[self.state % 2] == self.start_position[self.state % 2]
        ):
            self.state = (self.state + 2) % 4
        return 0

    @override
    def is_walkable(self) -> bool:
        """The enemy grid object is always walkable.
        Allows for episode termination by enemy contact.

        Returns
        -------
        bool
            :const:`True`
        """
        return True

    @override
    def render(self, canvas: pygame.Surface, pixelsquare: float) -> None:
        pygame.draw.polygon(
            canvas,
            IX_TO_COLOR[self.color],
            (
                pygame.Vector2(
                    *((self.position + np.array([0.25, 0.2])) * pixelsquare)
                ),
                pygame.Vector2(
                    *((self.position + np.array([0.75, 0.2])) * pixelsquare)
                ),
                pygame.Vector2(
                    *((self.position + np.array([0.85, 0.55])) * pixelsquare)
                ),
                pygame.Vector2(
                    *((self.position + np.array([0.5, 0.85])) * pixelsquare)
                ),
                pygame.Vector2(
                    *((self.position + np.array([0.15, 0.55])) * pixelsquare)
                ),
            ),
        )
        pygame.draw.circle(
            canvas,
            (255, 255, 255),
            pygame.Vector2(*((self.position + np.array([0.35, 0.37])) * pixelsquare)),
            5,
        )
        pygame.draw.circle(
            canvas,
            (255, 255, 255),
            pygame.Vector2(*((self.position + np.array([0.65, 0.37])) * pixelsquare)),
            5,
        )
        pygame.draw.line(
            canvas,
            (255, 255, 255),
            pygame.Vector2(*((self.position + np.array([0.4, 0.63])) * pixelsquare)),
            pygame.Vector2(*((self.position + np.array([0.6, 0.63])) * pixelsquare)),
            3,
        )
