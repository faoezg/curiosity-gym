from typing_extensions import override
from curiosity_gym.core.objects.agent import Agent 
from curiosity_gym.core.objects.grid_object import GridObject

import numpy as np
from curiosity_gym.utils.constants import IX_TO_COLOR
import pygame

class Key(GridObject):
    """Collectable grid object that is used to unlock :class:`Door` objects of the same color."""
    def __init__(self, position: tuple[int, int], color: int = 0, state: int = 0, use_colour: bool = True) -> None:
        super().__init__(position, color, state)
        self.use_colour = use_colour

    @override
    def interact(self, agent: Agent) -> bool:
        """Collect the key object and remove it from the environment.
        The agent changes color according to the key collected.

        Parameters
        ----------
        agent : :class:`~Agent`
            Agent object performing the interaction.
        """
        if (self.use_colour):
            agent.color = self.color
        else:
            agent.carrys_key = True

        self.position = np.array([-5, -5])
        return True

    @override
    def render(self, canvas: pygame.Surface, pixelsquare: float) -> None:
        pygame.draw.circle(
            canvas,
            IX_TO_COLOR[self.color],
            pygame.Vector2(*((self.position + np.array([0.5, 0.4])) * pixelsquare)),
            0.15 * pixelsquare,
            5,
        )
        pygame.draw.line(
            canvas,
            IX_TO_COLOR[self.color],
            pygame.Vector2(*((self.position + np.array([0.5, 0.55])) * pixelsquare)),
            pygame.Vector2(*((self.position + np.array([0.5, 0.8])) * pixelsquare)),
            5,
        )
        pygame.draw.line(
            canvas,
            IX_TO_COLOR[self.color],
            pygame.Vector2(*((self.position + np.array([0.5, 0.7])) * pixelsquare)),
            pygame.Vector2(*((self.position + np.array([0.4, 0.7])) * pixelsquare)),
            5,
        )
