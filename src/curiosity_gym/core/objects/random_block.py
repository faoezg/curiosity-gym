from typing_extensions import override
from curiosity_gym.core.objects.grid_object import GridObject

import numpy as np
from curiosity_gym.utils.enums import Action, SimplerAction
from curiosity_gym.utils.constants import IX_TO_COLOR
import pygame
import random

class RandomBlock(GridObject):
    """A non walkable grid object that randomly changes color at every time step. \n
    Can be used to test the `Noisy-TV problem <https://openai.com/index/reinforcement
    -learning-with-prediction-based-rewards/#the-noisy-tv-problem>`__, where certain
    curiosity-based RL algorithms get stuck when encountering stochastic environment
    components.
    """

    @override
    def step(
        self,
        action: Action | SimplerAction,
        front_object: GridObject | None = None,
        walkable: bool = False,
    ) -> float:
        """Randomly change grid object color.
        Available colors are taken from :const:`~curiosity_gym.utils.constants.IX_TO_COLOR`.
        """
        self.color = random.randint(0, len(IX_TO_COLOR) - 1)
        return 0

    def render(self, canvas: pygame.Surface, pixelsquare: float) -> None:
        pygame.draw.rect(
            canvas,
            IX_TO_COLOR[self.color],
            pygame.Rect(
                pygame.Vector2(
                    *((self.position) * pixelsquare)
                    + np.array([0.15, 0.15]) * pixelsquare
                ),
                (pixelsquare * 0.7, pixelsquare * 0.7),
            ),
        )
        font = pygame.font.SysFont("freesansbold", int(pixelsquare))
        img = font.render("?", True, (255, 255, 255))
        canvas.blit(
            img,
            pygame.Vector2(*((self.position + np.array([0.275, 0.2])) * pixelsquare)),
        )
