from typing_extensions import override
from curiosity_gym.core.objects.grid_object import GridObject
from curiosity_gym.core.objects.agent import Agent

import numpy as np
from curiosity_gym.utils.constants import IX_TO_COLOR
import pygame

class Door(GridObject):
    """Grid object with walkability depending on its state. \n
    A closed door cannot be walked or seen through. Doors can be opened
    by the agent through interaction. State 2 represents a locked door,
    that can only be opened if the agent has collected a :class:`Key`
    object of the same color. \n
    Can be used in combination with :class:`Key` objects to introduce
    additional reward sparcity to an environment, as more precice
    actions are required for the agent to get the reward.

    """

    def __init__(self, position: tuple[int, int], color: int = 0, state: int = 0, use_colour: bool = True) -> None:
        super().__init__(position, color, state)
        self.use_colour = use_colour

    @override
    def interact(self, agent: Agent):
        """Interact with the door.
        The result of the interaction depends on the door state prior to
        the interaction. Closed doors (state = 1) can always be opened
        by interaction. Locked doors (state = 2) require the prior
        collection of a :class:`Key` object of the same color to open.

        Parameters
        ----------
        agent : :class:`~Agent`
            Agent object performing the interaction.
        """
        if (self.use_colour):
            if self.state == 2 and agent.color != self.color:
                return
            if self.state == 2 and agent.color == self.color:
                self.state = 0
                agent.color = agent.start_color
            else:
                self.state = (self.state + 1) % 2
        else:
            if self.state == 2 and not agent.carrys_key:
                return
            if self.state == 2 and agent.carrys_key:
                self.state = 0
                agent.carrys_key = False
                #agent.color = agent.start_color
            else:
                self.state = (self.state + 1) % 2

    @override
    def render(self, canvas: pygame.Surface, pixelsquare: float) -> None:
        if self.state > 0:
            pygame.draw.rect(
                canvas,
                IX_TO_COLOR[self.color],
                pygame.Rect(
                    pygame.Vector2(*(pixelsquare * self.position)),
                    (pixelsquare, pixelsquare),
                ),
                5,
            )

            pygame.draw.circle(
                canvas,
                IX_TO_COLOR[self.color],
                pygame.Vector2(
                    *((self.position + np.array([0.75, 0.5])) * pixelsquare)
                ),
                0.1 * pixelsquare,
            )

        else:
            pygame.draw.rect(
                canvas,
                IX_TO_COLOR[self.color],
                pygame.Rect(
                    pygame.Vector2(*(pixelsquare * self.position)),
                    (pixelsquare * 0.2, pixelsquare),
                ),
                5,
            )

    @override
    def is_walkable(self) -> bool:
        """Return the walkability of a door, as determined by its state.
        The agent can only walk through the door if it is open.

        Returns
        -------
        bool
            Returns :const:`True` if the state of the door is 0,
            :const:`False` otherwise.
        """
        return self.state == 0
