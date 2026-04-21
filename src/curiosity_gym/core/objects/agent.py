from typing_extensions import override
from curiosity_gym.core.objects.grid_object import GridObject

import numpy as np
from curiosity_gym.utils.enums import Action, SimplerAction
from curiosity_gym.utils.constants import IX_TO_COLOR, STATE_TO_ROTATION
import pygame

class Agent(GridObject):
    """Grid object representing the RL agent. \n
    The color of the agent grid object cannot be specified at initialisation, as it is
    used to represent collected :class:`~Key` objects. The agent's color will always be
    initialised as 1 (orange).

    Parameters
    ----------
    position : tuple[int,int]
        Position in the grid where the agent object will be placed. Values must be in range
        (:attr:`~curiosity_gym.core.gridengine.env_settings.width` - 1,
        :attr:`~curiosity_gym.core.gridengine.env_settings.height` - 1).
    state : int
        :type:`Rotation` of the agent.
    """

    @override
    def __init__(self, position: tuple[int, int], state: int = 0) -> None:
        super().__init__(position, 1, state)

    @override
    def step(
        self,
        action: Action | SimplerAction,
        front_object: GridObject | None = None,
        walkable: bool = False,
    ) -> float:
        """Perform a given action. \n
        The agent is the main recipient of the action specified in the step function of
        the environment. For the action of moving forward, the agent considers the walkable
        parameter provided by the environment to determine if it can change its position
        accordingly. For the interaction action, the agent will call the :meth:`GridObject.interact`
        method of the object in front of it.

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

        #if action == Action.FORWARD and (walkable or front_object):
        if action == Action.FORWARD and walkable:
            self.position = self.position + STATE_TO_ROTATION[self.state] * np.array(
                [1, -1]
            )

        elif action == Action.TURN_RIGHT:
            self.state = (self.state - 1) % 4

        elif action == Action.TURN_LEFT:
            self.state = (self.state + 1) % 4

        elif action == Action.INTERACT and front_object:
            front_object.interact(self)
        
        elif action == SimplerAction.MOVE_RIGHT and walkable:
            self.state = (self.state - 1) % 4
            self.position = self.position + STATE_TO_ROTATION[self.state] * np.array(
                [1, -1]
            )
        elif action == SimplerAction.MOVE_LEFT and walkable:
            self.state = (self.state + 1) % 4
            self.position = self.position + STATE_TO_ROTATION[self.state] * np.array(
                [1, -1]
            )



        return 0

    @override
    def render(self, canvas: pygame.Surface, pixelsquare: float) -> None:
        # Constant parameters
        c = np.array([0.5, 0.5])
        d = 0.35
        r = STATE_TO_ROTATION[self.state]

        # Calculate points for triangle rotation
        p1 = (self.position + c - np.array([-d, d]) * r) * pixelsquare
        p2 = (self.position + c - [d, -d] * np.flip(r) + [-d, d] * r) * pixelsquare
        p3 = (self.position + c + [d, -d] * np.flip(r) + [-d, d] * r) * pixelsquare

        # Draw agent
        pygame.draw.polygon(
            canvas,
            IX_TO_COLOR[self.color],
            (pygame.Vector2(*p1), pygame.Vector2(*p2), pygame.Vector2(*p3)),
            0,  # filled triangle
        )

    def get_front(self) -> np.ndarray:
        """Calculate position in front of the agent grid object.

        Returns
        -------
        np.ndarray
            Grid coordinates of the position.
        """
        return self.position + STATE_TO_ROTATION[self.state] * np.array([1, -1])
