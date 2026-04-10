"""Definition of enums used in the grid environments."""

from enum import Enum


class Action(Enum):
    """Available actions in a curiosity-gym grid environment."""

    FORWARD = 0
    TURN_RIGHT = 1
    TURN_LEFT = 2
    INTERACT = 3

class SimplerAction(Enum):
    """Available actions in a curiosity-gym grid environment."""

    FORWARD = 0
    MOVE_RIGHT = 1
    MOVE_LEFT = 2
    INTERACT = 3

class Rotation(Enum):
    """Possible rotations in a curiosity-gym grid environment."""

    RIGHT = 0
    UP = 50
    LEFT = -50
    DOWN = 100
