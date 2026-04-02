# Node n has an option o_n which terminates in state s
# state s may be another Node
# Nodes keep track of where there outgoing options (N -> o_n -> s) terminates

import numpy as np
from .node import Node
from typing import override

# Special node to account for state which are not part of a Node yet, e.g. "falling of the edge of the graph"
class NodePhi(Node):

    def __init__(self) -> None:
        self.identifier = -1
        self.terminal_states = [] # the outgoing option o_n terminated here, |Eff(o_n)|
    
    @override
    def get_propabliity_to_terminate_in_state_after_option(self, state):
        propability = 0.0
        return propability

    @override
    def is_goal_achieved(self, state: np.ndarray) -> bool: # beta in the paper, the condition after which the nodes, rather the opations "goal" has been reached
        return False
    