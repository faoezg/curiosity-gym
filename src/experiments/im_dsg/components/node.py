# Node n has an option o_n which terminates in state s
# state s may be another Node
# Nodes keep track of where there outgoing options (N -> o_n -> s) terminates

import numpy as np

import random

class Node():

    _next_id = 0

    def __init__(self) -> None:
        self.identifier = Node._next_id
        Node._next_id += 1

        # TODO is this right? Is it the outgoing, or incomming ones?
        self.terminal_states: list[np.ndarray] = [] # the outgoing option o_n terminated here, |Eff(o_n)|
    
    def add_terminal_state(self, state: np.ndarray):
        if (not self.is_goal_achieved(state)):
            self.terminal_states.append(state)

    # TODO is this correct? What is this used for? The expansion node determination? the AMDP?
    def get_propabliity_to_terminate_in_state_after_option(self, state):
        propability = 0.0
        if (state in self.terminal_states):
            propability = 1/len(self.terminal_states)
        return propability

    def is_goal_achieved(self, state: np.ndarray) -> bool: # beta in the paper, the condition after which the nodes, rather the opations "goal" has been reached
        for terminal_state in self.terminal_states:
            if (np.all(np.equal(state, terminal_state))):
                return True
        return False
    
    def get_goal_state(self) -> np.ndarray:
        return random.choice(self.terminal_states) # TODO is this realy okay??
