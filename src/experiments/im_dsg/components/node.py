# Node n has an option o_n which terminates in state s
# state s may be another Node
# Nodes keep track of where there outgoing options (N -> o_n -> s) terminates

import numpy as np

class Node():

    _next_id = 1

    def __init__(self) -> None:
        self.identifier = Node._next_id
        Node._next_id += 1

        # TODO is this right? Is it the outgoing, or incomming ones?
        self.terminal_states = [] # the outgoing option o_n terminated here, |Eff(o_n)|
    
    # TODO is this correct? What is this used for? The expansion node determination? the AMDP?
    def get_propabliity_to_terminate_in_state_after_option(self, state):
        propability = 0.0
        if (state in self.terminal_states):
            propability = 1/len(self.terminal_states)
        return propability

    def get_goal_definition(self) -> np.ndarray: # beta in the paper, the condition after which the nodes, rather the opations "goal" has been reached
        return self.terminal_states[0] # TODO Perhaps do a random sample? The paper is not clear about this...
    