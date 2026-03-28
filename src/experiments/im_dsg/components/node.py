# Node n has an option o_n which terminates in state s
# state s may be another Node
# Nodes keep track of where there outgoing options (N -> o_n -> s) terminates

class Node():

    _next_id = 1

    def __init__(self) -> None:
        self.identifier = Node._next_id
        Node._next_id += 1

        self.terminal_states = [] # the outgoing option o_n terminated here, |Eff(o_n)|
    
    def get_propabliity_to_terminate_in_state_after_option(self, state):
        propability = 0.0
        if (state in self.terminal_states):
            propability = 1/len(self.terminal_states)
        return propability


    