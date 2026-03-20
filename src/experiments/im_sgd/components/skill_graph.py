from collections import defaultdict
from .node import Node
from .node_phi import NodePhi

class SkillGraph():

    def __init__(self) -> None:
        self.adjacency_list: dict[int, dict[int, float]] = defaultdict(defaultdict) # going with list, as this should end up as a sparse dag
        self.nodes: list[Node] = []

    def add_node(self, node: Node):
        self.adjacency_list[node.identifier] = {}
        self.nodes.append(node)

    def remove_node(self, node: Node):
        self.adjacency_list.pop(node.identifier)
        self.nodes.remove(node)
    
    def add_edge(self, from_node: int, to_node: int):
        self.adjacency_list[from_node][to_node] = self._get_transition_probability_of_nodes(from_node, to_node)
    
    def remove_edge(self, from_node: int, to_node: int):
        self.adjacency_list[from_node].pop(to_node)

    def map_state_to_nodes(self, state) -> list[Node]:
        mapped_nodes = [] 
        for node in self.nodes:
            if (state in node.terminal_states):
                mapped_nodes.append(node)
        
        if (len(mapped_nodes) == 0):
            phi = NodePhi()
            mapped_nodes.append(phi)
        
        return mapped_nodes

    def get_descendants_of_node(self, node_identifier: int) -> list[Node]:
        descendants = []
        for node in self.nodes:
            if (node.identifier != node_identifier):
                if (self.has_path(node_identifier, node.identifier)):
                    descendants.append(node)
        return descendants
    
    def _get_transition_probability_of_nodes(self, from_node: int, to_node: int) -> float:
        return 0.3
    
    def has_path(self, start_node: int, end_node: int):
        visited = [False] * (len(self.nodes) + 1)

        queue = []

        queue.append(start_node)
        visited[start_node] = True

        while queue:
            node_identifier = queue.pop(0)
            edges_dict = self.adjacency_list[node_identifier]
            for connected_node_identifier, weight in edges_dict.items():
                if (not visited[connected_node_identifier] and weight > 0.0):
                    if (connected_node_identifier == end_node):
                        return True
                    else:
                        queue.append(connected_node_identifier)
                        visited[connected_node_identifier]
        return False
