from collections import defaultdict
import numpy as np

from .node import Node
from .node_phi import NodePhi
from .value_model import ValueModel
from .q_model import QModel

from curiosity_gym.utils.enums import Action
from experiments.coin_flip import CoinFlipModel

class SkillGraph():

    def __init__(self,
                 coin_flip_model: CoinFlipModel,
                 value_model: ValueModel,
                 q_model: QModel,
                 intrinsic_reward_scalar: float,
                 edge_weight_scalar: float,
                 goal_value_threshold: float) -> None:
        self.adjacency_list: dict[int, dict[int, float]] = defaultdict(defaultdict) # going with list, as this should end up as a sparse dag
        self.edge_transition_counts: dict[int, dict[int, int]] = defaultdict(defaultdict) # going with list, as this should end up as a sparse dag
        self.node_visitation_counts: dict[int, int] = defaultdict(int)
        self.nodes: list[Node] = []

        self.coin_flip_model = coin_flip_model
        self.value_model = value_model 
        self.q_model = q_model
        self.intrinsic_reward_scalar = intrinsic_reward_scalar # b in the paper
        self.edge_weight_scalar = edge_weight_scalar
        self.goal_value_threshold = goal_value_threshold

    def add_node(self, node: Node):
        self.adjacency_list[node.identifier] = {}
        self.nodes.append(node)

    def remove_node(self, node: Node):
        self.adjacency_list.pop(node.identifier)
        self.nodes.remove(node)
    
    def _add_edge(self, from_node_id: int, to_node_id: int):
        self.adjacency_list[from_node_id][to_node_id] = 0.1 # set inital transition propability
        self.edge_transition_counts[from_node_id][to_node_id] = 0 # set inital edge transition count
    
    def remove_edge(self, from_node_id: int, to_node_id: int):
        self.adjacency_list[from_node_id].pop(to_node_id)

    # This method will also add a new edge between nodes, if not yet present
    # Thereby clearing the "within H steps" heuristic as any option will terminate after H steps
    # which will continue the agent logic and map the state to a node
    def update_edge(self, from_node_id: int, to_node_id: int, state):
        self.node_visitation_counts[from_node_id] += 1
        self.node_visitation_counts[to_node_id] += 1
        self.edge_transition_counts[from_node_id][to_node_id] += 1
        count = self.edge_transition_counts[from_node_id][to_node_id]
        new_weight = self.edge_weight_scalar * (1/np.sqrt(count))
        self.adjacency_list[from_node_id][to_node_id] = new_weight
    
    def add_edge(self, from_node_id: int, to_node_id: int, state):
        goal_value = self._calc_goal_conditioned_value(from_node_id, to_node_id, state)
        if (goal_value > self.goal_value_threshold):
            self._add_edge(from_node_id, to_node_id)
    
    def _calc_goal_conditioned_value(self, from_node_id: int, to_node_id: int, state: np.ndarray) -> float:
        goal_node = None
        for node in self.nodes:
            if (node.identifier == to_node_id):
                goal_node = node
                break
        
        if (goal_node is None or len(goal_node.terminal_states) == 0):
            print("THIS SHOULD NEVER BE POSSIBLE! - SOMETHING IS BROKEN")
            return 0.0
        
        goal_state = goal_node.terminal_states[0] # TODO which should this be? In the paper nodes map to multiple states...
        # action does not matter here, as the reward is in regards of the goal beeing reached, which is the state
        # thus if the
        pred_value = self.q_model.calc_q_value(state, goal_state, 0) 
        goal_conditioned_value = pred_value.item()
        count = min(self.node_visitation_counts[from_node_id], self.node_visitation_counts[to_node_id])
        return goal_conditioned_value - (self.edge_weight_scalar * 1/np.sqrt(count))

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
    
    def _calc_expansion_node(self, state: np.ndarray, action: Action | int):
        state_associated_nodes = self.map_state_to_nodes(state)
        state_utility = self._calc_utility_of_state(state, action)

        best_utility = 0.0
        best_node = None
        for node in state_associated_nodes:
            accumulated_descendant_utility = self._calc_utility_of_descendants(state, node.identifier, action)
            utility = state_utility / accumulated_descendant_utility
            if (utility >= best_utility):
                best_utility = utility 
                best_node = node
        
        return best_node
    
    def _calc_utility_of_descendants(self, state: np.ndarray, node_identifier: int, action: Action | int) -> float:
        node_descendants = self.get_descendants_of_node(node_identifier)

        accumulated_reward = 0.0
        for node in node_descendants:
            for state in node.terminal_states:
                accumulated_reward += self._calc_utility_of_state(state, action)
        return accumulated_reward
    
    
    def _calc_utility_of_state(self, state: np.ndarray, action: Action | int) -> float:
        r_novelty = self.coin_flip_model.calc_intrinsic_reward(state, action)
        extrinsic_reward = self.value_model.calc_extrinsic_reward(state, action)
        return extrinsic_reward + self.intrinsic_reward_scalar * r_novelty
    
    def has_path(self, start_node: int, end_node: int): # using bfs
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
