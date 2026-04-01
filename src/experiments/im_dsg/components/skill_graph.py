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
        self.edge_reward_matrix: dict[int, dict[int, float]] = defaultdict(defaultdict) # going with list, as this should end up as a sparse dag
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
        self.edge_reward_matrix[from_node_id][to_node_id] = 0.0 # set inital edge reward count
    
    def remove_edge(self, from_node_id: int, to_node_id: int):
        self.adjacency_list[from_node_id].pop(to_node_id)

    # This method will also add a new edge between nodes, if not yet present
    # Thereby clearing the "within H steps" heuristic as any option will terminate after H steps
    # which will continue the agent logic and map the state to a node
    def update_edge(self, from_node_id: int, to_node_id: int, accumulated_reward_during_option: float):
        self.node_visitation_counts[from_node_id] += 1
        self.node_visitation_counts[to_node_id] += 1
        self.edge_transition_counts[from_node_id][to_node_id] += 1
        self.edge_reward_matrix[from_node_id][to_node_id] += accumulated_reward_during_option

        count = self.edge_transition_counts[from_node_id][to_node_id]
        new_weight = self.edge_weight_scalar * (1/np.sqrt(count))
        self.adjacency_list[from_node_id][to_node_id] = new_weight
    
    def reset_edge_reward(self, from_node_id: int, to_node_id: int):
        self.edge_reward_matrix[from_node_id][to_node_id] = 0 
    
    def _reset_all_edge_rewards(self):
        for from_node_identifier in self.edge_reward_matrix.keys():
            for to_node_identifier in self.edge_reward_matrix[from_node_identifier].keys():
                self.edge_reward_matrix[from_node_identifier][to_node_identifier] = 0

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
    
    def calc_expansion_node(self, state: np.ndarray, action: Action | int) -> Node:
        state_associated_nodes = self.map_state_to_nodes(state)
        state_utility = self._calc_utility_of_state(state, action)

        best_utility = 0.0
        best_node = NodePhi() 
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
        r_novelty = self.coin_flip_model.calc_intrinsic_reward(state, action) # this also trains the cfn
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
    
    def get_abstract_policy(self):
        q_matrix = self._solve_amdp_with_value_iteration()

        abstract_policy = {}
        for node in self.nodes:
            # greedy policy, indexing q gives a node
            # togher with the iteration over all nodes we get a one-step policy for every node
            # where the action is given by the edge, that is to say, the option
            best_option = int(q_matrix[node.identifier].argmax())
            abstract_policy[node.identifier] = best_option
        
        return abstract_policy
    
    def _solve_amdp_with_value_iteration(self):
        CONVERGENCE_THRESHOLD = 0.001
        is_converged = False
        abstract_transition_matrix, abstract_reward_matrix, abstract_gamma = self._build_abstract_mdp()

        value_matrix = np.zeros(abstract_transition_matrix.shape[0])
        q_matrix = np.zeros(abstract_transition_matrix.shape)

        while(not is_converged):
            q_matrix = abstract_reward_matrix + (abstract_gamma * (abstract_transition_matrix @ value_matrix[:, None]))
            updated_value_matrix = q_matrix.max(axis=1)
            value_matrices_distance = np.max(np.abs(updated_value_matrix - value_matrix))
            if (value_matrices_distance <= CONVERGENCE_THRESHOLD):
                is_converged = True
            value_matrix = updated_value_matrix

        return q_matrix

    def _build_abstract_mdp(self) -> tuple[np.ndarray, np.ndarray, float]:
        node_count = len(self.nodes) + 1 # + 1 for the "falling of" NodePhi
        # Yes, the matricies are duplicate data, no, I do not care.
        # We need those square matrices for nice and easy calculation later
        transition_matrix = np.zeros((node_count, node_count)) # square and sparse matrix, can not be botherd to optimizse as it is small
        reward_matrix = np.zeros((node_count, node_count)) # square and sparse matrix, can not be botherd to optimizse as it is small


        for i, node in enumerate(self.nodes):
            all_outgoing_edges_of_node = self.adjacency_list[node.identifier].items()
            for j, weight in all_outgoing_edges_of_node:
                transition_matrix[i,j] = weight
                reward_matrix[i,j] = self.edge_reward_matrix[i][j]
            # chance to fall of the graph then leaving a certain node_i
            chance_to_fall_of_the_graph_coming_from_node = (1.0 - transition_matrix[i, :].sum())
            transition_matrix[i, -1] = max(0.0, chance_to_fall_of_the_graph_coming_from_node)

        gamma = transition_matrix[:, -1].sum() /  node_count # TODO not sure about this one chief

        return reward_matrix, transition_matrix, gamma

    def get_node_by_identifier(self, node_identifier: int) -> Node:
        for node in self.nodes:
            if (node.identifier == node_identifier):
                return node
        return NodePhi()
    
    def reset(self):
        self._reset_all_edge_rewards()
