from collections import defaultdict
import numpy as np
import random

from .node import Node
from .node_phi import NodePhi
from .value_model import ValueModel
from .q_model import QModel
from .r2d2 import R2D2Model

from curiosity_gym.utils.enums import Action
from experiments.components import IntrinsicMotivationModel

class SkillGraph():

    def __init__(self,
                 intrinsic_motivation_model: IntrinsicMotivationModel,
                 value_model: ValueModel,
                 option_model: QModel | R2D2Model,
                 intrinsic_reward_scalar: float,
                 edge_weight_scalar: float,
                 goal_value_threshold: float) -> None:
        self.adjacency_list: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float)) # going with list, as this should end up as a sparse dag
        self.edge_transition_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int)) # going with list, as this should end up as a sparse dag
        self.edge_reward_matrix: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float)) # going with list, as this should end up as a sparse dag
        self.node_visitation_counts: dict[int, int] = defaultdict(int)
        self.nodes: list[Node] = []

        self.intrnsic_motivation_model = intrinsic_motivation_model
        self.value_model = value_model 
        self.option_model = option_model
        self.intrinsic_reward_scalar = intrinsic_reward_scalar # b in the paper
        self.edge_weight_scalar = edge_weight_scalar
        self.goal_value_threshold = goal_value_threshold
    
    def init_graph(self):
        inital_node = NodePhi()
        self.add_node(inital_node)

    def add_node(self, node: Node):
        self.adjacency_list[node.identifier] = {}
        self.nodes.append(node)
        self.node_visitation_counts[node.identifier] = 1

    def remove_node(self, node: Node):
        self.adjacency_list.pop(node.identifier)
        self.nodes.remove(node)
    
    def _add_edge(self, from_node_id: int, to_node_id: int):
        self.adjacency_list[from_node_id][to_node_id] = 0.0 # set inital transition propability
        self.edge_transition_counts[from_node_id][to_node_id] = 0 # set inital edge transition count
        self.edge_reward_matrix[from_node_id][to_node_id] =  0.0 # set inital edge reward count
    
    def remove_edge(self, from_node_id: int, to_node_id: int):
        self.adjacency_list[from_node_id].pop(to_node_id)
        # TODO add the other lists, though this is not used for now

    # This method will also add a new edge between nodes, if not yet present
    # Thereby clearing the "within H steps" heuristic as any option will terminate after H steps
    # which will continue the agent logic and map the state to a node
    def update_edge(self, from_node_id: int, to_node_id: int, accumulated_reward_during_option: float):
        self.node_visitation_counts[from_node_id] += 1
        self.node_visitation_counts[to_node_id] += 1
        self.edge_transition_counts[from_node_id][to_node_id] += 1
        self.edge_reward_matrix[from_node_id][to_node_id] += accumulated_reward_during_option

        count = self.edge_transition_counts[from_node_id][to_node_id]
        from_state = self.get_node_by_identifier(from_node_id).get_goal_state()
        to_state = self.get_node_by_identifier(to_node_id).get_goal_state()
        new_weight = self.option_model.calc_q_value(from_state, to_state, 0).detach().item() + self.edge_weight_scalar * (1/np.sqrt(count))
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
            self.update_edge(from_node_id, to_node_id, 0)

    def add_edges_between_nodes(self):
        for from_node in self.nodes[1:]:
            for to_node in self.nodes[1:]:
                is_connected = False
                for connected_node, _ in self.adjacency_list[from_node.identifier].items():
                    if (connected_node == to_node.identifier):
                        is_connected = True
                if (not is_connected and from_node.identifier != to_node.identifier):
                    self.add_edge(from_node.identifier, to_node.identifier, to_node.get_goal_state())
        #print(f"Edges: ", self.adjacency_list.items())

    def _calc_goal_conditioned_value(self, from_node_id: int, to_node_id: int, state: np.ndarray) -> float:
        goal_node = None
        for node in self.nodes:
            if (node.identifier == to_node_id):
                goal_node = node
                break
        if (goal_node is None or len(goal_node.terminal_states) == 0):
            print("THIS SHOULD NEVER BE POSSIBLE! - SOMETHING IS BROKEN")
            return 0.0
        
        goal_state = goal_node.get_goal_state() # TODO which should this be? In the paper nodes map to multiple states...
        # action does not matter here, as the reward is in regards of the goal beeing reached, which is the state
        pred_value = self.option_model.calc_q_value(state, goal_state, 0) 
        goal_conditioned_value = pred_value.item()
        count = min(self.node_visitation_counts[from_node_id], self.node_visitation_counts[to_node_id])
        return goal_conditioned_value - (self.edge_weight_scalar * 1/np.sqrt(count))

    def map_state_to_nodes(self, state: np.ndarray) -> list[Node]:
        mapped_nodes = [] 
        for node in self.nodes:
            if (node.is_goal_achieved(state)):
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
    
    def calc_expansion_node(self) -> Node:
        default_action = 0 # action shouldn't matter here
        nodes = []
        utilitys = []
        for node in self.nodes[1:]: # ignore node phi
            goal_state = node.get_goal_state()
            state_utility = self._calc_utility_of_state(goal_state,default_action)
            accumulated_descendant_utility = self._calc_utility_of_descendants(node.get_goal_state(), node.identifier, default_action)
            utility = state_utility / accumulated_descendant_utility
            utility = utility.detach().item() # type: ignore
            nodes.append(node)
            if (utility < 0):# negative utility shoudn't be choosen, thus setting it to 0, allows for probs in [0,1]
                utility = 0
            utilitys.append(utility)

        if (len(nodes) == 0):
            return NodePhi()
        
        node_probs = []
        total_utility = sum(utilitys)
        if (total_utility == 0):
            return random.choice(nodes)

        for utility in utilitys:
            prob = round(abs(utility / total_utility), 2)
            node_probs.append(prob)

        return random.choices(nodes, node_probs)[0] # type: ignore

       # state_associated_nodes = self.map_state_to_nodes(state)
       # state_utility = self._calc_utility_of_state(state, action)

       # best_utility = -float("inf")
       # best_node = self.nodes[-1] # TODO is this okay? Assume lattest added node is a good contender
       # for node in state_associated_nodes:
       #     accumulated_descendant_utility = self._calc_utility_of_descendants(state, node.identifier, action)
       #     utility = state_utility / accumulated_descendant_utility
       #     if (utility >= best_utility):
       #         best_utility = utility 
       #         best_node = node
       # return best_node
    
    def _calc_utility_of_descendants(self, state: np.ndarray, node_identifier: int, action: Action | int) -> float:
        node_descendants = self.get_descendants_of_node(node_identifier)

        accumulated_reward = 1e-4 # ensure non zero
        for node in node_descendants:
            for terminal_state in node.terminal_states:
                accumulated_reward += self._calc_utility_of_state(terminal_state, action)
        return accumulated_reward
    
    
    def _calc_utility_of_state(self, state: np.ndarray, action: Action | int) -> float:
        r_novelty = self.intrnsic_motivation_model.calc_intrinsic_reward(state, action, with_training=False) # this also trains the cfn
        extrinsic_reward = self.value_model.calc_extrinsic_reward(state, action)
        return extrinsic_reward + self.intrinsic_reward_scalar * r_novelty
    
    def has_path(self, start_node: int, end_node: int): # using bfs
        visited = [False] * (len(self.nodes) + 1)

        queue = []

        queue.append(start_node)
        visited[start_node] = True

        while len(queue) > 0:
            node_identifier = queue.pop(0)
            edges_dict = self.adjacency_list[node_identifier]
            for connected_node_identifier, weight in edges_dict.items():
                if (not visited[connected_node_identifier] and weight > 0.0):
                    if (connected_node_identifier == end_node):
                        return True
                    else:
                        queue.append(connected_node_identifier)
                        visited[node_identifier] = True
        return False
    
    def get_abstract_policy(self, expansion_node: Node):
        q_matrix = self._solve_amdp_with_value_iteration(expansion_node)

        #abstract_policy = {}
        #for idx, node in enumerate(self.nodes):
        #    # greedy policy, indexing q gives a node
        #    # togher with the iteration over all nodes we get a one-step policy for every node
        #    # where the action is given by the edge, that is to say, the option
        #    best_option = int(q_matrix[idx])
        #    abstract_policy[node.identifier] = self.nodes[best_option].identifier
        return q_matrix
    
    def _solve_amdp_with_value_iteration(self, expansion_node: Node):
        # CONVERGENCE_THRESHOLD = 0.001
        # is_converged = False
        # abstract_reward_matrix, abstract_transition_matrix, abstract_gamma = self._build_abstract_mdp(expansion_node)

        # print("Transiton Matrix", abstract_transition_matrix)
        # print("Reward Matrix", abstract_reward_matrix)

        # value_matrix = np.zeros(abstract_transition_matrix.shape[0])
        # q_matrix = np.zeros(abstract_transition_matrix.shape)

        # while True:
        #     delta = 0
        #     for from_node_index in range(len(self.nodes)):
        #         value = value_matrix[from_node_index]
        #         value_matrix[from_node_index] = max(
        #                 abstract_transition_matrix[from_node_index, to_node_index] *
        #                 (abstract_reward_matrix[from_node_index, to_node_index] + (abstract_gamma * value_matrix[to_node_index]))
        #                 for to_node_index in range(len(self.nodes))
        #             )
        #         delta = max(delta, abs(value - value_matrix[from_node_index]))
        #     if delta < CONVERGENCE_THRESHOLD:
        #         break
        
        # abstract_policy = {}
        # for from_node_index in range(-1,len(self.nodes)):
        #     abstract_policy[from_node_index] = max(
        #         range(len(self.nodes)),
        #         key=lambda a: sum(
        #             abstract_transition_matrix[from_node_index, a] *
        #             (abstract_reward_matrix[from_node_index, a] + (abstract_gamma * value_matrix[a])) for _ in range(-1,len(self.nodes))
        #         )
        #     )

        #while(not is_converged): # TODO this might have caused a value voerflow in matmul??
        #for _ in range(200):
        #     q_matrix = abstract_reward_matrix + (abstract_gamma * (abstract_transition_matrix @ value_matrix))
        #     updated_value_matrix = q_matrix.max(axis=1)
        #     value_matrices_distance = np.max(np.abs(updated_value_matrix - value_matrix))
        #     if (value_matrices_distance <= CONVERGENCE_THRESHOLD):
        #         break
        #         is_converged = True
        #     value_matrix = updated_value_matrix

        # TODO why is something like this not in the paper?
        #q_matrix[:, 0] = -np.inf # ensure phi is never a goal

        # AI SLOP
                # ------------------------------------------------------------------
        # 1️⃣  Build the AMDP (reward, transition, discount)
        # ------------------------------------------------------------------
        reward_mat, trans_mat, gamma = self._build_abstract_mdp(expansion_node)

        n_nodes = trans_mat.shape[0]                     # number of abstract states
        V = np.zeros(n_nodes)                           # state‑value vector
        CONVERGENCE_THRESHOLD = 1e-4
        MAX_ITER = 5000                                 # safety guard

        # ------------------------------------------------------------------
        # 2️⃣  Value‑iteration (Bellman backup)
        # ------------------------------------------------------------------
        delta = 0
        for it in range(MAX_ITER):
            V_old = V.copy()
            # Vectorised Bellman backup:
            #   Q_i(j) = T_ij * (R_ij + gamma * V_j)
            Q = trans_mat * (reward_mat + gamma * V_old[None, :])
            V = Q.max(axis=1)                           # V_i = max_j Q_i(j)

            # Convergence test
            delta = np.max(np.abs(V - V_old))
            if delta < CONVERGENCE_THRESHOLD:
                break
        else:
            # If we exit the for‑loop without breaking, warn the user.
            print("[IM‑DSG] WARNING: value iteration did not converge after "
                f"{MAX_ITER} iterations (Δ={delta:.6f}).")

        # ------------------------------------------------------------------
        # 3️⃣  Extract the deterministic abstract policy
        # ------------------------------------------------------------------
        # For each abstract state i we pick the successor j that maximises
        #   T_ij * (R_ij + gamma * V_j)
        # (exactly the same expression used in the Bellman backup)
        Q = trans_mat * (reward_mat + gamma * V[None, :])
        abstract_policy = {self.nodes[i].identifier: self.nodes[
            int(np.random.choice(np.flatnonzero(Q[i] == Q[i].max())))
            ].identifier for i in range(n_nodes)} # random tiebreak
        # print("Q Matrix", Q)
        # print("Policy", abstract_policy)
        return abstract_policy 

    def _build_abstract_mdp(self, expansion_node: Node) -> tuple[np.ndarray, np.ndarray, float]:
        node_count = len(self.nodes)
        # Yes, the matricies are duplicate data, no, I do not care.
        # We need those square matrices for nice and easy calculation later
        transition_matrix = np.zeros((node_count, node_count)) # square and sparse matrix, can not be botherd to optimizse as it is small
        reward_matrix = np.zeros((node_count, node_count)) # square and sparse matrix, can not be botherd to optimizse as it is small

        for from_node_index, node in enumerate(self.nodes):
            all_outgoing_edges_of_node = self.adjacency_list[node.identifier].items()
            for to_node_identifier, weight in all_outgoing_edges_of_node:
                weight_node_index = self._get_idx_of_node_identifier(to_node_identifier)
                transition_matrix[from_node_index, weight_node_index] = weight
                reward_matrix[from_node_index, weight_node_index] = self.edge_reward_matrix[from_node_index][weight_node_index]

            # chance to fall of the graph then leaving a certain node_i
            # TODO this is not correct. the sum of the transition_matrix weights is not in [0,1]
            # and the noramlization later thus, in most cases, forces the chance to fall of to be 0
            chance_to_fall_of_the_graph_coming_from_node = (1.0 - np.absolute(transition_matrix[from_node_index, :]).sum())
            transition_matrix[from_node_index, 0] = max(0.0, chance_to_fall_of_the_graph_coming_from_node)

        transition_matrix = self._normalize_matrix_rows(transition_matrix)
        
        expansion_index = self._get_idx_of_node(expansion_node)
        reward_matrix[:, expansion_index] += 1
        gamma = transition_matrix[:, 0].sum() / node_count # TODO not sure about this one chief

        return reward_matrix, transition_matrix, gamma
    
    def _normalize_matrix_rows(self, matrix: np.ndarray) -> np.ndarray:

        np.maximum(matrix, 0, out=matrix)
        row_sums = matrix.sum(axis=1, keepdims=True)

        non_zero_mask = row_sums != 0
        divisor =  np.where(non_zero_mask, row_sums, 1)

        return matrix / divisor
    
    def _get_idx_of_node(self, node_to_check: Node) -> int:
        for idx, node in enumerate(self.nodes):
            if (node.identifier == node_to_check.identifier):
                return idx
        return -2

    def _get_idx_of_node_identifier(self, node_identifier: int) -> int:
        for idx, node in enumerate(self.nodes):
            if (node.identifier == node_identifier):
                return idx
        return -2

    def get_node_by_identifier(self, node_identifier: int) -> Node:
        for node in self.nodes:
            if (node.identifier == node_identifier):
                return node
        return NodePhi()
    
    def reset(self):
        self._reset_all_edge_rewards()
