import numpy as np

from .node import Node
from .skill_graph import SkillGraph
from .value_model import ValueModel
from .q_model import QModel

from experiments.coin_flip import CoinFlipModel


class SkillGraphAgent():

    def __init__(self,
                 coin_flip_model: CoinFlipModel,
                 value_model: ValueModel,
                 q_model: QModel,
                 intrinsic_reward_scalar: float = 0.01,
                 edge_weight_scalar: float = 0.1,
                 goal_value_threshold: float = 0.8) -> None:
        self.skill_graph = SkillGraph(coin_flip_model,
                                      value_model,
                                      q_model,
                                      intrinsic_reward_scalar,
                                      edge_weight_scalar,
                                      goal_value_threshold)
    
    def handle_state(self, state: np.ndarray):
        current_nodes = self.skill_graph.map_state_to_nodes(state)
        option = self._choose_option(current_nodes)
        terminted_state = self._execute_option(option)
        nodes_after_option_execution = self.skill_graph.map_state_to_nodes(terminted_state)
        self.skill_graph.update_edge(current_nodes[0].identifier, nodes_after_option_execution[0].identifier, state)
    
    def _choose_option(self, nodes: list[Node]):
        pass

    def _execute_option(self, option) -> np.ndarray:
        state_after_option = np.zeros(9) # TODO PLACEHODLER
        return state_after_option
