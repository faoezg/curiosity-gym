import numpy as np

from .node import Node
from .node_phi import NodePhi
from .skill_graph import SkillGraph
from .value_model import ValueModel
from .q_model import QModel

from curiosity_gym.core.gridengine import GridEngine
from experiments.coin_flip import CoinFlipModel


class SkillGraphAgent():

    def __init__(self,
                 env: GridEngine,
                 coin_flip_model: CoinFlipModel,
                 value_model: ValueModel,
                 q_model: QModel,
                 option_horizon: int,
                 intrinsic_reward_scalar: float = 0.01,
                 edge_weight_scalar: float = 0.1,
                 goal_value_threshold: float = 0.8) -> None:
        self.q_model = q_model
        self.option_horizon = option_horizon
        self.env = env
        self.current_state, _ = self.env.reset()
    
        self.skill_graph = SkillGraph(coin_flip_model,
                                      value_model,
                                      q_model,
                                      intrinsic_reward_scalar,
                                      edge_weight_scalar,
                                      goal_value_threshold)
   
    def run_agent(self, episodes: int, episode_steps: int):

        for _ in range(episodes):
            self.current_expansion_node = self.skill_graph.calc_expansion_node(self.current_state, 0) # inital action does not matter
            abstract_policy = self.skill_graph.get_abstract_policy() # abstract_policy is fixed per episode
            for _ in range(episode_steps):
                start_nodes = self.skill_graph.map_state_to_nodes(self.current_state)
                goal_node_identifier = abstract_policy[start_nodes[0].identifier]
                goal_node = self.skill_graph.get_node_by_identifier(goal_node_identifier)
                accumulated_reward = self._follow_edge(goal_node.get_goal_definition())
                nodes_after_option_execution = self.skill_graph.map_state_to_nodes(self.current_state)

                has_fallen_of_the_graph = isinstance(nodes_after_option_execution[0], NodePhi)
                if (not has_fallen_of_the_graph):
                    has_reached_expansion_node = self.current_expansion_node.identifier == nodes_after_option_execution[0].identifier
                    if (has_reached_expansion_node):
                        accumulated_reward += 1
                        self._follow_novelty_policy()
                    self.skill_graph.update_edge(start_nodes[0].identifier, nodes_after_option_execution[0].identifier, accumulated_reward)
                else:
                    pass
        
    def _follow_edge(self, goal_state: np.ndarray) -> float:
        accumulated_reward = 0.0
        step_count = 0
        while (step_count < self.option_horizon and np.not_equal(self.current_state, goal_state)):
            action = self.q_model.get_action_from_greedy_policy(self.current_state, goal_state)
            state, extrinsic_reward, terminated, truncated, info = self.env.step(action)
            self.current_state = state
            accumulated_reward += extrinsic_reward
            step_count += 1

            if (truncated):
                self._reset()
                return 0.0

        return accumulated_reward
    
    # TODO
    # also, for how long???
    def _follow_novelty_policy(self):
        pass

    def _reset(self):
        state, _ = self.env.reset()
        self.current_state = state
        self.skill_graph.reset()
