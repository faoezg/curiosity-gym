import numpy as np
import random

from .components.node import Node
from .components.node_phi import NodePhi
from .components.skill_graph import SkillGraph
from .components.value_model import ValueModel
from .components.q_model import QModel
from .components.exploration_model import ExplorationModel

from curiosity_gym.core.gridengine import GridEngine
from curiosity_gym.utils.enums import Action
from experiments.components import IntrinsicMotivationModel


class SkillGraphAgent():

    def __init__(self,
                 env: GridEngine,
                 intrinsic_motivation_model: IntrinsicMotivationModel,
                 value_model: ValueModel,
                 q_model: QModel,
                 exploration_model: ExplorationModel,
                 q_model_batch_size: int = 10,
                 exploration_model_batch_size: int = 32,
                 option_horizon: int = 10,
                 intrinsic_reward_scalar: float = 0.01,
                 edge_weight_scalar: float = 0.1,
                 goal_value_threshold: float = 0.0,
                 intrinsic_std_deviaton_scalar: float = 1,
                 hindsight_k: int = 4) -> None:
        self.intrinsic_motivation_model = intrinsic_motivation_model 
        self.intrinsic_reward_scalar = intrinsic_reward_scalar
        self.value_model = value_model
        self.q_model = q_model
        self.q_model_batch_size = q_model_batch_size
        self.hindsight_k = hindsight_k
        self.exploration_model = exploration_model
        self.exploration_model_batch_size = exploration_model_batch_size
        self.option_horizon = option_horizon
        self.env = env
        self.current_state, _ = self.env.reset()
    
        self.skill_graph = SkillGraph(intrinsic_motivation_model,
                                      value_model,
                                      q_model,
                                      intrinsic_reward_scalar,
                                      edge_weight_scalar,
                                      goal_value_threshold)
        self.skill_graph.init_graph()

        self.intrinsic_std_deviaton_scalar = intrinsic_std_deviaton_scalar
        self.intrinisc_reward_mean = 0.0
        self.intrinsic_running_std_deviaton_squared = 0.0
        self.intrinsic_reward_count = 0
   
    def run_agent(self, episodes: int, episode_steps: int):
        for i in range(episodes):
            print("Start episode: ", i)
            self.current_expansion_node = self.skill_graph.calc_expansion_node(self.current_state, 0) # inital action does not matter
            if (isinstance(self.current_expansion_node, NodePhi)):
                self._expand_graph()
                continue

            for j in range(episode_steps):
                print("Start Episode step", j)
                abstract_policy = self.skill_graph.get_abstract_policy() # abstract_policy is fixed per episode
                self._try_reaching_expansion_node(abstract_policy)
 
    def _try_reaching_expansion_node(self, abstract_policy: dict):
        start_nodes = self.skill_graph.map_state_to_nodes(self.current_state)
        has_reached_expansion_node = self.current_expansion_node.identifier == start_nodes[0].identifier
        print("policy", abstract_policy)
        goal_node_identifier = abstract_policy[start_nodes[0].identifier]
        print("Goal Node", goal_node_identifier, start_nodes[0].identifier)
        for node in self.skill_graph.nodes:
            print("Node Id: ", node.identifier)
        goal_node = self.skill_graph.get_node_by_identifier(goal_node_identifier)

        accumulated_reward = self._follow_edge(start_nodes[0], goal_node)
        nodes_after_option_execution = self.skill_graph.map_state_to_nodes(self.current_state)

        has_reached_expansion_node = self.current_expansion_node.identifier == nodes_after_option_execution[0].identifier
        if (has_reached_expansion_node):
            accumulated_reward += 1
            self._expand_graph()

        if (not isinstance(start_nodes[0], NodePhi)):
            self.skill_graph.update_edge(start_nodes[0].identifier, nodes_after_option_execution[0].identifier, accumulated_reward)

    def _expand_graph(self):
        print("Expanding graph")
        trajectory = self._follow_novelty_policy()
        self._add_new_node_from_trajectory_to_graph(trajectory)
        print("Done expaning the graph")

    def _follow_edge(self, start_node: Node, goal_node: Node) -> float:
        accumulated_reward = 0.0
        step_count = 0
        current_goal_node = goal_node

        while (step_count < self.option_horizon and not current_goal_node.is_goal_achieved(self.current_state)):
            if (isinstance(start_node, NodePhi) or isinstance(goal_node, NodePhi)): # TODO this shouldn't be needed?
                current_goal_node = self._get_closest_goal_to_current_state() # "shortest path back onto the graph"

            goal_state = current_goal_node.get_goal_state()
            action = self.q_model.get_action_from_greedy_policy(self.current_state, goal_state)
            new_state, extrinsic_reward, terminated, truncated, info = self.env.step(action)
            print("Followed option:", Action(action))
            self.value_model.train_network(extrinsic_reward, self.current_state, action)
            print("Trained value model")
            self._process_q_model_transition(new_state, goal_state, action)
            print("Procced Q transition")
            self.current_state = new_state 
            accumulated_reward += extrinsic_reward
            step_count += 1

            if (truncated or terminated):
                print("Reseting")
                self._reset()
                return 0.0
            
        #if (current_goal_node.is_goal_achieved(self.current_state)):
        print("Adding terminal state to goalnode")
        current_goal_node.add_terminal_state(self.current_state) # TODO is this okay??
        
        if (len(self.q_model.buffer) >= self.q_model_batch_size):
            self._apply_hindsight_replay_to_q_model_buffer()
            print("training q model")
            self.q_model.train_network(self.q_model_batch_size)

        print("finished following edge")
        return accumulated_reward
    
    def _is_goal_met(self, current_state: np.ndarray, goal_state: np.ndarray):
        return np.all(np.equal(current_state, goal_state)) # TODO is this okay??
    
    def _process_q_model_transition(self, next_state: np.ndarray, goal_state: np.ndarray, action: Action | int):
        reward = 0.0
        if (self._is_goal_met(next_state, goal_state)):
            reward = 1.0
        self.q_model.store_transition(self.current_state, action, reward, next_state, goal_state) # goal-conditioned thus not the reward from env
    
    def _apply_hindsight_replay_to_q_model_buffer(self):
        transitions = list(self.q_model.buffer.buffer)[-self.option_horizon:]
        for idx, transition in enumerate(transitions):
            future_indicies = list(range(idx+1, len(transitions)))
            selected_future_idx = random.choices(future_indicies, k=min(self.hindsight_k, len(future_indicies))) # TODO this will blow up if k > len(transitions), but what ever

            for future_idx in selected_future_idx:
                hindsight_goal_state = transitions[future_idx].goal_state # type: ignore
                self._process_q_model_transition(transition.next_state, hindsight_goal_state, transition.action)
    
    def _follow_novelty_policy(self) -> list[tuple[np.ndarray, Action | int, float, float, np.ndarray]]:
        print("Exploring")
        trajectory = []
        step_count = 0
        done = False

        while (step_count < self.option_horizon and not done): # TODO is the option_horizon okay here? It might be!
            action = self.exploration_model.get_action_from_greedy_policy(self.current_state)
            new_state, extrinsic_reward, terminated, truncated, info = self.env.step(action)
            print("training value model (exploring)")
            self.value_model.train_network(extrinsic_reward, self.current_state, action)
            print("procces exploration transition")
            self._process_exploration_model_transition(new_state, action, extrinsic_reward, trajectory)
            self._add_edge_during_exploration(self.current_state, new_state, extrinsic_reward)

            self.current_state = new_state
            done = truncated or terminated
            step_count += 1
        
       
        if (len(self.exploration_model.buffer) >= self.exploration_model_batch_size):
            self.exploration_model.train_network(self.exploration_model_batch_size)

        if (done):
            print("Resting (explorationg)")
            self._reset()
            return trajectory
        
        print("Done exploring")
        return trajectory
    
    # TODO should this be a thing?
    # should this be done for every action in the exploration, or after every exploration?
    # in any case, it must be with the option_horizon
    def _add_edge_during_exploration(self, start_state: np.ndarray, following_state: np.ndarray, reward: float):
        start_nodes = self.skill_graph.map_state_to_nodes(start_state)
        end_nodes = self.skill_graph.map_state_to_nodes(following_state)

        for start_node in start_nodes:
            for end_node in end_nodes:
                if (not isinstance(start_node, NodePhi) and not isinstance(end_node, NodePhi) and start_node.identifier != end_node.identifier):
                    self.skill_graph.update_edge(start_node.identifier, end_node.identifier, reward)

    def _process_exploration_model_transition(self, next_state: np.ndarray, action: Action | int, extrinsic_reward: float, trjectory: list):
        current_state = self.current_state
        intrinsic_reward = self.intrinsic_motivation_model.calc_intrinsic_reward(current_state, action) # this also trains the cfn
        reward = extrinsic_reward + (self.intrinsic_reward_scalar * intrinsic_reward)
        self.exploration_model.store_transition(current_state, action, reward, next_state)
        trjectory.append((self.current_state, action, extrinsic_reward, intrinsic_reward, next_state))
    
    def _add_new_node_from_trajectory_to_graph(self, trajectory: list[tuple[np.ndarray, Action | int, float, float, np.ndarray]]):
        intrisic_rewards = [element[3] for element in trajectory]
        best_idx = int(np.argmax(intrisic_rewards))

        best_state, best_action, best_extrinsic_reward, best_intrinsic_reward, best_next_state = trajectory[best_idx]
        self._update_intrinsic_reward_statistics(best_intrinsic_reward)

        intrisic_reward_threshold = self.intrinisc_reward_mean + (self.intrinsic_std_deviaton_scalar * self._calc_intrinsic_std_deviation())
        if (best_intrinsic_reward >= intrisic_reward_threshold):
            new_node = Node()
            new_node.terminal_states.append(best_state)
            print("ADD NODE: ", new_node.identifier)
            self.skill_graph.add_node(new_node)

            for node in self.skill_graph.nodes: # TODO should it be like this??
                if (not isinstance(node, NodePhi)):
                    self.skill_graph.add_edge(node.identifier, new_node.identifier, self.current_state)
                    # this makes the graph bidirectional though
                    #self.skill_graph.add_edge(new_node.identifier, node.identifier, self.current_state)
        print("Done handling trajectory")
    
    def _update_intrinsic_reward_statistics(self, intrinsic_reward: float) -> None:
        self.intrinsic_reward_count += 1
        delta = intrinsic_reward - self.intrinisc_reward_mean
        self.intrinisc_reward_mean += delta / self.intrinsic_reward_count
        new_delta = intrinsic_reward - self.intrinisc_reward_mean
        self.intrinsic_running_std_deviaton_squared += delta * new_delta
    
    def _calc_intrinsic_std_deviation(self) -> float:
        if (self.intrinsic_reward_count < 2):
            return 0.0
        return np.sqrt((self.intrinsic_running_std_deviaton_squared / (self.intrinsic_reward_count - 1)))

    # assuming that the goal_conditioned q_value is trained correctly
    # it should represent some kind of distance metric towards a goal
    # if it were regarded as the goal given the current_state
    # which therefore should be the "closest" to the current_state that is still on the graph
    def _get_closest_goal_to_current_state(self) -> Node:
        best_node = NodePhi() 
        best_q_value = -float("inf")
        for node in self.skill_graph.nodes:
            if (len(node.terminal_states) != 0):
                goal_state = node.get_goal_state()
                q_value = self.q_model.calc_q_value(self.current_state, goal_state, 0) # action dosn't matter here

                if (q_value > best_q_value):
                    best_q_value = q_value
                    best_node = node
        return best_node
    
    def _reset(self):
        state, _ = self.env.reset()
        self.current_state = state
        self.skill_graph.reset()
