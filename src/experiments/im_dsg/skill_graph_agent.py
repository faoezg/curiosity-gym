import numpy as np
import random

from .components.node import Node
from .components.node_phi import NodePhi
from .components.skill_graph import SkillGraph
from .components.value_model import ValueModel
from .components.q_model import QModel, SGDTransition
from .components.r2d2 import R2D2Model, R2D2Transition
from .components.exploration_model import ExplorationModel

from curiosity_gym.core.gridengine import GridEngine
from curiosity_gym.utils.enums import Action
from experiments.components import IntrinsicMotivationModel


class SkillGraphAgent():

    def __init__(self,
                 env: GridEngine,
                 intrinsic_motivation_model: IntrinsicMotivationModel,
                 value_model: ValueModel,
                 option_model: QModel | R2D2Model,
                 exploration_model: ExplorationModel,
                 option_model_batch_size: int = 10,
                 exploration_model_batch_size: int = 10,
                 option_horizon: int = 10,
                 intrinsic_reward_scalar: float = 0.01,
                 edge_weight_scalar: float = 0.9,
                 goal_value_threshold: float = 0.8,
                 intrinsic_std_deviaton_scalar: float = 1,
                 hindsight_k: int = 4) -> None:
        self.intrinsic_motivation_model = intrinsic_motivation_model 
        self.intrinsic_reward_scalar = intrinsic_reward_scalar
        self.value_model = value_model
        self.option_model = option_model
        self.option_model_batch_size = option_model_batch_size
        self.hindsight_k = hindsight_k
        self.exploration_model = exploration_model
        self.exploration_model_batch_size = exploration_model_batch_size
        self.option_horizon = option_horizon
        self.env = env
        self.current_state, _ = self.env.reset()
    
        self.skill_graph = SkillGraph(intrinsic_motivation_model,
                                      value_model,
                                      option_model,
                                      intrinsic_reward_scalar,
                                      edge_weight_scalar,
                                      goal_value_threshold)
        self.skill_graph.init_graph()

        self.intrinsic_std_deviaton_scalar = intrinsic_std_deviaton_scalar
        self.intrinisc_reward_mean = 0.0
        self.intrinsic_running_std_deviaton_squared = 0.0
        self.intrinsic_reward_count = 0
        self.trajectory: list[tuple[np.ndarray, Action | int, float, float, np.ndarray]] = []
   
    def run_agent(self, episodes: int, episode_steps: int):
        self.episode_counter = 0
        while (self.episode_counter <= episodes):
            self.current_expansion_node = self.skill_graph.calc_expansion_node()
            abstract_policy = self.skill_graph.get_abstract_policy(self.current_expansion_node) # abstract_policy is fixed per episode
            if (isinstance(self.current_expansion_node, NodePhi) or self._has_fallen_of_the_graph()):
                self._expand_graph()
                # no break needed, as this can not result in new state without recalculating the policy
                continue

            has_reached_expansion_node = self._try_reaching_expansion_node(abstract_policy, episode_steps)
            if (has_reached_expansion_node):
                print("Reached Expansion Node!")
                self._expand_graph()
            
            self.skill_graph.add_edges_between_nodes()
            if (len(self.option_model.buffer) >= self.option_model_batch_size):
                self.option_model.train_network(self.option_model_batch_size)


 
    def _try_reaching_expansion_node(self, abstract_policy: dict, episode_steps: int):
        print(f"Policy: {abstract_policy}, Expansion Node: {self.current_expansion_node.identifier}, Current Node: {self._get_current_node().identifier}")
        has_reached_expansion_node = self.current_expansion_node.is_goal_achieved(self.current_state)
        step_count = 0

        while (not has_reached_expansion_node and step_count <= episode_steps):
            if (not self._has_fallen_of_the_graph()):
                current_node = self._get_current_node()
                goal_node_identifier = abstract_policy[current_node.identifier]
                goal_node = self.skill_graph.get_node_by_identifier(goal_node_identifier)
                if (isinstance(goal_node, NodePhi) or goal_node.is_goal_achieved(self.current_state)):
                    self._expand_graph()
                    break

                self._follow_edge(goal_node)

                has_reached_expansion_node = self.current_expansion_node.is_goal_achieved(self.current_state)
            else:
                goal_node = self._get_closest_goal_to_current_state()
                self._follow_edge(goal_node)
                has_reached_expansion_node = self.current_expansion_node.is_goal_achieved(self.current_state)

            step_count += 1

        return has_reached_expansion_node
    
    def _has_fallen_of_the_graph(self, state: np.ndarray | None = None):
        if (state is not None):
            current_node = self.skill_graph.map_state_to_nodes(state)[0]
        else:
            current_node = self._get_current_node()
        if (isinstance(current_node, NodePhi)):
            return True
        return False
    
    def _get_current_node(self) -> Node:
        return self.skill_graph.map_state_to_nodes(self.current_state)[0]

    def _expand_graph(self):
        self._follow_novelty_policy()
        self._add_new_node_from_trajectory_to_graph()

    def _follow_edge(self, goal_node: Node):
        accumulated_reward = 0.0
        step_count = 0
        current_goal_node = goal_node
        goal_state = current_goal_node.get_goal_state()
        start_node = self._get_current_node()
        goal_achived = False

        while (step_count <= self.option_horizon):
            if(goal_achived):
                print(f"Option successfull, From {start_node.identifier} -> {goal_node.identifier}, Expansion Node: {self.current_expansion_node.identifier}")
                break
            #print(f"Following Edge {self._get_current_node().identifier} -> {current_goal_node.identifier}")
            action = self.option_model.get_action_from_greedy_policy(self.current_state, goal_state)
            # print("Action: ",Action(action))
            new_state, extrinsic_reward, terminated, truncated, info = self.env.step(action)
            done = truncated or terminated
            self.value_model.train_network(extrinsic_reward, self.current_state, action)
            self._process_option_model_transition(self.current_state, new_state, goal_state, action, done)
            self._apply_hindsight_replay_to_option_model_buffer()
            #if (self._has_fallen_of_the_graph(new_state)):
            #    self._process_exploration_model_transition(new_state, action, extrinsic_reward)
            #    current_goal_node = self._get_closest_goal_to_current_state()
            #    goal_state = current_goal_node.get_goal_state()
            #else:
            if (start_node.identifier != self.skill_graph.map_state_to_nodes(new_state)[0].identifier and not isinstance(start_node, NodePhi)):
                self._add_edge_during_exploration(start_node.get_goal_state(), new_state, extrinsic_reward)
            self.current_state = new_state 
            accumulated_reward += extrinsic_reward
            step_count += 1
            goal_achived = goal_node.is_goal_achieved(self.current_state)

            if (done):
                self._reset()
                return 0.0
        if (not goal_achived):
            print(f"Option failed, From {start_node.identifier} -> {self._get_current_node().identifier}, wished for {goal_node.identifier}")
    
        #if (len(self.option_model.buffer) >= self.option_model_batch_size):
        #    self._apply_hindsight_replay_to_option_model_buffer()
        #    self.option_model.train_network(self.option_model_batch_size)


        #if (not current_goal_node.is_goal_achieved(self.current_state)):
            # TODO is this okay?? If this is in, there is an infinite loop
            # as moving from node_1 -> node_1 will then add the state to the terminal state
            # thus ensuring that is_goal_achieved is always true
        # current_goal_node.add_terminal_state(self.current_state) # increases termination region
    
    def _is_goal_met(self, current_state: np.ndarray, goal_state: np.ndarray):
        return np.all(np.equal(current_state, goal_state)) # TODO is this okay??
    
    def _process_option_model_transition(self, old_state: np.ndarray, current_state: np.ndarray, goal_state: np.ndarray, action: Action | int, done: bool, goal_node: Node | None = None):
        reward = 1.0
        if ((goal_node is not None and goal_node.is_goal_achieved(current_state)) or self._is_goal_met(current_state, goal_state)):
            reward = 1.0
        self.option_model.handle_transition(old_state, action, reward, current_state, goal_state, done) # goal-conditioned thus not the reward from env
    
    def _apply_hindsight_replay_to_option_model_buffer(self):
        transitions = list(self.option_model.buffer.buffer)[-self.hindsight_k:]
        for idx, transition in enumerate(transitions):
            if (transition.reward == 0 and not self._is_goal_met(transition.state,transition.next_state)):
                future_indicies = list(range(idx+1, len(transitions)))
                selected_future_idx = random.sample(future_indicies, k=min(self.hindsight_k, len(future_indicies))) # TODO this will blow up if k > len(transitions), but what ever

                for future_idx in selected_future_idx:
                    hindsight_goal_state = transitions[future_idx].next_state # type: ignore
                    if (isinstance(transition, SGDTransition)):
                        self._process_option_model_transition(transition.state, transition.next_state, hindsight_goal_state, transition.action, False) # done is ignored in this case
                    if (isinstance(transition, R2D2Transition)):
                        self._process_option_model_transition(transition.state, transition.next_state, hindsight_goal_state, transition.action, transition.done)
    
    def _follow_novelty_policy(self):
        print("Exploring")
        self.trajectory = [] # reset trajectory
        step_count = 0
        done = False
        start_node = self._get_current_node()

        while (step_count < self.option_horizon and not done): # TODO is the option_horizon okay here? It might be!
            action = self.exploration_model.get_action_from_greedy_policy(self.current_state)
            new_state, extrinsic_reward, terminated, truncated, info = self.env.step(action)
            self.value_model.train_network(extrinsic_reward, self.current_state, action)
            self._process_exploration_model_transition(new_state, action, extrinsic_reward)
            if (not isinstance(start_node, NodePhi)):
                    self._add_edge_during_exploration(start_node.get_goal_state(), new_state, extrinsic_reward)
            else:
                start_node = self._get_current_node()
            self.current_state = new_state
            done = truncated or terminated
            step_count += 1
        
        if (len(self.exploration_model.buffer) >= self.exploration_model_batch_size):
            self.exploration_model.train_network(self.exploration_model_batch_size)

        if (done):
            self._reset()
    
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

    def _process_exploration_model_transition(self, next_state: np.ndarray, action: Action | int, extrinsic_reward: float):
        current_state = self.current_state
        intrinsic_reward = self.intrinsic_motivation_model.calc_intrinsic_reward(current_state, action) # this also trains the cfn
        reward = extrinsic_reward + (self.intrinsic_reward_scalar * intrinsic_reward)
        self.exploration_model.store_transition(current_state, action, reward, next_state)
        self.trajectory.append((self.current_state, action, extrinsic_reward, intrinsic_reward, next_state))
    
    def _add_new_node_from_trajectory_to_graph(self):
        intrisic_rewards = [element[3] for element in self.trajectory]
        top_n = min(len(self.trajectory), 3)
        best_indicies = np.argsort(intrisic_rewards)[-top_n:][:-1]

        for idx in best_indicies:
            best_state, best_action, best_extrinsic_reward, best_intrinsic_reward, best_next_state = self.trajectory[idx]
            intrisic_reward_threshold = self.intrinisc_reward_mean + (self.intrinsic_std_deviaton_scalar * self._calc_intrinsic_std_deviation())
            if (best_intrinsic_reward >= intrisic_reward_threshold):
                for node in self.skill_graph.nodes:
                    if (node.is_goal_achieved(best_state)):
                        break

                self._update_intrinsic_reward_statistics(best_intrinsic_reward)
                new_node = Node()
                new_node.terminal_states.append(best_state)
                print("ADD NODE: ", new_node.identifier)
                self.skill_graph.add_node(new_node)

                for node in self.skill_graph.nodes[1:]: # TODO should it be like this??
                    if (node.identifier != new_node.identifier):
                        self.skill_graph.add_edge(node.identifier, new_node.identifier, best_state)
                        # this makes the graph bidirectional though
                        # self.skill_graph.add_edge(new_node.identifier, node.identifier, self.current_state)

                break # added the best node we could from the trajectorie, so we stop and ignore the overs
        
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
                action = random.choice([action.value for action in Action])
                q_value = self.option_model.calc_q_value(self.current_state, goal_state, action) # action dosn't matter here

                if (q_value > best_q_value):
                    best_q_value = q_value
                    best_node = node
        return best_node
    
    def _reset(self):
        state, _ = self.env.reset()
        self.current_state = state
        self.episode_counter += 1
        self.skill_graph.reset()
