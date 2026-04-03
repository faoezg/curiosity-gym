import torch
from torch import device, optim
import numpy as np
import random

from curiosity_gym.utils.enums import Action
from experiments.components import PriorityReplayBuffer
from .q_network import QNetwork
from .buffer import SGDTransition

class QModel():
    def __init__(self,
                 state_dim: int,
                 action_dim: int = 1,
                 hidden_dim: int = 64,
                 td_target_gamma: float = 0.001,
                 eps: float = 0.1,
                 buffer_size: int = 64,
                 buffer_alpha: float = 0.5,
                 device: device | str = "cpu") -> None:
        self.device = device
        self.td_target_gamma = td_target_gamma
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self._init_target_network(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(params=self.q_network.parameters(), lr=0.0001)

        self.buffer = PriorityReplayBuffer(buffer_size, buffer_alpha)
        self.eps = eps
    
    def get_action_from_greedy_policy(self, state: np.ndarray, goal_state: np.ndarray) -> Action | int:
        action_list = [element.value for element in Action] # Why is there no method for this?
        if (random.random() < self.eps):
            return random.choice(action_list)


        state_tensor = self._to_tensor(state)
        goal_state_tensor = self._to_tensor(goal_state)

        best_q_value = -float("inf")
        best_action = 0
        for action in action_list:
            action_tensor = self._to_tensor(action)
            pred_q_value = self.q_network(state_tensor, goal_state_tensor, action_tensor)
            if (pred_q_value > best_q_value):
                best_q_value = pred_q_value
                best_action = action
        
        return best_action


    def calc_q_value(self, state: np.ndarray, goal_state: np.ndarray, action: Action | int):
        self.q_network.eval()

        with torch.no_grad():
            state_tensor = self._to_tensor(state).unsqueeze(0)
            goal_state_tensor = self._to_tensor(goal_state).unsqueeze(0)
            action_tensor = self._to_tensor(action).unsqueeze(0)

        extrinsic_reward = self.q_network(state_tensor, goal_state_tensor, action_tensor)

        self.q_network.train()
        return extrinsic_reward

    def store_transition(self,
                        state: np.ndarray,
                        action: Action | int,
                        reward: float,
                        next_state: np.ndarray,
                        goal_state: np.ndarray,
                        priority: float = 1.0):
        transition = SGDTransition(
            state,
            action,
            reward,
            next_state,
            priority,
            goal_state
        )
        self.buffer.add(transition)

    def train_network(self, batch_size: int = 64):
        if (len(self.buffer) < batch_size):
            return 0.0

        samples, indices, weights = self.buffer.sample(batch_size)
        weights_tensor = torch.from_numpy(weights)
        state_tensor_batch = torch.stack([self._to_tensor(transition.state) for transition in samples])
        action_tensor_batch = torch.stack([self._to_tensor(transition.action) for transition in samples])
        reward_tensor_batch = torch.stack([self._to_tensor(transition.reward) for transition in samples]) # type: ignore
        next_state_tensor_batch = torch.stack([self._to_tensor(transition.next_state) for transition in samples])
        goal_tensor_batch = torch.stack([self._to_tensor(transition.goal_state) for transition in samples]) # type: ignore

        with torch.no_grad():
            next_state_target_pred_q = self.target_network(next_state_tensor_batch, goal_tensor_batch, action_tensor_batch)
            td_target = reward_tensor_batch + self.td_target_gamma * next_state_target_pred_q
        
        state_pred_q = self.q_network(state_tensor_batch, goal_tensor_batch, action_tensor_batch)
        loss = (weights_tensor * (state_pred_q - td_target) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        new_prios = (td_target - state_pred_q).abs().detach().squeeze().numpy() + 1e-6 # non-zero needed
        self.buffer.update_prioritites(indices, new_prios)

        self._update_target_network()
    
    def _update_target_network(self, tau = 0.005):
        for q_parameters, target_parameters in zip(self.q_network.parameters(), self.target_network.parameters()):
            target_parameters.data.copy_(tau * q_parameters + (1.0 - tau) * target_parameters) # slightly different than byol, as we can just override them
    
    def _init_target_network(self, state_dim: int, action_dim: int, hidden_dim: int):
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

    def _to_tensor(self, input: np.ndarray | Action | int | float):
        if (not isinstance(input, np.ndarray)):
            input = np.array(input, ndmin=1)
        return torch.tensor(input, dtype=torch.float32, device=self.device)
