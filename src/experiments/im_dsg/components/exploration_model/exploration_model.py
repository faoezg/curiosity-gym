import torch
from torch import device, optim
import numpy as np
import random

from curiosity_gym.utils.enums import Action
from experiments.components import PriorityReplayBuffer, PriorizedTransition
from .exploration_network import ExplorationNetwork


# TODO 
# refactor this and QModel to use same super?
# can I be bothered?
class ExplorationModel():
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
        self.exploration_network = ExplorationNetwork(state_dim, action_dim, hidden_dim).to(device)
        self._init_target_network(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(params=self.exploration_network.parameters(), lr=0.0001)

        self.buffer = PriorityReplayBuffer(buffer_size, buffer_alpha)
        self.eps = eps

    def get_action_from_greedy_policy(self, state: np.ndarray) -> Action | int:
        action_list = [element.value for element in Action] # Why is there no method for this?
        if (random.random() < self.eps):
            return random.choice(action_list)

        state_tensor = self._to_tensor(state)
        action_tensor = torch.zeros_like(state_tensor) # action dosn't matter
        pred_q_value = self.exploration_network(state_tensor, action_tensor)

        return int(pred_q_value.argmax(dim=1).detach().item())

    def store_transition(self,
                        state: np.ndarray,
                        action: Action | int,
                        reward: float,
                        next_state: np.ndarray,
                        priority: float = 1.0):
        transition = PriorizedTransition(
            state,
            action,
            reward,
            next_state,
            priority,
        )
        self.buffer.add(transition)

    def train_network(self, batch_size: int = 64):
        if (len(self.buffer) < batch_size):
            return 0.0

        samples, indices, weights = self.buffer.sample(batch_size)
        state_tensor_batch = torch.stack([self._to_tensor(transition.state) for transition in samples])
        action_tensor_batch = torch.stack([self._to_tensor(transition.action) for transition in samples])
        reward_tensor_batch = torch.stack([self._to_tensor(transition.reward) for transition in samples])
        next_state_tensor_batch = torch.stack([self._to_tensor(transition.next_state) for transition in samples])

        with torch.no_grad():
            next_state_target_pred_q = self.target_network(next_state_tensor_batch, action_tensor_batch)
            td_target = reward_tensor_batch + self.td_target_gamma * next_state_target_pred_q
        
        state_pred_q = self.exploration_network(state_tensor_batch, action_tensor_batch)
        loss = (weights * (state_pred_q - td_target) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        new_prios = (td_target - state_pred_q).abs().detach().numpy() + 1e-6 # non-zero needed
        self.buffer.update_prioritites(indices, new_prios)

        self._update_target_network()
    
    def _update_target_network(self, tau = 0.005):
        for q_parameters, target_parameters in zip(self.exploration_network.parameters(), self.target_network.parameters()):
            target_parameters.data.copy_(tau * q_parameters + (1.0 - tau) * target_parameters) # slightly different than byol, as we can just override them
    
    def _init_target_network(self, state_dim: int, action_dim: int, hidden_dim: int):
        self.target_network = ExplorationNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.exploration_network.state_dict())
        self.target_network.eval()

    def _to_tensor(self, input: np.ndarray | Action | int | float):
        return torch.tensor(input, dtype=torch.float32, device=self.device)
