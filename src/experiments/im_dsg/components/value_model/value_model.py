import torch
from torch import nn
from torch import device, optim
import numpy as np

from curiosity_gym.utils.enums import Action
from .value_network import ValueNetwork

class ValueModel():
    def __init__(self,
                 state_dim: int,
                 hidden_dim: int = 64,
                 device: device | str = "cpu") -> None:
        self.device = device
        self.value_network = ValueNetwork(state_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(params=self.value_network.parameters(), lr=0.0001)
    
    def calc_extrinsic_reward(self, state: np.ndarray, action: Action | int):
        state_tensor = self._to_tensor(state)
        action_tensor = self._to_tensor(action)

        extrinsic_reward = self.value_network(state_tensor, action_tensor)
        return extrinsic_reward

    def train_network(self, true_reward: float, predicted_reward: torch.Tensor):
        true_reward_tensor = self._to_tensor(true_reward)
        loss = nn.functional.mse_loss(true_reward_tensor, predicted_reward, reduction="none")
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _to_tensor(self, input: np.ndarray | Action | int | float):
        return torch.tensor(input, dtype=torch.float32, device=self.device)
