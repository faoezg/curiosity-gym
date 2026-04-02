import torch
from torch import nn

# network to take state, action and return reward
class ValueNetwork(nn.Module):
    def __init__(self,
                 state_dim: int,
                 hidden_dim: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.output_dim = 1 # action_dim
        self.network = self._init_network()

    def _init_network(self):
        input_dim = self.state_dim + self.output_dim 
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        input_tensor = torch.concat((state, action))
        value_pred = self.network(input_tensor) 
        return value_pred
