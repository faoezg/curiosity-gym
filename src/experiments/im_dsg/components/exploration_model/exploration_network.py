import torch
from torch import nn

class ExplorationNetwork(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.input_dim = self.state_dim + self.action_dim
        self.hidden_dim = hidden_dim
        self.output_dim = 1
        self.network = self._init_network()

    def _init_network(self):
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        if (state.ndim == 1): # Non-batched query
            input_tensor = torch.concat((state, action))
        else:
            input_tensor = torch.concat((state, action), dim=1)
        value_pred = self.network(input_tensor) 
        return value_pred
