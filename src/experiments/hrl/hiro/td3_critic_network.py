import torch.nn as nn
import torch

# inspired by https://github.com/watakandai/hiro_pytorch/blob/master/hiro/models.py
class TD3Critic(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dim,
                 goal_dim,) -> None:
        
        self.network = self._init_network(state_dim, action_dim, hidden_dim, goal_dim)
        self.target_network = self._init_network(state_dim, action_dim, hidden_dim, goal_dim)

    def _init_network(self, state_dim, action_dim, hidden_dim, goal_dim):
        return nn.Sequential(
            nn.Linear(state_dim + goal_dim + action_dim,  hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor, goal: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        input = torch.cat( (state,goal,action), dim=-1)
        value = self.network(input)
        return value
