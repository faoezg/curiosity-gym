import torch.nn as nn
import torch

# inspired by https://github.com/watakandai/hiro_pytorch/blob/master/hiro/models.py
class TD3Actor(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dim,
                 goal_dim,) -> None:
        
        self.network = self._init_network(state_dim, action_dim, hidden_dim, goal_dim)


    def _init_network(self, state_dim, action_dim, hidden_dim, goal_dim):
        return nn.Sequential(
            nn.Linear(state_dim + goal_dim,  hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        input = torch.cat((state, goal), dim=-1)
        action_logits = self.network(input)
        return action_logits
