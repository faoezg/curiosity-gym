import torch.nn as nn
import torch
import torch.nn.functional as F

class OptionCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, option_dim, hidden_dim):
        super().__init__()
        self.action_dim = action_dim
        self.option_dim = option_dim

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.policy_over_options_head = nn.Linear(hidden_dim, option_dim)
        self.beta_head  = nn.Linear(hidden_dim, option_dim)
        self.option_policy_head = nn.Linear(hidden_dim, option_dim * action_dim)
    
    def get_hidden(self, state: torch.Tensor):
        return self.backbone(state)
    
    def get_policy_over_options(self, hidden: torch.Tensor):
        return self.policy_over_options_head(hidden)
    
    def get_terminations(self, hidden: torch.Tensor):
        return self.beta_head(hidden)
    
    def forward(self, state_tensor):
        hidden = self.backbone(state_tensor)

        policy = self.policy_over_options_head(hidden)

        option_logits = self.option_policy_head(hidden)
        option = option_logits.view(-1, self.n_options, self.n_actions)
        option_actions = F.softmax(option, dim=-1)
        beta = torch.sigmoid(self.beta_head(hidden))

        return policy, option_actions, beta
