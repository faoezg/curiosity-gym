import torch
from torch import nn, device
import torch.nn.functional as F
import copy

from experiments.components import IntrinsicMotivationModel
from .byol_network import ByolExploreNetwork
from .reward_normaliser import RewardNormaliser

class ByolExploreModel(IntrinsicMotivationModel):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int,
                 latent_rep_dim: int,
                 time_horizon: int,
                 device: device | str,
                 alpha: float = 0.9999,
                 lambda_byol: float = 5.0, # lambda_byol in the paper
                 reward_norm_decay: float = 0.99
                ) -> None:
        super().__init__()
        self.byol_network = ByolExploreNetwork(state_dim, action_dim, hidden_dim, latent_rep_dim, time_horizon, device, alpha).to(device)
        self.reward_normaliser = RewardNormaliser(decay=reward_norm_decay)

        self.lambda_byol = lambda_byol
        self.optimizer = torch.optim.Adam(self.byol_network.parameters(), lr=0.0001)

    def calc_intrinsic_reward(self, state_buffer: torch.Tensor, action_buffer: torch.Tensor):
        byol_loss, raw_intrinsic_reward = self.byol_network(state_buffer, action_buffer)
        raw_intrinsic_reward = raw_intrinsic_reward.squeeze(0)
        norm_intrinsic_reward = self.reward_normaliser(raw_intrinsic_reward).detach()
        intrinsic_reward = norm_intrinsic_reward[0].item()
        return intrinsic_reward, byol_loss

    def _train_network(self, byol_loss: torch.Tensor):
        self.optimizer.zero_grad()
        (byol_loss * self.lambda_byol).backward()
        self.optimizer.step()
        self.byol_network.update_target_model()
