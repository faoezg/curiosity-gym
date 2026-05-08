import torch
import torch.nn.functional as F
import numpy as np
from torch import device
from experiments.components import IntrinsicMotivationModel, Transition, RewardNormalizer

from .icm_network import ICMNetwork

class ICMModel(IntrinsicMotivationModel):
    """Very basic ICM implementation following Pathak et. al. 2017 - Curiosity-driven Exploration by Self-supervised PredictionSelf-Supervised

    Parameters
    ----------
    state_dim : int
        Dimension of the state-space, that is, the observation-space of the env
    action_dim: int
        Dimenstion of the action-space
    hidden_dim: int
        Dimension of the hidden layer of the encoder, forward and invers model
    beta: float
        Weighting of the inverse loss
    eta: float
        Scaler of the intrinsic reward
    """
    def __init__(self,
                 device: device | str,
                 state_dim: int,
                 action_dim: int,
                 latent_rep_dim: int,
                 hidden_dim: int,
                 beta: float,
                 eta: float,
                 icm_lr: float = 1e-3,
                 ):
        super().__init__()
        self.icm_network = ICMNetwork(device, state_dim, action_dim, latent_rep_dim, hidden_dim, beta, eta).to(device)
        self.optimizer = torch.optim.Adam(self.icm_network.parameters(), lr=icm_lr)
        self.action_dim = action_dim
        self.device = device
        self.reward_normalizer = RewardNormalizer()

    def calc_intrinsic_reward(self, state, next_state, action) -> float:
        """
        Calculates the intrinsic reward following Pathak et. al. 2017
        """

        self.icm_network.eval() # no gradiants
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.icm_network.device)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.icm_network.device)
            action_tensor = torch.tensor(action, dtype=torch.long, device=self.icm_network.device)
            _, phi_next, forward_pred, _ = self.icm_network.forward(state_tensor, next_state_tensor, action_tensor)
            intrinsic_reward = self.calc_forward_loss(forward_pred, phi_next).item()
            intrinsic_reward = self.icm_network.eta * intrinsic_reward

        return self.reward_normalizer.normalize_reward(intrinsic_reward)
    
    def calc_forward_loss(self, forward_pred: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        return 0.5 * F.cosine_similarity(forward_pred, next_state, dim=-1) ** 2
    
    def calc_icm_loss(self, state: torch.Tensor, next_state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, phi_next, forward_pred, inv_logits = self.icm_network.forward(state, next_state, action)
        action_one_hot = F.one_hot(action, self.action_dim).to(torch.float32).to(self.device)
        inv_loss = F.cross_entropy(inv_logits, action_one_hot, reduction="sum")
        forward_loss = self.calc_forward_loss(forward_pred, phi_next)
        return inv_loss, forward_loss
    
    def _train_network_with_batch(self, batch: list[Transition]):
        self.icm_network.train()
        prev_state_list = []
        next_state_list = []
        action_list = []

        for transition in batch:
            prev_state_list.append(transition.state)
            action_list.append(transition.action)
            next_state_list.append(transition.next_state)

        state_tensor = torch.tensor(np.asarray(prev_state_list), dtype=torch.float32, device=self.icm_network.device)
        action_tensor = torch.tensor(np.asarray(action_list), dtype=torch.long, device=self.icm_network.device)
        next_state_tensor = torch.tensor(np.asarray(next_state_list), dtype=torch.float32, device=self.icm_network.device)

        inv_loss, forward_loss = self.calc_icm_loss(state_tensor, next_state_tensor, action_tensor)
        icm_beta = self.icm_network.beta
        loss = (1 - icm_beta) * inv_loss + icm_beta * forward_loss
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _train_network(self, prev_state, state, action):
        self.icm_network.train()
        state_tensor = torch.tensor(prev_state, dtype=torch.float32, device=self.icm_network.device)
        next_state_tensor = torch.tensor(state, dtype=torch.float32, device=self.icm_network.device)
        action_tensor = torch.tensor(action, dtype=torch.long, device=self.icm_network.device)
        inv_loss, forward_loss = self.calc_icm_loss(state_tensor, next_state_tensor, action_tensor)
        icm_beta = self.icm_network.beta
        loss = (1 - icm_beta) * inv_loss + icm_beta * forward_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
