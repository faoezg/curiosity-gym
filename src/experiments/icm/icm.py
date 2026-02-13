import numpy as np
import torch
from torch import Tensor, device
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ICMModel(nn.Module):
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
                 device: device,
                 state_dim: int,
                 action_dim: int,
                 latent_rep_dim: int,
                 hidden_dim: int,
                 beta: float,
                 eta: float
                 ):
                 super(ICMModel, self).__init__()
                 self.device = device
                 self.action_dim = action_dim
                 self.beta = beta
                 self.eta = eta

                 self.encoder_model = self.create_encoder_model(state_dim, hidden_dim, latent_rep_dim)
                 self.forward_model = self.create_forward_model(latent_rep_dim, action_dim, hidden_dim)
                 self.invers_model  = self.create_invers_model(latent_rep_dim, action_dim, hidden_dim)
    
    def forward(
                self,
                state: torch.Tensor,
                next_state: torch.Tensor,
                action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
          """
          Returns
          -------
          phi: torch.Tensor
          Latent representation of the state (phi(s)) in (Batch,State-Space)-dimension

          phi_next: torch.Tensor
          Latent representation of the next state (phi(s')) in (Batch,State-Space)-dimension

          forward_model_pred: torch.Tensor
          Latent representation of the expected next state (phi_hat(s')) in (Batch,State-Space)-dimension

          inv_logits:: torch.Tensor
          Inversed logits of the two latent representations phi(s) and phi(s')
          """

          phi, phi_next = self.encode_states(state, next_state)
          inv_logits = self.pass_through_inverse_model(phi, phi_next)
          forward_pred = self.pass_through_forward_model(phi, action)
          return phi, phi_next, forward_pred, inv_logits

    def encode_states(self, state, next_state):
        phi = self.encoder_model(state)           # Dim (Batch, state-space)
        phi_next = self.encoder_model(next_state) # Dim (Batch, state-space)
        return phi, phi_next

    def pass_through_forward_model(self, state: Tensor, action: Tensor):
        action_onehot = F.one_hot(action, num_classes=self.action_dim).float()
        forward_input = torch.cat([state, action_onehot], dim=1)
        return self.forward_model(forward_input)

    def pass_through_inverse_model(self, state, next_state):
        inv_input = torch.cat([state, next_state], dim=1) # Dim (Batch, 2 * state-space)
        inv_logits = self.invers_model(inv_input)         # Dim (Batch, state-space)
        return inv_logits

    def calc_intrinsic_reward(self, state, next_state, action) -> float:
        """
        Calculates the intrinsic reward following Pathak et. al. 2017
        """

        self.eval() # no gradiants
        with torch.no_grad():
         state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).flatten(0).unsqueeze(0)
         next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.device).flatten(0).unsqueeze(0)
         action_tensor = torch.tensor([action], dtype=torch.long, device=self.device)

         _, phi_next, forward_pred, _ = self.forward(state_tensor, next_state_tensor, action_tensor)

         intrinsic_reward = self.calc_forward_loss(forward_pred, phi_next).item()
         return self.eta * intrinsic_reward
    
    def calc_forward_loss(self, forward_pred: Tensor, next_state: Tensor) -> Tensor:
        return 0.5 * F.mse_loss(forward_pred, next_state, reduction="mean")
    
    def calc_icm_loss(self, state: Tensor, next_state: Tensor, action: Tensor) -> Tuple[Tensor, Tensor]:
        _, phi_next, forward_pred, inv_logits = self.forward(state, next_state, action)
        inv_loss = F.cross_entropy(inv_logits, action, reduction="mean")
        forward_loss = self.calc_forward_loss(forward_pred, phi_next)
        return inv_loss, forward_loss
    
    def create_encoder_model(self, state_dim, hidden_dim, latent_rep_dim) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_rep_dim)
        )

    def create_forward_model(self, latent_rep_dim, action_dim, hidden_dim) -> nn.Sequential:
        """(phi(s), a) -> phi_hat(s')"""
        return nn.Sequential(
            nn.Linear(latent_rep_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_rep_dim)
        )

    def create_invers_model(self, latent_rep_dim, action_dim, hidden_dim) -> nn.Sequential:
        """(phi(s), phi(s')) -> action_hat"""
        return nn.Sequential(
            nn.Linear(2 * latent_rep_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
