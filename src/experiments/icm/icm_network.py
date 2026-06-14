import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import device

from experiments.components.icm_encoder import ICMEncoder


class ICMNetwork(nn.Module):
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
                 hidden_dim_forward: int,
                 hidden_dim_inverse: int,
                 hidden_dim_encoder: int,
                 beta: float,
                 eta: float,
                 stride: int,
                 use_1d_cnn_encoder: bool,
                 use_cnn_encoder: bool,
                 use_id_encoder: bool):
                 super().__init__()
                 self.device = device
                 self.action_dim = action_dim
                 self.beta = beta
                 self.eta = eta
                 
                 self.encoder = ICMEncoder(
                       device=device,
                       state_dim=state_dim,
                       latent_rep_dim=latent_rep_dim,
                       hidden_dim_encoder=hidden_dim_encoder,
                       stride=stride,
                       use_1d_cnn_encoder=use_1d_cnn_encoder,
                       use_cnn_encoder=use_cnn_encoder,
                       use_id_encoder=use_id_encoder
                 )
                 self.forward_model = self._create_forward_model(latent_rep_dim, action_dim, hidden_dim_forward)
                 self.invers_model  = self._create_invers_model(latent_rep_dim, action_dim, hidden_dim_inverse)
    
    def forward(
                self,
                state: torch.Tensor,
                next_state: torch.Tensor,
                action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

          phi, phi_next = self.encoder.encode_states(state, next_state)
          #phi = state
          #phi_next = next_state
          if (len(phi.shape) == 1):
            phi = phi.unsqueeze(0)
            phi_next = phi_next.unsqueeze(0)
          
          phi = phi
          phi_next = phi_next 
          forward_pred = self._pass_through_forward_model(phi, action)
          inv_logits = self._pass_through_inverse_model(phi, phi_next)
          return phi, phi_next, forward_pred, inv_logits

    def _pass_through_forward_model(self, state: torch.Tensor, action: torch.Tensor):
        action_onehot = F.one_hot(action, num_classes=self.action_dim).float()
        forward_input = torch.cat([state, action_onehot], dim=-1)
        return self.forward_model(forward_input)

    def _pass_through_inverse_model(self, state, next_state):
        inv_input = torch.cat([state, next_state], dim=-1)
        inv_logits = self.invers_model(inv_input)
        return inv_logits

    def _create_forward_model(self, latent_rep_dim, action_dim, hidden_dim) -> nn.Sequential:
        """(phi(s), a) -> phi_hat(s')"""
        return nn.Sequential(
            nn.Linear(latent_rep_dim + action_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_rep_dim),
        )

    def _create_invers_model(self, latent_rep_dim, action_dim, hidden_dim) -> nn.Sequential:
        """(phi(s), phi(s')) -> action_hat"""
        return nn.Sequential(
            nn.Linear(2 * latent_rep_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim),
        )
