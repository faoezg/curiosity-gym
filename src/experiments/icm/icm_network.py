import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import device


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
                 eta: float):
                 super().__init__()
                 self.device = device
                 self.action_dim = action_dim
                 self.beta = beta
                 self.eta = eta
                 self.use_cnn_encoder = False

                 if (not self.use_cnn_encoder):
                    self.encoder_model = self._create_encoder_model(state_dim, hidden_dim_encoder, latent_rep_dim)
                 else:
                    self.encoder_model = self._create_cnn_encoder_model(3, state_dim, latent_rep_dim)
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

          phi, phi_next = self._encode_states(state, next_state)
          #phi = state
          #phi_next = next_state
          forward_pred = self._pass_through_forward_model(phi, action)
          inv_logits = self._pass_through_inverse_model(phi, phi_next)
          #for i, para in enumerate(self.invers_model.parameters()):
          #      print(f'{i + 1}th parameter tensor:', para.shape)
          #      print(para)
          #      print(para.grad)
          return phi, phi_next, forward_pred, inv_logits

    def _encode_states(self, state, next_state):
        if (self.use_cnn_encoder):
            state = torch.swapdims(state, 0, 1)
            next_state = torch.swapdims(next_state, 0, 1)
        phi = self.encoder_model(state)
        phi_next = self.encoder_model(next_state)
        return phi, phi_next

    def _pass_through_forward_model(self, state: torch.Tensor, action: torch.Tensor):
        action_onehot = F.one_hot(action, num_classes=self.action_dim).float()
        forward_input = torch.cat([state, action_onehot], dim=-1)
        return self.forward_model(forward_input)

    def _pass_through_inverse_model(self, state, next_state):
        inv_input = torch.cat([state, next_state], dim=-1)
        inv_logits = self.invers_model(inv_input)
        return inv_logits

    def _create_encoder_model(self, state_dim, hidden_dim, latent_rep_dim) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ELU(),
            nn.Linear(hidden_dim // 4, latent_rep_dim),
        )
    
    def _create_cnn_encoder_model(self, state_channel_dim, state_dim, latent_rep_dim) -> nn.Sequential:
         return nn.Sequential(
              nn.Conv1d(in_channels=state_channel_dim, out_channels=state_channel_dim, kernel_size=1),
              #nn.ReLU(),
              #nn.Conv1d(in_channels=state_channel_dim, out_channels=state_channel_dim, kernel_size=1, groups=3),
              nn.ReLU(),
              nn.Conv1d(in_channels=state_channel_dim, out_channels=1, kernel_size=1), # MLP/NIN
              nn.Linear(state_dim, latent_rep_dim) # linear transfrom into latent dims
         )

    def _create_forward_model(self, latent_rep_dim, action_dim, hidden_dim) -> nn.Sequential:
        """(phi(s), a) -> phi_hat(s')"""
        return nn.Sequential(
            nn.Linear(latent_rep_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            #nn.Linear(hidden_dim, hidden_dim),
            #nn.ELU(),
            #nn.Linear(hidden_dim, hidden_dim),
            #nn.ELU(),
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
