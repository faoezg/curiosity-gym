import torch
from torch import nn, device
import torch.nn.functional as F
import copy

from experiments.components.icm_encoder import ICMEncoder

# following Z. Guo et. al. 2022
class ByolExploreNetwork(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int,
                 latent_rep_dim: int,
                 time_horizon: int,
                 stride: int,
                 device: device | str,
                 alpha: float = 0.99,
                 ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.DEVICE = device
        self.time_horizon = time_horizon
        self.latent_rep_dim = latent_rep_dim
        self.action_dim = action_dim

        self.encoder_model = ICMEncoder(device, state_dim, latent_rep_dim, hidden_dim, stride, True, False, False)
        self._init_recurrent_model(action_dim, latent_rep_dim, hidden_dim)
        self.predictor_model = self._create_predictor_model(hidden_dim, latent_rep_dim)
        self._init_target_model(alpha)

    def forward(self, state_buffer: torch.Tensor, action_buffer: torch.Tensor):
        B, T, C = state_buffer.shape 

        state_encoding = self.encoder_model.encode_state(state_buffer.view(B * T, C)).view(B, T, self.latent_rep_dim)
        # state_projection = self.projection_model(state_encoding)

        h_closed_hist = self._calc_closed_loop_history_states(B, action_buffer, state_encoding)
        byol_loss, intrinsic_rewards = self._calc_loss_and_intrinsic_rewards(B, T, C, h_closed_hist, action_buffer, state_buffer)

        return byol_loss, intrinsic_rewards

    def _calc_closed_loop_history_states(self, batch_dim: int, action_buffer: torch.Tensor, state_encoding: torch.Tensor) -> torch.Tensor:
        h_closed = torch.zeros(batch_dim, self.hidden_dim, device=self.DEVICE, dtype=torch.float32)
        previous_action = nn.functional.one_hot(action_buffer[:, 0], self.action_dim).to(torch.float32).to(self.DEVICE)
        input = torch.cat([state_encoding[:,0], previous_action], dim=-1).to(self.DEVICE) # (B, latent_rep_dim + ACTION_EMBEDDING_DIM)
        h_closed = self.close_gru(input, h_closed)
        return h_closed # (B, hidden_dim)
    
    def _calc_loss_and_intrinsic_rewards(self,
                                         batch_dim: int,
                                         end_time: int,
                                         channel: int,
                                         h_closed: torch.Tensor,
                                         action_buffer: torch.Tensor,
                                         state_buffer: torch.Tensor) -> tuple[torch.Tensor | float, torch.Tensor]:
        cos_loss = 0
        count = 0
        intrinsic_rewards = torch.zeros(batch_dim, end_time, device=self.DEVICE, dtype=torch.float32)

        # For reasons of efficency, we could progressivly shift the starting state over the trajecorie
        # thereby learning more from a single trajectory, though, this does also progressivly reduce the time horizon
        # for t in range(end_time):
        h_open = h_closed # b_t
        for k in range(1, self.time_horizon + 1):
            if k >= end_time:
                break
            future_action = nn.functional.one_hot(action_buffer[:, k-1], self.action_dim).to(torch.float32).to(self.DEVICE)
            h_open = self.open_gru(future_action, h_open) # (B, hidden_dim)

            pred = self.predictor_model(h_open)

            with torch.no_grad():
                target_encoded = self.target_encoder_model.encode_state(state_buffer[:, k].view(batch_dim, channel)).view(batch_dim, self.latent_rep_dim)
            
            pred_normalised = F.normalize(pred, dim=-1)
            target_normalised = F.normalize(target_encoded, dim=-1)
            timestep_loss = 2 - 2 * (pred_normalised * target_normalised).sum(dim=-1) # cos-similarity is the dot-product of two unit vectors
            cos_loss += timestep_loss.mean() # take the mean as batch size is not fixed
            count += 1
            intrinsic_rewards[:, k] += timestep_loss

        byol_loss = cos_loss / max(count, 1) # average by k, but count is not necessarily non-zero
        intrinsic_rewards = intrinsic_rewards.cumsum(dim=1)
        return byol_loss, intrinsic_rewards
    
    @torch.no_grad()
    def update_target_model(self):
        for encoder_parameters, target_parameters in zip(self.encoder_model.encoder_model.parameters(), self.target_encoder_model.encoder_model.parameters()):
            target_parameters.data.mul_(self.alpha)
            target_parameters.data.add_(encoder_parameters.data * (1 - self.alpha)) # EMA for Byol
 
    def _init_target_model(self, alpha: float):
        self.target_encoder_model = copy.deepcopy(self.encoder_model)
        self.target_network = copy.deepcopy(self.encoder_model.encoder_model)
        for parameter in self.target_network.parameters():
            parameter.requires_grad = False

        self.target_network.eval()
        self.alpha = alpha
        self.target_encoder_model.encoder_model = self.target_network

    # h in the paper
    def _init_recurrent_model(self, action_dim: int, latent_rep_dim: int, hidden_dim: int):
        self.close_gru = nn.GRUCell(latent_rep_dim + action_dim, hidden_dim, dtype=torch.float32, device=self.DEVICE)
        self.open_gru  = nn.GRUCell(action_dim, hidden_dim, dtype=torch.float32, device=self.DEVICE)

    # g in the paper
    def _create_predictor_model(self, hidden_dim: int, latent_rep_dim: int):
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_rep_dim),
        ).to(torch.float32)
