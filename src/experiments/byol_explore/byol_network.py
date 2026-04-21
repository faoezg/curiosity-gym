import torch
from torch import nn, device
import torch.nn.functional as F
import copy

# following Z. Guo et. al. 2022
class ByolExploreNetwork(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int,
                 latent_rep_dim: int,
                 time_horizon: int,
                 device: device | str,
                 alpha: float = 0.99,
                 ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.DEVICE = device
        self.time_horizon = time_horizon
        self.latent_rep_dim = latent_rep_dim
        self.action_dim = action_dim

        self.encoder_model = self._create_encoder_model(state_dim, hidden_dim, latent_rep_dim)
        self.projection_model = self._create_projection_model(latent_rep_dim)
        self._init_recurrent_model(action_dim, latent_rep_dim, hidden_dim)
        self.predictor_model = self._create_predictor_model(hidden_dim, latent_rep_dim)
        self._init_target_model(alpha)
    
    def forward(self, state_buffer: torch.Tensor, action_buffer: torch.Tensor):
        B, T, C = state_buffer.shape 

        state_encoding = self.encoder_model(state_buffer.view(B * T, C)).view(B, T, self.latent_rep_dim)
        # state_projection = self.projection_model(state_encoding)

        h_closed_hist = self._calc_closed_loop_history_states(B, T, action_buffer, state_encoding)
        byol_loss, intrinsic_rewards = self._calc_loss_and_intrinsic_rewards(B, T, h_closed_hist, action_buffer, state_buffer)

        return byol_loss, intrinsic_rewards

    def _calc_closed_loop_history_states(self, batch_dim: int, end_time: int, action_buffer: torch.Tensor, state_encoding: torch.Tensor) -> torch.Tensor:
        h_hist = []
        h_closed = torch.zeros(batch_dim, self.hidden_dim, device=self.DEVICE, dtype=torch.float32)
        for t in range(end_time):
            previous_action = nn.functional.one_hot(action_buffer[:, t], self.action_dim).to(torch.float32).to(self.DEVICE)
            input = torch.cat([state_encoding[:,t], previous_action], dim=-1).to(self.DEVICE) # (B, latent_rep_dim + ACTION_EMBEDDING_DIM)
            h_closed = self.close_gru(input, h_closed)
            h_hist.append(h_closed)
        return torch.stack(h_hist, dim=1) # (B, T, hidden_dim)
    
    def _calc_loss_and_intrinsic_rewards(self,
                                         batch_dim: int,
                                         end_time: int,
                                         h_hist: torch.Tensor,
                                         action_buffer: torch.Tensor,
                                         state_buffer: torch.Tensor) -> tuple[torch.Tensor | float, torch.Tensor]:
        cos_loss = 0
        count = 0
        intrinsic_rewards = torch.zeros(batch_dim, end_time, device=self.DEVICE, dtype=torch.float32)

        # For reasons of efficency, we could progressivly shift the starting state over the trajecorie
        # thereby learning more from a single trajectory, though, this does also progressivly reduce the time horizon
        # for t in range(end_time):
        h_open = h_hist[:, 0] # b_t
        for k in range(1, self.time_horizon + 1):
            if k >= end_time:
                break
            future_action = nn.functional.one_hot(action_buffer[:, k], self.action_dim).to(torch.float32).to(self.DEVICE)
            h_open = self.open_gru(future_action, h_open) # (B, hidden_dim)

            pred = self.predictor_model(h_open)

            with torch.no_grad():
                target_encoded = self.target_encoder_model(state_buffer[:, k].flatten())
                target_projection = self.target_projection_model(target_encoded)
            
            pred_normalised = F.normalize(pred, dim=-1)
            target_normalised = F.normalize(target_projection, dim=-1)
            timestep_loss = 2 - 2 * (pred_normalised * target_normalised).sum(dim=-1) # cos-similarity is the dot-product of two unit vectors
            cos_loss += timestep_loss.mean() # take the mean as batch size is not fixed
            count += 1

            intrinsic_rewards[:, 0] = intrinsic_rewards[:, 0] + timestep_loss

        byol_loss = cos_loss / max(count, 1) # average by k, but count is not necessarily non-zero
        return byol_loss, intrinsic_rewards
    
    @torch.no_grad()
    def update_target_model(self):
        for encoder_parameters, target_parameters in zip(self.encoder_model.parameters(), self.target_encoder_model.parameters()):
            target_parameters.data.mul_(self.alpha).add(encoder_parameters.data * (1.0 - self.alpha)) # EMA for Byol

        for projection_parameters, target_parameters in zip(self.projection_model.parameters(), self.target_projection_model.parameters()):
            target_parameters.data.mul_(self.alpha).add(projection_parameters.data * (1.0 - self.alpha))
 
 
    # f in the paper, no conv needed as we have a very simply state to begin with
    def _create_encoder_model(self, state_dim: int, hidden_dim: int, latent_rep_dim: int):
        return nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_rep_dim)
        ).to(torch.float32).to(self.DEVICE)
    
    def _init_target_model(self, alpha: float):
        self.target_encoder_model = copy.deepcopy(self.encoder_model)
        self.target_projection_model = copy.deepcopy(self.projection_model)
        for parameter in self.target_encoder_model.parameters():
            parameter.requires_grad = False
        for parameter in self.target_projection_model.parameters():
            parameter.requires_grad = False

        self.target_encoder_model.eval()
        self.target_projection_model.eval()
        self.alpha = alpha

    # h in the paper
    def _init_recurrent_model(self, action_dim: int, latent_rep_dim: int, hidden_dim: int):
        self.close_gru = nn.GRUCell(latent_rep_dim + action_dim, hidden_dim, dtype=torch.float32, device=self.DEVICE)
        self.open_gru  = nn.GRUCell(action_dim, hidden_dim, dtype=torch.float32, device=self.DEVICE)

    # g in the paper
    def _create_predictor_model(self, hidden_dim: int, latent_rep_dim: int):
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_rep_dim)
        ).to(torch.float32)

    # not in the byol-explore but the byol paper to help with model collapse on finding trivial solution for encoding
    def _create_projection_model(self, latent_rep_dim: int):
        return nn.Sequential(
            nn.Linear(latent_rep_dim, latent_rep_dim),
            nn.LayerNorm(latent_rep_dim),
            nn.ReLU(),
            nn.Linear(latent_rep_dim, latent_rep_dim),
        ).to(torch.float32)
