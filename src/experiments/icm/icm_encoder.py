import torch.nn as nn
import torch

class ICMEncoder():

    def __init__(
            self,
            device,
            state_dim: int,
            latent_rep_dim: int,
            hidden_dim_encoder: int,
            stride: int,
            use_1d_cnn_encoder: bool,
            use_cnn_encoder: bool
        ) -> None:

        self.use_1d_cnn_encoder = use_1d_cnn_encoder
        self.use_cnn_encoder = use_cnn_encoder 

        if (self.use_1d_cnn_encoder):
            self.encoder_model = self._create_1d_cnn_encoder_model(1, stride, state_dim, latent_rep_dim).to(device)
        elif (self.use_cnn_encoder):
            self.encoder_model = self._create_cnn_encoder_model(1, stride, state_dim, latent_rep_dim).to(device)
            self.encoder_fc = nn.Linear(256 * 7 * 10,  256).to(device) # linear transfrom into latent dims
        else:
            self.encoder_model = self._create_encoder_model(state_dim, hidden_dim_encoder, latent_rep_dim).to(device)

    def encode_states(self, state, next_state):
        phi = self.encode_state(state)
        phi_next = self.encode_state(next_state)

        return phi, phi_next

    def encode_state(self, state):
        if (self.use_1d_cnn_encoder):
            s1 = state.clone()
            s1 = s1.unsqueeze(1)
            if (len(state.shape) == 1):
                s1 = torch.swapdims(s1, 0, 1)
 
            phi = self.encoder_model(s1)
            phi = phi.squeeze()
            return phi
        elif (self.use_cnn_encoder):

            state = torch.swapdims(state, 1, -1)
            state = torch.swapdims(state, 2, -1)
 
            phi = self.encoder_model(state)
            phi = phi.reshape(phi.size(0), -1)
            
            phi = self.encoder_fc(phi)
            return phi

        else:
            phi = self.encoder_model(state)
            return phi


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
    
    def _create_1d_cnn_encoder_model(self, state_channel_dim, stride, state_dim, latent_rep_dim) -> nn.Sequential:
         return nn.Sequential(
              nn.Conv1d(in_channels=1, out_channels=1, kernel_size=stride, stride=stride),
              #nn.ReLU(),
              #nn.Conv1d(in_channels=state_channel_dim, out_channels=state_channel_dim, kernel_size=1, groups=3),
              nn.ELU(),
              nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1), # MLP/NIN
              nn.Linear(24, latent_rep_dim // 2), # linear transfrom into latent dims
              nn.Linear(latent_rep_dim // 2, latent_rep_dim) # linear transfrom into latent dims
         )

    def _create_cnn_encoder_model(self, state_channel_dim, stride, state_dim, latent_rep_dim) -> nn.Sequential:
         return nn.Sequential(
              nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4, padding=0),
              nn.ELU(),
              nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0),
              nn.ELU(),
              nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=0),
              nn.ELU(),
              nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=0),
              nn.ELU(),
              nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=0),
              nn.ELU(),
         )
