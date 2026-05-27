import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.icm.icm_encoder import ICMEncoder

# AI SUPPORTED CODE - OpenAI GPT OSS 120B

#def normalized_columns_initializer(tensor: torch.Tensor, std: float = 1.0) -> None:
#    out = np.random.randn(*tensor.shape).astype(np.float32)
#    out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True) + 1e-8)
#    with torch.no_grad():
#        tensor.copy_(torch.from_numpy(out))

class IcmLSTMPolicy(nn.Module):
    def __init__(
            self,
            device,
            state_dim,
            action_dim,
            latent_dim,
            hidden_dim_encoder = 1024,
            stride = 1,
            use_1d_cnn_encoder = False,
            use_cnn_encoder = False
                 ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.device = device

        self.encoder = ICMEncoder(
            device=device,
            state_dim=state_dim,
            latent_rep_dim=latent_dim,
            hidden_dim_encoder=hidden_dim_encoder,
            stride=stride,
            use_1d_cnn_encoder=use_1d_cnn_encoder,
            use_cnn_encoder=use_cnn_encoder
        )

        self.lstm = nn.LSTMCell(latent_dim, latent_dim).to(self.device)
        self.action_network = nn.Linear(latent_dim, action_dim).to(self.device)
        self.value_network  = nn.Linear(latent_dim, 1).to(self.device)
    
    def init_lstm_state(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, self.latent_dim, device=self.device)
        c = torch.zeros(batch_size, self.latent_dim, device=self.device)
        return hidden , c

    def forward(self, state, lstm_state: tuple[torch.Tensor, torch.Tensor] | None = None):

        if (lstm_state == None):
            hidden, c = self.init_lstm_state(state.shape[0])
        else:
            hidden, c = lstm_state
        
        phi = self.encoder.encode_state(state)

        hidden_next, c_next = self.lstm(phi, (hidden,c))

        logits = self.action_network(hidden_next)
        probs = torch.softmax(logits, dim=-1)

        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample()
        sample_one_hot = F.one_hot(action_idx, num_classes=self.action_dim)

        value = self.value_network(hidden_next).squeeze(-1)

        return probs, sample_one_hot, value, (hidden_next, c_next)

    def get_action(self, state, lstm_state=None):
        _, sample_one_hot, _, lstm_state_next = self.forward(state, lstm_state)
        action = sample_one_hot.argmax(dim=-1)

        return action, lstm_state_next
