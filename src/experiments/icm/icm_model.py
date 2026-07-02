import torch
import torch.nn.functional as F
import numpy as np
from torch import device
from experiments.components import IntrinsicMotivationModel, Transition

from .icm_network import ICMNetwork

class ICMModel(IntrinsicMotivationModel):
    """Very basic ICM implementation following Pathak et. al. 2017 - Curiosity-driven Exploration by Self-supervised PredictionSelf-Supervised
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
                 icm_lr: float = 1e-3,
                 use_1d_cnn_encoder: bool = False,
                 use_cnn_encoder: bool = False,
                 use_id_encoder: bool = False,
                 share_memory: bool = False
                 ):
        super().__init__("icm")
        self.latent_rep_dim = latent_rep_dim
        self.icm_network = ICMNetwork(device,
                                      state_dim,
                                      action_dim,
                                      latent_rep_dim,
                                      hidden_dim_forward,
                                      hidden_dim_inverse,
                                      hidden_dim_encoder,
                                      beta,
                                      eta,
                                      stride,
                                      use_1d_cnn_encoder,
                                      use_cnn_encoder,
                                      use_id_encoder=use_id_encoder).to(device)

        self.use_cnn_encoder = use_cnn_encoder
        if (share_memory):
            self.icm_network.share_memory()
            self.icm_network.encoder.encoder_model.share_memory()
            self.optimizer = torch.optim.Adam([
                {"params": self.icm_network.encoder.encoder_model.parameters(), "lr": 1e-3},
                {"params": self.icm_network.forward_model.parameters(), "lr": 1e-3},
                {"params": self.icm_network.invers_model.parameters(), "lr": 1e-3},
            ])

        else:
            self.optimizer = torch.optim.Adam([
                {"params": self.icm_network.encoder.encoder_model.parameters(), "lr": 1e-3},
                {"params": self.icm_network.forward_model.parameters(), "lr": 1e-3},
                {"params": self.icm_network.invers_model.parameters(), "lr": 1e-3},
            ])
            #self.optimizer = torch.optim.Adam(self.icm_network.parameters())
            #self.optimizer = torch.optim.SGD(self.icm_network.parameters())

        self.action_dim = action_dim
        self.device = device

        self.inv_loss_function = torch.nn.CrossEntropyLoss(reduction="none")
    
    def calc_intrinsic_reward(self, state, next_state, action) -> float:
        """
        Calculates the intrinsic reward following Pathak et. al. 2017
        """

        self.icm_network.eval() # no gradiants
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.icm_network.device).unsqueeze(0)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=self.icm_network.device).unsqueeze(0)
            action_tensor = torch.tensor(action, dtype=torch.long, device=self.icm_network.device).unsqueeze(0)

            if (self.use_cnn_encoder):
                state_tensor /= 255.0
                next_state_tensor /= 255.0
            _, phi_next, forward_pred, _ = self.icm_network.forward(state_tensor, next_state_tensor, action_tensor)
            intrinsic_reward = self.calc_forward_loss(forward_pred, phi_next).mean().item()
            intrinsic_reward = self.icm_network.eta * intrinsic_reward

        return intrinsic_reward
    
    def calc_forward_loss(self, forward_pred: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        return 0.5 * F.mse_loss(forward_pred, next_state, reduction="none") * self.latent_rep_dim
    
    def calc_icm_loss(self, state: torch.Tensor, next_state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if (self.use_cnn_encoder):
            state /= 255.0
            next_state /= 255.0
        _, phi_next, forward_pred, inv_logits = self.icm_network.forward(state, next_state, action)

        forward_loss = self.calc_forward_loss(forward_pred, phi_next) #* self.latent_rep_dim # scaling taken from the implementation provided by the paper
        inv_loss = self.inv_loss_function(inv_logits, action)
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

        state_tensor = torch.tensor(np.asarray(prev_state_list, dtype=np.uint8), dtype=torch.float32, device=self.icm_network.device)
        action_tensor = torch.tensor(np.asarray(action_list), dtype=torch.long, device=self.icm_network.device)
        next_state_tensor = torch.tensor(np.asarray(next_state_list, dtype=np.uint8), dtype=torch.float32, device=self.icm_network.device)
        if (self.use_cnn_encoder):
            state_tensor /= 255.0
            next_state_tensor /= 255.0

        inv_loss, forward_loss = self.calc_icm_loss(state_tensor, next_state_tensor, action_tensor)
        forward_loss_per_batch = torch.mean(forward_loss, dim=-1)
        icm_beta = self.icm_network.beta
        loss = (1 - icm_beta) * inv_loss + icm_beta * forward_loss_per_batch
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.icm_network.parameters(), max_norm=2)
        #fwd_grad_norm = sum(p.grad.norm().item() for p in self.icm_network.forward_model.parameters())
        #inv_grad_norm = sum(p.grad.norm().item() for p in self.icm_network.invers_model.parameters())
        #enc_grad_norm = sum(p.grad.norm().item() for p in self.icm_network.encoder.encoder_model.parameters())
        #print(f"fwd grad norm = {fwd_grad_norm:.4f} inv grad norm = {inv_grad_norm:.4f}, enc grad norm = {enc_grad_norm:.4f}")
        #torch.nn.utils.clip_grad_norm_(self.icm_network.parameters(), max_norm=40)
        self.optimizer.step()

        return inv_loss.mean().item(), forward_loss.mean().item()

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
        torch.nn.utils.clip_grad_norm_(self.icm_network.parameters(), max_norm=2)
        self.optimizer.step()

        #enc_grad_norm = sum(p.grad.norm().item() for p in self.icm_network.encoder_model.parameters())
        #print('Encoder grad norm:', enc_grad_norm)
        return inv_loss.item(), forward_loss.item()
