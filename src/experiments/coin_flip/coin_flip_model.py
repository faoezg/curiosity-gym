import torch
from torch import nn
from torch import device, optim
from .coin_flip_network import CoinFlipNetwork
from .buffer import CFNTransition
from .rademacher_generator import RademacherDistGenerator
from experiments.components import PriorityReplayBuffer, IntrinsicMotivationModel, RewardNormalizer
from curiosity_gym.utils.enums import Action
import numpy as np

class CoinFlipModel(IntrinsicMotivationModel):
    def __init__(self,
                 state_dim: int,
                 hidden_dim: int = 32,
                 d_dim: int = 10,
                 reward_scale: float = 0.8, # set it high, as extrnsic is sparse in experiments
                 batch_size: int = 32,
                 buffer_size: int = 1000,
                 min_req_buffer_population: int = 32,
                 priority_alpha: float = 0.5,
                 update_period: int = 1,
                 p_replace: float = 1.0,
                 device: device | str = "cuda" if torch.cuda.is_available() else "cpu"
                 ) -> None:
        self.device = device
        self.min_req_buffer_population = min_req_buffer_population
        self.update_period = update_period
        self.batch_size = batch_size
        self.priority_alpha = priority_alpha
        self.reward_scale = reward_scale

        self.coin_flip_network = CoinFlipNetwork(state_dim, hidden_dim, d_dim).to(device)
        self.optimizer = optim.Adam(params=self.coin_flip_network.parameters(), lr=0.0001)
        self.rademacher_generator = RademacherDistGenerator(d_dim, p_replace)
        self.buffer = PriorityReplayBuffer(buffer_size, priority_alpha)

        self.step_count = 0
        self.num_updates = np.zeros(buffer_size, dtype=np.float32)

        self.reward_normalizer = RewardNormalizer()

        self.reset()

        super().__init__("cfn")

    def _np_array_to_tensor(self, state: np.ndarray):
        return torch.tensor(state, dtype=torch.float32, device=self.device)

    def calc_intrinsic_reward(self, state: np.ndarray, action: Action | int, with_training: bool = True):
        state_tensor = self._np_array_to_tensor(state)
        rademacher_sample = self.rademacher_generator()

        if (with_training):
            prev_state = self.prev_state
            if not prev_state is None:
                transition = CFNTransition(
                    state=prev_state,
                    next_state=state,
                    reward=0.0,
                    coin_flip_vector=rademacher_sample,
                    action=action,
                    priority=1.0
                )
                self.buffer.add(transition)
            
            self.prev_state = state
            self.step_count += 1
            isBufferPopulatedEnough = len(self.buffer) >= self.min_req_buffer_population
            if (isBufferPopulatedEnough and len(self.buffer) >= self.min_req_buffer_population) and self.step_count % self.update_period == 0:
                self._train_network()
        else:
            isBufferPopulatedEnough = len(self.buffer) >= self.min_req_buffer_population

        intrinsic_reward = 0.0
        if (isBufferPopulatedEnough):
            with torch.no_grad():
                _, _, _, one_over_counts = self.coin_flip_network(state_tensor)
                intrinsic_reward_tensor = one_over_counts
                intrinsic_reward = intrinsic_reward_tensor.squeeze().detach().item()

                if (with_training):
                    intrinsic_reward = self.reward_normalizer.normalize_reward(intrinsic_reward)
        
        return intrinsic_reward * self.reward_scale

    
    def _train_network(self):
        samples, indicies, weights = self.buffer.sample(self.batch_size)

        state_batch = []
        rademacher_sample_batch = []
        for sample in samples:
            state_batch.append(self._np_array_to_tensor(sample.state))
            rademacher_sample_batch.append(sample.coin_flip_vector) # type: ignore
        
        state_batch = torch.stack(state_batch).to(self.device)
        rademacher_sample_batch = torch.from_numpy(np.array(rademacher_sample_batch)).float().to(self.device).detach()
        weights_tensor = torch.from_numpy(weights).float().to(self.device)

        coin_flip_preds, _, _, one_over_counts = self.coin_flip_network(state_batch)

        #loss = nn.functional.mse_loss(coin_flip_preds, rademacher_sample_batch, reduction="none")
        #loss = torch.sum(loss * weights_tensor.unsqueeze(1), 1).mean()

        element_wise = (coin_flip_preds - rademacher_sample_batch) ** 2
        loss = (torch.sum(element_wise, dim=1) * weights_tensor).mean()
    
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._update_buffer_prioritise(one_over_counts, indicies)
    
    def _update_buffer_prioritise(self, one_over_counts_tensor, indicies):
        num_updates = self.num_updates[indicies]
        pseudo_counts_tensor = one_over_counts_tensor ** 2
        pseudo_counts = pseudo_counts_tensor.squeeze().cpu().detach().numpy()
        new_priorities = (self.priority_alpha / (num_updates + 1)) + (1 - self.priority_alpha) * pseudo_counts
        self.buffer.update_prioritites(indicies, new_priorities)
        self.num_updates[indicies] += 1

    def reset(self):
        self.prev_state = None
        self.step_count = 0
        self.rademacher_generator.reset()
        self.reward_normalizer.reset()
