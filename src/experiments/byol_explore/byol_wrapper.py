from collections import deque
from curiosity_gym.core.gridengine import GridEngine
from .byol_explore import ByolExploreModel
from .reward_normaliser import RewardNormaliser
import gymnasium as gym
import torch
from torch import device

class ByolExploreWrapper(gym.Wrapper):
    def __init__(
        self,
        env: GridEngine,
        byol_explore_model: ByolExploreModel,
        device: device,
        lambda_byol: float = 5.0, # λ_byol in the paper
        reward_norm_decay: float = 0.99,
    ):
        super().__init__(env)
        self.device = device 
        self.model = byol_explore_model
        self.opt = torch.optim.Adam(self.model.parameters())

        self.norm = RewardNormaliser(decay=reward_norm_decay)

        self.lambda_byol = lambda_byol
        self.buffer = deque(maxlen=byol_explore_model.time_horizon + 1)
        self.prev_action = None

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        return state, info

    def step(self, action):
        state, extrinsic_reward, terminated, truncated, info = self.env.step(action)
        self.buffer.append((state, action))

        if len(self.buffer) == self.model.time_horizon + 1:
            state_buffer, action_buffer = self._create_trajectory()

            intrinsic_reward, byol_loss = self._calc_intrinsic_reward_and_loss(state_buffer, action_buffer)
            reward = extrinsic_reward + self.lambda_byol * intrinsic_reward # type: ignore

            self._train_byol_explore_model(byol_loss)
        else:
            reward = extrinsic_reward
            intrinsic_reward = 0.0
        
        info["extrinsic_reward"] = extrinsic_reward
        info["intrinsic_reward"] = intrinsic_reward
        info["total_reward"] = reward 

        # print(info)

        return state, reward, terminated, truncated, info 

    def _create_trajectory(self):
        state_tensors = []
        actions = []
        for state, action in self.buffer:
            state_tensors.append(torch.from_numpy(state))
            actions.append(action)

        state_buffer = torch.stack(state_tensors,dim=0,).unsqueeze(0) # (B=1, T=2, N, 3)
        state_buffer = state_buffer.to(torch.float32).to(self.device)

        action_buffer = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0) # (B=1, T=2)

        return state_buffer, action_buffer

    def _calc_intrinsic_reward_and_loss(self, state_buffer: torch.Tensor, action_buffer: torch.Tensor):
        byol_loss, raw_intrinsic_reward = self.model(state_buffer, action_buffer) # raw_intr: (B,T)
        raw_intrinsic_reward = raw_intrinsic_reward.squeeze(0)                    # (T,)
        norm_intrinsic_reward = self.norm(raw_intrinsic_reward).detach()
        intrinsic_reward = norm_intrinsic_reward[0].item()
        return intrinsic_reward, byol_loss

    def _train_byol_explore_model(self, byol_loss: torch.Tensor):
        self.opt.zero_grad()
        (byol_loss * self.lambda_byol).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        self.model.update_target_model()
