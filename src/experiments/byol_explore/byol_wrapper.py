import gymnasium as gym
import torch
from torch import device
from collections import deque

from curiosity_gym.core.gridengine import GridEngine
from .byol_model import ByolExploreModel
from experiments.components import IntrinsicMotivationModelWrapper
from experiments.components.buffers import ReplayBuffer, Transition

class ByolExploreWrapper(IntrinsicMotivationModelWrapper):
    def __init__(
        self,
        env: GridEngine,
        byol_explore_model: ByolExploreModel,
        device: device,
        intrinsic_reset_threshold: float = 0.5,
    ):
        super().__init__(env, byol_explore_model, device, intrinsic_reset_threshold)
        self.intrinsic_model: ByolExploreModel = self.intrinsic_model
        self.buffer = ReplayBuffer(size=self.intrinsic_model.byol_network.time_horizon + 1)

    def step(self, action):
        state, extrinsic_reward, terminated, truncated, info, raw_global_state = self.env.step_with_global_state(action)

        transition = Transition(self.prev_state, state, action)
        self.buffer.add(transition)

        if self.buffer.is_fully_populated():
            state_buffer, action_buffer = self._create_trajectory()
            intrinsic_reward, byol_loss = self.intrinsic_model.calc_intrinsic_reward(state_buffer, action_buffer)
            self._handle_new_intrinsic_reward(intrinsic_reward, raw_global_state)
            reward = extrinsic_reward + intrinsic_reward # type: ignore

            self.intrinsic_model._train_network(byol_loss)
        else:
            reward = extrinsic_reward
            intrinsic_reward = 0.0
        
        info["extrinsic_reward"] = extrinsic_reward
        info["intrinsic_reward"] = intrinsic_reward
        info["total_reward"] = reward 

        return state, reward, terminated, truncated, info 


    def _create_trajectory(self):
        state_tensors = []
        actions = []
        samples = self.buffer.sample(len(self.buffer))

        for transition in samples:
            state_tensor = torch.from_numpy(transition.next_state)
            state_tensors.append(state_tensor)
            actions.append(transition.action)

        state_buffer = torch.stack(state_tensors, dim=0).unsqueeze(0) # (B=1, T=2, N, 3)
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
