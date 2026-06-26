import torch
from torch import device

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
        allow_global_state_reset: bool = False,
        max_training_steps: int = 500,
        max_episodes: int = 1000,
        batch_dim = 32,
        buffer_size = 5000,
    ):
        super().__init__(env, byol_explore_model,
                         device,
                         intrinsic_reset_threshold,
                         allow_global_state_reset,
                         max_training_steps,
                         max_episodes)
        self.intrinsic_model: ByolExploreModel = self.intrinsic_model
        self.batch_dim = batch_dim

        self.time_horizon = self.intrinsic_model.byol_network.time_horizon

        buffer_size = (self.intrinsic_model.byol_network.time_horizon + 1) * buffer_size
        self.buffer = ReplayBuffer(size=buffer_size)
        self.running_buffer = ReplayBuffer(size=self.time_horizon + 1)
    
    def reset(self, **kwargs):
        self.running_buffer.buffer.clear()
        return super().reset(**kwargs)

    def step(self, action):
        if (isinstance(self.env, GridEngine)):
            state, extrinsic_reward, terminated, truncated, info, raw_global_state = self.env.step_with_global_state(action)
        else:
            state, extrinsic_reward, terminated, truncated, info = self.env.step(action)

        transition = Transition(self.prev_state, action, extrinsic_reward, state) # type: ignore
        self.buffer.add(transition)
        self.running_buffer.add(transition)

        if ((len(self.buffer) >= self.batch_dim * (self.intrinsic_model.byol_network.time_horizon + 1)) and self.training_step % 12 == 0):
            state_buffer, action_buffer = self._create_trajectorys_for_batch()
            # This is fine as the buffer only every has one trajecotry. Thus o_t+1 is the latest in the buffer
            # Therefore, if the sample the entire buffer the loss should be correctly associated this the current transition
            intrinsic_reward, byol_loss = self.intrinsic_model.calc_intrinsic_reward(state_buffer, action_buffer, False)
            self.intrinsic_model._train_network(byol_loss)

        if (len(self.running_buffer) == (self.time_horizon + 1)):
            state_buffer, action_buffer = self._create_trajectory_for_running_buffer()
            # This is fine as the buffer only every has one trajecotry. Thus o_t+1 is the latest in the buffer
            # Therefore, if the sample the entire buffer the loss should be correctly associated this the current transition
            intrinsic_reward, byol_loss = self.intrinsic_model.calc_intrinsic_reward(state_buffer, action_buffer, False)
            intrinsic_reward = intrinsic_reward[0, 1].item()
            if (isinstance(self.env, GridEngine)):
                self._handle_new_intrinsic_reward(intrinsic_reward, raw_global_state)
            reward = extrinsic_reward + intrinsic_reward # type: ignore

        else:
            reward = extrinsic_reward
            intrinsic_reward = 0.0
            self.all_intrinsisc_rewards.append(intrinsic_reward)

        self.all_extrinsisc_rewards.append(extrinsic_reward)

        if (isinstance(self.env, GridEngine)):
            self.state_space_visited[self.env.objects.agent.position.tobytes()] = 1
            self.all_visited_cell_counts_by_step.append(len(self.state_space_visited))

        self.total_extrinsic_reward += extrinsic_reward # type: ignore
        
        self._log_training(extrinsic_reward, intrinsic_reward)
        
        info["extrinsic_reward"] = extrinsic_reward
        self.total_episode_extrinsic_reward += extrinsic_reward # type: ignore
        info["intrinsic_reward"] = intrinsic_reward
        self.total_episode_intrinsic_reward += intrinsic_reward
        info["total_reward"] = reward 

        self.prev_state = state
        self.training_step += 1
        self.total_training_step += 1

        return state, reward, terminated, truncated, info 
    
    def _create_trajectory_for_running_buffer(self):
        time_horizon = self.intrinsic_model.byol_network.time_horizon
        state_tensors = []
        actions = []
        samples = self.running_buffer.sample_continues_slices(1, time_horizon + 1, self.env.env_settings.max_steps) # not sampling as that breaks causality

        for batch in samples:
            batch_state_list = []
            batch_action_list = []
            for idx, transition in enumerate(batch):
                state_tensor = torch.from_numpy(transition.next_state)
                batch_state_list.append(state_tensor)
                if (idx != self.time_horizon):
                    batch_action_list.append(transition.action)
            state_tensors.append(torch.stack(batch_state_list, dim=0).to(self.device))
            actions.append(torch.tensor(batch_action_list, dtype=torch.long, device=self.device))

        state_buffer = torch.stack(state_tensors, dim=0) # (B, T=2, N)
        state_buffer = state_buffer.to(torch.float32).to(self.device)

        action_buffer = torch.stack(
            actions,
        ).to(self.device) # (B, T=2)

        return state_buffer, action_buffer

    def _create_trajectorys_for_batch(self):
        time_horizon = self.intrinsic_model.byol_network.time_horizon
        state_tensors = []
        actions = []
        samples = self.buffer.sample_continues_slices(self.batch_dim, time_horizon + 1, self.env.env_settings.max_steps) # not sampling as that breaks causality

        for batch in samples:
            batch_state_list = []
            batch_action_list = []
            for transition in batch:
                state_tensor = torch.from_numpy(transition.next_state)
                batch_state_list.append(state_tensor)
                batch_action_list.append(transition.action)
            state_tensors.append(torch.stack(batch_state_list, dim=0).to(self.device))
            actions.append(torch.tensor(batch_action_list, dtype=torch.long, device=self.device))

        state_buffer = torch.stack(state_tensors, dim=0) # (B, T=2, N)
        state_buffer = state_buffer.to(torch.float32).to(self.device)

        action_buffer = torch.stack(
            actions,
        ).to(self.device) # (B, T=2)

        return state_buffer, action_buffer
