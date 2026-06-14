# inspired by https://github.com/watakandai/hiro_pytorch/blob/master/hiro/models.py

import torch
import torch.nn.functional as F
import numpy as np

from .td3_actor_network import TD3Actor
from .td3_critic_network import TD3Critic
from .buffer import HIROReplayBuffer, HIROTransition

class TD3Model():
    def __init__(self,
                 device,
                 state_dim,
                 action_dim,
                 goal_dim,
                 hidden_dim_actor,
                 hidden_dim_critic,
                 actor_lr = 1e-3,
                 critic_lr = 1e-3,
                 expl_noise = 0.1,
                 policy_noise = 0.2,
                 noise_clip = 0.5,
                 gamma = 0.99,
                 policy_update_freq = 2,
                 tau= 5e-3
                 ) -> None:
        self.device = device
        self.expl_noise = expl_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.gamma = gamma
        self.policy_update_freq = 2
        self.tau = tau

        self.actor = TD3Actor(state_dim, action_dim, hidden_dim_actor, goal_dim)
        self.target_actor = TD3Actor(state_dim, action_dim, hidden_dim_actor, goal_dim)
        self._init_target_network(self.actor.network, self.target_actor.network)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = TD3Critic(state_dim, action_dim, hidden_dim_actor, goal_dim)
        self.target_critic = TD3Critic(state_dim, action_dim, hidden_dim_actor, goal_dim)
        self._init_target_network(self.critic.network, self.target_critic.network)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.total_training_iteration = 0

    def _init_target_network(self, original_network: torch.nn.Sequential, target_network: torch.nn.Sequential):
        self._update_target_network(original_network, target_network, 1.0)

    def _update_target_network(self, original_network: torch.nn.Sequential, target_network: torch.nn.Sequential, tau: float):
        for target_param, original_param in zip(target_network.parameters(), original_network.parameters()):
            target_param.data.copy_(tau * original_param.data + (1.0 - tau) * target_param.data)

    def _train(self,
               states: torch.Tensor,
               goals: torch.Tensor,
               actions: torch.Tensor,
               rewards: torch.Tensor,
               next_states: torch.Tensor,
               next_goals: torch.Tensor,
               not_done: torch.Tensor):
        self.total_training_iteration += 1

        with torch.no_grad():
            n_actions = self.target_actor(next_states, next_goals)

            target_value = self.target_critic(next_states, next_goals, n_actions)
            target_value_detached = (rewards + not_done * self.gamma * target_value).detach()
        
        current_value = self.critic(states, goals, actions)
        critic_loss = F.smooth_l1_loss(current_value, target_value_detached)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_training_iteration % self.policy_update_freq == 0:
            choosen_actions = self.actor(states, goals)
            value = self.critic(states, goals, choosen_actions)
            actor_loss = -value.mean() # gradient ascent

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._update_target_network(self.critic.network, self.target_critic.network, self.tau)
            self._update_target_network(self.actor.network, self.target_actor.network, self.tau)

    def train(self, sample: list[HIROTransition]):
        states = []
        actions = []
        rewards = []
        next_states = []
        goals = []
        not_dones = []

        for transition in sample:
            states.append(self._to_tensor(transition.state))
            actions.append(self._to_tensor(transition.action, torch.long)) # type: ignore
            rewards.append(self._to_tensor(transition.reward))
            next_states.append(self._to_tensor(transition.next_state))
            goals.append(self._to_tensor(transition.goal))
            not_dones.append(self._to_tensor(transition.not_done))
        
        states_tensor = torch.stack(states).to(self.device)
        actions_tensor = torch.stack(actions).to(self.device)
        rewards_tensor = torch.stack(rewards).to(self.device)
        next_states_tensor = torch.stack(next_states).to(self.device)
        goals_tensor = torch.stack(goals).to(self.device)
        not_dones_tensor = torch.stack(not_dones).to(self.device)
        
        self._train(states_tensor, goals_tensor, actions_tensor, rewards_tensor, next_states_tensor, goals_tensor, not_dones_tensor)
    
    def get_action(self, state, goal):
        return self.actor(state, goal)
    
    def _to_tensor(self, input: np.ndarray | float, dtype = torch.float16):
        return torch.tensor(input, dtype=torch.float16, device=self.device)
