from .td3_model import TD3Model
from .buffer import HIROReplayBuffer, HighHIROTransition, LowHIROTransition

import numpy as np
import random

class HIROAgent():
    def __init__(self,
                 device,
                 state_dim,
                 action_dim,
                 goal_dim,
                 hidden_dim_actor,
                 hidden_dim_critic,
                 training_freq,
                 reward_scaling,
                 high_policy_training_freq,
                 low_policy_training_freq,
                 replay_buffer_size = 50000,
                 batch_size = 32,
                 ) -> None:
        self.high_policy_model = TD3Model(
            device,
            state_dim,
            action_dim,
            goal_dim,
            hidden_dim_actor,
            hidden_dim_critic,
            policy_update_freq=high_policy_training_freq
        )

        self.low_policy_model = TD3Model(
            device,
            state_dim,
            action_dim,
            goal_dim,
            hidden_dim_actor,
            hidden_dim_critic,
            policy_update_freq=low_policy_training_freq
        )

        self.high_replay_buffer = HIROReplayBuffer(replay_buffer_size)
        self.low_replay_buffer = HIROReplayBuffer(replay_buffer_size)

        self.batch_size = batch_size
        
        self.reward_scale = self.reward_scale
        self.episode_subreward = 0
        self.subreward = 0

        self.final_goal = np.array([-1] * state_dim) 
        self.subgoal = random.randint(0,3)
    
    def step(self, state, env):
        action = self._choose_action(state)
        state, extrinsic_reward, terminated, truncated, info = env.step(action)

        

        return state, extrinsic_reward, terminated, truncated, info
    
    def add_transition(self,
                       state: np.ndarray,
                       action: int | float,
                       reward: float,
                       next_state: np.ndarray,
                       goal: np.ndarray,
                       not_done: bool):
        transition = HIROTransition(state, action, reward, next_state, goal, not_done)
        self.replay_buffer.add(transition)
    
    def train(self):
        if (len(self.replay_buffer) >= self.batch_size):
            self.model.train(self.replay_buffer.sample(self.batch_size))

    def _choose_action(self, state):
        return self.low_policy_model.get_action(state, self.subgoal)
