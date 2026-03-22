from abc import abstractmethod
from typing import Any
import gymnasium as gym
from torch import device
from curiosity_gym.core.gridengine import GridEngine

from .intrinsic_motiavtion_model import IntrinsicMotivationModel

class IntrinsicMotivationModelWrapper(gym.Wrapper):
    def __init__(self,
                 env: GridEngine, 
                 intrinsic_model: IntrinsicMotivationModel,
                 device: device | str,
                 intrinsic_reset_threshold: float = -1.0) -> None:
        super().__init__(env)
        self.env: GridEngine = self.env # getting ride of bad type hint as casting isn't a real thing in python
        self.intrinsic_model = intrinsic_model
        self.device = device

        self.last_best_global_state = None
        self.last_best_intrinsic_reward = 0.0
        self.intrinsic_reset_threshold = intrinsic_reset_threshold

        self.reset()

    def reset(self, **kwargs):
        if self.last_best_global_state is None:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, info = self.env.reset_to_specific_global_state(self.last_best_global_state, **kwargs)
        self.prev_state = obs
        return obs, info

    def step(self, action):
        state, extrinsic_reward, terminated, truncated, info, raw_global_state = self.env.step_with_global_state(action)
        intrinsic_reward = self._get_intrinsic_reward_from_model(state=state, action=action)
        self._handle_new_intrinsic_reward(intrinsic_reward, raw_global_state)
        reward = extrinsic_reward + intrinsic_reward

        info["extrinsic_reward"] = extrinsic_reward
        info["intrinsic_reward"] = intrinsic_reward
        info["total_reward"] = reward 

        self.prev_state = state

        return state, reward, terminated, truncated, info
    
    def _handle_new_intrinsic_reward(self, intrinsic_reward, raw_global_state):
        if (self.last_best_intrinsic_reward <= self.intrinsic_reset_threshold):
            self.last_best_global_state = None
        elif (intrinsic_reward >= self.last_best_intrinsic_reward):
            self.last_best_intrinsic_reward = intrinsic_reward
            self.last_best_global_state = raw_global_state

    @abstractmethod
    def _get_intrinsic_reward_from_model(self, *args, **kwargs) -> float | Any:
        pass
