import gymnasium as gym
from torch import device

from curiosity_gym.core.gridengine import GridEngine
from experiments.count_based.unified_count import UnifiedCountModel
from experiments.components import IntrinsicMotivationModelWrapper
from curiosity_gym.core.gridengine import Action, SimplerAction
import numpy as np

class UnifiedCountWrapper(IntrinsicMotivationModelWrapper):
    def __init__(self,
                 device: device,
                 env: GridEngine,
                 count_model: UnifiedCountModel,
                 intrinsic_reset_threshold: float = 0.5,
                 allow_global_state_reset: bool = False,
                 max_training_steps: int = 500,
                 max_episodes: int = 1000):
        super().__init__(env,
                         count_model,
                         device,
                         intrinsic_reset_threshold,
                         allow_global_state_reset,
                         max_training_steps,
                         max_episodes)
        self.intrinsic_model: UnifiedCountModel = self.intrinsic_model

    def reset(self, **kwargs):
        self.intrinsic_model.reset()
        return super().reset(**kwargs)

    def _get_intrinsic_reward_from_model(self, state, action):
        intrinsic_reward = self.intrinsic_model.calc_intrinsic_reward(state)
        self.intrinsic_model._train_network(state)
        return intrinsic_reward

    def _get_intrinsic_reward_from_model_no_training(self, state, action):
        return self.intrinsic_model.calc_intrinsic_reward(state)
