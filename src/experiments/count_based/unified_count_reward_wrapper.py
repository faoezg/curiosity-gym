import gymnasium as gym
from experiments.count_based.unified_count import UnifiedCountModel
from curiosity_gym.core.gridengine import Action, SimplerAction
import numpy as np

class UnifiedCountWrapper(gym.Wrapper):
    def __init__(self,
                 env: gym.Env,
                 count_model: UnifiedCountModel,
                 beta: float = 0.05,
                 eps: float = 0.01,
                 clip_range = 0.02):
        super().__init__(env)
        self.count_model = count_model
        self.beta = beta
        self.eps = eps
        self.clip_range = clip_range
    
    def step(self, action: Action | SimplerAction):
        state, extrinsic_reward, terminated, truncated, info = self.env.step(action)

        intrinsic_reward = self._calc_intrinsic_reward(state)
        self.count_model.update_observation(state)
        reward = extrinsic_reward + intrinsic_reward # type: ignore

        info["extrinsic_reward"] = extrinsic_reward
        info["intrinsic_reward"] = intrinsic_reward
        info["total_reward"] = reward 

        print(info)

        return state, reward, terminated, truncated, info
    
    def _calc_intrinsic_reward(self, state: np.ndarray) -> float:
        rho = self.count_model.calc_visitation_prob(state)
        rho_after = self.count_model.calc_visitation_prob_after_observation(state)

        if (rho_after <= rho):
            pseudo_count = 0.0
        else:
            pseudo_count = (rho * (1.0 - rho_after) / (rho_after - rho))

        intrinsic_reward = self.beta / np.sqrt(pseudo_count + self.eps)

        if (intrinsic_reward <= self.clip_range):
            return 0.0
        else:
            return intrinsic_reward
