import gymnasium as gym
from torch import device
from .coin_flip_model import CoinFlipModel

class CoinFlipWrapper(gym.Wrapper):
    def __init__(
        self,
        device: device,
        env: gym.Env,
        cfm: CoinFlipModel,
        intrinsic_reset_threshold: float = 0.5
    ):
        super().__init__(env)
        self.device = device
        self.cfm = cfm
        self.last_best_global_state = None
        self.last_best_intrinsic_reward = 0.0
        self.intrinsic_reset_threshold = intrinsic_reset_threshold

    def reset(self, **kwargs):
        if self.last_best_global_state is None:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, info = self.env.reset_to_specific_global_state(self.last_best_global_state, **kwargs)
        self.prev_state = obs
        return obs, info

    def step(self, action):
        state, extrinsic_reward, terminated, truncated, info, raw_global_state = self.env.step_with_global_state(action) # type: ignore
        intrinsic_reward = self.cfm.calc_intrinsic_reward(state, action)
        if (self.last_best_intrinsic_reward <= self.intrinsic_reset_threshold):
            self.last_best_global_state = None
        elif (intrinsic_reward >= self.last_best_intrinsic_reward):
            self.last_best_intrinsic_reward = intrinsic_reward
            self.last_best_global_state = raw_global_state
        reward = extrinsic_reward + intrinsic_reward # type: ignore

        info["extrinsic_reward"] = extrinsic_reward
        info["intrinsic_reward"] = intrinsic_reward
        info["total_reward"] = reward 

        self.prev_state = state

        return state, reward, terminated, truncated, info
