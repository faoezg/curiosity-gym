import gymnasium as gym
from torch import device
from .coin_flip_model import CoinFlipModel

class CoinFlipWrapper(gym.Wrapper):
    def __init__(
        self,
        device: device,
        env: gym.Env,
        cfm: CoinFlipModel,
    ):
        super().__init__(env)
        self.device = device
        self.cfm = cfm

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_state = obs
        return obs, info

    def step(self, action):
        state, extrinsic_reward, terminated, truncated, info = self.env.step(action)
        intrinsic_reward = self.cfm.calc_intrinsic_reward(state, action)
        reward = extrinsic_reward + intrinsic_reward # type: ignore

        info["extrinsic_reward"] = extrinsic_reward
        info["intrinsic_reward"] = intrinsic_reward
        info["total_reward"] = reward 

        self.prev_state = state

        return state, reward, terminated, truncated, info
