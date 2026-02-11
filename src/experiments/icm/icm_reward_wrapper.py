import gymnasium as gym
import torch
from stable_baselines3.common.callbacks import BaseCallback
from experiments.icm.icm import ICMModel

class ICMCuriosityWrapper(gym.Wrapper):
    """
    Wraps any gym.Env and adds an intrinsic curiosity bonus to the reward.
    The ICM module is trained online on the same transitions that the agent sees.
    """
    def __init__(
        self,
        env: gym.Env,
        icm: ICMModel,
        icm_lr: float = 1e-3,
    ):
        super().__init__(env)
        self.icm = icm
        self.optimizer = torch.optim.Adam(self.icm.parameters(), lr=icm_lr)

        self.prev_state = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_state = obs
        return obs, info

    def step(self, action):
        state, extrinsic_reward, terminated, truncated, info = self.env.step(action)

        intrinsic_reward = self.icm.calc_intrinsic_reward(self.prev_state, state, action)
        reward = extrinsic_reward + intrinsic_reward # type: ignore

        info["extrinsic_reward"] = extrinsic_reward
        info["intrinsic_reward"] = intrinsic_reward
        info["total_reward"] = reward 

        self._train_icm_model(self.prev_state, state, action)
        self.prev_state = state

        return state, reward, terminated, truncated, info

    def _train_icm_model(self, prev_state, state, action):
        self.icm.train() # TODO this is single batch, perhaps add replay buffer?
        state_tensor = torch.tensor(prev_state, dtype=torch.float32).flatten(0)
        next_state_tensor = torch.tensor(state, dtype=torch.float32).flatten(0)
        action_tensor = torch.tensor(action, dtype=torch.long)
        inv_loss, forward_loss = self.icm.calc_icm_loss(state_tensor, next_state_tensor, action_tensor)
        icm_beta = self.icm.beta
        loss = (1 - icm_beta) * inv_loss + icm_beta * forward_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
