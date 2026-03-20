import gymnasium as gym
import torch

from curiosity_gym.core.gridengine import GridEngine
from .coin_flip_model import CoinFlipModel 
from .coin_flip_wrapper import CoinFlipWrapper


def make_coin_flip_env(
    *,
    base_env_id: str = "SparseEnv",
    base_env_pov: str = "global",
    device: str = "cpu",
    reward_scale: float = 0.01,
    hidden_dim: int = 300,
    d_dim: int = 100,
    priority_alpha: float = 0.5,
    render_mode: str | None = None
) -> gym.Env:
    raw_env: GridEngine = gym.make(base_env_id, render_mode=render_mode, agentPOV="local_2")

    cfm = CoinFlipModel(state_dim=raw_env.observation_space.shape[0], # type: ignore
                        hidden_dim=hidden_dim,
                        d_dim=d_dim,
                        priority_alpha=priority_alpha,
                        reward_scale=reward_scale,
                        device=device)


    return CoinFlipWrapper(
        env=raw_env,
        cfm=cfm,
        device=torch.device(device))
