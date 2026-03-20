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
    lambda_byol: float = 5.0,
    reward_norm_decay: float = 0.99,
    hidden_dim: int = 8,
    d_dim: int = 4,
    time_horizon: int = 5,
    alpha: float = 0.9999,
    render_mode: str = None
) -> gym.Env:
    raw_env: GridEngine = gym.make(base_env_id, render_mode=render_mode, agentPOV="local_2")

    cfm = CoinFlipModel(state_dim=raw_env.observation_space.shape[0], # type: ignore
                            device=device)


    return CoinFlipWrapper(
        env=raw_env,
        cfm=cfm,
        device=torch.device(device))
