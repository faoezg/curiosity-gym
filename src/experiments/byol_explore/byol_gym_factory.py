import gymnasium as gym
import numpy as np
import torch

from curiosity_gym.core.gridengine import GridEngine
from .byol_explore import ByolExploreModel
from .byol_wrapper import ByolExploreWrapper


def make_byol_env(
    *,
    base_env_id: str = "SparseEnv",
    base_env_pov: str = "global",
    device: str = "cpu",
    lambda_byol: float = 5.0,
    reward_norm_decay: float = 0.99,
    hidden_dim: int = 8,
    latent_rep_dim: int = 4,
    time_horizon: int = 5,
    alpha: float = 0.9999,
    render_mode: str = None
) -> gym.Env:
    raw_env: GridEngine = gym.make(base_env_id, render_mode=render_mode, agentPOV="local_2")

    byol_model = ByolExploreModel(state_dim=raw_env.observation_space.shape[0], # type: ignore
                            action_dim=raw_env.action_space.n, # type: ignore
                            hidden_dim=hidden_dim,
                            latent_rep_dim=latent_rep_dim,
                            time_horizon=time_horizon,
                            device=device,
                            alpha=alpha).to(device)


    return ByolExploreWrapper(
        env=raw_env,
        byol_explore_model=byol_model,
        device=torch.device(device),
        lambda_byol=lambda_byol,
        reward_norm_decay=reward_norm_decay,
    )
