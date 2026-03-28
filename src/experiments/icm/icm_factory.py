import gymnasium as gym
import torch

from curiosity_gym.core.gridengine import GridEngine
from .icm_model import ICMModel 
from .icm_wrapper import ICMCuriosityWrapper 


def make_icm_env(
    *,
    base_env_id: str = "SparseEnv",
    base_env_pov: str = "global",
    device: str = "cpu",
    latent_rep_dim: int = 32,
    hidden_dim: int = 64,
    intrinsic_reset_threshold: float = 0.5,
    render_mode: str | None = None
) -> gym.Env:
    raw_env: GridEngine = gym.make(base_env_id, render_mode=render_mode, agentPOV="local_2") # type: ignore

    icm = ICMModel(
            device = device,
            state_dim=raw_env.observation_space.shape[0], # type: ignore
            action_dim=raw_env.action_space.n, # type: ignore
            latent_rep_dim=32,
            hidden_dim=64,
            beta=.2,
            eta=0.03,
        )

    return ICMCuriosityWrapper(device, raw_env, icm, 0.1)
