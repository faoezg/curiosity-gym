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
    allow_global_state_reset: bool = False,
    render_mode: str | None = None,
    max_episodes: int = 1,
    max_training_steps: int = 500,
    use_simple_obs: bool = True,
    use_globaly_unique_id: bool = True
) -> gym.Env:
    raw_env: GridEngine = gym.make(base_env_id,
                                   render_mode=render_mode,
                                   agentPOV=base_env_pov,
                                   simple_obs=use_simple_obs,
                                   use_globaly_unique_id=use_globaly_unique_id
                                   ) # type: ignore

    icm = ICMModel(
            device = device,
            state_dim=raw_env.observation_space.shape[0], # type: ignore
            action_dim=raw_env.action_space.n, # type: ignore
            latent_rep_dim=128,
            hidden_dim=256,
            beta=.2,
            eta=0.3,
            icm_lr=5e-6
        )

    return ICMCuriosityWrapper(device,
                               raw_env,
                               icm,
                               intrinsic_reset_threshold,
                               allow_global_state_reset,
                               max_training_steps=max_training_steps,
                               max_episodes=max_episodes)
