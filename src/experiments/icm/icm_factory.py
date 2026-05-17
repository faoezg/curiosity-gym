import gymnasium as gym
import torch

from curiosity_gym.core.gridengine import GridEngine
from .icm_model import ICMModel 
from .icm_wrapper import ICMCuriosityWrapper 


def make_icm_env(
    *,
    base_env_id: str = "SparseEnv",
    base_env_pov: str = "local_2",
    device: str = "cpu",
    latent_rep_dim: int = 512,
    hidden_dim_forward: int = 1024,
    hidden_dim_inverse: int = 32,
    hidden_dim_encoder: int = 1024,
    intrinsic_reset_threshold: float = 0.5,
    allow_global_state_reset: bool = False,
    render_mode: str | None = None,
    max_episodes: int = 1,
    max_training_steps: int = 500,
    use_simple_obs: bool = True,
    use_simple_actions: bool = False,
    use_globaly_unique_id: bool = True,
    is_atari: bool = False,
) -> gym.Env:

    if (not is_atari):
        raw_env: GridEngine = gym.make(base_env_id,
                                    render_mode=render_mode,
                                    agentPOV=base_env_pov,
                                    simple_obs=use_simple_obs,
                                    simple_actions=use_simple_actions,
                                    use_globaly_unique_id=use_globaly_unique_id
                                    ) # type: ignore
    else:
        raw_env = gym.make(base_env_id, render_mode="rgb_array") # type: ignore


    icm = ICMModel(
            device = device,
            state_dim=raw_env.observation_space.shape[0], # type: ignore
            action_dim=raw_env.action_space.n, # type: ignore
            latent_rep_dim=latent_rep_dim,#raw_env.observation_space.shape[0], # type: ignore
            hidden_dim_forward=hidden_dim_forward,
            hidden_dim_inverse=hidden_dim_inverse,
            hidden_dim_encoder=hidden_dim_encoder,
            beta=.2,
            eta=5,
            icm_lr=1e-6
        )
    new_env = ICMCuriosityWrapper(device,
                               raw_env,
                               icm,
                               intrinsic_reset_threshold,
                               allow_global_state_reset,
                               max_training_steps=max_training_steps,
                               max_episodes=max_episodes)
    new_env = gym.wrappers.RecordVideo(new_env, f"videos/tmp", episode_trigger=lambda x: x % 5== 0)

    return new_env
