import gymnasium as gym
import numpy as np
import torch

from curiosity_gym.core.gridengine import GridEngine
from .byol_model import ByolExploreModel
from .byol_wrapper import ByolExploreWrapper


def make_byol_env(
    *,
    base_env_id: str = "SparseEnv",
    base_env_pov: str = "local_2",
    device: str = "cpu",
    lambda_byol: float = 5.0,
    reward_norm_decay: float = 0.999,
    hidden_dim: int = 1024,
    latent_rep_dim: int = 1024,
    time_horizon: int = 2,
    alpha: float = 0.99999,
    render_mode: str = "rgb_array",
    max_episodes: int = 5000,
    max_training_steps: int = 500,
    use_simple_obs: bool = True,
    use_globaly_unique_id: bool = True,
    is_atari: bool = False
) -> gym.Env:
    if (not is_atari):
        raw_env: GridEngine = gym.make(base_env_id,
                                    render_mode=render_mode,
                                    agentPOV=base_env_pov,
                                    simple_obs=use_simple_obs,
                                    use_globaly_unique_id=use_globaly_unique_id
                                    ) # type: ignore
    else:
        raw_env = gym.make(base_env_id, render_mode="rgb_array") # type: ignore



    byol_model = ByolExploreModel(state_dim=raw_env.observation_space.shape[0], # type: ignore
                            action_dim=raw_env.action_space.n, # type: ignore
                            hidden_dim=hidden_dim,
                            latent_rep_dim=latent_rep_dim,
                            time_horizon=time_horizon,
                            device=device,
                            alpha=alpha,
                            lambda_byol=lambda_byol,
                            reward_norm_decay=reward_norm_decay)



    new_env = ByolExploreWrapper(
        env=raw_env,
        byol_explore_model=byol_model,
        device=torch.device(device),
        max_episodes=max_episodes,
        max_training_steps=max_training_steps
    )

    new_env = gym.wrappers.RecordVideo(new_env, f"videos/tmp", name_prefix=f"{byol_model.name}_{raw_env.name}", episode_trigger=lambda x: x % 1000 == 0)

    return new_env
