import gymnasium as gym
import torch

from curiosity_gym.core.gridengine import GridEngine
from .coin_flip_model import CoinFlipModel 
from .coin_flip_wrapper import CoinFlipWrapper

def make_coin_flip_env(
    *,
    base_env_id: str = "SparseEnv",
    base_env_pov: str = "local_2",
    device: str = "cpu",
    reward_scale: float = 10,
    hidden_dim: int = 300,
    d_dim: int = 20,
    priority_alpha: float = 0.5,
    render_mode: str | None = None,
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



    cfm = CoinFlipModel(state_dim=raw_env.observation_space.shape[0], # type: ignore
                        hidden_dim=hidden_dim,
                        d_dim=d_dim,
                        priority_alpha=priority_alpha,
                        reward_scale=reward_scale,
                        device=device,
                        update_period=15,
                        buffer_size=50000,
                        batch_size=512)

    new_env = CoinFlipWrapper(
        env=raw_env,
        cfm=cfm,
        device=torch.device(device),
        max_training_steps=max_training_steps,
        max_episodes=max_episodes)

    new_env = gym.wrappers.RecordVideo(new_env, f"videos/tmp", name_prefix=f"{cfm.name}_{raw_env.unwrapped.name}", episode_trigger=lambda x: x % 1000 == 0)

    return new_env
