import gymnasium as gym
import torch

from curiosity_gym.core.gridengine import GridEngine
from .unified_count import UnifiedCountModel
from .unified_count_reward_wrapper import UnifiedCountWrapper

def make_unified_count_env(
    *,
    base_env_id: str = "SparseEnv",
    base_env_pov: str = "local_2",
    device: str = "cpu",
    render_mode: str | None = None,
    max_episodes: int = 10000,
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

    unified_count_model = UnifiedCountModel(raw_env.observation_space.shape[0] ** 2, # type: ignore
                                            clip_range=0,
                                            eps=0.5)
    new_env = UnifiedCountWrapper(
        env=raw_env,
        count_model=unified_count_model,
        device=torch.device(device),
        max_training_steps=max_training_steps,
        max_episodes=max_episodes)

    new_env = gym.wrappers.RecordVideo(new_env, f"videos/tmp", episode_trigger=lambda x: x % 1000 == 0)

    return new_env
