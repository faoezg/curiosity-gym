import gymnasium as gym

from curiosity_gym.core.gridengine import GridEngine
from .logging_wrapper import LoggingWrapper


def make_experiment_env(
    *,
    base_env_id: str = "SparseEnv",
    base_env_pov: str = "local_2",
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


    new_env = LoggingWrapper(
        env=raw_env,
        training_steps=raw_env.unwrapped.env_settings.max_steps,
        training_episodes=max_episodes
        )

    new_env = gym.wrappers.RecordVideo(new_env, f"videos/tmp", episode_trigger=lambda x: x % 1000 == 0)

    return new_env
