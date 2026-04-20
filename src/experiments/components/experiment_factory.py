import gymnasium as gym

from curiosity_gym.core.gridengine import GridEngine
from .logging_wrapper import LoggingWrapper


def make_experiment_env(
    *,
    base_env_id: str = "SparseEnv",
    base_env_pov: str = "global",
    render_mode: str | None = None,
    max_episodes: int = 5000,
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


    return LoggingWrapper(
        env=raw_env,
        training_steps=raw_env.env_settings.max_steps,
        training_episodes=max_episodes
        )
