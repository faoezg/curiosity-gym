import gymnasium as gym

from curiosity_gym.core.gridengine import GridEngine
from .logging_wrapper import LoggingWrapper


def make_experiment_env(
    *,
    base_env_id: str = "SparseEnv",
    render_mode: str | None = None,
    max_episodes: int = 5000,
    max_training_steps: int = 500
) -> gym.Env:
    raw_env: GridEngine = gym.make(base_env_id, render_mode=render_mode, agentPOV="local_2") # type: ignore

    return LoggingWrapper(
        env=raw_env,
        training_steps=raw_env.env_settings.max_steps,
        training_episodes=max_episodes
        )
