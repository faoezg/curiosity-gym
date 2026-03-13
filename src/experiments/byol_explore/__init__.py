import gymnasium as gym

gym.register(
    id="SparseEnv-ByolExplore",
    entry_point="experiments.byol_explore.byol_gym_factory:make_byol_env",
    kwargs={
        "base_env_id":"SparseEnv"
    }
)

gym.register(
    id="DistractiveEnv-ByolExplore",
    entry_point="experiments.byol_explore.byol_gym_factory:make_byol_env",
    kwargs={
        "base_env_id":"DistractiveEnv"
    }
)

gym.register(
    id="MultitaskEnv-ByolExplore",
    entry_point="experiments.byol_explore.byol_gym_factory:make_byol_env",
    kwargs={
        "base_env_id":"MultitaskEnv"
    }
)
