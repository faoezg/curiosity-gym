import gymnasium as gym

gym.register(
    id="SparseEnv-ByolExplore",
    entry_point="experiments.byol_explore.byol_gym_factory:make_byol_env",
    kwargs={
        "base_env_id":"SparseEnv",
    }
)

gym.register(
    id="DistractiveEnv-ByolExplore",
    entry_point="experiments.byol_explore.byol_gym_factory:make_byol_env",
    kwargs={
        "base_env_id":"DistractiveEnv",
        "time_horizon":10
    }
)

gym.register(
    id="MultitaskEnv-ByolExplore",
    entry_point="experiments.byol_explore.byol_gym_factory:make_byol_env",
    kwargs={
        "base_env_id":"MultitaskEnv"
    }
)

gym.register(
    id="MountainCar-ByolExplore",
    entry_point="experiments.byol_explore.byol_gym_factory:make_byol_env",
    kwargs={
        "base_env_id":"MountainCar-v0",
        "max_episodes": 500,
        "is_atari": True
    }
)
