import gymnasium as gym

gym.register(
    id="SparseEnv-ByolExplore",
    entry_point="experiments.byol_explore.byol_gym_factory:make_byol_env",
    kwargs={
        "base_env_id":"SparseEnv",
        "max_episodes": 1000,
        "max_training_steps": 500_000
    }
)

gym.register(
    id="SimpleSparseEnv-ByolExplore",
    entry_point="experiments.byol_explore.byol_gym_factory:make_byol_env",
    kwargs={
        "base_env_id":"SimpleSparseEnv",
        "max_episodes": 1000,
        "max_training_steps": 500_000
    }
)

gym.register(
    id="DistractiveEnv-ByolExplore",
    entry_point="experiments.byol_explore.byol_gym_factory:make_byol_env",
    kwargs={
        "base_env_id":"DistractiveEnv",
        "max_episodes": 10000,
        "max_training_steps": 500_000
    }
)

gym.register(
    id="MultitaskEnv-ByolExplore",
    entry_point="experiments.byol_explore.byol_gym_factory:make_byol_env",
    kwargs={
        "base_env_id":"MultitaskEnv",
        "max_episodes": 1000,
        "max_training_steps": 500_000
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

gym.register(
    id="DetachmentEnv-ByolExplore",
    entry_point="experiments.byol_explore.byol_gym_factory:make_byol_env",
    kwargs={
        "base_env_id":"DetachmentEnv",
        "max_episodes": 1000,
        "max_training_steps": 500_000
    }
)

gym.register(
    id="DerailmentEnv-ByolExplore",
    entry_point="experiments.byol_explore.byol_gym_factory:make_byol_env",
    kwargs={
        "base_env_id":"DerailmentEnv",
        "max_episodes": 1000,
        "max_training_steps": 500_000
    }
)
