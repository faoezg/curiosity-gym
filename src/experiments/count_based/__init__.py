import gymnasium as gym

gym.register(
    id="SparseEnv-UnifiedCount",
    entry_point="experiments.count_based.unified_count_factory:make_unified_count_env",
    kwargs={
        "base_env_id":"SparseEnv",
    }
)

gym.register(
    id="DistractiveEnv-UnifiedCount",
    entry_point="experiments.count_based.unified_count_factory:make_unified_count_env",
    kwargs={
        "base_env_id":"DistractiveEnv",
    }
)

gym.register(
    id="MultitaskEnv-UnifiedCount",
    entry_point="experiments.count_based.unified_count_factory:make_unified_count_env",
    kwargs={
        "base_env_id":"MultitaskEnv"
    }
)

gym.register(
    id="DetachmentEnv-UnifiedCount",
    entry_point="experiments.count_based.unified_count_factory:make_unified_count_env",
    kwargs={
        "base_env_id":"DetachmentEnv",
    }
)

gym.register(
    id="DerailmentEnv-UnifiedCount",
    entry_point="experiments.count_based.unified_count_factory:make_unified_count_env",
    kwargs={
        "base_env_id":"DerailmentEnv",
        "max_episodes": 500 
    }
)

gym.register(
    id="Montezuma-UnifiedCount",
    entry_point="experiments.count_based.unified_count_factory:make_unified_count_env",
    kwargs={
        "base_env_id":"ALE/MontezumaRevenge-v5",
        "max_episodes": 500,
        "is_atari": True
    }
)

gym.register(
    id="MountainCar-UnifiedCount",
    entry_point="experiments.count_based.unified_count_factory:make_unified_count_env",
    kwargs={
        "base_env_id":"MountainCar-v0",
        "max_episodes": 500,
        "is_atari": True
    }
)
