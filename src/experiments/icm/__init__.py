from .icm_network import ICMNetwork as ICMNetwork 
from .icm_model import ICMModel as ICMModel 
from .icm_wrapper import ICMCuriosityWrapper as ICMCuriosityWrapper

import gymnasium as gym

gym.register(
    id="SparseEnv-Icm",
    entry_point="experiments.icm.icm_factory:make_icm_env",
    kwargs={
        "base_env_id":"SparseEnv",
        "max_episodes": 1000,
        "max_training_steps": 500_000
    }
)

gym.register(
    id="SimpleSparseEnv-Icm",
    entry_point="experiments.icm.icm_factory:make_icm_env",
    kwargs={
        "base_env_id":"SimpleSparseEnv",
        "max_episodes": 1,
        "max_training_steps": 500_000
    }
)



gym.register(
    id="DistractiveEnv-Icm",
    entry_point="experiments.icm.icm_factory:make_icm_env",
    kwargs={
        "base_env_id":"DistractiveEnv",
        "max_episodes": 10000,
        "max_training_steps": 500_000
    }
)

gym.register(
    id="MultitaskEnv-Icm",
    entry_point="experiments.icm.icm_factory:make_icm_env",
    kwargs={
        "base_env_id":"MultitaskEnv",
        "max_episodes": 1000,
        "max_training_steps": 500_000
    }
)

gym.register(
    id="DetachmentEnv-Icm",
    entry_point="experiments.icm.icm_factory:make_icm_env",
    kwargs={
        "base_env_id":"DetachmentEnv"
    }
)

gym.register(
    id="DerailmentEnv-Icm",
    entry_point="experiments.icm.icm_factory:make_icm_env",
    kwargs={
        "base_env_id":"DerailmentEnv",
        "max_episodes": 1000,
        "max_training_steps": 500_000
    }
)

gym.register(
    id="MountainCar-Icm",
    entry_point="experiments.icm.icm_factory:make_icm_env",
    kwargs={
        "base_env_id":"MountainCar-v0",
        "max_episodes": 500,
        "is_atari": True
    }
)

gym.register(
    id="CartPole-Icm",
    entry_point="experiments.icm.icm_factory:make_icm_env",
    kwargs={
        "base_env_id":"CartPole-v1",
        "max_episodes": 500,
        "is_atari": True
    }
)
