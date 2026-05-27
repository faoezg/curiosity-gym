from .buffer import CFNTransition as CFNTransition
from .coin_flip_network import CoinFlipNetwork as CoinFlipNetwork
from .rademacher_generator import RademacherDistGenerator as RademacherDistGenerator
from .coin_flip_model import CoinFlipModel as CoinFlipModel

import gymnasium as gym

gym.register(
    id="SparseEnv-CoinFlip",
    entry_point="experiments.coin_flip.coin_flip_factory:make_coin_flip_env",
    kwargs={
        "base_env_id":"SparseEnv",
        "max_episodes": 1000,
        "max_training_steps": 500_000
    }
)

gym.register(
    id="SimpleSparseEnv-CoinFlip",
    entry_point="experiments.coin_flip.coin_flip_factory:make_coin_flip_env",
    kwargs={
        "base_env_id":"SimpleSparseEnv",
        "max_episodes": 1,
        "max_training_steps": 500_000
    }
)

gym.register(
    id="DistractiveEnv-CoinFlip",
    entry_point="experiments.coin_flip.coin_flip_factory:make_coin_flip_env",
    kwargs={
        "base_env_id":"DistractiveEnv",
        "max_episodes": 10000,
        "max_training_steps": 500_000
    }
)

gym.register(
    id="MultitaskEnv-CoinFlip",
    entry_point="experiments.coin_flip.coin_flip_factory:make_coin_flip_env",
    kwargs={
        "base_env_id":"MultitaskEnv",
        "max_episodes": 1000,
        "max_training_steps": 500_000
    }
)

gym.register(
    id="DetachmentEnv-CoinFlip",
    entry_point="experiments.coin_flip.coin_flip_factory:make_coin_flip_env",
    kwargs={
        "base_env_id":"DetachmentEnv",
    }
)

gym.register(
    id="DerailmentEnv-CoinFlip",
    entry_point="experiments.coin_flip.coin_flip_factory:make_coin_flip_env",
    kwargs={
        "base_env_id":"DerailmentEnv",
        "max_episodes": 500 
    }
)

gym.register(
    id="Montezuma-CoinFlip",
    entry_point="experiments.coin_flip.coin_flip_factory:make_coin_flip_env",
    kwargs={
        "base_env_id":"ALE/MontezumaRevenge-v5",
        "max_episodes": 500,
        "is_atari": True
    }
)

gym.register(
    id="MountainCar-CoinFlip",
    entry_point="experiments.coin_flip.coin_flip_factory:make_coin_flip_env",
    kwargs={
        "base_env_id":"MountainCar-v0",
        "max_episodes": 500,
        "is_atari": True
    }
)
