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
    }
)

gym.register(
    id="DistractiveEnv-CoinFlip",
    entry_point="experiments.coin_flip.coin_flip_factory:make_coin_flip_env",
    kwargs={
        "base_env_id":"DistractiveEnv",
        "time_horizon":10
    }
)

gym.register(
    id="MultitaskEnv-CoinFlip",
    entry_point="experiments.coin_flip.coin_flip_factory:make_coin_flip_env",
    kwargs={
        "base_env_id":"MultitaskEnv"
    }
)
