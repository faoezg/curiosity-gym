from .icm_network import ICMNetwork as ICMNetwork 
from .icm_model import ICMModel as ICMModel 
from .icm_wrapper import ICMCuriosityWrapper as ICMCuriosityWrapper

import gymnasium as gym

gym.register(
    id="SparseEnv-Icm",
    entry_point="experiments.icm.icm_factory:make_icm_env",
    kwargs={
        "base_env_id":"SparseEnv",
    }
)

gym.register(
    id="DistractiveEnv-Icm",
    entry_point="experiments.icm.icm_factory:make_icm_env",
    kwargs={
        "base_env_id":"DistractiveEnv",
    }
)

gym.register(
    id="MultitaskEnv-Icm",
    entry_point="experiments.icm.icm_factory:make_icm_env",
    kwargs={
        "base_env_id":"MultitaskEnv"
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
        "base_env_id":"DerailmentEnv"
    }
)
