from .buffers import ReplayBuffer as ReplayBuffer
from .buffers import PriorityReplayBuffer as PriorityReplayBuffer
from .buffers import Transition as Transition
from .buffers import PriorizedTransition as PriorizedTransition
from .intrinsic_motiavtion_model import IntrinsicMotivationModel as IntrinsicMotivationModel
from .intrinsic_motiavtion_model_wrapper import IntrinsicMotivationModelWrapper as IntrinsicMotivationModelWrapper

import gymnasium as gym

gym.register(
    id="SparseEnv-Experiment",
    entry_point="experiments.components.experiment_factory:make_experiment_env",
    kwargs={
        "base_env_id":"SparseEnv",
    }
)

gym.register(
    id="DistractiveEnv-Experiment",
    entry_point="experiments.components.experiment_factory:make_experiment_env",
    kwargs={
        "base_env_id":"DistractiveEnv",
    }
)

gym.register(
    id="MultitaskEnv-Experiment",
    entry_point="experiments.components.experiment_factory:make_experiment_env",
    kwargs={
        "base_env_id":"MultitaskEnv"
    }
)

gym.register(
    id="DetachmentEnv-Experiment",
    entry_point="experiments.components.experiment_factory:make_experiment_env",
    kwargs={
        "base_env_id":"DetachmentEnv",
        "max_episodes":1
    }
)

gym.register(
    id="DerailmentEnv-Experiment",
    entry_point="experiments.components.experiment_factory:make_experiment_env",
    kwargs={
        "base_env_id":"DerailmentEnv",
        "max_episodes": 500 
    }
)
