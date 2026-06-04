import time
from experiments.clean_rl import Args, run_clean_rl_ppo_model, AtariArgs, run_clean_rl_ppo_atari_model
from curiosity_gym import DistractiveEnv, SparseEnv, SimpleSparseEnv, MultitaskEnv, DetachmentEnv, DerailmentEnv
from curiosity_gym.core.gridengine import GridEngine 

from experiments.count_based.unified_count import UnifiedCountModel
from experiments.count_based.unified_count_reward_wrapper import UnifiedCountWrapper
from experiments.coin_flip import CoinFlipModel
from experiments.icm import ICMModel
from experiments.byol_explore.byol_model import ByolExploreModel


import torch
import gymnasium as gym

SB3_DEVICE = "cpu"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running own models on: ", DEVICE)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

ENV_PREFIXES = ["DistractiveEnv", "MultitaskEnv", "SparseEnv", "SimpleSparseEnv", "DetachmentEnv", "DerailmentEnv"]
#MODELS_SUFFIXES = ["Experiment", "UnifiedCount", "CoinFlip", "Icm", "ByolExplore"]
MODELS_SUFFIXES = ["Icm", "ByolExplore"]


def tmp():
    for prefix in ENV_PREFIXES:
        for suffix in MODELS_SUFFIXES:
            vec_env =  make_vec_env(f"{prefix}-{suffix}", n_envs=1)
            model = PPO("MlpPolicy", vec_env, verbose=0, device="cpu")
            model.learn(total_timesteps=500_000)
            time.sleep(30) # wait till all I/O Operations finish
            vec_env.close()

tmp()
