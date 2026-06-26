from experiments.clean_rl import Args, run_clean_rl_ppo_model, AtariArgs, run_clean_rl_ppo_atari_model
from curiosity_gym import DistractiveEnv, SparseEnv, MultitaskEnv
from curiosity_gym.core.gridengine import GridEngine 

from experiments.coin_flip import CoinFlipModel

import gymnasium as gym
import torch
import ale_py

SB3_DEVICE = "cpu"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAINING_EPISODES = 100 
print("Running own models on: ", DEVICE)

gym.register_envs(ale_py)

def train_clean_rl_model() -> None:
    env_name = "MountainCar-CoinFlip"
    env = gym.make(env_name,
                   max_episodes=5000,
                   max_training_steps=1,
                   base_env_pov="local_2", # "forward_2_3",
                   render_mode="rgb_array",
                   device=DEVICE)

    args = Args(
        env_name,
        env_id=env_name,
        env=env,
        num_envs=1,
        num_steps=TRAINING_EPISODES,
        capture_video=True,
    )
    run_clean_rl_ppo_model(args)

#train_clean_rl_model()

def train_clean_rl_atari_model() -> None:
    env_name = "Montezuma-CoinFlip"
    env = gym.make(env_name,
                   max_episodes=1,
                   max_training_steps=1,
                   base_env_pov="local_2", # "forward_2_3",
                   render_mode="rgb_array",
                   device=DEVICE)

    args = AtariArgs(
        env_name,
        env_id=env_name,
        env=env,
        num_envs=1,
        num_steps=TRAINING_EPISODES,
        capture_video=True,
        batch_size=12
    )
    run_clean_rl_ppo_atari_model(args)

#train_clean_rl_atari_model()

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def tmp():
    vec_env =  make_vec_env("SimpleSparseEnv-CoinFlip", n_envs=1)
    model = PPO("MlpPolicy", vec_env, verbose=0, device="cpu", seed=42)
    model.learn(total_timesteps=500_000)
    vec_env.close()

tmp()
