from experiments.clean_rl.ppo import Args, run_clean_rl_ppo_model
from curiosity_gym import DistractiveEnv, SparseEnv, MultitaskEnv, DetachmentEnv, DerailmentEnv
from curiosity_gym.core.gridengine import GridEngine 

from experiments.icm import ICMModel

import torch
import gymnasium as gym

SB3_DEVICE = "cpu"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAINING_EPISODES = 10
print("Running own models on: ", DEVICE)


def train_clean_rl_model() -> None:
    env_name = "SimpleSparseEnv-Icm"
    #env_name = "MountainCar-Icm"
    #env_name = "CartPole-Icm"
    env = gym.make(env_name,
                   use_simple_actions=False,
                   use_globaly_unique_id=True,
                   max_episodes=1000,
                   max_training_steps=500_000,
                   base_env_pov="local_2",
                   render_mode="rgb_array",
                   device=DEVICE)

    args = Args(
        exp_name=env_name,
        env_id=env_name,
        env=env,
        num_steps=20,
        capture_video=True,
        learning_rate=0.001,
        total_timesteps=500_000,
        num_envs=1,
    )
    run_clean_rl_ppo_model(args)
    env.close()

train_clean_rl_model()

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def tmp():
    vec_env =  make_vec_env("SimpleSparseEnv-Icm", n_envs=1)
    model = PPO("MlpPolicy", vec_env, verbose=0, device="cpu", learning_rate=1e-3)
    #model = PPO("CnnPolicy", vec_env, verbose=0, device="cpu", learning_rate=1e-3)
    model.learn(total_timesteps=500_000)
    vec_env.close()

#tmp()
