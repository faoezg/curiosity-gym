from experiments.clean_rl.ppo import Args, run_clean_rl_ppo_model
from curiosity_gym import DistractiveEnv, SparseEnv, MultitaskEnv, DetachmentEnv, DerailmentEnv
from curiosity_gym.core.gridengine import GridEngine 

import torch
import gymnasium as gym

SB3_DEVICE = "cpu"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAINING_STEPS = 500_000
EVAL_EPISODES = 5
print("Running own models on: ", DEVICE)

#import gymnasium as gym
#env = gym.make("DerailmentEnv", render_mode="rgb_array")
#env.print_inital_env_state_as_pdf()



def train_clean_rl_model() -> None:
    env_name = "SparseEnv-Experiment"
    env = gym.make(env_name,
                   max_episodes=10000,
                   max_training_steps=1,
                   base_env_pov="local_2", # "forward_2_3",
                   render_mode="rgb_array")


    args = Args(
        env_name,
        env_id=env_name,
        env=env,
        num_envs=1,
        num_steps=TRAINING_STEPS,
        capture_video=False
    )
    run_clean_rl_ppo_model(args)

#train_clean_rl_model()

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def tmp():
    vec_env =  make_vec_env("SimpleSparseEnv-Experiment", n_envs=1)
    model = PPO("MlpPolicy", vec_env, verbose=0, device="cpu")
    model.learn(total_timesteps=500_000)

tmp()
