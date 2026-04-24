from experiments.clean_rl.ppo import Args, run_clean_rl_ppo_model
from curiosity_gym import DistractiveEnv, SparseEnv, MultitaskEnv
from curiosity_gym.core.gridengine import GridEngine 

from experiments.byol_explore.byol_model import ByolExploreModel
from experiments.byol_explore.byol_wrapper import ByolExploreWrapper

import torch
import gymnasium as gym

SB3_DEVICE = "cpu"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAINING_EPISODES = 100
print("Running own models on: ", DEVICE)

def train_clean_rl_model() -> None:
    env_name = "MountainCar-ByolExplore"
    env = gym.make(env_name,
                   max_episodes=1,
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

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def tmp():
    vec_env =  make_vec_env("SparseEnv-ByolExplore", n_envs=1)
    model = PPO("MlpPolicy", vec_env, verbose=1, device="cpu")
    model.learn(total_timesteps=500_000)

    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, _, _, _ = vec_env.step(action)
        vec_env.render("human")

tmp()
