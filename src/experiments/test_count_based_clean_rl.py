from experiments.clean_rl.ppo import Args, run_clean_rl_ppo_model
from curiosity_gym import DistractiveEnv, SparseEnv, MultitaskEnv, DetachmentEnv, DerailmentEnv
from curiosity_gym.core.gridengine import GridEngine 

from experiments.count_based.unified_count import UnifiedCountModel
from experiments.count_based.unified_count_reward_wrapper import UnifiedCountWrapper

import torch
import gymnasium as gym

SB3_DEVICE = "cpu"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAINING_EPISODES = 100 
print("Running own models on: ", DEVICE)

def train_clean_rl_model() -> None:
    env_name = "SparseEnv-UnifiedCount"
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
        batch_size=12
    )
    run_clean_rl_ppo_model(args)

train_clean_rl_model()
