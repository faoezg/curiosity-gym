from experiments.clean_rl.ppo import Args, run_clean_rl_ppo_model
from curiosity_gym import DistractiveEnv, SparseEnv, MultitaskEnv
from curiosity_gym.core.gridengine import GridEngine 

from experiments.experiment_setup.harness import ExperimentHarness
from experiments.experiment_setup.experiment_model import ExperimentModel
from experiments.experiment_setup.experiment_evaluator import ExperimentEvaluator

from experiments.byol_explore.byol_model import ByolExploreModel
from experiments.byol_explore.byol_wrapper import ByolExploreWrapper

import torch
import gymnasium as gym

SB3_DEVICE = "cpu"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAINING_STEPS = 50_000
EVAL_EPISODES = 5
print("Running own models on: ", DEVICE)

def train_clean_rl_model() -> None:
    env_name = "DistractiveEnv"
    args = Args(
        env_name,
        env_id=env_name+"-ByolExplore",
        num_envs=1,
        num_steps=TRAINING_STEPS,
        capture_video=True
    )
    run_clean_rl_ppo_model(args)

train_clean_rl_model()
