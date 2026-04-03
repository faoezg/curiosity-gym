import torch
import gymnasium as gym

from experiments.im_dsg import SkillGraphAgent, ValueModel, QModel, ExplorationModel
from experiments.coin_flip import CoinFlipModel
from experiments.icm import ICMModel
from curiosity_gym import SparseEnv, MultitaskEnv, DistractiveEnv
from curiosity_gym.core.gridengine import GridEngine

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAINING_STEPS = 500 
EVAL_EPISODES = 3

pov = "local_2"
render_mode = "rgb_array" 
env = SparseEnv(agentPOV=pov,render_mode=render_mode)
vide_env = env = gym.wrappers.RecordVideo(env, f"videos/im_dsg")
state_dim = env.observation_space.shape[0] # type: ignore
value_model = ValueModel(state_dim=state_dim)
q_model = QModel(state_dim=state_dim)
exploration_model = ExplorationModel(state_dim=state_dim)
coin_flip_model = CoinFlipModel(state_dim=state_dim,
                        hidden_dim=100,
                        d_dim=20,
                        priority_alpha=0.5,
                        reward_scale=0.3,
                        device=DEVICE)

# TODO check if coin_flip_model is correctly calculating the reward
graph_agent = SkillGraphAgent(
    env=env,
    intrinsic_motivation_model=coin_flip_model,
    value_model=value_model,
    q_model=q_model,
    exploration_model=exploration_model
)

graph_agent.run_agent(1500, TRAINING_STEPS)
