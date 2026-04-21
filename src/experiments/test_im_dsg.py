import torch
import gymnasium as gym

from experiments.im_dsg import SkillGraphAgent, ValueModel, QModel, ExplorationModel, R2D2Model
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
video_env = env = gym.wrappers.RecordVideo(env, f"videos/im_dsg/1", episode_trigger=lambda x: x % 10 == 0)
state_dim = env.observation_space.shape[0] # type: ignore
value_model = ValueModel(state_dim=state_dim, device=DEVICE)
q_model = QModel(state_dim=state_dim, hidden_dim=254, device=DEVICE)
r2d2_model = R2D2Model(state_dim=state_dim, action_dim=env.action_space.n, device=DEVICE, burn_in_transition_count=2, buffer_size=5000)
exploration_model = ExplorationModel(state_dim=state_dim)

coin_flip_model = CoinFlipModel(state_dim=state_dim,
                        hidden_dim=100,
                        d_dim=20,
                        priority_alpha=0.5,
                        reward_scale=0.3,
                        device=DEVICE)

graph_agent = SkillGraphAgent(
    env=video_env,
    intrinsic_motivation_model=coin_flip_model,
    value_model=value_model,
    option_model=r2d2_model,
    option_model_batch_size=80,
    exploration_model=exploration_model,
    goal_value_threshold=0.5,
    intrinsic_std_deviaton_scalar=0.2,
    option_horizon=5,
    hindsight_k=10
)

graph_agent.run_agent(400, TRAINING_STEPS)
