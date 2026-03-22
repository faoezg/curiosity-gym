from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from curiosity_gym import DistractiveEnv, SparseEnv, MultitaskEnv
from curiosity_gym.core.gridengine import GridEngine 

from experiments.experiment_setup.harness import ExperimentHarness
from experiments.experiment_setup.experiment_model import ExperimentModel
from experiments.experiment_setup.experiment_evaluator import ExperimentEvaluator

from experiments.byol_explore.byol_model import ByolExploreModel
from experiments.byol_explore.byol_wrapper import ByolExploreWrapper

import torch

SB3_DEVICE = "cpu"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAINING_STEPS = 50_000
EVAL_EPISODES = 5
print("Running own models on: ", DEVICE)

def _setup_byold_explore_model_for_env(env: GridEngine) -> ByolExploreModel:
    observation_space_shape = env.observation_space.shape
    return ByolExploreModel(state_dim=observation_space_shape[0], # type: ignore
                            action_dim=env.action_space.n, # type: ignore
                            hidden_dim=8,
                            latent_rep_dim=4,
                            time_horizon=5,
                            device=DEVICE,
                            alpha=0.9999,
                            reward_norm_decay=0.2)

def _setup_base_envs() -> list[GridEngine]:
    envs = []
    pov =  "local_2"
    render_mode = "rgb_array" 
    env_sparse = SparseEnv(agentPOV=pov, render_mode=render_mode, simple_actions=True, simple_obs=True)
    # env_distractive = DistractiveEnv(agentPOV=pov,render_mode=render_mode, simple_actions=True)
    #env_multitask1 = MultitaskEnv(agentPOV=pov, task=1, render_mode=render_mode)
    #env_multitask2 = MultitaskEnv(agentPOV=pov, task=2, render_mode=render_mode)

    envs.append(env_sparse)
    # envs.append(env_distractive)
    #envs.append(env_multitask1)
    #envs.append(env_multitask2)
    return envs

def _setup_wrapped_envs() -> list[tuple[ByolExploreWrapper, str]]:
    wrapped_envs = []

    base_envs = _setup_base_envs()
    for env in base_envs:
        count_model = _setup_byold_explore_model_for_env(env)
        wrapped_envs.append((ByolExploreWrapper(env, count_model, DEVICE), env.name))
    return wrapped_envs

def _setup_vec_envs() -> list[tuple[VecNormalize, str]]:
    wrapped_envs = _setup_wrapped_envs()

    vec_envs = []
    for wrapped_env, env_name in wrapped_envs:
        dummy_env = DummyVecEnv([lambda: wrapped_env])
        normalized_vec_env = VecNormalize(dummy_env, norm_obs=True, norm_reward=False)
        vec_envs.append((normalized_vec_env, env_name))
    return vec_envs

def _setup_sb3_ppo_models() -> dict[str, PPO]:
    vec_envs = _setup_vec_envs()

    model_dict = {}
    for vec_env, env_name in vec_envs:
        model_dict[env_name] = PPO("MlpPolicy", vec_env, verbose=0, batch_size=16, normalize_advantage=False, device=SB3_DEVICE)
    return model_dict

def _train_ipo_models_with_icm(models: dict[str, PPO], timesteps: int):
    for model in models.values():
        model.learn(total_timesteps=timesteps, progress_bar=True)

def run_experiment():
    base_envs = _setup_base_envs()
    model_dict = _setup_sb3_ppo_models()
    _train_ipo_models_with_icm(model_dict, TRAINING_STEPS)

    harness = ExperimentHarness(base_envs)
    experiment_models = []
    for env_name, model in model_dict.items():
        experiment_models.append(ExperimentModel(model, f"trained on {env_name}"))
    
    for experiment_model in experiment_models:
        harness.run_model_n_episodes_per_environment(experiment_model, EVAL_EPISODES)

    evaluator = ExperimentEvaluator(harness)
    evaluator.evaluate_entire_expermient()
    evaluator.print_summary()
    for env in base_envs:
        evaluator.save_environment_heatmaps(env.name)
        evaluator.save_environment_gif(env.name)

run_experiment()
