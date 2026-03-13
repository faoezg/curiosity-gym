from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from curiosity_gym import DistractiveEnv, SparseEnv, MultitaskEnv
from curiosity_gym.core.gridengine import GridEngine 

from experiments.experiment_setup.harness import ExperimentHarness
from experiments.experiment_setup.experiment_model import ExperimentModel
from experiments.experiment_setup.experiment_evaluator import ExperimentEvaluator

from experiments.count_based.unified_count import UnifiedCountModel
from experiments.count_based.unified_count_reward_wrapper import UnifiedCountWrapper

import torch

SB3_DEVICE = "cpu"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAINING_STEPS = 100_000
EVAL_EPISODES = 3

def _setup_count_model_for_env(env: GridEngine) -> UnifiedCountModel:
    observation_space_shape = env.observation_space.shape
    total_observation_space = observation_space_shape[0] ** observation_space_shape[1] # type: ignore
    return UnifiedCountModel(total_observation_space)

def _setup_base_envs() -> list[GridEngine]:
    envs = []
    # TODO global bricks evaluation suit
    # ValueError: Error: Unexpected observation shape (161, 3) for Box environment, 
    # please use (165, 3) or (n_env, 165, 3) for the observation shape
    pov = "local_2"
    render_mode = "rgb_array" 
    #env_sparse = SparseEnv(agentPOV=pov, render_mode=render_mode)
    #env_distractive = DistractiveEnv(agentPOV=pov,render_mode=render_mode, simple_actions=True)
    #env_multitask1 = MultitaskEnv(agentPOV=pov, task=1, render_mode=render_mode)
    env_multitask2 = MultitaskEnv(agentPOV=pov, task=2, render_mode=render_mode, simple_actions=True)

    # envs.append(env_sparse)
    #envs.append(env_distractive)
    #envs.append(env_multitask1)
    envs.append(env_multitask2)
    return envs

def _setup_wrapped_envs() -> list[tuple[UnifiedCountWrapper, str]]:
    wrapped_envs = []

    base_envs = _setup_base_envs()
    for env in base_envs:
        count_model = _setup_count_model_for_env(env)
        wrapped_envs.append((UnifiedCountWrapper(env, count_model, clip_range=0, eps=0.5, beta=5), env.name))
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
        model_dict[env_name] = PPO("MlpPolicy", vec_env,
                                   verbose=0,
                                   learning_rate=1e-3,
                                   batch_size=16,
                                   normalize_advantage=False,
                                   device=SB3_DEVICE)
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
