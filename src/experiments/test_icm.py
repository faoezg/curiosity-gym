from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from curiosity_gym import DistractiveEnv, SparseEnv, MultitaskEnv
from curiosity_gym.core.gridengine import GridEngine 

from experiments.experiment_setup.harness import ExperimentHarness
from experiments.experiment_setup.experiment_model import ExperimentModel
from experiments.experiment_setup.experiment_evaluator import ExperimentEvaluator

from experiments.icm.icm import ICMModel
from experiments.icm.icm_reward_wrapper import ICMCuriosityWrapper

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ICM_LR = 1e-3
TRAINING_STEPS = 30000
EVAL_EPISODES = 1

def _setup_icm_for_env(env: GridEngine) -> ICMModel:
    # TODO FIX TYPE IGNORE
    observation_space_shape = env.observation_space.shape
    flattened_observation_space = observation_space_shape[0] * observation_space_shape[1]
    icm = ICMModel(
        device = DEVICE,
        state_dim=flattened_observation_space, # type: ignore
        action_dim=env.action_space.n, # type: ignore
        latent_rep_dim=32,
        hidden_dim=64,
        beta = .2,
        eta = 0.01 
    ).to(DEVICE)
    return icm

def _setup_base_envs() -> list[GridEngine]:
    envs = []
    # TODO global bricks evaluation suit
    # ValueError: Error: Unexpected observation shape (161, 3) for Box environment, 
    # please use (165, 3) or (n_env, 165, 3) for the observation shape
    pov = "local_2"
    render_mode = "human"
    env_sparse = SparseEnv(agentPOV=pov, render_mode=render_mode)
    #env_distractive = DistractiveEnv(agentPOV=pov,render_mode=render_mode)
    #env_multitask1 = MultitaskEnv(agentPOV=pov, task=1, render_mode=render_mode)
    #env_multitask2 = MultitaskEnv(agentPOV=pov, task=2, render_mode=render_mode)

    envs.append(env_sparse)
    #envs.append(env_distractive)
    #envs.append(env_multitask1)
    #envs.append(env_multitask2)
    return envs

def _setup_wrapped_envs() -> list[tuple[ICMCuriosityWrapper, str]]:
    wrapped_envs = []

    base_envs = _setup_base_envs()
    for env in base_envs:
        icm = _setup_icm_for_env(env)
        wrapped_envs.append((ICMCuriosityWrapper(DEVICE, env, icm, ICM_LR), env.name))
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
        model_dict[env_name] = PPO("MlpPolicy", vec_env, verbose=1, clip_range=0.05, device=DEVICE)
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

run_experiment()
