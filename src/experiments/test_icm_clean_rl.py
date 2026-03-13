from experiments.clean_rl.ppo import Args, run_clean_rl_ppo_model
from curiosity_gym import DistractiveEnv, SparseEnv, MultitaskEnv
from curiosity_gym.core.gridengine import GridEngine 

from experiments.experiment_setup.harness import ExperimentHarness
from experiments.experiment_setup.experiment_model import ExperimentModel
from experiments.experiment_setup.experiment_evaluator import ExperimentEvaluator

from experiments.byol_explore.byol_explore import ByolExploreModel
from experiments.byol_explore.byol_wrapper import ByolExploreWrapper

import torch
import gymnasium as gym

SB3_DEVICE = "cpu"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAINING_STEPS = 50_000
EVAL_EPISODES = 5
print("Running own models on: ", DEVICE)

def _setup_byold_explore_model_for_env(env: GridEngine) -> ByolExploreModel:
    observation_space_shape = env.observation_space.shape
    flattened_observation_space = observation_space_shape[0] * observation_space_shape[1] # type: ignore
    return ByolExploreModel(state_dim=flattened_observation_space, # type: ignore
                            action_dim=env.action_space.n, # type: ignore
                            hidden_dim=8,
                            latent_rep_dim=4,
                            time_horizon=5,
                            device=DEVICE,
                            alpha=0.9999).to(DEVICE)

def _setup_base_envs() -> list[GridEngine]:
    envs = []
    pov =  "local_2"
    render_mode = "rgb_array" 
    env_sparse = SparseEnv(agentPOV=pov, render_mode=render_mode, simple_actions=True)
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
        wrapped_envs.append((ByolExploreWrapper(env, count_model, DEVICE, reward_norm_decay=0.2), env.name))
    return wrapped_envs


def _setup_clean_rl_ppo_models() -> dict[str, Args]:
    envs = _setup_base_envs()

    model_dict = {}
    for env in envs:
        args = Args(
            env.name,
            env_id="SparseEnv-ByolExplore",
            num_envs=1,
            num_steps=TRAINING_STEPS,
            capture_video=True
        )
        model_dict[env.name] = args
    return model_dict

def _train_ppo_models_with_byol(models: dict[str, Args]):
    for model in models.values():
        run_clean_rl_ppo_model(model)


def run_experiment():
    base_envs = _setup_base_envs()
    model_dict = _setup_clean_rl_ppo_models()
    _train_ppo_models_with_byol(model_dict)

   # harness = ExperimentHarness(base_envs)
   # experiment_models = []
   # for env_name, model in model_dict.items():
   #     experiment_models.append(ExperimentModel(model, f"trained on {env_name}"))
   # 
   # for experiment_model in experiment_models:
   #     harness.run_model_n_episodes_per_environment(experiment_model, EVAL_EPISODES)

   # evaluator = ExperimentEvaluator(harness)
   # evaluator.evaluate_entire_expermient()
   # evaluator.print_summary()
   # for env in base_envs:
   #     evaluator.save_environment_heatmaps(env.name)
   #     evaluator.save_environment_gif(env.name)

run_experiment()
