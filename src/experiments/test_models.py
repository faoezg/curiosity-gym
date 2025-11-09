from abc import ABC
from modles.model_types import Models

from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from curiosity_gym import DistractiveEnv, SparseEnv, MultitaskEnv
from curiosity_gym.core.gridengine import GridEngine 

# Define environments
pov = "local_2"
env_sparse = SparseEnv(agentPOV=pov)
env_distractive = DistractiveEnv(agentPOV=pov)
env_multitask1 = MultitaskEnv(agentPOV=pov, task=1)
env_multitask2 = MultitaskEnv(agentPOV=pov, task=2)

model_sparse = PPO("MlpPolicy", env_sparse, verbose=1)
model_distractive = PPO("MlpPolicy", env_distractive, verbose=1)
model_multitask1 = PPO("MlpPolicy", env_multitask1, verbose=1)
model_multitask2 = PPO("MlpPolicy", env_multitask1, verbose=1)

def train_model(model: BaseAlgorithm, timesteps: int):
    model.learn(total_timesteps=timesteps)

# Define test function
def test(model, env, n):
    score = 0
    
    for _ in range(n):
        obs, info = env.reset()
        terminated, truncated = False, False
        returns = 0
        while not (terminated or truncated):
            action = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action[0])
            returns += reward
        score += returns
    
    return score / n


from experiments.experiment_setup.harness import ExperimentHarness
from experiments.experiment_setup.experiment_model import ExperimentModel
from experiments.experiment_setup.experiment_evaluator import ExperimentEvaluator

train_model(model_sparse, 200)
train_model(model_distractive, 200)
train_model(model_multitask1, 200)
train_model(model_multitask2, 200)

n = 1
harness = ExperimentHarness([env_sparse, env_distractive, env_multitask1, env_multitask2])
experiment_sparse = ExperimentModel(model_sparse, "Sparse Model")
experiment_dist = ExperimentModel(model_distractive, "Distractvie Model")
experiment_multi1 = ExperimentModel(model_multitask1, "Multi1 Model")
experiment_multi2 = ExperimentModel(model_multitask2, "Multi2 Model")
harness.run_model_n_episodes_per_environment(experiment_sparse, n)
harness.run_model_n_episodes_per_environment(experiment_dist, n)
harness.run_model_n_episodes_per_environment(experiment_multi1, n)
harness.run_model_n_episodes_per_environment(experiment_multi2, n)

evaluator = ExperimentEvaluator(harness)
evaluator.evaluate_entire_expermient()
evaluator.print_summary()
evaluator.save_environment_heatmaps(env_multitask2.name)


# print("PPO - Sparse: ", test(model_sparse, env_sparse, n))
# print("PPO - Distractive: ", test(model_distractive, env_distractive, n))
# print("PPO - Multitask 1: ", test(model_multitask1, env_multitask1, n))
# print("PPO - Multitask 2: ", test(model_multitask2, env_multitask2, n))
