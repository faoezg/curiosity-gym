from abc import ABC
import copy
from tqdm import tqdm

from curiosity_gym.core.gridengine import GridEngine

# TODO change to own implementaiton super later on?
from stable_baselines3.common.base_class import BaseAlgorithm

from experiments.experiment_setup.experiment_model import ExperimentModel
from experiments.experiment_setup.simulation_report import SimulationReport


class ExperimentHarness(ABC):

    """
    Core class of the experiments feature. Instances of this class will cary enviroments and povs.
    The models, if training is actviated, will be trained.

    Any supplied model will be evaluated for all environments present.

    :param environments: A list of all enviroments to be tested, must be type :class:`~curiosity_gym.core.gridengine.GridEngine`
    """
    def __init__(self, environments: list[GridEngine]) -> None:
        super().__init__()
        self._environments: list[GridEngine] = environments
        self._simulation_reports: dict[str, dict[str, list[SimulationReport]]] = {} # this struct arises to allow both evaluation based on env or model
        self._init_simulation_reports()

    def run_model_n_episodes_per_environment(self, experiment_model: ExperimentModel, episode_count: int) -> None:
        elapsed_episodes = 0
        with tqdm(total=episode_count) as episode_progressbar:
            while (elapsed_episodes < episode_count):
                for env in (environment_progressbar := tqdm(self._environments)):
                    environment_progressbar.set_description(f"Processing environment {env.name}")
                    self._simulate_next_step(experiment_model.model, env)
                    self._store_new_simulation_report(env, experiment_model.model_name)
                episode_progressbar.set_description(f"Processing Episode {elapsed_episodes}")
                elapsed_episodes += 1
                episode_progressbar.update(1)

    def run_model_n_time_steps_per_environment(self, experiment_model: ExperimentModel, evaluation_timesteps: int) -> None:
        for env in (environment_progressbar := tqdm(self._environments)):
            environment_progressbar.set_description(f"Processing environment {env.name}")
            total_elapsed_time_steps_in_environment = self._get_total_elapsed_time_steps_in_environment(env.name, experiment_model.model_name)
            with tqdm(total=evaluation_timesteps - total_elapsed_time_steps_in_environment) as timestep_progessbar:
                while (total_elapsed_time_steps_in_environment < evaluation_timesteps):
                    timestep_progessbar.set_description(f"Processing Timestep {self._current_environment_steps_taken}")
                    self._simulate_next_step(experiment_model.model, env)
                    timestep_progessbar.update(1)
                    if (total_elapsed_time_steps_in_environment + self._current_environment_steps_taken < evaluation_timesteps):
                        self._store_new_simulation_report(env, experiment_model.model_name)

    def _simulate_next_step(self, model: BaseAlgorithm, env: GridEngine) -> None:
        self._init_env(env)
        while not (self._current_environment_goal_state_reached or self._current_environment_truncated):
            action = model.predict(observation=self._current_environment_observation)
            # TODO allow ExperimentModel to implement the model.predict() methode which returns an Action of the CuriosityGym
            self._current_environment_observation, reward, self._current_environment_goal_state_reached, self._current_environment_truncated, _ = env.step(action[0])
            self._current_environment_steps_taken += 1
            self._total_current_environment_reward += reward
    
    def _store_new_simulation_report(self, environment: GridEngine, model_name: str) -> None:
        deep_env_copy = copy.deepcopy(environment)
        simulation_report = SimulationReport(model_name=model_name, environment=deep_env_copy)
        simulation_report.total_reward = self._total_current_environment_reward
        env_model_dict = self._simulation_reports[environment.name]
        if(model_name not in env_model_dict.keys()):
            env_model_dict[model_name] = [simulation_report]
        else:
            env_model_dict[model_name] += [simulation_report]
    
    def _init_simulation_reports(self) -> None:
        for environment in self._environments:
            self._simulation_reports[environment.name] = {}

    def _init_env(self, env: GridEngine) -> None:
        self._current_environment_observation, _ = env.reset()
        self._current_environment_goal_state_reached = False
        self._current_environment_truncated = False
        self._total_current_environment_reward = 0
        self._current_environment_steps_taken = 0
    
    # TODO perhaps add manager for simulation reports later?? Potential use in harness and evaluator
    def _get_total_elapsed_time_steps_in_environment(self, environment_name: str, model_name: str) -> int:
        model_reports = self._simulation_reports[environment_name]
        simulation_reports = model_reports[model_name]
        total_elapsed_time_steps = 0
        for simulation_report in simulation_reports:
            total_elapsed_time_steps += simulation_report.environment_steps_taken
        return total_elapsed_time_steps

    @property
    def environments(self) -> list[GridEngine]:
        """Getter for the environments list."""
        return self._environments

    @property
    def simulation_reports(self) -> dict[str, dict[str, list[SimulationReport]]]:
        """Getter for the simulation reports dict."""
        return self._simulation_reports
