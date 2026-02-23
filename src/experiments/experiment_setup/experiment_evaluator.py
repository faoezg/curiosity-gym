from abc import ABC
import imageio
import numpy as np

from experiments.experiment_setup.harness import ExperimentHarness
from experiments.experiment_setup.simulation_report import SimulationReport

class ExperimentEvaluator(ABC):
    """
    ExperimentEvaluator evaluates a given experiment established through an experimetn harness

    In doing so, it will
    """
    def __init__(self, experiment_harness: ExperimentHarness) -> None:
        super().__init__()
        self._experiment_harness: ExperimentHarness = experiment_harness
        self._best_runs_per_environment: dict[str, SimulationReport | None] = {}
        self._worst_runs_per_environment: dict[str, SimulationReport | None] = {}
  
    def evaluate_entire_expermient(self) -> None:
        simulation_reports = self._experiment_harness.simulation_reports
        for environment_name, model_reports in simulation_reports.items():
            self._init_best_and_worst_runs_per_environment(environment_name)
            self._evaluate_environment(environment_name, model_reports)

    def _init_best_and_worst_runs_per_environment(self, environment_name: str) -> None:
        self._best_runs_per_environment[environment_name] = None
        self._worst_runs_per_environment[environment_name] = None

    def _evaluate_environment(self, environment_name: str, model_reports: dict[str, list[SimulationReport]]) -> None:
        for model, simulation_reports in model_reports.items():
            for simulation_report in simulation_reports:
                self._update_best_run(environment_name, simulation_report)
                self._update_worst_run(environment_name, simulation_report)

    def _update_best_run(self, environment_name:str, simulation_report: SimulationReport) -> None:
        best_run = self._best_runs_per_environment[environment_name]
        if (best_run is None or best_run.total_reward <= simulation_report.total_reward):
            self._best_runs_per_environment[environment_name] = simulation_report

    def _update_worst_run(self, environment_name:str, simulation_report: SimulationReport) -> None:
        worst_run = self._worst_runs_per_environment[environment_name]
        if (worst_run is None or worst_run.total_reward >= simulation_report.total_reward):
            self._worst_runs_per_environment[environment_name] = simulation_report

    # TODO pull into own class? 
    def save_environment_heatmaps(self, environment_name: str) -> None:
        simulation_reports = self._experiment_harness.simulation_reports
        for model_name, reports in simulation_reports[environment_name].items():
            for idx, simulation_report in enumerate(reports):
                file_path = f"{environment_name}_{model_name}_Episode{idx}.png"
                figure = simulation_report.environment.heatmap()
                if (figure is not None):
                    figure.savefig(file_path)
                    figure.clear()
                else:
                    # TODO make proper error?
                    print("Exporting Heatmaps failed, as no figure was able to be created")

    def save_environment_gif(self, environment_name: str) -> None:
        simulation_reports = self._experiment_harness.simulation_reports
        for model_name, reports in simulation_reports[environment_name].items():
            for idx, simulation_report in enumerate(reports):
                images = simulation_report.images
                if (len(images) == 0):
                    print("Exporting Gifs failed, as no images were found")
                gif_name = f"{environment_name}_{model_name}_Episode{idx}.gif"
                imageio.mimsave(gif_name, [np.array(img) for i, img in enumerate(simulation_report.images) if i%2 == 0], fps=7)

    def best_model_in_environment(self, environment_name: str) -> str:
        """Getter for the best model of a given environment"""
        best_run = self._best_runs_per_environment[environment_name]
        if (best_run is not None):
            return best_run.model_name
        else:
            return f"Environment {environment_name} has no evaluation data"

    def worst_model_in_environment(self, environment_name: str) -> str:
        """Getter for the worst model of a given environment"""
        worst_run = self._worst_runs_per_environment[environment_name]
        if (worst_run is not None):
            return worst_run.model_name
        else:
            return f"Environment {environment_name} has no evaluation data"

    def print_summary(self):
        # TODO these loops keep reoccuring perhaps make the manage class, or write methode to get iterators
        simulation_reports = self._experiment_harness.simulation_reports
        for environment_name, model_reports in simulation_reports.items():
            print(f"The best model in {environment_name} is {self.best_model_in_environment(environment_name)}")
            print(f"The worst model in {environment_name} is {self.worst_model_in_environment(environment_name)}")
            for reports in model_reports.values():
                for report in reports:
                    print(report)

    @property
    def best_runs_per_environment(self) -> dict[str, SimulationReport | None]:
        """Getter for the dictonary holding the best runs per environment"""
        return self._best_runs_per_environment

    @property
    def worst_runs_per_environment(self) -> dict[str, SimulationReport | None]:
        """Getter for the dictonary holding the worst runs per environment"""
        return self._worst_runs_per_environment
