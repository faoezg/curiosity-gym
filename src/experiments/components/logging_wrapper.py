import gymnasium as gym
from typing import override

from curiosity_gym.core.gridengine import GridEngine

class LoggingWrapper(gym.Wrapper):
    def __init__(self, env: GridEngine, training_steps: int, training_episodes: int):
        super().__init__(env)
        self.env: GridEngine = self.env
        self.max_episodes = training_episodes
        self.max_training_steps = training_steps
        self.training_step = 0
        self.episode_count = 0

    @override
    def step(self, action):
        self.training_step += 1
        print("Training Step:", self.training_step)
        return super().step(action)

    @override
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.episode_count += 1
        print("Episode Count:", self.episode_count)

        if (self.episode_count >= self.max_episodes or self.training_step >= self.max_training_steps):
            self._save_environment_heatmaps()

        return obs, info

    def _save_environment_heatmaps(self) -> None:
        file_path = f"{self.env.name}_Heatmap.png"
        figure = self.env.heatmap()
        if (figure is not None):
            figure.savefig(file_path)
            figure.clear()
        else:
            # TODO make proper error?
            print("Exporting Heatmaps failed, as no figure was able to be created")
