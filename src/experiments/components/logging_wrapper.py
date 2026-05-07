import gymnasium as gym
from typing import override
import numpy as np
from collections import deque, defaultdict

from curiosity_gym.core.gridengine import GridEngine
from curiosity_gym.envs.multitaskenv import MultitaskEnv
import matplotlib.pyplot as plt

class LoggingWrapper(gym.Wrapper):
    def __init__(self, env: GridEngine, training_steps: int, training_episodes: int):
        super().__init__(env)
        self.env: GridEngine = self.env.unwrapped
        self.max_episodes = training_episodes
        self.max_training_steps = training_steps
        self.training_step = 0
        self.episode_count = 0

        self.state_space_visited = defaultdict(int)
        self.total_extrinsic_reward = 0
        self.total_episode_extrinsic_reward = 0
        self.last_n_extrinsic_rewards = deque(maxlen=10)
        self.last_n_extrinsic_rewards_before_task_switch = None

    @override
    def step(self, action):
        self.training_step += 1
        #print("Training Step:", self.training_step)

        state, extrinsic_reward, terminated, truncated, info = super().step(action)
        self.total_extrinsic_reward += extrinsic_reward # type: ignore
        self.total_episode_extrinsic_reward += extrinsic_reward # type: ignore
        if (isinstance(self.env, GridEngine)):
            self.state_space_visited[self.env.objects.agent.position.tobytes()] = 1

        return state, extrinsic_reward, terminated, truncated, info

    @override
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.episode_count += 1
        self.last_n_extrinsic_rewards.append(self.total_episode_extrinsic_reward)
        self.total_episode_extrinsic_reward = 0

        print("Episode Count:", self.episode_count)

        if (isinstance(self.env, MultitaskEnv) and self.episode_count >= self.max_episodes * 0.56 and self.env.task == 1): # should be after 280.000 trainin steps
            self.last_n_extrinsic_rewards_before_task_switch = self.last_n_extrinsic_rewards.copy()
            self.env.task = 2 # switch task during training to see adaptation
            print("SWITCHED TASK")


        if (self.episode_count >= self.max_episodes):
            self._save_environment_heatmaps()
            self._calc_stats()

        return obs, info

    def _save_environment_heatmaps(self) -> None:
        file_path = f"{self.env.name}_Heatmap.png"
        overlay_file_path = f"{self.env.name}_Heatmap_overlay.png"
        figure = self.env.heatmap()
        overlay_figure = self.env.overlay_heatmap()
        if (figure is not None):
            figure.savefig(file_path)
            figure.clear()
        if (overlay_figure is not None):
            overlay_figure.savefig(overlay_file_path)
            overlay_figure.clear()
        else:
            # TODO make proper error?
            print("Exporting Heatmaps failed, as no figure was able to be created")

        plt.close()

    def _calc_stats(self):
        walkable_cords = self.env.get_every_wakable_cor()
        total_visited_states = 0
        for cord in walkable_cords:
            total_visited_states += self.state_space_visited[np.array(cord).tobytes()]
        lower_bound_visited_state_space = total_visited_states / len(walkable_cords)
        avg_extrinsic_reward = self.total_extrinsic_reward / self.training_step
        avg_extrinsic_reward_last_10_episodes = sum([reward for reward in self.last_n_extrinsic_rewards]) / len(self.last_n_extrinsic_rewards)

        print("############# STATS ################")
        print(f"Lower Bound of visited Statespace: {lower_bound_visited_state_space}")
        print(f"Avg. ext. Reward over all Trainingsteps: {avg_extrinsic_reward}")
        print(f"Avg. ext. Reward over last 10 Episodes: {avg_extrinsic_reward_last_10_episodes}")
        if (self.last_n_extrinsic_rewards_before_task_switch is not None):
            avg_extrinsic_reward_last_10_episodes_before_switch = sum([reward for reward in self.last_n_extrinsic_rewards_before_task_switch]) / len(self.last_n_extrinsic_rewards_before_task_switch)
            print(f"Avg. ext. Reward over last 10 Episodes before Task switch: {avg_extrinsic_reward_last_10_episodes_before_switch}")
