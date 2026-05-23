from abc import abstractmethod
from typing import Any
import gymnasium as gym
from torch import device
from torch.utils.tensorboard import SummaryWriter
import random
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import numpy as np

from curiosity_gym.core.gridengine import GridEngine
from curiosity_gym.envs.multitaskenv import MultitaskEnv
from curiosity_gym.utils.enums import Action, Rotation
from curiosity_gym.core.objects import Key


from .intrinsic_motiavtion_model import IntrinsicMotivationModel
from experiments.byol_explore.byol_model import ByolExploreModel
from experiments.components import ReplayBuffer, Transition

class IntrinsicMotivationModelWrapper(gym.Wrapper):
    def __init__(self,
                 env: GridEngine, 
                 intrinsic_model: IntrinsicMotivationModel,
                 device: device | str,
                 intrinsic_reset_threshold: float = 0.5,
                 allow_global_state_reset: bool = False,
                 max_training_steps: int = 500,
                 max_episodes: int = 1000,
                 use_rgb_state: bool = False) -> None:
        super().__init__(env)
        self.env: GridEngine | gym.Env = self.env.unwrapped # getting ride of bad type hint as casting isn't a real thing in python
        self.intrinsic_model = intrinsic_model
        self.device = device
        self.replay_buffer = ReplayBuffer(size=10000)
        self.rollout_buffer = deque(maxlen=20)
        self.use_rgb_state = use_rgb_state

        self.last_best_global_state = None
        self.last_best_intrinsic_reward = 0.0
        self.intrinsic_reset_threshold = intrinsic_reset_threshold
        self.allow_global_state_reset = allow_global_state_reset

        self.max_training_steps = max_training_steps
        self.max_episodes = max_episodes
        self.training_step = 0
        self.total_training_step = 0
        self.episode_count = -1 # given inital reset
        self.absolute_episode_count = -1 

        self.state_space_visited = defaultdict(int)
        self.total_extrinsic_reward = 0
        self.total_episode_extrinsic_reward = 0
        self.total_episode_intrinsic_reward = 0
        self.last_n_extrinsic_rewards = deque(maxlen=10)
        self.last_n_intrinsic_rewards = deque(maxlen=10)
        self.last_n_extrinsic_rewards_before_task_switch = None
        self.writer = SummaryWriter()

    def reset(self, **kwargs):
        self.episode_count += 1
        self.absolute_episode_count += 1
        self.last_n_extrinsic_rewards.append(self.total_episode_extrinsic_reward)
        self.last_n_intrinsic_rewards.append(self.total_episode_intrinsic_reward)

        print("Current Episode: ", self.episode_count)

        if not self.allow_global_state_reset or self.last_best_global_state is None:
            if (not self.use_rgb_state):
                obs, info = self.env.reset(**kwargs)
            else:
                obs, info = self.env.reset_with_rgb_state(**kwargs)
        else:
            obs, info = self.env.reset_to_specific_global_state(self.last_best_global_state, **kwargs)

        if (isinstance(self.env, GridEngine)):
            is_trainig_done = self.episode_count >= self.max_episodes or self.total_training_step >= self.max_training_steps
            if (is_trainig_done and self.absolute_episode_count % 300 == 0):
                # TODO THINK ABOUT BYOL
                # ORDER MATTERS BECOUSE OF AGENT STATE/COLOUR CHANGE ON RESET
                if (not isinstance(self.intrinsic_model, ByolExploreModel)):
                    self.print_intrinsic_heatmap()
                self._save_environment_heatmaps()
                self._calc_stats()
        
        self.total_episode_extrinsic_reward = 0
        self.total_episode_intrinsic_reward = 0
        self.training_step = 0

        if (isinstance(self.env, MultitaskEnv) and self.absolute_episode_count >= self.max_episodes * 0.56 and self.env.task == 1): # should be after 280.000 trainin steps
            self.last_n_extrinsic_rewards_before_task_switch = self.last_n_extrinsic_rewards.copy()
            self.env.task = 2 # switch task during training to see adaptation
            print("SWITCHED TASK")

       
        self.prev_state = obs # type: ignore
        return obs, info # type: ignore

    def step(self, action):
        if (isinstance(self.env, GridEngine)):
            if (not self.use_rgb_state):
                state, extrinsic_reward, terminated, truncated, info, raw_global_state = self.env.step_with_global_state(action)
            else:
                state, extrinsic_reward, terminated, truncated, info = self.env.step_with_rgb_state(action)
                raw_global_state = None
        else:
            state, extrinsic_reward, terminated, truncated, info = self.env.step(action)
        self._store_transition(self.prev_state, action, extrinsic_reward, state)
        intrinsic_reward = self._get_intrinsic_reward_from_model(state=state, action=action)

        if (isinstance(self.env, GridEngine)):
            self._handle_new_intrinsic_reward(intrinsic_reward, raw_global_state) # type: ignore

        reward = extrinsic_reward + intrinsic_reward # type: ignore

        info["extrinsic_reward"] = extrinsic_reward
        self.total_episode_extrinsic_reward += extrinsic_reward # type: ignore
        info["intrinsic_reward"] = intrinsic_reward
        self.total_episode_intrinsic_reward += intrinsic_reward
        info["total_reward"] = reward 

        if (isinstance(self.env, GridEngine)):
            self.state_space_visited[self.env.objects.agent.position.tobytes()] = 1

        self.total_extrinsic_reward += extrinsic_reward # type: ignore

        self.writer.add_scalar("Reward/Intrinsic", intrinsic_reward, self.total_training_step)
        self.writer.add_scalar("Reward/Extrinsic", extrinsic_reward, self.total_training_step)

        self.prev_state = state
        self.training_step += 1
        self.total_training_step += 1

        return state, reward, terminated, truncated, info
    
    def _handle_new_intrinsic_reward(self, intrinsic_reward, raw_global_state):
        if (self.last_best_intrinsic_reward <= self.intrinsic_reset_threshold):
            self.last_best_global_state = None
        elif (intrinsic_reward >= self.last_best_intrinsic_reward):
            self.last_best_intrinsic_reward = intrinsic_reward
            self.last_best_global_state = raw_global_state

    @abstractmethod
    def _get_intrinsic_reward_from_model(self, *args, **kwargs) -> float | Any:
        pass

    @abstractmethod
    def _get_intrinsic_reward_from_model_no_training(self, *args, **kwargs) -> float | Any:
        pass

    def _store_transition(self, old_state, action, reward, new_state):
        transition = Transition(old_state, action, reward, new_state)
        self.replay_buffer.add(transition)
        self.rollout_buffer.append(transition)
    
    def _sample_batch(self, batch_size = 32):
        return self.replay_buffer.sample(batch_size)
    
    def _get_rollout(self):
        rollout = list(self.rollout_buffer)
        self.rollout_buffer.clear()
        return rollout

    # TODO make nicer and normalize intrinsic reward for comparisons??
    def _calc_intrinsic_reward_map(self) -> tuple[list[dict[tuple[int,int], float]], list[int]]:
        intrinsic_maps = []
        colours = []

        for other_obj in self.env.objects.other:
            if (isinstance(other_obj, Key)):
                colour = other_obj.color
                colours.append(colour)

        colours.append(self.env.objects.agent.start_color)

        for colour in colours:
            colour_intrinsic_map = self._calc_intrinsic_map(colour)
            intrinsic_maps.append(colour_intrinsic_map)

        return intrinsic_maps, colours
    
    def _calc_intrinsic_map(self, colour: int):
        intrinsic_map = defaultdict(float)
        walkable_cor = self.env.get_every_wakable_cor()

        rotation_list = [element.value for element in Rotation]

        action_list = [element.value for element in Action] # Why is there no method for this?
        total_reward = 0
        for cor in walkable_cor:
            cor_intrinsic_reward = 0
            for rotation in rotation_list:
                obs = self.env.get_obs_by_state_and_agent_pos(cor, colour, rotation) # type: ignore
                random_action = random.choice(action_list)
                intrinsic_reward = self._get_intrinsic_reward_from_model_no_training(state=obs, action=random_action)
                #cor_intrinsic_reward += intrinsic_reward
                cor_intrinsic_reward = min(intrinsic_reward, cor_intrinsic_reward) if cor_intrinsic_reward > 0 else intrinsic_reward
            #intrinsic_reward = cor_intrinsic_reward / len(rotation_list)
            #total_reward += intrinsic_reward
            #intrinsic_map[cor] += intrinsic_reward
            total_reward += cor_intrinsic_reward 
            intrinsic_map[cor] += cor_intrinsic_reward

        if (total_reward > 0):
            for cor, value in intrinsic_map.items():
                intrinsic_map[cor] = (value/total_reward) * 100
        
        return intrinsic_map

    # TODO OH WHAT UGLY CODE!
    def print_intrinsic_heatmap(self):
        intrinsic_maps, colours = self._calc_intrinsic_reward_map()
        for idx, map in enumerate(intrinsic_maps):
            file_path = f"heatmaps/{self.env.name}_episode_{self.episode_count}_colour_{colours[idx]}_intrinsic_Heatmap.png"
            overlay_file_path = f"heatmaps/{self.env.name}_episode_{self.episode_count}_colour_{colours[idx]}_intrinsic_Heatmap_overlay.png"
            raw_figure = self.env.heatmap_from_data(map)
            overlay_figure = self.env.overlay_heatmap_from_data(map)
            if (raw_figure is not None):
                raw_figure.savefig(file_path, transparent=True)
                raw_figure.clear()
            if (overlay_figure is not None):
                overlay_figure.savefig(overlay_file_path, transparent=True)
                overlay_figure.clear()
            else:
                # TODO make proper error?
                print("Exporting Heatmaps failed, as no figure was able to be created")

        plt.close()

    def _save_environment_heatmaps(self) -> None:
        file_path = f"heatmaps/{self.env.name}_episode_{self.episode_count}_Heatmap.png"
        overlay_file_path = f"heatmaps/{self.env.name}_episode_{self.episode_count}_Heatmap_overlay.png"
        figure = self.env.heatmap()
        overlay_figure = self.env.overlay_heatmap()
        if (figure is not None):
            figure.savefig(file_path, transparent=True)
            figure.clear()
        if (overlay_figure is not None):
            overlay_figure.savefig(overlay_file_path, transparent=True)
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
        avg_intrinsic_reward_last_10_episodes = sum([reward for reward in self.last_n_intrinsic_rewards]) / len(self.last_n_intrinsic_rewards)

        with open(f"stats_{self.env.name}_episode_{self.absolute_episode_count}.txt", "a") as f:
            f.write("############# STATS ################ \n")
            f.write(f"Lower Bound of visited Statespace: {lower_bound_visited_state_space} \n")
            f.write(f"Avg. ext. Reward over all Trainingsteps: {avg_extrinsic_reward} \n")
            f.write(f"Avg. ext. Reward over last 10 Episodes: {avg_extrinsic_reward_last_10_episodes} \n")
            f.write(f"Avg. intr. Reward over last 10 Episodes: {avg_intrinsic_reward_last_10_episodes} \n")
            if (self.last_n_extrinsic_rewards_before_task_switch is not None):
                avg_extrinsic_reward_last_10_episodes_before_switch = sum([reward for reward in self.last_n_extrinsic_rewards_before_task_switch]) / len(self.last_n_extrinsic_rewards_before_task_switch)
                f.write(f"Avg. ext. Reward over last 10 Episodes before Task switch: {avg_extrinsic_reward_last_10_episodes_before_switch} \n")

    def _calc_training_stats(self):
        action_list = [element.value for element in Action] # Why is there no method for this?
        obs = self.env._get_obs() # type: ignore
        random_action = random.choice(action_list)
        intrinsic_reward = self._get_intrinsic_reward_from_model_no_training(state=obs, action=random_action)

        with open(f"stats_{self.env.name}_episode_{self.absolute_episode_count}_training.txt", "a") as f:
            f.write(f"############# STATS for Step {self.total_training_step} ################ \n")
            f.write(f"Intrinsic Reward {intrinsic_reward} \n")
