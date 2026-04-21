from abc import abstractmethod
from typing import Any
import gymnasium as gym
from torch import device
import random
import matplotlib.pyplot as plt
from collections import defaultdict

from curiosity_gym.core.gridengine import GridEngine
from curiosity_gym.utils.enums import Action, Rotation
from curiosity_gym.core.objects import Key


from .intrinsic_motiavtion_model import IntrinsicMotivationModel
from experiments.byol_explore.byol_model import ByolExploreModel

class IntrinsicMotivationModelWrapper(gym.Wrapper):
    def __init__(self,
                 env: GridEngine, 
                 intrinsic_model: IntrinsicMotivationModel,
                 device: device | str,
                 intrinsic_reset_threshold: float = 0.5,
                 allow_global_state_reset: bool = False,
                 max_training_steps: int = 500,
                 max_episodes: int = 1000) -> None:
        super().__init__(env)
        self.env: GridEngine | gym.Env = self.env # getting ride of bad type hint as casting isn't a real thing in python
        self.intrinsic_model = intrinsic_model
        self.device = device

        self.last_best_global_state = None
        self.last_best_intrinsic_reward = 0.0
        self.intrinsic_reset_threshold = intrinsic_reset_threshold
        self.allow_global_state_reset = allow_global_state_reset

        self.max_training_steps = max_training_steps
        self.max_episodes = max_episodes
        self.training_step = 0
        self.episode_count = -1 # given inital reset
        self.absolute_episode_count = -1 

        self.state_space_visited = defaultdict(int)
        self.total_reward = 0

    def reset(self, **kwargs):
        self.training_step = 0
        self.episode_count += 1
        self.absolute_episode_count += 1

        print("Current Episode: ", self.episode_count)

        if not self.allow_global_state_reset or self.last_best_global_state is None:
            obs, info = self.env.reset(**kwargs)
        elif (isinstance(self.env, GridEngine)):
            obs, info = self.env.reset_to_specific_global_state(self.last_best_global_state, **kwargs)

            is_trainig_done = self.episode_count >= self.max_episodes or self.training_step >= self.max_training_steps
            if (is_trainig_done and self.absolute_episode_count % 50 == 0):
                # TODO THINK ABOUT BYOL
                # ORDER MATTERS BECOUSE OF AGENT STATE/COLOUR CHANGE ON RESET
                if (not isinstance(self.intrinsic_model, ByolExploreModel)):
                    self.print_intrinsic_heatmap()
                self._save_environment_heatmaps()
            
        self.prev_state = obs # type: ignore
        return obs, info # type: ignore

    def step(self, action):
        if (isinstance(self.env, GridEngine)):
            state, extrinsic_reward, terminated, truncated, info, raw_global_state = self.env.step_with_global_state(action)
        else:
            state, extrinsic_reward, terminated, truncated, info = self.env.step(action)
        intrinsic_reward = self._get_intrinsic_reward_from_model(state=state, action=action)
        if (isinstance(self.env, GridEngine)):
            self._handle_new_intrinsic_reward(intrinsic_reward, raw_global_state) # type: ignore
        if (intrinsic_reward > 0.1):
            reward = extrinsic_reward + intrinsic_reward # type: ignore
        else:
            reward = extrinsic_reward

        info["extrinsic_reward"] = extrinsic_reward
        info["intrinsic_reward"] = intrinsic_reward
        info["total_reward"] = reward 

        if (isinstance(self.env, GridEngine)):
            self.state_space_visited[self.env.objects.agent.position.tobytes()] = 1

        self.total_reward += reward

        print(info)


        self.prev_state = state
        self.training_step += 1

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
                cor_intrinsic_reward += intrinsic_reward
            intrinsic_reward = cor_intrinsic_reward / 4
            total_reward += intrinsic_reward
            intrinsic_map[cor] += intrinsic_reward

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
                raw_figure.savefig(file_path)
                raw_figure.clear()
            if (overlay_figure is not None):
                overlay_figure.savefig(overlay_file_path)
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
            figure.savefig(file_path)
            figure.clear()
        if (overlay_figure is not None):
            overlay_figure.savefig(overlay_file_path)
            overlay_figure.clear()
        else:
            # TODO make proper error?
            print("Exporting Heatmaps failed, as no figure was able to be created")

        plt.close()
