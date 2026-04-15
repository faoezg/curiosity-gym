from abc import abstractmethod
from typing import Any
import gymnasium as gym
from torch import device
import random
from curiosity_gym.core.gridengine import GridEngine
from curiosity_gym.utils.enums import Action

from .intrinsic_motiavtion_model import IntrinsicMotivationModel

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
        self.env: GridEngine = self.env # getting ride of bad type hint as casting isn't a real thing in python
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

    def reset(self, **kwargs):
        self.training_step = 0
        self.episode_count += 1

        is_trainig_done = self.episode_count >= self.max_episodes or self.training_step >= self.max_training_steps
        if (is_trainig_done):
            # TODO THINK ABOUT BYOL
            # ORDER MATTRES BECOUSE OF AGENT POSITION SHIFTS...
            self.print_intrinsic_heatmap()
            self._save_environment_heatmaps()

        print("Current Episode: ", self.episode_count)

        if not self.allow_global_state_reset or self.last_best_global_state is None:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, info = self.env.reset_to_specific_global_state(self.last_best_global_state, **kwargs)
        
        self.prev_state = obs
        return obs, info

    def step(self, action):
        state, extrinsic_reward, terminated, truncated, info, raw_global_state = self.env.step_with_global_state(action)
        intrinsic_reward = self._get_intrinsic_reward_from_model(state=state, action=action)
        self._handle_new_intrinsic_reward(intrinsic_reward, raw_global_state)
        reward = extrinsic_reward + intrinsic_reward

        info["extrinsic_reward"] = extrinsic_reward
        info["intrinsic_reward"] = intrinsic_reward
        info["total_reward"] = reward 

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
    def _calc_intrinsic_reward_map(self) -> dict[tuple[int,int], float]:
        intrinsic_map = {}
        curr_global_state = self.env.get_state()
        walkable_cor = self.env.get_every_wakable_cor()

        action_list = [element.value for element in Action] # Why is there no method for this?
        total_reward = 0
        for cor in walkable_cor:
            obs = self.env.get_obs_by_state_and_agent_pos(curr_global_state, cor) # type: ignore
            random_action = random.choice(action_list)
            intrinsic_reward = self._get_intrinsic_reward_from_model_no_training(state=obs, action=random_action)
            total_reward += intrinsic_reward
            intrinsic_map[cor] = intrinsic_reward

        if (total_reward > 0):
            for cor, value in intrinsic_map.items():
                intrinsic_map[cor] = (value/total_reward) * 100
        return intrinsic_map
    
    def print_intrinsic_heatmap(self):
        file_path = f"{self.env.name}_episode_{self.episode_count}_intrinsic_Heatmap.png"
        intrinsic_map = self._calc_intrinsic_reward_map()
        figure = self.env.heatmap_from_data(intrinsic_map)
        if (figure is not None):
            figure.savefig(file_path)
            figure.clear()
        else:
            # TODO make proper error?
            print("Exporting Heatmaps failed, as no figure was able to be created")

    def _save_environment_heatmaps(self) -> None:
        file_path = f"{self.env.name}_episode_{self.episode_count}_Heatmap.png"
        figure = self.env.heatmap()
        if (figure is not None):
            figure.savefig(file_path)
            figure.clear()
        else:
            # TODO make proper error?
            print("Exporting Heatmaps failed, as no figure was able to be created")
