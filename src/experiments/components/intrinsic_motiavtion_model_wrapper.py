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
from experiments.components.plotter import Plotter

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
        self.plotter = Plotter()
        self.all_extrinsisc_rewards = []
        self.all_intrinsisc_rewards = []
        self.all_visited_cell_counts_by_step = []

        self.all_walkable_cords = self.env.get_every_wakable_cor()

    def reset(self, **kwargs):
        self.episode_count += 1
        self.absolute_episode_count += 1
        self.last_n_extrinsic_rewards.append(self.total_episode_extrinsic_reward)
        self.last_n_intrinsic_rewards.append(self.total_episode_intrinsic_reward)

        print(f"Current Model: {self.intrinsic_model.name}, Current Env: {self.env.name}, Current Episode: {self.episode_count}")

        if not self.allow_global_state_reset or self.last_best_global_state is None:
            if (not self.use_rgb_state):
                obs, info = self.env.reset(**kwargs)
            else:
                obs, info = self.env.reset_with_rgb_state(**kwargs)
        else:
            obs, info = self.env.reset_to_specific_global_state(self.last_best_global_state, **kwargs)

        #if (isinstance(self.env, GridEngine)):
        #    is_trainig_done = self.episode_count >= self.max_episodes or self.total_training_step >= self.max_training_steps
        #    if (is_trainig_done and (self.absolute_episode_count % 200 == 0 or self.total_training_step == self.max_training_steps)):
        #        # TODO THINK ABOUT BYOL
        #        if (not isinstance(self.intrinsic_model, ByolExploreModel)):
        #            self.print_intrinsic_heatmap()
        #        self._save_environment_heatmaps()
        #        self._calc_stats()

        self.total_episode_extrinsic_reward = 0
        self.total_episode_intrinsic_reward = 0
        self.training_step = 0

        if (isinstance(self.env, MultitaskEnv) and self.total_training_step >= self.max_training_steps * 0.56 and self.env.task == 1): # should be after 280.000 trainin steps
            self.last_n_extrinsic_rewards_before_task_switch = self.last_n_extrinsic_rewards.copy()
            for value in list(self.last_n_extrinsic_rewards):
                self.last_n_extrinsic_rewards_before_task_switch.append(value)
            self.env.task = 2 # switch task during training to see adaptation
            print("SWITCHED TASK")

       
        self.prev_state = obs # type: ignore
        return obs, info # type: ignore

    def close(self) -> None:
        self.env.reset()
        print("DONE TRAINING")
        if (not isinstance(self.intrinsic_model, ByolExploreModel)):
            self.print_intrinsic_heatmap()
        self._save_environment_heatmaps()
        self._calc_stats()
        exploration_fig = self._make_exploration_fig()
        ext_fig, int_fig = self._make_reward_figs()
        self.plotter.print_figure(exploration_fig, f"{self.intrinsic_model.name}_{self.env.name}_exploration")
        self.plotter.print_figure(ext_fig, f"{self.intrinsic_model.name}_{self.env.name}_ext_reward")
        self.plotter.print_figure(int_fig, f"{self.intrinsic_model.name}_{self.env.name}_int_reward")

        self.writer.close()

        return super().close()

    def step(self, action):
        if (isinstance(self.env, GridEngine)):
            if (not self.use_rgb_state):
                state, extrinsic_reward, terminated, truncated, info, raw_global_state = self.env.step_with_global_state(action)
            else:
                state, extrinsic_reward, terminated, truncated, info = self.env.step_with_rgb_state(action)
                raw_global_state = None
        else:
            state, extrinsic_reward, terminated, truncated, info = self.env.step(action)
        self.all_extrinsisc_rewards.append(extrinsic_reward)
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
            self.all_visited_cell_counts_by_step.append(len(self.state_space_visited))

        self.total_extrinsic_reward += extrinsic_reward # type: ignore

        self._log_training(extrinsic_reward, intrinsic_reward)

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
        self.all_intrinsisc_rewards.append(intrinsic_reward)

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
        self.replay_buffer.buffer.clear() # to safe some memory
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
                obs_doors_closed, frame = self.env.get_obs_by_state_and_agent_pos(cor, colour, rotation, locked_doors=False) # type: ignore
                obs_doors_closed_no_key, frame = self.env.get_obs_by_state_and_agent_pos(cor, colour, rotation, locked_doors=False, no_keys=True) # type: ignore
                obs_doors_open, frame = self.env.get_obs_by_state_and_agent_pos(cor, colour, rotation, open_doors=True) # type: ignore
                obs_doors_open_no_key, frame = self.env.get_obs_by_state_and_agent_pos(cor, colour, rotation, open_doors=True, no_keys=True) # type: ignore
                obs_doors_locked, frame = self.env.get_obs_by_state_and_agent_pos(cor, colour, rotation) # type: ignore
                obs_no_small_reward, frame = self.env.get_obs_by_state_and_agent_pos(cor, colour, rotation, no_small_reward=True) # type: ignore
                obs_no_key, frame = self.env.get_obs_by_state_and_agent_pos(cor, colour, rotation, no_keys=True) # type: ignore

                random_action = random.choice(action_list)
                intrinsic_reward_doors_closed = self._get_intrinsic_reward_from_model_no_training(state=obs_doors_closed, action=random_action)
                intrinsic_reward_doors_closed_no_key = self._get_intrinsic_reward_from_model_no_training(state=obs_doors_closed_no_key, action=random_action)
                intrinsic_reward_doors_opened = self._get_intrinsic_reward_from_model_no_training(state=obs_doors_open, action=random_action)
                intrinsic_reward_doors_opened_no_key = self._get_intrinsic_reward_from_model_no_training(state=obs_doors_open_no_key, action=random_action)
                intrinsic_reward_doors_locked = self._get_intrinsic_reward_from_model_no_training(state=obs_doors_locked, action=random_action)
                intrinsic_reward_no_small_reward = self._get_intrinsic_reward_from_model_no_training(state=obs_no_small_reward, action=random_action)
                intrinsic_reward_no_key = self._get_intrinsic_reward_from_model_no_training(state=obs_no_key, action=random_action)

                intrinsic_reward = min(intrinsic_reward_doors_closed,
                                       intrinsic_reward_doors_closed_no_key,
                                       intrinsic_reward_doors_opened,
                                       intrinsic_reward_doors_opened_no_key,
                                       intrinsic_reward_doors_locked,
                                       intrinsic_reward_no_small_reward,
                                       intrinsic_reward_no_key
                                       )
                #if (intrinsic_reward > 0.07):
                #    print(f"Closed: {intrinsic_reward_doors_closed}, Open: {intrinsic_reward_doors_opened}, Cor: {cor}, cor_rewar: {cor_intrinsic_reward}, rotation: {rotation}")
                #    _, axes = plt.subplots(figsize=(self.env.env_settings.width, self.env.env_settings.height))
                #    axes.imshow(frame, zorder=0) # type: ignore
                #    figure = axes.get_figure()
                #    if (figure is not None):
                #        figure.savefig(f"tmp_{cor}_{round(intrinsic_reward)}_{rotation}.png")
                #        figure.clear()
                #        plt.close()

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
            file_path = f"heatmaps/{self.intrinsic_model.name}_{self.env.name}_episode_{self.episode_count}_colour_{colours[idx]}_intrinsic_Heatmap.png"
            overlay_file_path = f"heatmaps/{self.intrinsic_model.name}_{self.env.name}_episode_{self.episode_count}_colour_{colours[idx]}_intrinsic_Heatmap_overlay.png"
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

            plt.close(raw_figure)
            plt.close(overlay_figure)

    def _save_environment_heatmaps(self) -> None:
        file_path = f"heatmaps/{self.intrinsic_model.name}_{self.env.name}_episode_{self.episode_count}_Heatmap.png"
        overlay_file_path = f"heatmaps/{self.intrinsic_model.name}_{self.env.name}_episode_{self.episode_count}_Heatmap_overlay.png"
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

        plt.close(figure)
        plt.close(overlay_figure)

    def _calc_stats(self):
        total_visited_states = 0
        for cord in self.all_walkable_cords:
            total_visited_states += self.state_space_visited[np.array(cord).tobytes()]
        lower_bound_visited_state_space = total_visited_states / len(self.all_walkable_cords)
        avg_extrinsic_reward = self.total_extrinsic_reward / self.training_step
        avg_extrinsic_reward_last_10_episodes = sum([reward for reward in self.last_n_extrinsic_rewards]) / len(self.last_n_extrinsic_rewards)
        avg_intrinsic_reward_last_10_episodes = sum([reward for reward in self.last_n_intrinsic_rewards]) / len(self.last_n_intrinsic_rewards)

        with open(f"stats_{self.intrinsic_model.name}_{self.env.name}_episode_{self.absolute_episode_count}.txt", "a") as f:
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

    def _log_training(self, extrinsic_reward, intrinsic_reward):
        self.writer.add_scalar("Reward/Intrinsic", intrinsic_reward, self.total_training_step)
        self.writer.add_scalar("Reward/Extrinsic", extrinsic_reward, self.total_training_step)
        self.writer.add_scalars("Exploration/Zellen", 
                            {
                                "Zellen":len(self.state_space_visited),
                                    "Alle Zellen": len(self.all_walkable_cords)
                                }, self.total_training_step)

    def _make_exploration_fig(self):
        return self.plotter.make_exploration_figure(
            self.all_visited_cell_counts_by_step,
            [len(self.all_walkable_cords)] * (self.total_training_step),
            list(range(self.total_training_step)),
            f"Explorationsverlauf\n der {self.env._full_name}",
        )

    def _make_reward_figs(self):
        with_marker = False
        if (self.total_extrinsic_reward > 0.0):
            with_marker = True

        if (not isinstance(self.env, MultitaskEnv)):
            ext_fig = self.plotter.make_reward_figure(
                        self.all_extrinsisc_rewards,
                        list(range(self.total_training_step)),
                        f"Extrinsische Belohnung über alle Trainingsschritte\n der {self.env._full_name}",
                        "ext.",
                        with_marker
                    )
            int_fig = self.plotter.make_reward_figure(
                    self.all_intrinsisc_rewards,
                    list(range(self.total_training_step)),
                    f"Intrinsische Belohnung über alle Trainingsschritte\n der {self.env._full_name}",
                    "int."
                )
        else:
            ext_fig = self.plotter.make_reward_figure_with_vertical(
                        self.all_extrinsisc_rewards,
                        list(range(self.total_training_step)),
                        f"Extrinsische Belohnung über alle Trainingsschritte\n der {self.env._full_name}",
                        "ext.",
                        280_000,
                        "Multitask Aufgabenwechsel",
                        with_marker,
                )

            int_fig = self.plotter.make_reward_figure_with_vertical(
                    self.all_intrinsisc_rewards,
                    list(range(self.total_training_step)),
                    f"Intrinsische Belohnung über alle Trainingsschritte\n der {self.env._full_name}",
                    "int.",
                    280_000,
                    "Multitask Aufgabenwechsel",
                )
        
        return ext_fig, int_fig
