from experiments.clean_rl.ppo import Args, run_clean_rl_ppo_model
from curiosity_gym import TestEnv
from curiosity_gym.core.gridengine import GridEngine 
import matplotlib.pyplot as plt

import gymnasium as gym

#import gymnasium as gym
#env = gym.make("DerailmentEnv", render_mode="rgb_array")
#env.print_inital_env_state_as_pdf()

def tmp():
    env_name = "TestEnv"
    env = gym.make(env_name,
                   agentPOV="local_2",
                   render_mode="rgb_array",
                   simple_obs=True,
                   simple_actions=False,
                   use_globaly_unique_id=False,
                )
    
    grid_env: GridEngine = env.unwrapped # type: ignore
    grid_env.print_inital_env_state_as_pdf(True)
    frame = grid_env._render_frame(with_view_overlay=True, with_obj_id=True)
    _, axes = plt.subplots(figsize=(grid_env.env_settings.width, grid_env.env_settings.height))
    axes.imshow(frame, zorder=0) # type: ignore
    figure = axes.get_figure()
    if (figure is not None):
        figure.savefig("tmp.png")
        figure.clear()

tmp()
