import gymnasium as gym

gym.register(
    id="SparseEnv",
    entry_point="curiosity_gym.envs.sparseenv:SparseEnv", 
    kwargs={
        "render_mode": "rgb_array"
    }
)

gym.register(
    id="DistractiveEnv",
    entry_point="curiosity_gym.envs.distractiveenv:DistractiveEnv",
    kwargs={
        "render_mode": "rgb_array"
    }
)

gym.register(
    id="MultitaskEnv",
    entry_point="curiosity_gym.envs.multitaskenv:MultitaskEnv",
    kwargs={
        "render_mode": "rgb_array"
    }
)
