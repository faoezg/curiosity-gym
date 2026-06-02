import gymnasium as gym

gym.register(
    id="SparseEnv",
    entry_point="curiosity_gym.envs.sparseenv:SparseEnv", 
    kwargs={
        "render_mode": "rgb_array"
    }
)

gym.register(
    id="SimpleSparseEnv",
    entry_point="curiosity_gym.envs.simple_sparseenv:SimpleSparseEnv", 
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

gym.register(
    id="DetachmentEnv",
    entry_point="curiosity_gym.envs.detachment_env:DetachmentEnv",
    kwargs={
        "render_mode": "rgb_array"
    }
)

gym.register(
    id="DerailmentEnv",
    entry_point="curiosity_gym.envs.derailment_env:DerailmentEnv",
    kwargs={
        "render_mode": "rgb_array"
    }
)

gym.register(
    id="TestEnv",
    entry_point="curiosity_gym.envs.test_env:TestEnv",
    kwargs={
        "render_mode": "rgb_array"
    }
)
