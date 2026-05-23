from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import A2C

from curiosity_gym import DistractiveEnv, SparseEnv, MultitaskEnv, DetachmentEnv, DerailmentEnv
from curiosity_gym.core.gridengine import GridEngine 

from experiments.icm import ICMModel
from experiments.icm.icm_factory import make_icm_env 

import torch
import gymnasium as gym

# ----------------------------------------------------------------------
# 0️⃣  Global configuration (you can move this to a config file)
# ---------------------
from functools import partial

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_ENVS = 8                     # number of asynchronous workers (SB3 default)
SEED = 42
TOTAL_TIMESTEPS = 500_000 * NUM_ENVS

SB3_DEVICE = "cpu"
print("Running own models on: ", DEVICE)

def _build_shared_icm():
    """
    Build ONE ICM instance and put its parameters in shared memory.
    This function is called **inside the __main__ guard**, so the
    tensors are created before any subprocess is spawned.
    """
    # We need a *temporary* raw env only to read the observation / action sizes.
    tmp_raw = gym.make(
        "SimpleSparseEnv",
        render_mode=None,
        agentPOV="local_2",
        simple_obs=True,
        simple_actions=False,
        use_globaly_unique_id=True,
        use_colours=True,
    )  # type: ignore

    icm = ICMModel(
        device=DEVICE,
        state_dim=tmp_raw.observation_space.shape[0],
        action_dim=tmp_raw.action_space.n,
        latent_rep_dim=256,
        hidden_dim_forward=64,
        hidden_dim_inverse=128,
        hidden_dim_encoder=1024,
        beta=0.6,
        eta=100.0,
        stride=tmp_raw.unwrapped.label_count_per_cell,
        icm_lr=1e-6,
    )

    # Put every parameter tensor into shared memory so that all workers
    # see the *same* ICM weights.
    for p in icm.icm_network.parameters():
        p.share_memory_()

    tmp_raw.close()
    return icm


def make_env(rank: int, shared_icm: ICMModel) -> gym.Env:
    """
    Returns a *Gymnasium* environment ready to be used by SubprocVecEnv.
    """
    env = make_icm_env(
        base_env_id="SimpleSparseEnv",
        base_env_pov="local_2",
        device=DEVICE,
        latent_rep_dim=126,
        hidden_dim_forward=64,
        hidden_dim_inverse=64,
        hidden_dim_encoder=512,
        intrinsic_reset_threshold=0.5,
        allow_global_state_reset=False,
        max_episodes=1,
        max_training_steps=500_000,
        use_simple_obs=True,
        use_simple_actions=False,
        use_globaly_unique_id=True,
        use_colours=True,
        is_atari=False,
        shared_icm=shared_icm,          # <-- reuse the SAME ICM instance
        rank=rank
    )
    # Give each worker its own seed (helps decorrelation)

    if (rank == NUM_ENVS - 1):
        env = gym.wrappers.RecordVideo(env, f"videos/tmp", episode_trigger=lambda x: x % 20 == 0)
    return env


# ------------------------------------------------------------------
# 1️⃣  Main entry point – everything that spawns processes lives here
# ------------------------------------------------------------------
if __name__ == "__main__":
    # --------------------------------------------------------------
    # (a)  Make the ICM model *once* and share its memory
    # --------------------------------------------------------------
    shared_icm = _build_shared_icm()

    # --------------------------------------------------------------
    # (b)  Build a list of environment factories for SubprocVecEnv
    # --------------------------------------------------------------
    env_fns = [partial(make_env, i, shared_icm) for i in range(NUM_ENVS)]
    vec_env = SubprocVecEnv(env_fns)          # ← asynchronous workers

    # --------------------------------------------------------------
    # (c)  Create the SB3 learner (A2C or A3C)
    # --------------------------------------------------------------
    a2c = A2C(
        policy="MlpPolicy",                  # you can also use "CnnPolicy" if you feed raw images
        env=vec_env,
        gamma=0.99,
        n_steps=5,                           # steps per rollout *inside* each worker
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        tensorboard_log="./a2c_icm_tensorboard/",
        seed=SEED,
        device=SB3_DEVICE,
        learning_rate=1e-3
    )

    # --------------------------------------------------------------
    # (d)  Train!
    # --------------------------------------------------------------
    print(f"Starting training with {NUM_ENVS} async workers …")
    a2c.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=100)

    # --------------------------------------------------------------
    # (e)  Save everything (policy + ICM)
    # --------------------------------------------------------------
    a2c.save("a2c_icm_policy.zip")
    torch.save(shared_icm.state_dict(), "shared_icm.pt")
    print("Training finished – models saved.")
