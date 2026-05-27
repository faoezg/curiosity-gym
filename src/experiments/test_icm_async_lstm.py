from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import A2C

from curiosity_gym import DistractiveEnv, SparseEnv, MultitaskEnv, DetachmentEnv, DerailmentEnv
from curiosity_gym.core.gridengine import GridEngine 

from experiments.icm import ICMModel
from experiments.icm.icm_factory import make_icm_env 
from experiments.icm.icm_lstm_policy import IcmLSTMPolicy

import torch
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import gc

# ----------------------------------------------------------------------
# 0️⃣  Global configuration (you can move this to a config file)
# ---------------------
from functools import partial

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_ENVS = 1                     # number of asynchronous workers (SB3 default)
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
        use_globaly_unique_id=False,
        use_colours=True,
        use_rgb_state=False
    )  # type: ignore

    state_dim=tmp_raw.observation_space.shape[0]
    action_dim=tmp_raw.action_space.n
    latent_rep_dim=256

    icm = ICMModel(
        device=DEVICE,
        state_dim=state_dim,
        action_dim=action_dim,
        latent_rep_dim=latent_rep_dim,
        hidden_dim_forward=256,
        hidden_dim_inverse=256,
        hidden_dim_encoder=1024,
        beta=0.2,
        eta=0.1,
        stride=tmp_raw.unwrapped.label_count_per_cell,
        icm_lr=1e-6,
        share_memory=True,
        use_cnn_encoder=False
    )

    # Put every parameter tensor into shared memory so that all workers
    # see the *same* ICM weights.

    tmp_raw.close()
    return icm, state_dim, action_dim, latent_rep_dim


def make_env(rank: int, shared_icm: ICMModel) -> gym.Env:
    """
    Returns a *Gymnasium* environment ready to be used by SubprocVecEnv.
    """
    env = make_icm_env(
        base_env_id="SimpleSparseEnv",
        base_env_pov="local_2",
        device=DEVICE,
        latent_rep_dim=256,
        hidden_dim_forward=256,
        hidden_dim_inverse=256,
        hidden_dim_encoder=1024,
        intrinsic_reset_threshold=0.5,
        allow_global_state_reset=False,
        max_episodes=1,
        max_training_steps=500_000,
        use_simple_obs=True,
        use_simple_actions=False,
        use_globaly_unique_id=False,
        use_colours=True,
        use_rgb_state=False,
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
    shared_icm, state_dim, action_dim, latent_rep_dim = _build_shared_icm()

    # --------------------------------------------------------------
    # (b)  Build a list of environment factories for SubprocVecEnv
    # --------------------------------------------------------------
    env_fns = [partial(make_env, i, shared_icm) for i in range(NUM_ENVS)]
    vec_env = SubprocVecEnv(env_fns)          # ← asynchronous workers

    # --------------------------------------------------------------
    # (c)  Create the SB3 learner (A2C or A3C)
    # --------------------------------------------------------------

    policy = IcmLSTMPolicy(
        device=DEVICE,
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=latent_rep_dim,
        use_cnn_encoder=False
    )

    policy.share_memory()
    policy.encoder.encoder_model.share_memory()

    policy_optimizer = torch.optim.Adam([
            {"params": policy.encoder.encoder_model.parameters(), "lr": 1e-3},
            {"params": policy.parameters(), "lr": 1e-3},
        ])

    GAMMA = 0.99
    ENTROPY_COEF = 0.01
    VALUE_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    N_STEPS = 20

    global_step = 0

    hidden, c = policy.init_lstm_state(batch_size=NUM_ENVS)

    obs = vec_env.reset()
    obs = torch.from_numpy(obs).to(torch.float32).to(DEVICE)
    while global_step < TOTAL_TIMESTEPS:
        obs_buf = []
        actions_buf = []
        rewards_buf = []
        values_buf = []
        logprob_buf = []
        h_buf, c_buf = [], []

        for _  in range(N_STEPS):
            probs, sample_one_hot, value, (hidden, c) = policy.forward(obs, (hidden, c))
            action = sample_one_hot.argmax(dim=-1).cpu().numpy()
            next_obs, reward, done, info = vec_env.step(action)

            obs_buf.append(obs.detach())
            actions_buf.append(action)
            rewards_buf.append(torch.tensor(reward, dtype=torch.float32, device=DEVICE).detach())
            values_buf.append(value.detach())
            action_tensor = torch.from_numpy(action).long().to(DEVICE).unsqueeze(1)
            logprog = torch.log(probs.gather(1, action_tensor))
            logprob_buf.append(logprog.detach())
            h_buf.append(hidden.detach())
            c_buf.append(c.detach())

            done_tensor = torch.tensor(done, dtype=torch.float32, device=DEVICE)
            mask = (1.0 - done_tensor).unsqueeze(1)

            hidden = hidden * mask
            c = c * mask

            obs = next_obs
            obs = torch.from_numpy(obs).to(torch.float32).to(DEVICE)

        with torch.no_grad():
            _, _, next_value, _ = policy.forward(obs, (hidden, c)) # type: ignore
            next_value = next_value.squeeze()
        
        returns = []
        advs = []
        gae = 0.0
        for step in reversed(range(N_STEPS)):
            mask = 1.0 - torch.tensor(done, dtype=torch.float32, device=DEVICE)  # type: ignore # 0 if episode ended
            delta = rewards_buf[step] + GAMMA * next_value * mask - values_buf[step]
            gae = delta + GAMMA * 0.95 * mask * gae          # λ = 0.95 (you can expose it)
            advs.insert(0, gae)
            next_value = values_buf[step]
            returns.insert(0, gae + values_buf[step])

        obs_batch      = torch.cat(obs_buf, dim=0).detach()
        actions_batch  = torch.from_numpy(np.concatenate(actions_buf)).to(DEVICE).detach()
        returns_batch  = torch.cat(returns).detach()
        advs_batch     = torch.cat(advs).detach()
        logprob_batch  = torch.cat(logprob_buf, dim=0)
        values_batch   = torch.cat(values_buf, dim=0)

        policy_optimizer.zero_grad()
        policy_loss = -(logprob_batch * advs_batch).mean()
        value_loss = F.mse_loss(values_batch, returns_batch)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean() # type: ignore
        entropy_loss = -ENTROPY_COEF * entropy

        total_loss = policy_loss + VALUE_COEF * value_loss + entropy_loss
        torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
        total_loss.backward()
        policy_optimizer.step()

        hidden = hidden.detach()
        c = c.detach()

        global_step += NUM_ENVS * N_STEPS

        if global_step % (NUM_ENVS * 1000) == 0:
            print(f"Step {global_step:,} | policy loss {policy_loss.item():.4f} | "
                f"value loss {value_loss.item():.4f} | entropy {entropy.item():.4f}")

        gc.collect()
        torch.cuda.empty_cache()
   
    print("Training finished – models saved.")
