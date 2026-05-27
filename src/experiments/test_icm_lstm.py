from experiments.clean_rl.ppo import Args, run_clean_rl_ppo_model
from curiosity_gym import DistractiveEnv, SparseEnv, MultitaskEnv, DetachmentEnv, DerailmentEnv
from curiosity_gym.core.gridengine import GridEngine 

from experiments.icm import ICMModel

import torch
import gymnasium as gym

SB3_DEVICE = "cpu"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAINING_EPISODES = 10
print("Running own models on: ", DEVICE)


def train_clean_rl_model() -> None:
    #env_name = "SparseEnv-Icm"
    #env_name = "MountainCar-Icm"
    env_name = "CartPole-Icm"
    env = gym.make(env_name,
                   max_episodes=1,
                   max_training_steps=1,
                   base_env_pov="local_2",
                   render_mode="rgb_array",
                   device=DEVICE)

    args = Args(
        exp_name=env_name,
        env_id=env_name,
        env=env,
        num_steps=TRAINING_EPISODES,
        capture_video=True,
        learning_rate=0.001
    )
    run_clean_rl_ppo_model(args)

#train_clean_rl_model()

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from experiments.icm.icm_lstm_policy import IcmLSTMPolicy
from experiments.icm.icm_factory import make_icm_env
import numpy as np

def tmp():
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

    tmp_raw.close()

    env = make_icm_env(
        base_env_id="SimpleSparseEnv",
        base_env_pov="local_2",
        device=DEVICE,
        latent_rep_dim=latent_rep_dim,
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
    )

    policy = IcmLSTMPolicy(
        device=DEVICE,
        state_dim=state_dim,
        action_dim=action_dim,
        latent_dim=latent_rep_dim,
        hidden_dim_encoder=1024,
        use_cnn_encoder=False
    )

    policy_optimizer = torch.optim.Adam([
            {"params": policy.encoder.encoder_model.parameters(), "lr": 1e-3},
            {"params": policy.parameters(), "lr": 1e-3},
        ])

    GAMMA = 0.99
    ENTROPY_COEF = 0.01
    VALUE_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    N_STEPS = 32

    global_step = 0

    hidden, c = policy.init_lstm_state(batch_size=1)

    TOTAL_TIMESTEPS = 500_000

    obs, _ = env.reset()
    obs = torch.from_numpy(obs).to(torch.float32).to(DEVICE)
    obs = obs.unsqueeze(0)
    while global_step < TOTAL_TIMESTEPS:
        obs_buf = []
        actions_buf = []
        rewards_buf = []
        values_buf = []
        logprob_buf = []
        h_buf, c_buf = [], []

        for _ in range(N_STEPS):
            probs, sample_one_hot, value, (hidden, c) = policy.forward(obs, (hidden, c))
            action = sample_one_hot.argmax(dim=-1).cpu().numpy()
            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

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
            mask = (1.0 - done_tensor)

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
        value_loss = torch.nn.functional.mse_loss(values_batch, returns_batch)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean() # type: ignore
        entropy_loss = -ENTROPY_COEF * entropy

        total_loss = policy_loss + VALUE_COEF * value_loss + entropy_loss
        torch.nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
        total_loss.backward()
        policy_optimizer.step()

        hidden = hidden.detach()
        c = c.detach()

        global_step += N_STEPS

        if global_step % (1000) == 0:
            print(f"Step {global_step:,} | policy loss {policy_loss.item():.4f} | "
                f"value loss {value_loss.item():.4f} | entropy {entropy.item():.4f}")

tmp()
