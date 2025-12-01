
import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from copy import deepcopy
from typing import Tuple
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==========================
# System parameters
# ==========================

c = 3e8
fc = 28e9
lam = c / fc
alpha_const = c / (4 * np.pi * fc)

n_eff = 1.4
lam0 = lam / n_eff

sigma2_dBm = -90
sigma2 = 10 ** ((sigma2_dBm - 30) / 10.0)

Gamma_th_dB = 10
Gamma_th = 10 ** (Gamma_th_dB / 10.0)

delta_min = lam / 2.0

M = 6
N_pinching = 3
K = 1

T_slots = 15
Delta_t = 1.0

d_ant_height = 3.0
pin_feed_point_x = 0.0

AREA_MAX_X = 150.0
AREA_MAX_Y = 150.0

P0_dBm = 20.0
P0_max = 10 ** ((P0_dBm - 30) / 10.0)

E_max = 6.0

UE_positions = np.array([
    [33.6309,  89.9410, 0.0],
    [39.6081,  17.1488, 0.0],
    [146.6042, 121.5036, 0.0],
    [148.9406, 34.7796, 0.0],
    [149.2488, 148.6565, 0.0],
    [85.7213,  82.8948, 0.0],
], dtype=np.float64)

Target_positions = np.array([
    [65.8152, 13.0490, 0.0],
], dtype=np.float64)


def project_positions_min_spacing(pos, delta, x_max):
    N = len(pos)
    idx = np.argsort(pos)
    p_sorted = np.clip(pos[idx], 0.0, x_max)

    for i in range(1, N):
        if p_sorted[i] - p_sorted[i - 1] < delta:
            p_sorted[i] = p_sorted[i - 1] + delta

    for i in range(N - 2, -1, -1):
        if p_sorted[i + 1] - p_sorted[i] < delta:
            p_sorted[i] = p_sorted[i + 1] - delta

    p_sorted = np.clip(p_sorted, 0.0, x_max)

    proj = np.zeros_like(pos)
    proj[idx] = p_sorted
    return proj


def compute_rate_and_snr_slot(p_m_t, x_pin_t):
    epsd = 1e-9

    pin_pos = np.vstack([
        x_pin_t,
        np.zeros(N_pinching),
        d_ant_height * np.ones(N_pinching),
    ])

    dist_from_feed = np.abs(x_pin_t - pin_feed_point_x)
    theta_n = 2 * np.pi * dist_from_feed / lam0

    total_rate = 0.0
    Hm_vals = np.zeros(M, dtype=np.complex128)

    for m in range(M):
        user_pos = UE_positions[m, :].reshape(3, 1)
        dist_mn = np.linalg.norm(user_pos - pin_pos, axis=0) + epsd

        H_m = np.sum(
            alpha_const
            * np.exp(-1j * 2 * np.pi / lam * dist_mn) / dist_mn
            * np.exp(-1j * theta_n)
        )
        Hm_vals[m] = H_m

        p_m = p_m_t[m]
        R_m = (1.0 / M) * np.log2(1.0 + (np.abs(H_m) ** 2 * p_m) / (N_pinching * sigma2))
        total_rate += np.real(R_m)

    SNR_targets = np.zeros(K)
    for k in range(K):
        tgt_pos = Target_positions[k, :].reshape(3, 1)
        dist_kn = np.linalg.norm(tgt_pos - pin_pos, axis=0) + epsd

        H_k = np.sum(
            alpha_const
            * np.exp(-1j * 2 * np.pi / lam * dist_kn) / dist_kn
            * np.exp(-1j * theta_n)
        )

        SNR_m = np.zeros(M)
        for m in range(M):
            p_m = p_m_t[m]
            num = np.abs(H_k) ** 2 * p_m / N_pinching
            denom = np.abs(Hm_vals[m]) ** 2 * p_m / N_pinching + sigma2
            SNR_m[m] = num / denom

        SNR_targets[k] = np.max(SNR_m)

    return total_rate, SNR_targets


class PinchingISACEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, beta_snr=1.0):
        super().__init__()

        self.beta_snr = beta_snr
        self.max_time = T_slots
        self.area_x = AREA_MAX_X

        obs_dim = (
            1 + 1 + N_pinching + M * 3 + K * 3
        )

        self.action_dim = N_pinching + M

        self.observation_space = spaces.Box(
            low=-np.ones(obs_dim, dtype=np.float32),
            high=np.ones(obs_dim, dtype=np.float32),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-np.ones(self.action_dim, dtype=np.float32),
            high=np.ones(self.action_dim, dtype=np.float32),
            dtype=np.float32,
        )

        self._ue_flat = UE_positions.reshape(-1)
        self._tgt_flat = Target_positions.reshape(-1)

        self.reset()

    def reset(self):
        self.t = 0
        self.energy_used = 0.0

        base_lin = np.linspace(10.0, self.area_x - 10.0, N_pinching)
        xpin0 = base_lin + np.random.randn(N_pinching) * 1.0
        xpin0 = np.clip(xpin0, 0.0, self.area_x)
        xpin0 = project_positions_min_spacing(xpin0, delta_min, self.area_x)
        self.x_pin_t = xpin0.copy()

        return self._get_obs()

    def _get_obs(self):
        time_norm = self.t / (self.max_time - 1.0)
        energy_remaining = max(E_max - self.energy_used, 0.0)
        energy_norm = energy_remaining / E_max
        xpin_norm = self.x_pin_t / self.area_x

        obs = np.concatenate([
            np.array([time_norm], dtype=np.float64),
            np.array([energy_norm], dtype=np.float64),
            xpin_norm.astype(np.float64),
            self._ue_flat,
            self._tgt_flat,
        ])

        obs_mapped = 2.0 * obs - 1.0
        return obs_mapped.astype(np.float32)

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        a_x = action[:N_pinching]
        a_p = action[N_pinching:]

        x_pin_t = (a_x + 1.0) / 2.0 * self.area_x
        x_pin_t = project_positions_min_spacing(x_pin_t, delta_min, self.area_x)

        p_m_t = (a_p + 1.0) / 2.0 * P0_max

        total_power = np.sum(p_m_t)
        remaining_energy = E_max - self.energy_used
        max_power_sum = remaining_energy / Delta_t

        if total_power * Delta_t > remaining_energy and total_power > 1e-9:
            scale = max_power_sum / total_power
            p_m_t = p_m_t * scale
            total_power = np.sum(p_m_t)

        slot_rate, snr_targets = compute_rate_and_snr_slot(p_m_t, x_pin_t)
        Gamma_k_t = float(snr_targets[0])

        reward = slot_rate + self.beta_snr * (Gamma_k_t - Gamma_th)

        self.x_pin_t = x_pin_t
        self.energy_used += total_power * Delta_t
        self.t += 1

        done = False
        if self.t >= self.max_time:
            done = True
        if self.energy_used >= E_max - 1e-6:
            done = True

        obs = self._get_obs()

        info = {
            "slot_rate": slot_rate,
            "slot_snr": Gamma_k_t,
            "energy_used": self.energy_used,
            "x_pin": x_pin_t.copy(),
            "p_m": p_m_t.copy(),
        }

        return obs, float(reward), done, info

    def render(self, mode="human"):
        pass


# --- Reusing classes from first cell ---
class Actor(nn.Module):
    """
    Renamed GaussianPolicy to Actor for consistency with DDPG/TD3 nomenclature.
    """
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh() # Actions are typically in [-1, 1] for DDPG/TD3
        )

    def forward(self, obs):
        return self.net(obs)


class Critic(nn.Module):
    """
    Renamed QNetwork to Critic for consistency with DDPG/TD3 nomenclature.
    """
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, float(done)))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, d = map(np.array, zip(*batch))
        return (
            torch.tensor(s, dtype=torch.float32, device=device),
            torch.tensor(a, dtype=torch.float32, device=device),
            torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(-1),
            torch.tensor(s_next, dtype=torch.float32, device=device),
            torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(-1),
        )

    def __len__(self):
        return len(self.buffer)


# OUNoise class (for optional Ornstein-Uhlenbeck exploration noise)
# Not strictly necessary for this run as `use_ou` is False, but included for completeness if desired later.
class OUNoise:
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

# --- End reusing classes ---

class TwinCritic(nn.Module):
    """
    Helper wrapper that contains two independent critics.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.q1 = Critic(obs_dim, act_dim, hidden_dim)
        self.q2 = Critic(obs_dim, act_dim, hidden_dim)

        # apply weight init to both (Critic class already applies weight_init)
        # but ensure both moved to device outside

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q1 = self.q1(obs, act)
        q2 = self.q2(obs, act)
        return q1, q2

class TD3Agent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        actor_lr: float = 1e-6,
        critic_lr: float = 5e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 500_000,
        batch_size: int = 256,
        hidden_dim: int = 256,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        exploration_noise: float = 0.1,
        use_ou: bool = False,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Actor (policy)
        self.actor = Actor(obs_dim, act_dim, hidden_dim).to(device)
        self.actor_target = deepcopy(self.actor).to(device)

        # Twin critics + targets
        self.critic = TwinCritic(obs_dim, act_dim, hidden_dim).to(device)
        self.critic_target = TwinCritic(obs_dim, act_dim, hidden_dim).to(device)
        # copy weights to targets
        self.critic_target.q1.load_state_dict(self.critic.q1.state_dict())
        self.critic_target.q2.load_state_dict(self.critic.q2.state_dict())

        # optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        # critic optimizer takes parameters of both critics
        self.critic_opt = optim.Adam(
            list(self.critic.q1.parameters()) + list(self.critic.q2.parameters()),
            lr=critic_lr,
        )

        # replay and exploration
        self.replay = ReplayBuffer(buffer_size)
        self.use_ou = use_ou
        if use_ou:
            # reuse OUNoise from DDPG code
            self.ou = OUNoise(act_dim, sigma=exploration_noise)
        else:
            self.ou = None
        self.exploration_noise = exploration_noise

        # TD3 specifics
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.total_it = 0

    def select_action(self, obs: np.ndarray, noise: bool = True) -> np.ndarray:
        """
        obs: numpy array state (already appropriate dtype)
        Returns action in [-1,1]^act_dim
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            a = self.actor(obs_t).cpu().numpy()[0]

        if noise:
            if self.use_ou and self.ou is not None:
                a = a + self.ou.sample()
            else:
                a = a + np.random.normal(0, self.exploration_noise, size=self.act_dim)

        return np.clip(a, -1.0, 1.0)

    def store_transition(self, s, a, r, s_next, done):
        self.replay.push(s, a, r, s_next, done)

    def update(self):
        """
        One update step per call (call after storing transition).
        Implements TD3 update:
        - update critics every step
        - update actor and target networks every policy_delay steps
        """
        if len(self.replay) < self.batch_size:
            return None, None

        self.total_it += 1

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        # -------------------------
        # Compute target actions with smoothing noise (policy smoothing)
        # -------------------------
        with torch.no_grad():
            # target policy action
            next_actions = self.actor_target(next_states)

            # add clipped noise to target (policy smoothing)
            noise = (torch.randn_like(next_actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (next_actions + noise).clamp(-1.0, 1.0)

            # compute target Q values from target twin critics
            q1_next = self.critic_target.q1(next_states, next_actions)
            q2_next = self.critic_target.q2(next_states, next_actions)
            q_next_min = torch.min(q1_next, q2_next)

            q_target = rewards + self.gamma * (1.0 - dones) * q_next_min

        # -------------------------
        # Critic update (both critics)
        # -------------------------
        q1_pred = self.critic.q1(states, actions)
        q2_pred = self.critic.q2(states, actions)

        critic_loss = nn.MSELoss()(q1_pred, q_target) + nn.MSELoss()(q2_pred, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # -------------------------
        # Delayed policy update
        # -------------------------
        actor_loss_val = None
        if self.total_it % self.policy_delay == 0:
            # actor loss: maximize Q (use q1)
            actor_actions = self.actor(states)
            actor_loss = -self.critic.q1(states, actor_actions).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # soft updates for targets
            self._soft_update(self.actor, self.actor_target)
            # soft update both critic targets
            for p, tp in zip(self.critic.q1.parameters(), self.critic_target.q1.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
            for p, tp in zip(self.critic.q2.parameters(), self.critic_target.q2.parameters()):
                tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

            actor_loss_val = actor_loss.item()

        return actor_loss_val, critic_loss.item()

    def _soft_update(self, net, target_net):
        for p, tp in zip(net.parameters(), target_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pth"))
        torch.save(self.critic.q1.state_dict(), os.path.join(path, "critic_q1.pth"))
        torch.save(self.critic.q2.state_dict(), os.path.join(path, "critic_q2.pth"))

    def load(self, path):
        self.actor.load_state_dict(torch.load(os.path.join(path, "actor.pth"), map_location=device))
        self.critic.q1.load_state_dict(torch.load(os.path.join(path, "critic_q1.pth"), map_location=device))
        self.critic.q2.load_state_dict(torch.load(os.path.join(path, "critic_q2.pth"), map_location=device))
        # copy to targets
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.q1.load_state_dict(self.critic.q1.state_dict())
        self.critic_target.q2.load_state_dict(self.critic.q2.state_dict())


# Training loop for TD3 
def train_td3(
    env,
    num_episodes: int = 2000,
    max_steps_per_episode: int = 100,
    seed: int = 42,
    actor_lr: float = 1e-6,
    critic_lr: float = 5e-4,
    gamma: float = 0.99,
    tau: float = 0.005,
    batch_size: int = 256,
    buffer_size: int = 500_000,
    policy_noise: float = 0.2,
    noise_clip: float = 0.5,
    policy_delay: int = 2,
    exploration_noise: float = 0.1,
    use_ou: bool = False,
    save_dir: str = "td3_checkpoints",
    log_interval: int = 10,
):
    # reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = TD3Agent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
        gamma=gamma,
        tau=tau,
        buffer_size=buffer_size,
        batch_size=batch_size,
        policy_noise=policy_noise,
        noise_clip=noise_clip,
        policy_delay=policy_delay,
        exploration_noise=exploration_noise,
        use_ou=use_ou,
    )

    rewards_history = []
    snr_history = []
    actor_losses = []
    critic_losses = []
    rate_history = []

    for ep in range(1, num_episodes + 1):
        obs = env.reset()
        ep_reward = 0.0
        ep_snr = []
        ep_rate = []
        if agent.use_ou and agent.ou is not None:
            agent.ou.reset()

        for step in range(max_steps_per_episode):
            action = agent.select_action(obs, noise=True)
            next_obs, reward, done, info = env.step(action)

            agent.store_transition(obs, action, reward, next_obs, done)
            a_loss, c_loss = agent.update()

            obs = next_obs
            ep_reward += reward
            ep_snr.append(info.get("slot_snr", 0.0))
            ep_rate.append(info.get("slot_rate", 0.0))

            if a_loss is not None:
                actor_losses.append(a_loss)
            if c_loss is not None:
                critic_losses.append(c_loss)

            if done:
                break

        rewards_history.append(ep_reward)
        snr_history.append(np.mean(ep_snr) if len(ep_snr) > 0 else 0.0)
        rate_history.append(sum(ep_rate))

        if ep % log_interval == 0 or ep == 1:
            avg_r = np.mean(rewards_history[-log_interval:])
            avg_snr = np.mean(snr_history[-log_interval:]) if len(snr_history) > 0 else 0.0
            print(f"EP {ep:4d} | AvgR {avg_r:.3f} | AvgSNR {avg_snr:.3f} | Replay {len(agent.replay)}")

        if ep % 200 == 0:
            agent.save(os.path.join(save_dir, f"ep{ep}"))

    agent.save(os.path.join(save_dir, "final"))

    return agent, rewards_history, snr_history, actor_losses, critic_losses, rate_history


if __name__ == "__main__":
    try:
        from pinching_env import PinchingISACEnv
    except Exception:
        try:
            PinchingISACEnv
        except NameError:
            raise RuntimeError("PinchingISACEnv not found. Ensure environment is imported correctly.")

    env = PinchingISACEnv(beta_snr=1.0)

    agent_td3, rew_td3, snr_td3, a_losses, c_losses, rates = train_td3(
        env,
        num_episodes=2000,
        max_steps_per_episode=env.max_time,
        seed=123,
        actor_lr=1e-6,
        critic_lr=5e-4,
        gamma=0.99,
        tau=0.01,
        batch_size=256,
        buffer_size=500_000,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
        exploration_noise=0.1,
        use_ou=False,
        save_dir="td3_checkpoints",
        log_interval=10,
    )

    # plotting
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(rew_td3)
        plt.title("TD3 Episode Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.show()
    except Exception:
        pass