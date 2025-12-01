import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
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





class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_dim, act_dim)
        self.log_std_layer = nn.Linear(hidden_dim, act_dim)

        self.LOG_STD_MIN = -20.0
        self.LOG_STD_MAX = 2.0

    def forward(self, obs):
        x = self.net(obs)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, obs):
        mean, _ = self.forward(obs)
        action = torch.tanh(mean)
        return action


class QNetwork(nn.Module):
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


class MERLAgent:
    def __init__(
        self,
        obs_dim,
        act_dim,
        gamma=0.99,
        rho=0.2,
        lr=1e-5,
        tau=0.005,
        buffer_size=500_000,
        batch_size=256,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.rho = rho
        self.tau = tau
        self.batch_size = batch_size

        self.policy = GaussianPolicy(obs_dim, act_dim).to(device)
        self.q1 = QNetwork(obs_dim, act_dim).to(device)
        self.q2 = QNetwork(obs_dim, act_dim).to(device)

        self.q1_target = QNetwork(obs_dim, act_dim).to(device)
        self.q2_target = QNetwork(obs_dim, act_dim).to(device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy_opt = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=lr)

        self.replay = ReplayBuffer(buffer_size)

    def select_action(self, obs_np, deterministic=False):
        obs = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        if deterministic:
            with torch.no_grad():
                a = self.policy.deterministic(obs)
        else:
            with torch.no_grad():
                a, _ = self.policy.sample(obs)
        return a.cpu().numpy()[0]

    def update(self):
        if len(self.replay) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)

            q1_next = self.q1_target(next_states, next_actions)
            q2_next = self.q2_target(next_states, next_actions)
            q_next_min = torch.min(q1_next, q2_next)

            y = rewards + self.gamma * (1.0 - dones) * (
                q_next_min - self.rho * next_log_probs
            )

        q1_pred = self.q1(states, actions)
        q2_pred = self.q2(states, actions)

        q1_loss = nn.MSELoss()(q1_pred, y)
        q2_loss = nn.MSELoss()(q2_pred, y)
        q_loss = q1_loss + q2_loss

        self.q1_opt.zero_grad()
        self.q2_opt.zero_grad()
        q_loss.backward()
        self.q1_opt.step()
        self.q2_opt.step()

        new_actions, log_probs = self.policy.sample(states)
        q1_new = self.q1(states, new_actions)
        q2_new = self.q2(states, new_actions)
        q_new_min = torch.min(q1_new, q2_new)

        policy_loss = (self.rho * log_probs - q_new_min).mean()

        self.policy_opt.zero_grad()
        policy_loss.backward()
        self.policy_opt.step()

        with torch.no_grad():
            for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


def train_merl(
    num_episodes=2000,
    max_steps_per_episode=T_slots,
    beta_snr=1.0,
    rho=0.2,
    lr=1e-5,
    log_interval=10,
):
    env = PinchingISACEnv(beta_snr=beta_snr)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = MERLAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        gamma=0.99,
        rho=rho,
        lr=lr,
        tau=0.01,
        buffer_size=500_000,
        batch_size=256,
    )

    episode_rewards = []
    episode_snr = []

    for ep in range(num_episodes):
        obs = env.reset()
        ep_reward = 0.0
        snr_list = []
        rate_list = []

        for step in range(max_steps_per_episode):
            action = agent.select_action(obs, deterministic=False)
            next_obs, reward, done, info = env.step(action)

            agent.replay.push(obs, action, reward, next_obs, done)
            agent.update()

            obs = next_obs
            ep_reward += reward
            snr_list.append(info["slot_snr"])
            rate_list.append(info["slot_rate"])

            if done:
                break

        episode_rewards.append(ep_reward)
        episode_snr.append(np.mean(snr_list))

        if (ep + 1) % log_interval == 0:
            print(
                f"Episode {ep + 1} "
                f"Reward {np.mean(episode_rewards[-log_interval:]):.3f} "
                f"SNR {np.mean(episode_snr[-log_interval:]):.3f}"
            )

    return agent, episode_rewards, episode_snr


def plot_rewards(episode_rewards):
    plt.figure(figsize=(8, 5))
    plt.plot(episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward over Episodes")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    agent_merl, rew_merl, snr_merl = train_merl(
        num_episodes=2000,
        beta_snr=1.0,
        rho=0.2,
        lr=1e-5,
    )
    plot_rewards(rew_merl)