# Baseline approach - single run
import pandas as pd
from openpyxl import Workbook
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



def random_baseline(
    num_episodes=2000,
    max_steps_per_episode=15,
    log_interval=10,
):

    env = PinchingISACEnv(beta_snr=1.0)

    rewards = []
    snrs = []
    rates = []

    for ep in range(num_episodes):

        env.reset()

        ep_reward = 0.0
        snr_list = []
        rate_list = []

        for t in range(max_steps_per_episode):

            # -------- BAD RANDOM BASELINE (SAFE) --------
            p_rand = np.random.uniform(0, 0.05 * P0_max, size=M)

            x_rand = np.random.uniform(0, AREA_MAX_X, size=N_pinching)
            x_rand = project_positions_min_spacing(x_rand, delta_min, AREA_MAX_X)

            # Convert back to (-1, 1) action
            a_x = (x_rand / AREA_MAX_X) * 2 - 1
            a_p = (p_rand / P0_max) * 2 - 1

            action = np.concatenate([a_x, a_p])
            # --------------------------------------------

            obs, reward, done, info = env.step(action)

            ep_reward += reward
            snr_list.append(info["slot_snr"])
            rate_list.append(info["slot_rate"])

            if done:
                break

        rewards.append(ep_reward)
        snrs.append(np.mean(snr_list) if len(snr_list) else 0)
        rates.append(np.mean(rate_list) if len(rate_list) else 0)

        if (ep + 1) % log_interval == 0:
            print(f"[Random Baseline] Ep {ep+1} | "
                  f"AvgR={np.mean(rewards[-log_interval:]):.3f} | "
                  f"AvgSNR={np.mean(snrs[-log_interval:]):.3f} | "
                  f"AvgRate={np.mean(rates[-log_interval:]):.3f}")

    # ===========================================
    # ---------- SAVE TO EXCEL FILE -------------
    # ===========================================

    df_reward = pd.DataFrame({"Reward": rewards})
    df_snr = pd.DataFrame({"SNR": snrs})
    df_rate = pd.DataFrame({"Rate": rates})

    with pd.ExcelWriter("RandomBaseline_results.xlsx", engine="openpyxl") as writer:
        df_reward.to_excel(writer, sheet_name="EpisodeReward", index=True)
        df_snr.to_excel(writer, sheet_name="EpisodeSNR", index=True)
        df_rate.to_excel(writer, sheet_name="EpisodeRate", index=True)

    print("✔ Random baseline results saved to RandomBaseline_results.xlsx")

    return rewards,snrs,rates

if __name__ == "__main__":

    rewards, snrs, rates = random_baseline(
            num_episodes=2000,
            max_steps_per_episode=15,
            log_interval=10
        )

    print("Finished running random baseline.")
