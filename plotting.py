import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Helper: Smooth the curves
# -----------------------------
def smooth_curve(values, window=20):
    if window < 2:
        return values
    return np.convolve(values, np.ones(window)/window, mode='same')


# -----------------------------
# Load Excel files
# -----------------------------
files = {
    "MERL": "MERL_results.xlsx",
    "TD3": "TD3_results.xlsx",
    "DDPG": "DDPG_results.xlsx",
    "Random": "RandomBaseline_results.xlsx",
}

colors = {
    "MERL": "blue",
    "TD3": "red",
    "DDPG": "green",
    "Random": "orange",
}

# Sheets are the same for all
sheet_reward = "EpisodeReward"
sheet_snr = "EpisodeSNR"
sheet_rate = "EpisodeRate"


# =====================================
# ⿡ REWARD PLOT
# =====================================
plt.figure(figsize=(9, 5))

for name, filename in files.items():
    df = pd.read_excel(filename, sheet_name=sheet_reward)
    rewards = smooth_curve(df["Reward"].values)
    plt.plot(rewards, label=name, color=colors[name], linewidth=2)

plt.xlabel("Episode")
plt.ylabel("Reward")
#plt.title("Reward vs Episode (All Approaches)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# =====================================
# ⿢ SNR PLOT
# =====================================
plt.figure(figsize=(9, 5))

for name, filename in files.items():
    df = pd.read_excel(filename, sheet_name=sheet_snr)
    snr = smooth_curve(df["SNR"].values)
    plt.plot(snr, label=name, color=colors[name], linewidth=2)

plt.xlabel("Episode")
plt.ylabel("SNR (dB)")
#plt.title("SNR vs Episode (All Approaches)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# =====================================
# ⿣ RATE PLOT
# =====================================
plt.figure(figsize=(9, 5))

for name, filename in files.items():
    df = pd.read_excel(filename, sheet_name=sheet_rate)
    rate = smooth_curve(df["Rate"].values)
    plt.plot(rate, label=name, color=colors[name], linewidth=2)

plt.xlabel("Episode")
plt.ylabel("Rate (bps/Hz)")
#plt.title("Rate vs Episode (All Approaches)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()