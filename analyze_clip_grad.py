import pandas as pd
import numpy as np

path = "/home/developbyarya/Projects/GAP-RL/runs/20260909_163641_sac8_state_egopoints_pd_ee_delta_pose_euler_YCB12_40_bezier2d_vary_AL_SAC_clip_grad_attn384/progress.csv"
df = pd.read_csv(path)

# Ensure columns exist
cols = ["time/episodes", "time/total_timesteps", "rollout/success_rate", "train/actor_loss", "train/critic_loss", "train/ent_coef_loss"]
for c in cols:
    if c not in df.columns:
        df[c] = np.nan

df = df.dropna(subset=["time/episodes", "rollout/success_rate"])

max_steps = df["time/total_timesteps"].max()
max_eps = df["time/episodes"].max()

success_df = df[df["rollout/success_rate"] > 0]
if len(success_df) > 0:
    first_success_step = success_df.iloc[0]["time/total_timesteps"]
    first_success_ep = success_df.iloc[0]["time/episodes"]
    max_success = df["rollout/success_rate"].max()
    max_success_step = df.loc[df["rollout/success_rate"].idxmax()]["time/total_timesteps"]
else:
    first_success_step = "N/A"
    first_success_ep = "N/A"
    max_success = 0.0
    max_success_step = "N/A"

max_critic = df["train/critic_loss"].max()
min_critic = df["train/critic_loss"].min()
final_critic = df["train/critic_loss"].iloc[-1]

print(f"--- AL-SAC with Gradient Clipping Run Summary ---")
print(f"Total Timesteps: {max_steps}")
print(f"Total Episodes: {max_eps}")
print(f"First Success: at timestep {first_success_step} (episode {first_success_ep})")
print(f"Max Success Rate: {max_success} at timestep {max_success_step}")
print(f"Critic Loss Spike (Max): {max_critic:.2f}")
print(f"Critic Loss (Final): {final_critic:.2f}")

# Group by 500k timesteps to see trends
print("\nTrend over time (per 500k timesteps):")
df["step_bin"] = (df["time/total_timesteps"] // 500000) * 500000
trend = df.groupby("step_bin").agg({
    "rollout/success_rate": ["mean", "max"],
    "train/critic_loss": "max",  # to see if spikes are controlled
    "train/actor_loss": "mean",
    "train/ent_coef_loss": "mean"
})
print(trend)

