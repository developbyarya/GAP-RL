import pandas as pd
import numpy as np

# Load datasets
baseline_path = "/home/developbyarya/Projects/GAP-RL/runs/baseline_raw_20260908_072301_seed1029/progress.csv"
alsac_path = "/home/developbyarya/Projects/GAP-RL/runs/al-sac/sanity_20260909_044730/20260909_044730_sac8_state_egopoints_pd_ee_delta_pose_euler_YCB12_40_bezier2d_vary_AL_SAC_sanity_check_attn384/progress.csv"

try:
    df_base = pd.read_csv(baseline_path)
    df_al = pd.read_csv(alsac_path)
except Exception as e:
    print(f"Error loading csv: {e}")
    exit(1)

def process_df(df, name):
    # Ensure necessary columns exist, else fill with NaN
    cols = ["time/episodes", "rollout/success_rate", "train/actor_loss", "train/critic_loss", "train/std", "train/ent_coef"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    
    # Drop rows where episodes is NaN
    df = df.dropna(subset=["time/episodes"])
    df["ep_bin"] = (df["time/episodes"] // 2000) * 2000
    
    # Aggregate
    agg = df.groupby("ep_bin").agg(
        episodes_end=("time/episodes", "max"),
        mean_success_rate=("rollout/success_rate", "mean"),
        max_success_rate=("rollout/success_rate", "max"),
        mean_actor_loss=("train/actor_loss", "mean"),
        mean_critic_loss=("train/critic_loss", "mean"),
        mean_std=("train/std", "mean"),
        mean_ent_coef=("train/ent_coef", "mean"),
    ).reset_index()
    
    # Calculate differences (losses changed)
    agg["actor_loss_change"] = agg["mean_actor_loss"].diff()
    agg["critic_loss_change"] = agg["mean_critic_loss"].diff()
    agg["Model"] = name
    return agg

df_base_agg = process_df(df_base, "Baseline")
df_al_agg = process_df(df_al, "AL-SAC")

# Merge for comparison
# We can just concat and sort, or merge on ep_bin
merged = pd.concat([df_base_agg, df_al_agg]).sort_values(by=["ep_bin", "Model"])

markdown = f"# AL-SAC vs Baseline Comparison Report\n\n"
markdown += "This report aggregates the training logs into bins of 2000 episodes.\n\n"

# Create a Markdown table
cols = ["Model", "ep_bin", "mean_success_rate", "max_success_rate", "mean_actor_loss", "actor_loss_change", "mean_critic_loss", "critic_loss_change", "mean_std"]
headers = ["Model", "Episode Range", "Avg Success Rate", "Max Success Rate", "Actor Loss", "Actor Loss \u0394", "Critic Loss", "Critic Loss \u0394", "Avg Std"]

markdown += "| " + " | ".join(headers) + " |\n"
markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"

for _, row in merged.iterrows():
    ep_range = f"{int(row['ep_bin'])} - {int(row['ep_bin']+1999)}"
    
    def fmt(val, decimals=4):
        return f"{val:.{decimals}f}" if not pd.isna(val) else "-"
        
    line = [
        row["Model"],
        ep_range,
        fmt(row["mean_success_rate"], 3),
        fmt(row["max_success_rate"], 3),
        fmt(row["mean_actor_loss"], 1),
        fmt(row["actor_loss_change"], 1),
        fmt(row["mean_critic_loss"], 1),
        fmt(row["critic_loss_change"], 1),
        fmt(row["mean_std"], 4)
    ]
    markdown += "| " + " | ".join(line) + " |\n"

out_path = "/home/developbyarya/.gemini/antigravity-cli/brain/fac523b2-8d7e-44cb-81f3-84c1d446ad39/comparison_report.md"
with open(out_path, "w") as f:
    f.write(markdown)

print("Report generated.")
