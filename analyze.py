import pandas as pd
import numpy as np

# Load data
df_test = pd.read_csv('runs/runs/reward_ablation_20260831_043757_iew0.3/progress.csv')
df_control = pd.read_csv('runs/runs/reward_ablation_20260831_064136_iew3.0/progress3-0.csv')

def get_stats(df, name):
    res = []
    res.append(f"### Run: {name}")
    
    # Check train/std (first 50k steps and overall)
    df_50k = df[df['time/total_timesteps'] <= 50000]
    std_50k = df_50k['train/std'].mean() if 'train/std' in df_50k else np.nan
    std_all = df['train/std'].mean() if 'train/std' in df else np.nan
    res.append(f"- **train/std (first 50k steps)**: {std_50k:.4f}")
    res.append(f"- **train/std (overall)**: {std_all:.4f}")

    # Success rate
    max_sr = df['rollout/success_rate'].max() * 100
    final_sr = df['rollout/success_rate'].iloc[-1] * 100
    res.append(f"- **rollout/success_rate (max)**: {max_sr:.2f}%")
    res.append(f"- **rollout/success_rate (final)**: {final_sr:.2f}%")

    # Reward components
    mean_info = df['reward/info_exist_reward'].mean()
    mean_app = df['reward/approach_reward'].mean()
    mean_grasp = df['reward/grasp_reward'].mean()
    mean_goal = df['reward/goal_reward'].mean()
    mean_static = df['reward/static_reward'].mean()
    
    total_rew = mean_info + mean_app + mean_grasp + mean_goal + mean_static
    
    res.append(f"- **reward/info_exist_reward**: {mean_info:.3f} ({mean_info/total_rew*100:.1f}%)")
    res.append(f"- **reward/approach_reward**: {mean_app:.3f} ({mean_app/total_rew*100:.1f}%)")
    res.append(f"- **reward/grasp_reward**: {mean_grasp:.3f} ({mean_grasp/total_rew*100:.1f}%)")
    
    # Entropy coef
    ent_coef = df['train/ent_coef'].mean()
    res.append(f"- **train/ent_coef (overall)**: {ent_coef:.5f}")
    
    res.append("")
    return "\n".join(res)

with open('docs/reward_diagnosis.md', 'w') as f:
    f.write("# Reward Ablation Diagnosis Results\n\n")
    f.write("We compared two training runs of ~300k timesteps to diagnose the `info_exist_reward` collapse hypothesis.\n\n")
    f.write(get_stats(df_control, "Control (info-exist-weight = 3.0)"))
    f.write(get_stats(df_test, "Test (info-exist-weight = 0.3)"))
