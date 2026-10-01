import os
import glob
import pandas as pd
import numpy as np

runs_dir = "/home/developbyarya/Projects/GAP-RL/runs"
run_folders = [f.path for f in os.scandir(runs_dir) if f.is_dir()]

data = []

for run in run_folders:
    run_name = os.path.basename(run)
    progress_file = os.path.join(run, "progress.csv")
    
    if not os.path.exists(progress_file):
        continue
        
    try:
        df = pd.read_csv(progress_file)
    except Exception as e:
        print(f"Failed to read {progress_file}: {e}")
        continue
        
    # Get max timesteps
    if 'time/total_timesteps' in df.columns:
        max_steps = df['time/total_timesteps'].max()
    else:
        max_steps = 0
        
    # Get max eval success rate
    max_eval_success = 0.0
    if 'eval/success_rate' in df.columns:
        eval_df = df.dropna(subset=['eval/success_rate'])
        if not eval_df.empty:
            max_eval_success = eval_df['eval/success_rate'].max()
            
    # Get max rollout reward
    max_reward = 0.0
    if 'rollout/ep_rew_mean' in df.columns:
        rew_df = df.dropna(subset=['rollout/ep_rew_mean'])
        if not rew_df.empty:
            max_reward = rew_df['rollout/ep_rew_mean'].max()
            
    # Get critic loss at 1M steps
    critic_loss_1m = None
    if 'train/critic_loss' in df.columns and 'time/total_timesteps' in df.columns:
        loss_df = df.dropna(subset=['train/critic_loss'])
        around_1m = loss_df[(loss_df['time/total_timesteps'] >= 990000) & (loss_df['time/total_timesteps'] <= 1010000)]
        if not around_1m.empty:
            # Get the closest to 1,000,000
            closest = around_1m.iloc[(around_1m['time/total_timesteps'] - 1000000).abs().argsort()[:1]]
            critic_loss_1m = closest['train/critic_loss'].values[0]
            
    # Look for ycb_eval generalization results
    eval_dir = glob.glob(os.path.join(run, "simLoG_result_onlyhandcam_ycb_eval_*"))
    generalization_success = None
    if eval_dir:
        bezier_eval_file = os.path.join(eval_dir[0], "bezier2d", "success_rates_steps.txt")
        if os.path.exists(bezier_eval_file):
            try:
                with open(bezier_eval_file, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        if "success rates (mean, std):" in line:
                            # format: success rates (mean, std): (0.455, 0.264)
                            parts = line.split("(")[-1].split(")")[0]
                            mean_val = float(parts.split(",")[0].strip())
                            generalization_success = mean_val
            except Exception as e:
                print(f"Error parsing {bezier_eval_file}: {e}")
                
    # Classify the experiment type based on folder name
    exp_type = "Unknown"
    if "baseline" in run_name.lower():
        exp_type = "Baseline (No Aux)"
    elif "al_sac" in run_name.lower() or "al-sac" in run_name.lower():
        exp_type = "AL-SAC (Decaying Aux)"
    elif "unshared" in run_name.lower():
        exp_type = "Exp 4: Unshared Trunk + Shrink&Perturb"
    elif "twin_arch" in run_name.lower():
        exp_type = "Exp 3: TwinSAC"
    elif "experiment1" in run_name.lower():
        # Distinguish the lucky seed vs dead seeds based on max success
        if max_eval_success > 0.1:
            exp_type = "Exp 1: Frozen Aux + Shrink&Perturb (Lucky Seed)"
        else:
            exp_type = "Exp 1: Frozen Aux (Dead Seed)"
            
    data.append({
        "Experiment Name": run_name,
        "Experiment Type": exp_type,
        "Total Steps": max_steps,
        "Max Train Reward": max_reward,
        "Max Eval Success (Train Set)": max_eval_success,
        "YCB Generalization Success (Test Set)": generalization_success,
        "Critic Loss @ 1M Steps": critic_loss_1m
    })

df_out = pd.DataFrame(data)
df_out = df_out.sort_values(by="Experiment Type")
output_path = "/home/developbyarya/Projects/GAP-RL/runs/GAP_RL_Experiments_Summary.xlsx"
df_out.to_excel(output_path, index=False)
print(f"Saved excel to {output_path}")
