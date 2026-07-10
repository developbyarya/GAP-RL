import argparse
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import seaborn as sns
import yaml

sns.set_theme(style="whitegrid", context="talk")

def load_data(log_dir: str):
    log_dir = Path(log_dir)
    progress_csv = log_dir / "progress.csv"
    monitor_csv = log_dir / "monitor.csv"
    config_yaml = log_dir / "config.yaml"

    if not progress_csv.exists():
        raise FileNotFoundError(f"No progress.csv in {log_dir}")

    df_progress = pd.read_csv(progress_csv, skipinitialspace=True)
    config = {}
    if config_yaml.exists():
        with open(config_yaml) as f:
            config = yaml.safe_load(f)

    df_monitor = None
    if monitor_csv.exists():
        df_monitor = pd.read_csv(monitor_csv, comment="#", names=["r", "l", "t"], skiprows=0)
        df_monitor = df_monitor[df_monitor["r"] != "r"].copy()
        for col in ["r", "l", "t"]:
            df_monitor[col] = pd.to_numeric(df_monitor[col], errors="coerce")
        df_monitor = df_monitor.dropna().reset_index(drop=True)

    return df_progress, df_monitor, config


def plot_training_curves(df_progress, save_dir, run_name=""):
    os.makedirs(save_dir, exist_ok=True)

    if "time/total_timesteps" not in df_progress.columns:
        print("WARNING: 'time/total_timesteps' not found. Trying 'total_timesteps'...")
        if "total_timesteps" in df_progress.columns:
            x_col = "total_timesteps"
        else:
            x_col = df_progress.columns[2]
            print(f"Using fallback column: {x_col}")
    else:
        x_col = "time/total_timesteps"

    train_cols = [
        ("rollout/ep_rew_mean", "Mean Episode Reward", "Reward"),
        ("rollout/success_rate", "Success Rate", "Success Rate"),
        ("rollout/ep_len_mean", "Episode Length", "Episode Length"),
    ]

    rollout_present = [name for name, _, _ in train_cols if name in df_progress.columns]
    if rollout_present:
        fig, axes = plt.subplots(len(rollout_present), 1, figsize=(14, 4 * len(rollout_present)), sharex=True)
        if len(rollout_present) == 1:
            axes = [axes]
        for ax, (col, title, ylabel) in zip(axes, train_cols):
            if col not in df_progress.columns:
                continue
            ax.plot(df_progress[x_col], df_progress[col], linewidth=1.5)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.set_xlabel("Timesteps")
        fig.suptitle(f"Training Rollout Metrics — {run_name}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(save_dir, "rollout_curves.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    loss_cols = [
        ("train/critic_loss", "Critic Loss"),
        ("train/actor_loss", "Actor Loss"),
        ("train/ent_coef", "Entropy Coefficient"),
    ]
    train_present = [c for c, _ in loss_cols if c in df_progress.columns]
    if train_present:
        fig, axes = plt.subplots(len(train_present), 1, figsize=(14, 4 * len(train_present)), sharex=True)
        if len(train_present) == 1:
            axes = [axes]
        for ax, (col, title) in zip(axes, loss_cols):
            if col not in df_progress.columns:
                continue
            ax.plot(df_progress[x_col], df_progress[col], linewidth=1.5, color="C1")
            ax.set_ylabel(title)
            ax.set_title(f"Training {title}")
            ax.set_xlabel("Timesteps")
        fig.suptitle(f"Training Losses — {run_name}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(save_dir, "training_losses.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    if "time/fps" in df_progress.columns:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(df_progress[x_col], df_progress["time/fps"], linewidth=1.5, color="C2")
        ax.set_ylabel("FPS")
        ax.set_title(f"Training Speed (FPS) — {run_name}")
        ax.set_xlabel("Timesteps")
        rolling = df_progress["time/fps"].rolling(window=50, min_periods=1).mean()
        ax.plot(df_progress[x_col], rolling, linewidth=2, color="C3", linestyle="--", label="Rolling Avg (50)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "fps_curve.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved plots to {save_dir}")


def plot_monitor_analysis(df_monitor, save_dir, run_name=""):
    os.makedirs(save_dir, exist_ok=True)
    if df_monitor is None:
        return

    df = df_monitor.copy()
    df["episode"] = range(len(df))
    df["cumulative_reward"] = df["r"].cumsum()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].plot(df["episode"], df["r"], linewidth=0.6, alpha=0.7, color="C0")
    rolling = df["r"].rolling(window=50, min_periods=1).mean()
    axes[0, 0].plot(df["episode"], rolling, linewidth=2, color="C1", label="Rolling Avg (50 eps)")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Episode Reward")
    axes[0, 0].set_title("Per-Episode Reward")
    axes[0, 0].legend()

    axes[0, 1].hist(df["r"], bins=80, color="C0", alpha=0.7, edgecolor="white")
    axes[0, 1].axvline(df["r"].mean(), color="C1", linestyle="--", linewidth=2, label=f'Mean: {df["r"].mean():.1f}')
    axes[0, 1].axvline(df["r"].median(), color="C2", linestyle="-.", linewidth=2, label=f'Median: {df["r"].median():.1f}')
    axes[0, 1].set_xlabel("Episode Reward")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Reward Distribution")
    axes[0, 1].legend()

    axes[1, 0].plot(df["t"] / 60, df["r"], linewidth=0.6, alpha=0.7, color="C2")
    rolling_t = df["r"].rolling(window=100, min_periods=1).mean()
    axes[1, 0].plot(df["t"] / 60, rolling_t, linewidth=2, color="C3", label="Rolling Avg (100 eps)")
    axes[1, 0].set_xlabel("Training Time (minutes)")
    axes[1, 0].set_ylabel("Episode Reward")
    axes[1, 0].set_title("Reward vs Training Time")
    axes[1, 0].legend()

    df["reward_ma50"] = df["r"].rolling(window=50, min_periods=1).mean()
    df["reward_std50"] = df["r"].rolling(window=50, min_periods=1).std()
    axes[1, 1].plot(df["episode"], df["reward_ma50"], linewidth=1.5, color="C0", label="Mean (50 eps)")
    axes[1, 1].fill_between(
        df["episode"],
        df["reward_ma50"] - df["reward_std50"],
        df["reward_ma50"] + df["reward_std50"],
        alpha=0.2, color="C0", label="±1 Std"
    )
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Episode Reward")
    axes[1, 1].set_title("Reward Trend with Variability")
    axes[1, 1].legend()

    fig.suptitle(f"Monitor Analysis — {run_name}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(save_dir, "monitor_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Total episodes: {len(df)}")
    print(f"Mean reward: {df['r'].mean():.2f} ± {df['r'].std():.2f}")
    print(f"Median reward: {df['r'].median():.2f}")
    print(f"Min reward: {df['r'].min():.2f}")
    print(f"Max reward: {df['r'].max():.2f}")
    print(f"Last 100 episodes mean: {df['r'].tail(100).mean():.2f} ± {df['r'].tail(100).std():.2f}")


def print_summary(df_progress, config):
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    if "rollout/ep_rew_mean" in df_progress.columns:
        rewards = df_progress["rollout/ep_rew_mean"].dropna()
        if len(rewards) > 0:
            print(f"Reward: {rewards.iloc[0]:.1f} → {rewards.iloc[-1]:.1f}")
            best_idx = rewards.idxmax()
            best_val = rewards.max()
            print(f"Best reward: {best_val:.1f} at step {best_idx}")

    if "rollout/success_rate" in df_progress.columns:
        success = df_progress["rollout/success_rate"]
        print(f"Final success rate: {success.iloc[-1]:.3f} ({success.iloc[-1]*100:.1f}%)")

    if "time/total_timesteps" in df_progress.columns:
        total_steps = df_progress["time/total_timesteps"].iloc[-1]
        print(f"Total timesteps: {int(total_steps):,}")

    if "time/fps" in df_progress.columns:
        fps = df_progress["time/fps"].dropna()
        if len(fps) > 0:
            print(f"Average FPS: {fps.mean():.0f}")

    if config:
        print(f"\nConfiguration:")
        for k, v in config.items():
            print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser(description="Analyze RL training results")
    parser.add_argument("log_dir", type=str, help="Path to experiment directory (e.g., 20260701_003718_sac4_...)")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save plots (default: <log_dir>/analysis)")
    parser.add_argument("--no-monitor", action="store_true", help="Skip monitor.csv analysis")

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Error: directory {log_dir} does not exist")
        sys.exit(1)

    save_dir = args.save_dir or str(log_dir / "analysis")
    run_name = log_dir.name

    print(f"Loading data from: {log_dir}")
    df_progress, df_monitor, config = load_data(str(log_dir))

    print(f"Progress.csv has {len(df_progress)} rows, columns: {list(df_progress.columns)}")
    if df_monitor is not None:
        print(f"Monitor.csv has {len(df_monitor)} episodes")

    print_summary(df_progress, config)

    print("\nGenerating plots...")
    plot_training_curves(df_progress, save_dir, run_name)
    if df_monitor is not None and not args.no_monitor:
        plot_monitor_analysis(df_monitor, save_dir, run_name)

    print(f"\nAll analysis saved to: {save_dir}")


if __name__ == "__main__":
    main()
