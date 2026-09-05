#!/usr/bin/env python3
"""
Clean script to train and evaluate the RAW baseline GAP-RL model.
This script removes frame stacking and LSTM (by using standard SAC and standard config)
and trains for the default 2M steps. It then automatically evaluates the final model,
retaining the new failure evaluation metrics.

Usage:
    cd gap_rl/algorithms/scripts && \\
    python ../../scripts/train_baseline.py --total-timesteps 2000000 --seed 1029
"""

import os
import sys
import time
import argparse
import subprocess

import numpy as np
import yaml
import torch

# Ensure repo root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

ALGO_SCRIPTS = os.path.join(REPO_ROOT, "gap_rl", "algorithms", "scripts")
if ALGO_SCRIPTS not in sys.path:
    sys.path.insert(0, ALGO_SCRIPTS)

from gap_rl import ALGORITHM_DIR
from gap_rl.envs import *  # noqa: F401,F403
from gap_rl.utils.common import setup_seed
from gap_rl.algorithms.rl_utils import sb3_make_multienv
from gap_rl.algorithms.rl_utils import (
    CustomGraspPointGroupExtractor,
    CustomGraspExtractor,
    CustomObjPNExtractor,
)
from reward_component_logger import RewardComponentCallback

from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from typing import Dict, Type

extractor_aliases: Dict[str, Type[BaseFeaturesExtractor]] = {
    "state_grasp9d": CustomGraspExtractor,
    "state_egopoints": CustomGraspPointGroupExtractor,
    "state_eogkeypoints": CustomGraspPointGroupExtractor,
    "state_objpoints_rt": CustomObjPNExtractor,
}

def main():
    parser = argparse.ArgumentParser(description="Baseline Training for GAP-RL")
    # By default, use the clean egopoints_ur85_bezier2d config which has goal_aux=False
    parser.add_argument("--config-name", type=str, default="egopoints_ur85_bezier2d")
    parser.add_argument("--log-std-init", type=float, default=-3.67,
                        help="Initial value for gSDE log_std (default: -3.67)")
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=1029,
                        help="Random seed (default: 1029)")
    parser.add_argument("--eval", action="store_true", default=True,
                        help="Automatically run evaluation after training")
    args = parser.parse_args()

    np.set_printoptions(suppress=True, precision=4)

    # ---- Load configs ----
    config_file = ALGORITHM_DIR / f"config/{args.config_name}.yaml"
    with open(config_file, "r", encoding="utf-8") as fin:
        cfg = yaml.load(fin, Loader=yaml.FullLoader)

    env_cfg_file = ALGORITHM_DIR / "config/env_settings.yaml"
    with open(env_cfg_file, "r", encoding="utf-8") as fin:
        env_cfg = yaml.load(fin, Loader=yaml.FullLoader)

    share_feat = cfg.get("share_feat", True)

    env_id = env_cfg["ycb_train"]["env_id"]
    model_ids = env_cfg["ycb_train"]["model_ids"]

    seed = args.seed
    print(f"Seed: {seed}")
    print(f"total_timesteps: {args.total_timesteps}")
    setup_seed(seed)

    rl_feat_extract_class = extractor_aliases[cfg["obs_mode"]]

    # ---- Output directory ----
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_dir = os.path.join(
        REPO_ROOT, "runs",
        f"baseline_raw_{timestamp}_seed{seed}"
    )
    os.makedirs(log_dir, exist_ok=True)

    # Save config for reproducibility
    training_cfg = {**cfg, "log_std_init": args.log_std_init,
                    "total_timesteps": args.total_timesteps, "seed": seed}
    with open(os.path.join(log_dir, "config.yaml"), "w") as f:
        yaml.dump(training_cfg, f)

    # ---- Build vectorized env ----
    vec_env = SubprocVecEnv(
        [
            sb3_make_multienv(
                env_id=env_id,
                robot_id=cfg["robot_id"],
                robot_init_qpos_noise=cfg["robot_init_qpos_noise"],
                shader_dir=cfg["shader_dir"],
                model_ids=model_ids,
                num_grasps=cfg["num_grasps"],
                num_grasp_points=cfg["num_grasp_points"],
                grasp_points_mode=cfg["grasp_points_mode"],
                obj_init_rot_z=cfg["obj_init_rot_z"],
                obj_init_rot=cfg["obj_init_rot"],
                goal_thresh=cfg["goal_thresh"],
                robot_x_offset=cfg["robot_x_offset"],
                gen_traj_mode=cfg["gen_traj_mode"],
                vary_speed=cfg["vary_speed"],
                grasp_select_mode=cfg["grasp_select_mode"],
                obs_mode=cfg["obs_mode"],
                control_mode=cfg["control_mode"],
                reward_mode=cfg["reward_mode"],
                sim_freq=cfg["sim_freq"],
                control_freq=cfg["control_freq"],
                device=cfg["device"],
                rank=i,
                seed=seed,
            )
            for i in range(cfg["train_procs"])
        ],
        start_method="spawn",
    )

    vec_env = VecMonitor(vec_env, log_dir)

    # ---- Build pure SAC model (RAW GAP-RL, NO LSTM, NO CustomSAC) ----
    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.total_timesteps // (cfg["train_procs"] * 5), 1000),
        save_path=log_dir,
    )
    reward_cb = RewardComponentCallback(verbose=1)
    new_logger = configure(log_dir, ["stdout", "csv", "log", "tensorboard"])

    model = SAC(
        "MultiInputPolicy",
        vec_env,
        batch_size=512,
        ent_coef="auto_0.2",
        gamma=0.98,
        train_freq=64,
        gradient_steps=64,
        buffer_size=100000,
        learning_starts=800,
        use_sde=True,
        policy_kwargs=dict(
            log_std_init=args.log_std_init,
            net_arch=[256, 256],
            features_extractor_class=rl_feat_extract_class,
            features_extractor_kwargs=None,
            normalize_images=False,
            share_features_extractor=share_feat,
        ),
        tensorboard_log=os.path.join(log_dir, "tb/"),
        seed=seed,
        device=cfg["device"],
        verbose=1,
    )

    if getattr(model, "use_sde", False) and hasattr(model.policy, "actor"):
        with torch.no_grad():
            model.policy.actor.log_std.fill_(args.log_std_init)
            
    model.set_logger(new_logger)
    print(f"\n{'='*60}")
    print(f"Starting RAW Baseline GAP-RL training")
    print(f"  total_timesteps   = {args.total_timesteps}")
    print(f"  output_dir        = {log_dir}")
    print(f"{'='*60}\n")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_cb, reward_cb],
    )

    model.save(os.path.join(log_dir, "final_model"))
    print(f"\nTraining complete. Results saved in: {log_dir}")

    # ---- Run Evaluation ----
    if args.eval:
        print(f"\n{'='*60}")
        print("Launching Evaluation for Baseline Model...")
        print(f"{'='*60}\n")
        
        eval_script = os.path.join(REPO_ROOT, "scripts", "run_all_evals.sh")
        # Ensure run_all_evals is executable
        os.chmod(eval_script, 0o755)
        
        # Call the evaluation script on the newly trained baseline model
        cmd = [
            eval_script, 
            log_dir, 
            "final_model", # name of the model we just saved
            str(seed),
            "--force" # Force overwrite if needed
        ]
        
        # We need to run it from the repo root
        subprocess.run(cmd, cwd=REPO_ROOT)
        print("\nAll Evaluation finished!")


if __name__ == "__main__":
    main()
