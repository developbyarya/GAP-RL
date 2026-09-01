#!/usr/bin/env python3
"""
Temporary ablation script for info_exist_reward weight.

Thin wrapper around the existing training setup (env, CustomSAC, callbacks)
that adds --info-exist-weight and --total-timesteps flags and writes output
to a separate runs/reward_ablation_<timestamp>/ directory.

Usage (from gap_rl/algorithms/scripts/ directory, or set PYTHONPATH):

    cd gap_rl/algorithms/scripts && \\
    python ../../scripts/train_reward_ablation.py \\
        --config-name egopoints_ur85_bezier2d_goalaux \\
        --info-exist-weight 3.0 \\
        --total-timesteps 300000 \\
        --seed 1029

Does NOT modify any existing config, checkpoint, or output path.
"""

import os
import sys
import time
import argparse

import numpy as np
import yaml
import torch

# Ensure repo root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Also need gap_rl/algorithms/scripts on path for custom_sac imports
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
    CustomGraspPointExtractor,
    CustomObjPNExtractor,
)
from custom_sac import CustomSAC, FrameStackWrapper, default_stack_keys
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
    parser = argparse.ArgumentParser(description="Reward ablation training")
    parser.add_argument("--config-name", type=str, default="egopoints_ur85_bezier2d_goalaux")
    parser.add_argument("--info-exist-weight", type=float, default=3.0,
                        help="Weight for info_exist_reward (default: 3.0 = original)")
    parser.add_argument("--log-std-init", type=float, default=-3.67,
                        help="Initial value for gSDE log_std (default: -3.67)")
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: random)")
    args = parser.parse_args()

    np.set_printoptions(suppress=True, precision=4)

    # ---- Load configs (same as sac_train.py) ----
    config_file = ALGORITHM_DIR / f"config/{args.config_name}.yaml"
    with open(config_file, "r", encoding="utf-8") as fin:
        cfg = yaml.load(fin, Loader=yaml.FullLoader)

    env_cfg_file = ALGORITHM_DIR / "config/env_settings.yaml"
    with open(env_cfg_file, "r", encoding="utf-8") as fin:
        env_cfg = yaml.load(fin, Loader=yaml.FullLoader)

    is_goal_aux = cfg.get("goal_aux", False)
    share_feat = cfg.get("share_feat", True)
    n_stack = cfg.get("n_stack", 4 if is_goal_aux else 1)

    env_id = env_cfg["ycb_train"]["env_id"]
    model_ids = env_cfg["ycb_train"]["model_ids"]

    seed = args.seed if args.seed is not None else np.random.RandomState().randint(2**32)
    print(f"Seed: {seed}")
    print(f"info_exist_weight: {args.info_exist_weight}")
    print(f"total_timesteps: {args.total_timesteps}")
    setup_seed(seed)

    rl_feat_extract_class = extractor_aliases[cfg["obs_mode"]]

    # ---- Output directory ----
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_dir = os.path.join(
        REPO_ROOT, "runs",
        f"reward_ablation_{timestamp}_iew{args.info_exist_weight}"
    )
    os.makedirs(log_dir, exist_ok=True)

    # Save ablation config for reproducibility
    ablation_cfg = {**cfg, "info_exist_weight": args.info_exist_weight,
                    "log_std_init": args.log_std_init,
                    "total_timesteps": args.total_timesteps, "seed": seed}
    with open(os.path.join(log_dir, "ablation_config.yaml"), "w") as f:
        yaml.dump(ablation_cfg, f)

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
                # --- Ablation parameter ---
                info_exist_weight=args.info_exist_weight,
            )
            for i in range(cfg["train_procs"])
        ],
        start_method="spawn",
    )

    orig_obs_space = vec_env.observation_space
    stack_keys = default_stack_keys(orig_obs_space)
    if is_goal_aux:
        vec_env = FrameStackWrapper(vec_env, n_stack=n_stack, stack_keys=stack_keys)
        print(f"FrameStack: n_stack={n_stack}, stack_keys={stack_keys}")
    vec_env = VecMonitor(vec_env, log_dir)

    for obs_key, box_space in vec_env.observation_space.items():
        print(f"{obs_key}: {box_space.shape}")
    print("Action Space:", vec_env.action_space)

    # ---- Build SAC model (identical to sac_train.py) ----
    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.total_timesteps // (cfg["train_procs"] * 5), 1000),
        save_path=log_dir,
    )
    reward_cb = RewardComponentCallback(verbose=1)
    new_logger = configure(log_dir, ["stdout", "csv", "log", "tensorboard"])

    if is_goal_aux:
        model = CustomSAC(
            "CustomSACPolicy",
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
                extra_pred_dim=9,
                orig_observation_space=orig_obs_space,
                stack_keys=stack_keys,
            ),
            tensorboard_log=os.path.join(log_dir, "tb/"),
            seed=seed,
            device=cfg["device"],
            verbose=1,
        )
    else:
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

    model.set_logger(new_logger)
    print(f"\n{'='*60}")
    print(f"Starting ablation training")
    print(f"  info_exist_weight = {args.info_exist_weight}")
    print(f"  total_timesteps   = {args.total_timesteps}")
    print(f"  output_dir        = {log_dir}")
    print(f"{'='*60}\n")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_cb, reward_cb],
    )

    model.save(os.path.join(log_dir, "final_model"))
    print(f"\nTraining complete. Results in: {log_dir}")


if __name__ == "__main__":
    main()
