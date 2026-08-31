#!/usr/bin/env python3
"""
Diagnostic script: measure how often `is_info_exist` fires under a uniform
random policy, and break down reward components per step.

Usage (from repo root):
    python scripts/debug_info_exist_rate.py --episodes 20

No trained model or checkpoint is needed — just the simulator.
"""

import argparse
import os
import sys
import random
from collections import defaultdict

import numpy as np
import yaml

# Ensure the repo root is importable regardless of cwd
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import gym
from gap_rl import ALGORITHM_DIR
from gap_rl.envs import *  # noqa: F401,F403 — registers env IDs


def setup_seed(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser(description="Diagnose info_exist_reward under random policy")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes to run")
    parser.add_argument("--config-name", type=str, default="egopoints_ur85_bezier2d_goalaux",
                        help="Config YAML name (without .yaml) from gap_rl/algorithms/config/")
    parser.add_argument("--seed", type=int, default=1029, help="Random seed")
    args = parser.parse_args()

    setup_seed(args.seed)

    # ---------- Load config (same as sac_train.py) ----------
    config_file = ALGORITHM_DIR / f"config/{args.config_name}.yaml"
    with open(config_file, "r", encoding="utf-8") as fin:
        cfg = yaml.load(fin, Loader=yaml.FullLoader)

    env_cfg_file = ALGORITHM_DIR / "config/env_settings.yaml"
    with open(env_cfg_file, "r", encoding="utf-8") as fin:
        env_cfg = yaml.load(fin, Loader=yaml.FullLoader)

    env_id = env_cfg["ycb_train"]["env_id"]
    model_ids = env_cfg["ycb_train"]["model_ids"]

    # ---------- Create single env (matches training env kwargs) ----------
    env = gym.make(
        env_id,
        robot=cfg["robot_id"],
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
        device="cpu",
        renderer_kwargs=dict(offscreen_only=True),
    )
    env.seed(args.seed)

    # ---------- Tracking ----------
    step_records = defaultdict(list)
    total_steps = 0
    max_episode_steps = 100  # from @register_env max_episode_steps

    print(f"Running {args.episodes} episodes with uniformly random actions...")
    print(f"Config: {args.config_name}, env_id: {env_id}, objects: {len(model_ids)}")
    print("-" * 72)

    for ep in range(args.episodes):
        obs = env.reset()
        ep_rewards = defaultdict(float)

        for step in range(max_episode_steps):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)

            # Extract reward components from the cached info dict
            # (compute_dense_reward stores them in self._cache_info → merged
            # into info at step() line 1020)
            info_exist_val = info.get("info_exist_reward", 0.0)
            approach_val = info.get("approach_reward", 0.0)
            grasp_val = info.get("grasp_reward", 0.0)
            goal_val = info.get("goal_reward", 0.0)
            static_val = info.get("static_reward", 0.0)

            is_info_exist = 1.0 if info_exist_val > 0 else 0.0
            is_obj_grasp = 1.0 if grasp_val > 0 else 0.0
            evaluate_info = info.get("evaluate_info", np.zeros(4))
            is_obj_lift = float(evaluate_info[3]) if len(evaluate_info) > 3 else 0.0

            step_records["is_info_exist"].append(is_info_exist)
            step_records["is_obj_grasp"].append(is_obj_grasp)
            step_records["is_obj_lift"].append(is_obj_lift)
            step_records["info_exist_reward"].append(info_exist_val)
            step_records["approach_reward"].append(approach_val)
            step_records["grasp_reward"].append(grasp_val)
            step_records["goal_reward"].append(goal_val)
            step_records["static_reward"].append(static_val)
            step_records["total_reward"].append(reward)

            for k in ["info_exist_reward", "approach_reward", "grasp_reward",
                       "goal_reward", "static_reward"]:
                ep_rewards[k] += info.get(k, 0.0)

            total_steps += 1
            if done:
                break

        ep_total = sum(ep_rewards.values()) * 0.5  # matches the *0.5 scaling
        print(f"  Episode {ep+1:3d}/{args.episodes} | "
              f"steps={step+1:3d} | "
              f"info_exist={ep_rewards['info_exist_reward']:6.1f} | "
              f"approach={ep_rewards['approach_reward']:6.1f} | "
              f"grasp={ep_rewards['grasp_reward']:5.1f} | "
              f"goal={ep_rewards['goal_reward']:5.1f} | "
              f"static={ep_rewards['static_reward']:5.1f}")

    env.close()

    # ---------- Summary ----------
    print("\n" + "=" * 72)
    print(f"SUMMARY over {total_steps} total steps ({args.episodes} episodes)")
    print("=" * 72)

    # Boolean fractions
    for flag in ["is_info_exist", "is_obj_grasp", "is_obj_lift"]:
        frac = np.mean(step_records[flag])
        print(f"  {flag:20s} fraction: {frac:.4f}  ({int(sum(step_records[flag]))}/{total_steps} steps)")

    print()

    # Per-step means (pre-scaling, as stored in info dict)
    components = ["info_exist_reward", "approach_reward", "grasp_reward",
                   "goal_reward", "static_reward"]
    means = {}
    for comp in components:
        m = np.mean(step_records[comp])
        means[comp] = m
        print(f"  {comp:22s} mean/step: {m:.4f}")

    total_component_sum = sum(means.values())
    print()
    print(f"  {'Sum of components':22s} mean/step: {total_component_sum:.4f}")
    print(f"  {'Scaled reward (*0.5)':22s} mean/step: {np.mean(step_records['total_reward']):.4f}")

    print()
    print("  Reward share (% of total component sum):")
    if total_component_sum > 0:
        for comp in components:
            share = means[comp] / total_component_sum * 100
            print(f"    {comp:22s}: {share:6.2f}%")
    else:
        print("    (total is zero — no reward signal at all)")

    print()
    print("=" * 72)
    print("INTERPRETATION:")
    info_frac = np.mean(step_records["is_info_exist"])
    if info_frac > 0.8:
        print(f"  is_info_exist fires on {info_frac:.1%} of steps — near-unconditional.")
        print("  This SUPPORTS the hypothesis that info_exist_reward provides a cheap")
        print("  local optimum that can dominate the Q-landscape.")
    elif info_frac > 0.5:
        print(f"  is_info_exist fires on {info_frac:.1%} of steps — moderately frequent.")
        print("  Hypothesis is plausible but not overwhelming.")
    else:
        print(f"  is_info_exist fires on {info_frac:.1%} of steps — NOT near-unconditional.")
        print("  This WEAKENS the hypothesis.")
    print("=" * 72)


if __name__ == "__main__":
    main()
