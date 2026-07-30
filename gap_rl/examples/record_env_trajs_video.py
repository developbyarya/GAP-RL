"""Headless video recording of camera POV for pick env trajectories.

Based on test_env_trajs.py, but never opens a viewer window. Frames are taken
from the onboard hand camera (hand_realsense) with optional LoG grasp overlays.
"""

import argparse
import os
import random
from pathlib import Path

import gym
import numpy as np
import torch
import yaml
from gap_rl import ALGORITHM_DIR
from gap_rl.utils.visualization.misc import images_to_video


def setup_seed(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def capture_camera_rgb(
    env,
    camera_name="hand_realsense",
    view_workspace=True,
    view_traj=True,
    view_grasps=True,
    view_obj_bbdx=False,
):
    """Render one RGB frame from a scene camera with optional debug overlays."""
    e = env.unwrapped
    overlays = []

    if view_workspace and e.gen_traj_mode in ["random2d", "bezier2d"]:
        overlays.append(e._view_workspace())
    if view_traj:
        overlays.append(e._view_traj())
    if view_grasps and e.obs_mode in [
        "state_grasp9d",
        "state_egopoints",
        "state_egopoints_rt",
        "state_grasp9d_rt",
    ]:
        if e.pred_grasp_actor_critic is not None:
            overlays.append(e._view_pred_grasp())
        if e.grasps_mat is not None:
            overlays.append(e._view_anno_grasps())
        overlays.append(e._view_grasps())
    if view_grasps and e.obs_mode == "state_objpoints_rt":
        if e.grasps_mat is not None:
            overlays.append(e._view_anno_grasps())
        if e.pred_grasp_actor_critic is not None:
            overlays.append(e._view_pred_grasp())
    if view_obj_bbdx:
        overlays.append(e._view_obj_bbdx())

    try:
        e.update_render()
        if camera_name == "render_camera":
            cam = e._render_cameras["render_camera"]
            rgba = cam.get_images(take_picture=True)["Color"]
        else:
            e.take_picture()
            cam = e._cameras[camera_name]
            rgba = cam.get_images()["Color"]
        rgb = np.clip(rgba[..., :3] * 255, 0, 255).astype(np.uint8)
    finally:
        for lineset in overlays:
            e._remove_lineset(lineset)

    return rgb


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record headless camera-POV videos of env trajectories."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="videos/env_trajs",
        help="Directory to write mp4 files.",
    )
    parser.add_argument("--mode", type=str, default="ycb_train")
    parser.add_argument("--obs-mode", type=str, default="state_egopoints")
    parser.add_argument("--grasp-select-mode", type=str, default="near4")
    parser.add_argument("--control-mode", type=str, default="pd_ee_delta_pose_euler")
    parser.add_argument("--gen-traj-mode", type=str, default="random2d")
    parser.add_argument("--robot-id", type=str, default="ur5e_robotiq85_old")
    parser.add_argument("--num-grasps", type=int, default=40)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument(
        "--camera",
        type=str,
        default="hand_realsense",
        choices=["hand_realsense", "render_camera"],
        help="Camera POV to record. hand_realsense = wrist camera.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-view-workspace", action="store_true")
    parser.add_argument("--no-view-traj", action="store_true")
    parser.add_argument("--no-view-grasps", action="store_true")
    parser.add_argument("--view-obj-bbdx", action="store_true")
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="If set, only run this object id; otherwise cycle through mode model_ids.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    view_workspace = not args.no_view_workspace
    view_traj = not args.no_view_traj
    view_grasps = not args.no_view_grasps
    view_obj_bbdx = args.view_obj_bbdx

    env_cfg_file = ALGORITHM_DIR / "config/env_settings.yaml"
    with open(env_cfg_file, "r", encoding="utf-8") as fin:
        env_cfg = yaml.load(fin, Loader=yaml.FullLoader)
    env_id = env_cfg[args.mode]["env_id"]
    model_ids = env_cfg[args.mode]["model_ids"]
    if args.model_id is not None:
        model_ids = [args.model_id]
    print(env_id)
    print(model_ids)

    seed = args.seed
    if seed is None:
        seed = np.random.RandomState().randint(2**32)
    print("experiment random seed:", seed)
    setup_seed(seed)

    env = gym.make(
        env_id,
        shader_dir="ibl",
        robot=args.robot_id,
        model_ids=model_ids,
        obj_init_rot_z=True,
        obs_mode=args.obs_mode,
        reward_mode="dense",
        control_mode=args.control_mode,
        robot_x_offset=0.56,
        sim_freq=150,
        control_freq=5,
        vary_speed=True,
        num_grasps=args.num_grasps,
        gen_traj_mode=args.gen_traj_mode,
        grasp_select_mode=args.grasp_select_mode,
        renderer_kwargs=dict(offscreen_only=True),
    )
    env.seed(seed)
    print(env.action_space)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture_kwargs = dict(
        camera_name=args.camera,
        view_workspace=view_workspace,
        view_traj=view_traj,
        view_grasps=view_grasps,
        view_obj_bbdx=view_obj_bbdx,
    )

    for num in range(args.num_episodes):
        cur_id = int(num % len(model_ids))
        model_id = model_ids[cur_id]
        print(f"episode {num}: model_id={model_id}")
        env.reset(model_id=model_id)

        frames = [capture_camera_rgb(env, **capture_kwargs)]
        for step in range(args.max_steps):
            env.step(np.zeros(env.agent.action_space.sample().shape))
            frames.append(capture_camera_rgb(env, **capture_kwargs))
            print(f"episode {num}, step {step + 1}/{args.max_steps}", end="\r")
        print()

        video_name = f"{num:03d}_{model_id}_{args.camera}"
        images_to_video(
            frames,
            str(output_dir),
            video_name=video_name,
            fps=args.fps,
            verbose=True,
        )

    env.close()
    print(f"Saved {args.num_episodes} video(s) to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
