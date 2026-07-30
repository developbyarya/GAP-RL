"""Headless recording of camera POV trajectories.

Supports:
  - preview video (optional debug overlays)
  - clean training dumps: RGB, depth, LoG grasp poses (4x4 EE-frame)
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
from gap_rl.envs.pick_single import PickSingleYCBEnv  # noqa: F401  # registers envs
from gap_rl.utils.visualization.cv2_utils import images_to_video


def setup_seed(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _add_overlays(e, view_workspace, view_traj, view_grasps, view_obj_bbdx):
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
    return overlays


def capture_frame(
    env,
    camera_name="hand_realsense",
    view_workspace=False,
    view_traj=False,
    view_grasps=False,
    view_obj_bbdx=False,
    want_depth=True,
    want_grasps=True,
):
    """One render pass: RGB (+ optional depth / LoG grasps). Overlays for preview only."""
    e = env.unwrapped
    overlays = _add_overlays(
        e, view_workspace, view_traj, view_grasps, view_obj_bbdx
    )

    try:
        e.update_render()
        if camera_name == "render_camera":
            cam = e._render_cameras["render_camera"]
            images = cam.get_images(take_picture=True)
            cam_params = None
        else:
            e.take_picture()
            cam = e._cameras[camera_name]
            images = cam.get_images()
            cam_params = cam.get_params()

        rgb = np.clip(images["Color"][..., :3] * 255, 0, 255).astype(np.uint8)
        depth = None
        if want_depth and "Position" in images:
            depth = (-images["Position"][..., 2]).astype(np.float32)  # meters

        grasps_ee, grasps_scores = None, None
        tcp_pose = None
        if want_grasps:
            if e.obs_mode in ["state_egopoints", "state_grasp9d"]:
                grasps_ee, grasps_scores = e._compute_near_grasps()
            elif e.obs_mode in ["state_egopoints_rt", "state_grasp9d_rt"]:
                grasps_ee, grasps_scores = e._compute_near_grasps_rt()
            tcp_pose = e.tcp.pose.to_transformation_matrix().astype(np.float32)
    finally:
        for lineset in overlays:
            e._remove_lineset(lineset)

    return dict(
        rgb=rgb,
        depth=depth,
        grasps_ee=grasps_ee,
        grasps_scores=grasps_scores,
        tcp_pose=tcp_pose,
        cam_params=cam_params,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record headless camera-POV video and/or clean training data."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="videos/env_trajs",
        help="Directory for mp4 / npz outputs.",
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
    parser.add_argument(
        "--save-data",
        action="store_true",
        help="Save clean RGB/depth/grasps npz (no overlays). Recommended for training.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        default=True,
        help="Also write an mp4 preview (default: on).",
    )
    parser.add_argument(
        "--no-save-video",
        action="store_true",
        help="Skip mp4 preview.",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Draw workspace/traj/grasps on the preview video only (never in npz).",
    )
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
    save_video = args.save_video and not args.no_save_video
    # Overlays only affect preview video; training npz is always clean.
    use_overlay = args.overlay and save_video

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

    want_depth = args.save_data
    want_grasps = args.save_data

    for num in range(args.num_episodes):
        cur_id = int(num % len(model_ids))
        model_id = model_ids[cur_id]
        print(f"episode {num}: model_id={model_id}")
        env.reset(model_id=model_id)

        rgbs, depths, grasps_list, scores_list, tcp_list = [], [], [], [], []
        cam_params = None

        def record_step():
            nonlocal cam_params
            sample = capture_frame(
                env,
                camera_name=args.camera,
                view_workspace=use_overlay,
                view_traj=use_overlay,
                view_grasps=use_overlay,
                view_obj_bbdx=args.view_obj_bbdx and use_overlay,
                want_depth=want_depth,
                want_grasps=want_grasps,
            )
            rgbs.append(sample["rgb"])
            if sample["depth"] is not None:
                depths.append(sample["depth"])
            if sample["grasps_ee"] is not None:
                grasps_list.append(sample["grasps_ee"].astype(np.float32))
                scores_list.append(sample["grasps_scores"].astype(np.float32))
                tcp_list.append(sample["tcp_pose"])
            if cam_params is None and sample["cam_params"] is not None:
                cam_params = sample["cam_params"]

        record_step()
        for step in range(args.max_steps):
            env.step(np.zeros(env.agent.action_space.sample().shape))
            record_step()
            print(f"episode {num}, step {step + 1}/{args.max_steps}", end="\r")
        print()

        stem = f"{num:03d}_{model_id}_{args.camera}"

        if save_video:
            images_to_video(
                rgbs,
                str(output_dir),
                video_name=stem,
                fps=args.fps,
                verbose=True,
            )

        if args.save_data:
            payload = dict(
                rgb=np.stack(rgbs, axis=0),  # (T, H, W, 3) uint8, no overlays
                model_id=np.array(model_id),
                seed=np.array(seed, dtype=np.uint64),
                camera=np.array(args.camera),
                obs_mode=np.array(args.obs_mode),
                grasp_select_mode=np.array(args.grasp_select_mode),
            )
            if depths:
                payload["depth"] = np.stack(depths, axis=0)  # (T, H, W) meters
            if grasps_list:
                payload["grasps_ee"] = np.stack(grasps_list, axis=0)  # (T, N, 4, 4)
                payload["grasps_scores"] = np.stack(scores_list, axis=0)  # (T, N)
                payload["tcp_pose"] = np.stack(tcp_list, axis=0)  # (T, 4, 4) EE->world
            if cam_params is not None:
                for k, v in cam_params.items():
                    payload[f"cam_{k}"] = np.asarray(v)

            npz_path = output_dir / f"{stem}.npz"
            np.savez_compressed(npz_path, **payload)
            print(f"Saved data: {npz_path}")

    env.close()
    print(f"Done. Outputs in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
