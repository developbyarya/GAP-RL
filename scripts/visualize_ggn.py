import argparse
import os
import random
from pathlib import Path

import gym
import numpy as np
import torch
import yaml
import cv2
import matplotlib.pyplot as plt

from gap_rl import ALGORITHM_DIR
from gap_rl.envs.pick_single import PickSingleYCBEnv  # noqa: F401
from gap_rl.utils.visualization.cv2_utils import images_to_video
from gap_rl.utils.geometry import transform_points
from gap_rl.algorithms.Networks.pointnet import GraspPointAppGroup

def setup_seed(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def generate_gaussian_grasps_overlay(e):
    trans_ee2world = e.tcp.pose.to_transformation_matrix()
    grasps_mat_ee, grasps_scores = e._compute_near_grasps()
    
    wg = e.gripper_w
    num_pts = e.num_grasp_points
    
    rng = np.random.RandomState(42)
    gripper_pts_init = rng.normal(0.0, wg / 6, size=(num_pts, 3))
    
    draw_pts = []
    colors = []
    
    # Add identity grasp
    draw_pts.append(transform_points(trans_ee2world, gripper_pts_init))
    colors.append(np.repeat([[1, 0, 0, 1]], num_pts, axis=0)) # Red
    
    num_grasps = grasps_mat_ee.shape[0]
    for i in range(num_grasps):
        pts_ee = transform_points(grasps_mat_ee[i], gripper_pts_init)
        pts_world = transform_points(trans_ee2world, pts_ee)
        draw_pts.append(pts_world)
        colors.append(np.repeat([[0, 1, 0, 1]], num_pts, axis=0)) # Green
        
    draw_pts = np.concatenate(draw_pts, axis=0)
    colors = np.concatenate(colors, axis=0)
    
    cross_size = 0.002
    lines = []
    line_colors = []
    for p, c in zip(draw_pts, colors):
        lines.append([p - [cross_size, 0, 0], p + [cross_size, 0, 0]])
        lines.append([p - [0, cross_size, 0], p + [0, cross_size, 0]])
        lines.append([p - [0, 0, cross_size], p + [0, 0, cross_size]])
        line_colors.extend([c, c, c, c, c, c])
        
    lines = np.array(lines).reshape(-1, 3)
    line_colors = np.array(line_colors)
    
    renderer_context = e._renderer._internal_context
    traj_linesets = renderer_context.create_line_set(lines, line_colors)
    lineset = e._scene.renderer_scene._internal_scene.add_line_set(traj_linesets)
    return lineset, gripper_pts_init, grasps_mat_ee

def overlay_ggn_feature(img, feature, step):
    h, w, _ = img.shape
    
    fig, ax = plt.subplots(figsize=(4, 1.5), dpi=100)
    ax.bar(range(len(feature)), feature, color='b')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("GraspGroupNet Feature", fontsize=10)
    plt.tight_layout()
    
    fig.canvas.draw()
    plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    ph, pw, _ = plot_img.shape
    img[-ph:, w-pw:] = plot_img
    
    cv2.putText(img, f"Step: {step}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img

def main():
    setup_seed(1029)
    output_dir = Path("videos/ggn_vis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    env_cfg_file = ALGORITHM_DIR / "config/env_settings.yaml"
    with open(env_cfg_file, "r") as fin:
        env_cfg = yaml.load(fin, Loader=yaml.FullLoader)
    env_id = env_cfg["ycb_train"]["env_id"]
    model_ids = [env_cfg["ycb_train"]["model_ids"][0]]
    
    env = gym.make(
        env_id,
        shader_dir="ibl",
        robot="ur5e_robotiq85_old",
        model_ids=model_ids,
        obs_mode="state_egopoints",
        reward_mode="dense",
        control_mode="pd_ee_delta_pose_euler",
        robot_x_offset=0.56,
        sim_freq=150,
        control_freq=5,
        vary_speed=True,
        num_grasps=40,
        num_grasp_points=20,
        grasp_points_mode="gauss",
        gen_traj_mode="random2d",
        grasp_select_mode="near4",
        renderer_kwargs=dict(offscreen_only=True),
    )
    
    gpag = GraspPointAppGroup(
        in_ch=3,
        graspgroup_mlp_specs=[16, 32],
        group_mlp_specs=[64, 128],
    )
    gpag.eval()
    
    env.reset(seed=1029, model_id=model_ids[0])
    
    rgbs = []
    max_steps = 100
    for step in range(max_steps):
        e = env.unwrapped
        
        lineset, gripper_pts_init, grasps_mat_ee = generate_gaussian_grasps_overlay(e)
        
        try:
            e.update_render()
            e.take_picture()
            cam = e._cameras["hand_realsense"]
            images = cam.get_images()
            rgb = np.clip(images["Color"][..., :3] * 255, 0, 255).astype(np.uint8)
        finally:
            e._remove_lineset(lineset)
            
        # compute GGN feature
        x_init_tensor = torch.tensor(gripper_pts_init, dtype=torch.float32).unsqueeze(0)
        
        num_grasps = grasps_mat_ee.shape[0]
        pts_ee_all = []
        for i in range(num_grasps):
            pts_ee = transform_points(grasps_mat_ee[i], gripper_pts_init)
            pts_ee_all.append(pts_ee)
        if len(pts_ee_all) > 0:
            x_tensor = torch.tensor(np.array(pts_ee_all), dtype=torch.float32).unsqueeze(0)
        else:
            x_tensor = torch.zeros((1, 0, e.num_grasp_points, 3), dtype=torch.float32)
            
        with torch.no_grad():
            if len(pts_ee_all) > 0:
                feat = gpag(x_init_tensor, x_tensor)
                feat_np = feat[0].numpy()
            else:
                feat_np = np.zeros(128)
                
        rgb = overlay_ggn_feature(rgb, feat_np, step)
        rgbs.append(rgb)
        
        env.step(np.zeros(env.action_space.shape))
        print(f"Step {step+1}/{max_steps}", end="\r")
        
    print("\nSaving video...")
    images_to_video(rgbs, str(output_dir), video_name="ggn_visualization", fps=10, verbose=True)
    env.close()

if __name__ == "__main__":
    main()
