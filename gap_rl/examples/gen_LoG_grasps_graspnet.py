import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
import copy
from copy import deepcopy
import matplotlib.pyplot as plt
import open3d as o3d
from tqdm import tqdm
from sapien.core import Pose
import gym

from gap_rl.envs.pick_single import PickSingleGraspnetEnv
from gap_rl.localgrasp.LoG import lg_parse, LgNet, GraspGroup
from gap_rl.utils.geometry import homo_transfer, transform_points, sample_grasp_points_ee
from gap_rl.utils.io_utils import load_json, dump_json
from gap_rl.utils.sapien_utils import look_at
import gap_rl.utils.registration

def main(grasp_file, model_ids, stereo=False, vis=False, render=False, save=True):
    # Using gym.make to properly route all kwargs through the registry
    env = gym.make(
        "PickSingleGraspnet-v0",
        shader_dir="ibl",
        robot="ur5e_robotiq85_old",
        model_ids=["035"],
        obj_init_rot_z=False,
        obs_mode="state_egopoints",
        reward_mode="dense",
        control_mode="pd_ee_delta_pose",
        sim_freq=150,
        control_freq=5,
        gen_traj_mode="line",
        vary_speed=True,
        robot_x_offset=0,
    )

    env.reset()
    vis_dir = env.unwrapped.asset_root
    
    import argparse
    parser = argparse.ArgumentParser()
    parser = lg_parse(parser)
    lg_args = parser.parse_args(["--checkpoint_path", "gap_rl/localgrasp/epoch_11_acc_0.915_cover_0.765"])
    lgNet = LgNet(lg_args)
    
    model_grasps = {model_id: {} for model_id in model_ids}
    
    for model_id in model_ids:
        env.reset(model_id=model_id)
        
        obj_grasps_list = []
        for ind in tqdm(range(72)):
            env.unwrapped.obj.set_pose(Pose())
            angle = (ind % 72) * 5 / 180 * np.pi
            env.unwrapped.obj.set_pose(Pose(p=env.unwrapped.goal_pos, q=[np.cos(angle / 2), 0, 0, np.sin(angle / 2)]))
            env.unwrapped._scene.step()
            
            sample_cam_z = env.unwrapped.goal_pos[2] + 0.3
            sample_poscenter = [env.unwrapped.goal_pos[0], env.unwrapped.goal_pos[1], env.unwrapped.goal_pos[2]]
            pos_it = [sample_poscenter[0], sample_poscenter[1], sample_cam_z]
            
            data_cam_pose = look_at(eye=pos_it, target=sample_poscenter, up=[0, 0, 1])
            env.unwrapped._cameras["data_cam"].camera.set_pose(data_cam_pose)
            
            obs = env.unwrapped.get_state_objpoints_rt(action=None)
            obj_pc_ee, scene_pc_ee = obs["obj_pc_ee"], obs["scene_pc_ee"]
            
            if obj_pc_ee.shape[0] < 64:
                obj_grasps_list.append(None)
                continue
                
            pred_gg = lgNet.inference(
                obj_points=torch.from_numpy(obj_pc_ee).to("cuda:0"),
                scene_points=torch.from_numpy(scene_pc_ee.copy()).to("cuda:0"),
                num_grasps=64,
                scale=1.0,
            )
            
            if pred_gg.size == 0:
                obj_grasps_list.append(None)
                continue
                
            trans_cam2world = env.unwrapped._cameras["data_cam"].camera.get_model_matrix()
            trans_world2obj = env.unwrapped.obj_pose.inv().to_transformation_matrix()
            trans_ee2obj = trans_world2obj @ trans_cam2world @ np.linalg.inv(env.unwrapped.trans_cam2ee)
            
            T = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
            rot_mat = pred_gg.rotations
            pred_gg.rotations = np.einsum("ijk, kl->ijl", rot_mat, T)
            
            gg_homo = homo_transfer(R=pred_gg.rotations, T=pred_gg.translations)
            X_gg_homo = np.einsum("ij, kjl->kil", trans_ee2obj, gg_homo)
            trans_gg = deepcopy(pred_gg)
            trans_gg.translations = X_gg_homo[:, :3, 3]
            trans_gg.rotations = X_gg_homo[:, :3, :3]
            
            obj_grasp = GraspGroup(
                translations=trans_gg.translations,
                rotations=trans_gg.rotations,
                heights=trans_gg.heights,
                widths=trans_gg.widths,
                depths=trans_gg.depths,
                scores=trans_gg.scores,
            )
            obj_grasps_list.append(obj_grasp.to_list())
            
        model_grasps[model_id]["grasp"] = obj_grasps_list
        
    dump_json(vis_dir + grasp_file, model_grasps)
    print(f"Generated {grasp_file} in {vis_dir}")

if __name__ == "__main__":
    model_ids = ['035', '038', '041', '043', '045', '046', '048', '055', '057', '063', '070']
    main(
        grasp_file="/info_localgrasp_v0.json",
        model_ids=model_ids,
        stereo=False,
        vis=False,
        render=False,
        save=True,
    )
