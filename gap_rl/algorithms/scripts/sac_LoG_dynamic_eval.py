import os
import glob
import time
import argparse
import json
import yaml
from tqdm import tqdm
from copy import deepcopy
from collections import OrderedDict

import gym
import numpy as np
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from gap_rl import ALGORITHM_DIR
from gap_rl.envs import *
from gap_rl.utils.geometry import transform_points, pointcloud_filter, pc_bbdx_filter, homo_transfer
from gap_rl.utils.common import setup_seed
from gap_rl.utils.wrappers.common import NormalizeBoxActionWrapper
from gap_rl.utils.wrappers.observation import StackObservationWrapper, DictObservationStack
from gap_rl.utils.wrappers.record import RecordEpisode
from gap_rl.utils.trimesh_utils import get_articulation_meshes, merge_meshes
from gap_rl.utils.o3d_utils import draw_o3d_geometries
from gap_rl.localgrasp.LoG import lg_parse, LgNet, GraspGroup

from custom_sac import CustomSAC

import pickle
from eval_metrics import (
    EpisodeMetricsTracker,
    RunMetricsAggregator,
    ablate_observation,
    reinit_window_first_valid,
    get_valid_frames_count,
    compute_rotmat_grasp_error,
    write_jsonl_line,
    write_json_summary,
    write_curves_npz,
)


def filter_grasp(world_gg, ee_gg, ground_offset=0.05, z_rot_offset=0):
    assert world_gg.size == ee_gg.size
    mask_pos = world_gg.translations[:, 2] > ground_offset
    if z_rot_offset == 0:
        filter_mask = mask_pos
    else:
        mask_rot = ee_gg.rotations[:, 2, 2] > np.cos(z_rot_offset)
        filter_mask = mask_pos & mask_rot
    filter_ids = np.arange(world_gg.size)[filter_mask]
    return filter_ids


def gen_grasps(env, lgNet):
    ## gene fused world rgbpc
    scene_points_ee, obj_points_ee = env.get_objpoints_rt()
    if obj_points_ee.shape[0] < 64:
        # print('too few object points')
        return torch.zeros((0, 6)), torch.zeros((0, 6)), GraspGroup(), GraspGroup()

    # scale points
    scale = 1.0
    pred_gg_ee = lgNet.inference(
        obj_points=torch.from_numpy(obj_points_ee).to('cuda'),
        scene_points=torch.from_numpy(scene_points_ee).to('cuda'),
        num_grasps=64,
        scale=scale
    )
    trans_ee2world = env.tcp.pose.to_transformation_matrix()

    predgg_ee_homo = homo_transfer(
        R=pred_gg_ee.rotations, T=pred_gg_ee.translations
    )
    draw_gg_world = deepcopy(pred_gg_ee)
    draw_graspmat_world = np.einsum('ij, kjl -> kil', trans_ee2world, predgg_ee_homo)  # (N, 4, 4)
    draw_gg_world.translations = draw_graspmat_world[:, :3, 3]
    draw_gg_world.rotations = draw_graspmat_world[:, :3, :3]
    # if vis_grasp:
    #     draw_o3d_geometries([world_rgbpc] + draw_gg_world.to_open3d_geometry_list())

    pred_gg_ee.rotations = np.einsum('ijk, kl -> ijl', pred_gg_ee.rotations, np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]))  # (N, 4, 4)

    return scene_points_ee, obj_points_ee, pred_gg_ee, draw_gg_world


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=4)
    device = "cuda:0"

    parser = argparse.ArgumentParser()
    ## control kwargs
    parser.add_argument("--obj-test-num", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=100)
    # parser.add_argument("--ground-ws", type=json.loads, default=[[0.2, 1.0], [-0.8, 0.8], [0.081, 0.32]])
    parser.add_argument("--cam-mode", type=str, default="hand_realsense")
    parser.add_argument("--eval-datasets", type=str, nargs='+', required=True)  # "ycb_train", "ycb_eval", "graspnet_eval", "acronym_eval"
    parser.add_argument("--gen-traj-modes", type=str, nargs='+', required=True)  # line circle circular bezier2d random2d; random3d bezier3d
    parser.add_argument("--obs-mode", type=str, default="state_grasp9d_rt")  # state_grasp9d_rt, state_egopoints_rt
    parser.add_argument("--timestamp", type=str, default=None)
    parser.add_argument("--run-dir", type=str, default=None, help="Exact path to the run directory (e.g. runs/reward_ablation_...)")
    parser.add_argument("--config", type=str, default=None, help="Optional path to config.yaml (defaults to config.yaml inside run-dir)")
    parser.add_argument("--model-name", type=str, default="rl_model_2000000_steps", help="Name of the model file to load (without or with .zip)")
    parser.add_argument("--seeds", type=int, nargs='+', required=True)

    parser.add_argument("--vis-grasp", action='store_true')
    parser.add_argument("--save-all-grasp", action='store_true')
    parser.add_argument("--save-feat", action='store_true')
    parser.add_argument("--visualizer", action='store_true')
    parser.add_argument("--render", action='store_true')
    parser.add_argument("--save-video", action='store_true')
    parser.add_argument("--add-noise", action='store_true')

    ## extra evaluation metric & diagnostic flags
    parser.add_argument("--success-hold-steps", type=int, default=5, help="Number of consecutive steps success must be held (default: 5)")
    parser.add_argument("--near-trans-thresh", type=float, default=0.08, help="Translation distance threshold for near-target stage (default: 0.08)")
    parser.add_argument("--near-rot-thresh", type=float, default=0.25, help="Rotation distance threshold for near-target stage (default: 0.25)")
    parser.add_argument("--gripper-dim", type=int, default=6, help="Action dimension corresponding to gripper (default: 6)")
    parser.add_argument("--gripper-close-thresh", type=float, default=0.0, help="Action threshold for commanding gripper closure (default: 0.0)")
    parser.add_argument("--speed-bins", type=float, nargs="*", default=None, help="Fixed bin edges for object speed stratification (default: terciles)")
    parser.add_argument("--history-ablation", type=str, choices=["none", "repeat_last", "shuffle", "reverse"], default="none", help="Interventional window ablation during rollout")
    parser.add_argument("--save-latent", action="store_true", help="Extract and save latent representations h_t")
    parser.add_argument("--window-init", type=str, choices=["as_is", "replicate_first_valid"], default="as_is", help="Window initialization protocol")
    parser.add_argument("--save-curves", action="store_true", help="Save per-step distance and action curves to .npz")
    parser.add_argument("--save-eval-records", action="store_true", default=True, help="Save per-episode records to JSONL")

    # setup LoG
    parser = lg_parse(parser)
    args = parser.parse_args()
    
    # Fix checkpoint path when running from outside the localgrasp directory
    if hasattr(args, "checkpoint_path") and not os.path.exists(args.checkpoint_path):
        from gap_rl import LOCALGRASP_DIR
        fixed_path = str(LOCALGRASP_DIR / os.path.basename(args.checkpoint_path))
        if os.path.exists(fixed_path):
            args.checkpoint_path = fixed_path
            
    # args.vis = args.vis_grasp

    print("experiment random seeds: ", args.seeds)
    for rseed in args.seeds:
        setup_seed(rseed)
        lgNet = LgNet(args)
        print(args.eval_datasets, args.gen_traj_modes)
        noise_str = "noise" if args.add_noise else ""

        if args.run_dir is not None:
            log_dir = os.path.abspath(args.run_dir)
            log_path = args.run_dir.rstrip("/")
        elif args.timestamp is not None:
            log_path = glob.glob(f"{args.timestamp}*")[0]
            log_dir = str(ALGORITHM_DIR / f"scripts/{log_path}")
        else:
            raise ValueError("Must provide either --timestamp or --run-dir")

        if args.config is not None:
            config_file = os.path.abspath(args.config)
        else:
            config_file = os.path.join(log_dir, "config.yaml")
            if not os.path.exists(config_file):
                config_file = os.path.join(log_dir, "ablation_config.yaml")
            
        with open(config_file, 'r', encoding='utf-8') as fin:
            cfg = yaml.load(fin, Loader=yaml.FullLoader)
            
        env_cfg_file = ALGORITHM_DIR / f"config/env_settings.yaml"
        with open(env_cfg_file, 'r', encoding='utf-8') as fin:
            env_cfg = yaml.load(fin, Loader=yaml.FullLoader)

        filter_angle = np.pi / 3 if "filter" in cfg['grasp_select_mode'] else np.pi
        is_goal_aux = cfg.get('goal_aux', False)

        for eval_dataset in args.eval_datasets:
            if args.cam_mode == 'hand_realsense':
                camera_modes = ["hand_realsense"]  # "hand_realsense", "base_kinect"
                result_path = f'simLoG_result_onlyhandcam_{eval_dataset}_{rseed}_{noise_str}'
            elif args.cam_mode == 'both':
                camera_modes = ["hand_realsense", "base_kinect"]  # "hand_realsense", "base_kinect"
                result_path = f'simLoG_result_basehandcam_{eval_dataset}_{rseed}_{noise_str}'
            else:
                raise NotImplementedError

            sr_current_dataset = []
            st_current_dataset = []
            env_id = env_cfg[eval_dataset]['env_id']
            model_ids = env_cfg[eval_dataset]['model_ids']
            overall_aggregator = RunMetricsAggregator(speed_bins=args.speed_bins)
            for gen_traj_mode in args.gen_traj_modes:
                traj_aggregator = RunMetricsAggregator(speed_bins=args.speed_bins)
                env = gym.make(
                    env_id,
                    robot=cfg['robot_id'],
                    robot_init_qpos_noise=cfg['robot_init_qpos_noise'],
                    shader_dir=cfg['shader_dir'],
                    model_ids=model_ids,
                    num_grasps=cfg['num_grasps'],
                    num_grasp_points=cfg['num_grasp_points'],
                    grasp_points_mode=cfg['grasp_points_mode'],
                    obj_init_rot_z=cfg['obj_init_rot_z'],
                    obj_init_rot=cfg['obj_init_rot'],
                    goal_thresh=cfg['goal_thresh'],
                    robot_x_offset=cfg['robot_x_offset'],
                    gen_traj_mode=gen_traj_mode,
                    vary_speed=cfg['vary_speed'],
                    grasp_select_mode=cfg['grasp_select_mode'],
                    obs_mode=args.obs_mode,
                    control_mode=cfg['control_mode'],
                    reward_mode=cfg['reward_mode'],
                    sim_freq=cfg['sim_freq'],
                    control_freq=cfg['control_freq'],
                    device=cfg["device"],
                )
                env = NormalizeBoxActionWrapper(env)
                if is_goal_aux:
                    n_stack = cfg.get("n_stack", 4)
                    # window axis is required by the CustomSAC policy even when n_stack == 1
                    from custom_sac import FrameStackObsWrapper, default_stack_keys
                    stack_keys = default_stack_keys(env.observation_space)
                    env = FrameStackObsWrapper(env, n_stack=n_stack, stack_keys=stack_keys)
                else:
                    n_stack = 1
                    stack_keys = None
                para = {
                    "camera_modes": camera_modes,
                    "add_noise": args.add_noise
                }
                env.set_rt_paras(**para)
                record_env = RecordEpisode(
                    env=env,
                    output_dir='./',
                    save_trajectory=False,
                    trajectory_name=None,
                    save_video=args.save_video,
                    info_on_video=False,
                    render_mode="agent_cameras",
                    save_on_reset=True,
                    clean_on_close=True,
                )
                # Resolve checkpoint path
                raw_name = args.model_name
                if raw_name.endswith(".zip"):
                    raw_name = raw_name[:-4]

                candidates = [
                    os.path.join(log_dir, raw_name),
                    os.path.join(log_dir, "best_checkpoint", raw_name),
                    os.path.abspath(raw_name),
                ]
                chosen_path = None
                for cand in candidates:
                    if os.path.exists(cand + ".zip"):
                        chosen_path = cand
                        break
                if chosen_path is None:
                    # Fallback checks if requested name was not found
                    if os.path.exists(os.path.join(log_dir, "best_checkpoint", "best_model.zip")):
                        chosen_path = os.path.join(log_dir, "best_checkpoint", "best_model")
                    elif os.path.exists(os.path.join(log_dir, "best_model.zip")):
                        chosen_path = os.path.join(log_dir, "best_model")
                    elif os.path.exists(os.path.join(log_dir, "final_model.zip")):
                        chosen_path = os.path.join(log_dir, "final_model")
                    else:
                        chosen_path = os.path.join(log_dir, raw_name)

                model_path = chosen_path
                print(f"Loading RL checkpoint: {model_path}.zip")
                        
                rl_model = CustomSAC.load(model_path,
                                    env=record_env,
                                    print_system_info=True
                                    )

                for obs_key, box_space in record_env.observation_space.items():
                    print(f"{obs_key}: {box_space.shape} ")
                # print("Action Space: ", record_env.action_space)
                record_env.seed(rseed)
                # print("+++++++++++++ init env main seed: ", record_env.unwrapped._main_seed)

                success_rates = []
                success_steps = []

                if args.save_all_grasp:
                    preg_gg_ee_list = OrderedDict()
                if args.save_feat:
                    pn_feat_list = OrderedDict()

                for model_id in model_ids:
                    result_dir = f"{log_path}/{result_path}/{gen_traj_mode}/{model_id}/"
                    os.makedirs(result_dir, exist_ok=True)
                    record_env.reset_output_dir(result_dir)
                    is_success = []
                    exp_steps = []
                    if args.save_all_grasp:
                        preg_gg_ee_list[model_id] = OrderedDict()
                    if args.save_feat:
                        pn_feat_list[model_id] = OrderedDict()
                    model_aggregator = RunMetricsAggregator(speed_bins=args.speed_bins)
                    with torch.no_grad():
                        for num in tqdm(range(args.obj_test_num), desc=f'Processing 2M', colour='red', leave=True):
                            t = time.time()
                            # print("+++++++++++++ env main seed: ", record_env.unwrapped._main_seed)
                            _ = record_env.reset(model_id=model_id)
                            epi_seed = record_env.unwrapped._episode_seed
                            # print("+++++++++++++ episode seed: ", epi_seed)
                            # print("*************  points add noise: ", record_env.unwrapped.points_add_noise)
                            init_obj_exist = False
                            if "3d" in gen_traj_mode:
                                grasp_z_th = 0.01
                            else:
                                grasp_z_th = record_env.table_halfsize[2] * 2 + 0.01
                            # print("----- reset time: ", time.time() - t)
                            t = time.time()

                            if args.render:
                                viewer = record_env.render()
                                print("Press [e] to start")
                                while True:
                                    if viewer.window.key_down("e"):
                                        break
                                    record_env.render()
                            else:
                                record_env.update_render()

                            # print("----- init render time: ", time.time() - t)
                            t = time.time()

                            ## generate grasps in EE frame
                            scene_points_ee, obj_points_ee, pred_gg_ee, draw_gg_world = gen_grasps(record_env, lgNet)

                            if obj_points_ee.shape[0] == 0 or pred_gg_ee.size == 0:
                                # grasps_ee = np.eye(4)[None]  # (1, 4, 4)
                                grasps_ee = np.zeros((1, 4, 4))
                                grasps_scores = np.zeros(1)
                            else:
                                init_obj_exist = True

                                # filter_ids = filter_grasp(pred_gg_world, pred_gg_ee, ground_offset=grasp_z_th, z_rot_offset=filter_angle)
                                mask_rot = pred_gg_ee.rotations[:, 2, 2] > np.cos(filter_angle)
                                filter_ids = np.arange(pred_gg_ee.size)[mask_rot]
                                pred_gg_ee.select_from_ids(filter_ids)
                                draw_gg_world.select_from_ids(filter_ids)
                                if args.vis_grasp:
                                    trans_ee2world = record_env.tcp.pose.to_transformation_matrix()
                                    world_pc = transform_points(trans_ee2world, scene_points_ee.cpu().numpy())
                                    draw_o3d_geometries([world_pc] + draw_gg_world.to_open3d_geometry_list())
                                # print("----- init grasp generating time: ", time.time() - t)
                                t = time.time()
                                ## to obj frame, (N, 4, 4)
                                grasps_ee = np.repeat(np.eye(4)[None], pred_gg_ee.size, axis=0)  # (N, 4, 4)
                                grasps_ee[:, :3, 3] = pred_gg_ee.translations
                                grasps_ee[:, :3, :3] = pred_gg_ee.rotations
                                grasps_scores = pred_gg_ee.scores

                            # record_env.set_rt_graspmat_ee(grasps_ee)
                            para = {"grasps_mat_ee": grasps_ee, "grasps_scores": grasps_scores}
                            record_env.set_rt_paras(**para)
                            obs = record_env.get_obs(np.zeros(7))
                            # print("----- init get observation time: ", time.time() - t)

                            # Handle window initialization protocol (Sanity check)
                            if is_goal_aux and args.window_init == "replicate_first_valid":
                                reinit_window_first_valid(record_env)
                                cur_w = record_env
                                while cur_w is not None:
                                    if hasattr(cur_w, "observation") and hasattr(cur_w, "_history"):
                                        obs = cur_w.observation(obs)
                                        break
                                    cur_w = getattr(cur_w, "env", None)

                            tracker = EpisodeMetricsTracker(
                                max_steps=args.max_steps,
                                success_hold_steps=args.success_hold_steps,
                                near_trans_thresh=args.near_trans_thresh,
                                near_rot_thresh=args.near_rot_thresh,
                                gripper_dim=args.gripper_dim,
                                gripper_close_thresh=args.gripper_close_thresh,
                                n_stack=n_stack if is_goal_aux else 1,
                                stack_keys=stack_keys if is_goal_aux else None,
                            )

                            if args.save_all_grasp:
                                grasps_ee, grasps_scores = record_env.unwrapped._compute_near_grasps_rt()
                                preg_gg_ee_list[model_id][num] = [grasps_ee]
                            if args.save_feat:
                                pn_feat_list[model_id][num] = OrderedDict()
                                pn_feat_list[model_id][num] = OrderedDict()
                                pn_feat_list[model_id][num]["actor_pnfeat"] = []
                                pn_feat_list[model_id][num]["critic_pnfeat"] = []

                            success_flag = False
                            total_steps = 0

                            for step in tqdm(range(args.max_steps), desc=f'Processing 2M', colour='green', leave=False):
                                ## grasps_ee as mat form
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()
                                t_pred_start = time.perf_counter()
                                action, _states = rl_model.predict(obs, deterministic=True)
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()
                                predict_time_ms = (time.perf_counter() - t_pred_start) * 1000.0

                                # Counterfactual sensitivity (skip gracefully if n_stack <= 1)
                                cf_diffs = {}
                                if is_goal_aux and n_stack > 1:
                                    for abl_m in ["repeat_last", "shuffle", "reverse"]:
                                        abl_o = ablate_observation(obs, abl_m, stack_keys=stack_keys)
                                        abl_a, _ = rl_model.predict(abl_o, deterministic=True)
                                        cf_diffs[abl_m] = float(np.linalg.norm(abl_a - action))

                                # Interventional arms
                                rollout_action = action
                                if args.history_ablation != "none" and is_goal_aux and n_stack > 1:
                                    abl_rollout_obs = ablate_observation(obs, args.history_ablation, stack_keys=stack_keys)
                                    rollout_action, _ = rl_model.predict(abl_rollout_obs, deterministic=True)

                                # Latent diagnostics
                                latent_vec = None
                                if args.save_latent and hasattr(rl_model.actor, "_extract_windowed_features"):
                                    try:
                                        obs_t_dev = {k: torch.as_tensor(o[None], device=device) for k, o in obs.items()}
                                        h_t = rl_model.actor._extract_windowed_features(obs_t_dev)
                                        latent_vec = h_t[0].detach().cpu().numpy()
                                    except Exception:
                                        latent_vec = None

                                # Aux head predictions & critic Q
                                aux_metrics = None
                                critic_q_min = None
                                if is_goal_aux:
                                    obs_tensor = {k: torch.tensor(o[None], device=device) for k, o in obs.items()}
                                    action_tensor = torch.tensor(rollout_action[None], device=device)
                                    _, pred_pose_actor, pred_target_actor = rl_model.actor.action_log_prob(obs_tensor)
                                    q_vals, pred_pose_critic, pred_target_critic = rl_model.critic(obs_tensor, action_tensor)
                                    pred_grasp_actor_critic = torch.cat((pred_pose_actor, pred_pose_critic), dim=0).cpu().numpy()
                                    pred_target_actor_critic = torch.cat((torch.sigmoid(pred_target_actor[0]), torch.sigmoid(pred_target_critic[0]))).cpu().numpy()
                                    para = {"pred_grasp_actor_critic": pred_grasp_actor_critic, "pred_target_actor_critic": pred_target_actor_critic}
                                    record_env.set_rt_paras(**para)

                                    if isinstance(q_vals, (tuple, list)) and len(q_vals) >= 2:
                                        critic_q_min = float(torch.min(q_vals[0], q_vals[1]).item())
                                    elif isinstance(q_vals, torch.Tensor):
                                        critic_q_min = float(q_vals.item())

                                    goal_pose = obs.get("close_grasp_pose_ee", None)
                                    if goal_pose is not None:
                                        _, _, act_err = compute_rotmat_grasp_error(pred_pose_actor[0], goal_pose)
                                        _, _, crit_err = compute_rotmat_grasp_error(pred_pose_critic[0], goal_pose)
                                    else:
                                        act_err, crit_err = float("nan"), float("nan")

                                    eval_tgt = obs.get("eval_target", None)
                                    if eval_tgt is not None:
                                        tgt_arr = np.asarray(eval_tgt)
                                        pred_probs = torch.sigmoid(pred_target_actor[0]).detach().cpu().numpy()
                                        tgt_acc = float(np.mean((pred_probs > 0.5) == (tgt_arr > 0.5)))
                                    else:
                                        tgt_acc = float("nan")

                                    aux_metrics = {
                                        "actor_pose_err": act_err,
                                        "critic_pose_err": crit_err,
                                        "target_acc": tgt_acc,
                                    }

                                if args.save_feat:
                                    pn_feat_list[model_id][num]["actor_pnfeat"].append(rl_model.actor.features_forward(obs_tensor)[0, 128:].cpu().numpy())
                                    pn_feat_list[model_id][num]["critic_pnfeat"].append(rl_model.critic.features_forward(obs_tensor)[0, 128:].cpu().numpy())

                                ## generate & filter grasps in EE frame
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()
                                t_grasp_start = time.perf_counter()
                                scene_points_ee, obj_points_ee, pred_gg_ee, draw_gg_world = gen_grasps(record_env, lgNet)
                                # filter_ids = filter_grasp(pred_gg_world, pred_gg_ee, ground_offset=grasp_z_th, z_rot_offset=filter_angle)
                                mask_rot = pred_gg_ee.rotations[:, 2, 2] > np.cos(filter_angle)
                                filter_ids = np.arange(pred_gg_ee.size)[mask_rot]
                                pred_gg_ee.select_from_ids(filter_ids)
                                draw_gg_world.select_from_ids(filter_ids)
                                if args.vis_grasp:
                                    trans_ee2world = record_env.tcp.pose.to_transformation_matrix()
                                    world_pc = transform_points(trans_ee2world, scene_points_ee.cpu().numpy())
                                    draw_o3d_geometries([world_pc] + draw_gg_world.to_open3d_geometry_list())
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()
                                grasp_gen_time_ms = (time.perf_counter() - t_grasp_start) * 1000.0

                                if obj_points_ee.shape[0] == 0 or pred_gg_ee.size == 0:
                                    # grasps_ee = np.eye(4)[None]  # (1, 4, 4)
                                    grasps_ee = np.zeros((1, 4, 4))
                                    grasps_scores = np.zeros(1)
                                    # print(f"----- step {step}: no grasp generated: ", time.time() - t)
                                else:
                                    init_obj_exist = True
                                    ## to obj frame, (N, 4, 4)
                                    grasps_ee = np.repeat(np.eye(4)[None], pred_gg_ee.size, axis=0)  # (N, 4, 4)
                                    grasps_ee[:, :3, 3] = pred_gg_ee.translations
                                    grasps_ee[:, :3, :3] = pred_gg_ee.rotations
                                    grasps_scores = pred_gg_ee.scores

                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()
                                t_env_start = time.perf_counter()
                                if not init_obj_exist:
                                    obs, rewards, dones, info = record_env.step(np.zeros(7))
                                else:
                                    # record_env.set_rt_graspmat_ee(grasps_ee)
                                    para = {"grasps_mat_ee": grasps_ee, "grasps_scores": grasps_scores}
                                    record_env.set_rt_paras(**para)
                                    obs, rewards, dones, info = record_env.step(rollout_action)
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()
                                env_step_time_ms = (time.perf_counter() - t_env_start) * 1000.0

                                if args.save_all_grasp:
                                    grasps_ee, grasps_scores = record_env.unwrapped._compute_near_grasps_rt()
                                    preg_gg_ee_list[model_id][num].append(grasps_ee)

                                # Record step metrics
                                valid_frames = get_valid_frames_count(obs, n_stack=n_stack if is_goal_aux else 1)
                                tracker.record_step(
                                    step=step,
                                    action=rollout_action,
                                    info=info,
                                    timing_ms={
                                        "predict": predict_time_ms,
                                        "grasp_gen": grasp_gen_time_ms,
                                        "env_step": env_step_time_ms,
                                    },
                                    valid_frames_in_window=valid_frames,
                                    cf_diffs=cf_diffs if cf_diffs else None,
                                    latent_vector=latent_vec,
                                    aux_metrics=aux_metrics,
                                    critic_q_min=critic_q_min,
                                )

                                if not success_flag and info['is_success']:
                                    success_flag = True
                                    total_steps = step

                                if args.render:
                                    record_env.render()
                                else:
                                    record_env.update_render()
                                # print(f"----- step {step}: env render time: ", time.time() - t)
                                t = time.time()

                            is_success.append(np.array(success_flag).astype(float))
                            exp_steps.append(total_steps)
                            suffix = f"success_{total_steps}" if success_flag else "failure"
                            record_env.set_video_suffix(f"{epi_seed}_" + suffix)

                            # Finalize episode metrics
                            epi_record, curves_dict, latent_dict = tracker.finalize()
                            epi_record.update({
                                "run_name": log_path,
                                "model_name": args.model_name,
                                "dataset": eval_dataset,
                                "gen_traj_mode": gen_traj_mode,
                                "model_id": model_id,
                                "seed": rseed,
                                "episode_seed": int(epi_seed),
                                "episode_idx": num,
                            })
                            model_aggregator.add_episode(epi_record)
                            traj_aggregator.add_episode(epi_record)
                            overall_aggregator.add_episode(epi_record)

                            if args.save_eval_records:
                                ep_jsonl = f"{log_path}/{result_path}/{gen_traj_mode}/episodes.jsonl"
                                write_jsonl_line(ep_jsonl, epi_record)
                            if args.save_curves:
                                c_path = f"{result_dir}/curves_epi_{num}.npz"
                                write_curves_npz(c_path, curves_dict)
                            if args.save_latent and len(tracker.latent_ht) > 0:
                                l_path = f"{result_dir}/latent_epi_{num}.npz"
                                write_curves_npz(l_path, latent_dict)

                            if args.render:
                                viewer = record_env.render()
                                print("Press [t] to next test")
                                while True:
                                    if viewer.window.key_down("t"):
                                        break
                                    record_env.render()

                            # print(is_success, exp_steps)

                        success_rate = np.sum(is_success) / args.obj_test_num
                        exp_steps = np.array(exp_steps)
                        success_mask = exp_steps > 0
                        if np.any(success_mask):
                            mean_steps = np.mean(exp_steps[success_mask]).round(4)
                        else:
                            mean_steps = args.max_steps
                        print(f"2M, {model_id}, success_rate, mean_steps: {success_rate}, {mean_steps}")
                        success_rates.append(success_rate)
                        success_steps.append(mean_steps)

                if args.save_all_grasp:
                    with open(f"{log_path}/{result_path}/{gen_traj_mode}/graspgroup.pkl", "wb") as file:
                        pickle.dump(preg_gg_ee_list, file)
                if args.save_feat:
                    with open(f"{log_path}/{result_path}/{gen_traj_mode}/grasp_pnfeat.pkl", "wb") as file:
                        pickle.dump(pn_feat_list, file)

                results = [model_ids, success_rates, success_steps]
                mean_sr, mean_st = np.mean(success_rates).round(3), np.mean(success_steps).round(1)
                std_sr, std_st = np.std(success_rates).round(3), np.std(success_steps).round(1)
                print(model_ids)
                print('success rates (mean, std): ', mean_sr, std_sr)
                print('success steps (mean, std): ', mean_st, std_st)
                with open(f"{log_path}/{result_path}/{gen_traj_mode}/success_rates_steps.txt", "w") as f:
                    for lst in results:
                        f.write(f"{lst}\n")
                    f.write(f"success rates (mean, std): ({mean_sr}, {std_sr})\n")
                    f.write(f"success steps (mean, std): ({mean_st}, {std_st})\n")

                # Save trajectory mode summary metrics
                traj_summary = traj_aggregator.compute_summary()
                write_json_summary(f"{log_path}/{result_path}/{gen_traj_mode}/summary_metrics.json", traj_summary)

                sr_current_dataset.append(mean_sr)
                st_current_dataset.append(mean_st)

            # Save dataset summary metrics
            dataset_summary = overall_aggregator.compute_summary()
            write_json_summary(f"{log_path}/{result_path}/dataset_summary_metrics.json", dataset_summary)

            mean_sr_d, mean_st_d = np.mean(sr_current_dataset).round(3), np.mean(st_current_dataset).round(1)
            with open(f"{log_path}/{result_path}/mean_success_rates_steps.txt", "w") as f:
                f.write(f"{args.gen_traj_modes}\n")
                f.write(f"success rates: {sr_current_dataset}\n")
                f.write(f"success steps: {st_current_dataset}\n")
                f.write(f"avg (models) success rates, steps: {mean_sr_d}, {mean_st_d}\n")
