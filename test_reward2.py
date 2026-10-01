import sys
import yaml
import numpy as np
sys.path.append("/root/GAP-RL")
import gap_rl.envs
from gap_rl.algorithms.rl_utils import sb3_make_multienv

config_file = "/root/GAP-RL/gap_rl/algorithms/config/egopoints_ur85_bezier2d_goalaux.yaml"
with open(config_file, "r", encoding="utf-8") as fin:
    cfg = yaml.load(fin, Loader=yaml.FullLoader)

env = sb3_make_multienv(
    env_id="PickSingleGraspnet-v0",
    robot_id=cfg["robot_id"],
    robot_init_qpos_noise=cfg["robot_init_qpos_noise"],
    shader_dir=cfg["shader_dir"],
    model_ids=["035"],
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
    rank=0,
    seed=42
)()

obs = env.reset()
obs, reward, done, info = env.step(env.action_space.sample())

print("REWARD_MODE_ON_ENV:", env.unwrapped._reward_mode)
print("REWARD:", reward)
print("DONE:", done)
print("INFO_KEYS:", info.keys())
if "approach_reward" in info:
    print("APPROACH:", info["approach_reward"])
else:
    print("NO APPROACH REWARD")

