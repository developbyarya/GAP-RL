import sys
import yaml
import numpy as np
sys.path.append("/root/GAP-RL")
from gap_rl.envs.pick_single import PickSingleGraspnetEnv

config_file = "/root/GAP-RL/gap_rl/algorithms/config/egopoints_ur85_bezier2d_goalaux.yaml"
with open(config_file, "r", encoding="utf-8") as fin:
    cfg = yaml.load(fin, Loader=yaml.FullLoader)

env = PickSingleGraspnetEnv(
    model_ids=["035"],
    obs_mode=cfg["obs_mode"],
    reward_mode=cfg["reward_mode"],
    grasp_points_mode=cfg["grasp_points_mode"],
    num_grasps=cfg["num_grasps"],
    num_grasp_points=cfg["num_grasp_points"],
    robot_id=cfg["robot_id"],
    renderer_kwargs={"offscreen_only": True}
)

obs = env.reset()
action = env.action_space.sample()
obs, reward, done, info = env.step(action)

print("REWARD:", reward)
print("HAS NAN IN OBS:", any(np.isnan(v).any() for v in obs.values()))
