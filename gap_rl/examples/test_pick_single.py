import gymnasium as gym
import numpy as np
from gap_rl import ALGORITHM_DIR
from gap_rl.envs.pick_single import PickSingleYCBEnv
import yaml

with open(ALGORITHM_DIR / "config/env_settings.yaml", "r") as f:
    env_cfg = yaml.load(f, Loader=yaml.FullLoader)

env = gym.make(
    "PickSingleYCB-v0",
    shader_dir="ibl",
    robot="ur5e_robotiq85_old",
    model_ids=env_cfg["ycb_train"]["model_ids"][:2],
    obs_mode="state_egopoints",
    reward_mode="dense",
    control_mode="pd_ee_delta_pose_euler",
    robot_x_offset=0.56,
    sim_freq=150,
    control_freq=5,
    num_grasps=40,
)

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
print()

obs, _ = env.reset(model_id=env_cfg["ycb_train"]["model_ids"][0])
print("Obs keys:", list(obs.keys()))
for k, v in obs.items():
    print(f"  {k}: {v.shape}, dtype={v.dtype}")

for i in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(f"\nStep {i+1}: reward={reward:.4f}, done={done}")
    print(f"  action={np.round(action, 4)}")

env.close()
print("\nDone!")
