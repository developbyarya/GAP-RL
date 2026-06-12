import gymnasium as gym
import numpy as np
from gap_rl import ALGORITHM_DIR
from gap_rl.envs.pick_single import PickSingleYCBEnv
import yaml


def main():
    with open(ALGORITHM_DIR / "config/env_settings.yaml") as f:
        env_cfg = yaml.load(f, Loader=yaml.FullLoader)

    model_ids = env_cfg["ycb_train"]["model_ids"][:1]

    env = gym.make(
        "PickSingleYCB-v0",
        shader_dir="ibl",
        robot="ur5e_robotiq85_old",
        model_ids=model_ids,
        obs_mode="state_egopoints",
        reward_mode="dense",
        control_mode="pd_ee_delta_pose_euler",
        robot_x_offset=0.56,
        sim_freq=500,
        control_freq=20,
        num_grasps=40,
        max_episode_steps=1000000,
    )

    print(f"Object: {model_ids[0]}")
    print("Close the viewer window or press Ctrl+C to quit.\n")

    obs, _ = env.reset()

    try:
        while True:
            action = np.zeros(7, dtype=np.float32)
            obs, rew, terminated, truncated, info = env.step(action)
            env.unwrapped.render()
            if terminated or truncated:
                obs, _ = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        print("Exited.")


if __name__ == "__main__":
    main()
