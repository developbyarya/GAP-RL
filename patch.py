import re

with open('gap_rl/algorithms/scripts/sac_train.py', 'r') as f:
    content = f.read()

eval_env_code = """
    # Setup periodic evaluation environment (1 process)
    eval_vec_env = SubprocVecEnv(
        [
            sb3_make_multienv(
                env_id=env_id,
                robot_id=cfg["robot_id"],
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
                device=cfg["device"],
                rank=cfg.get("train_procs", 1),  # Offset rank
                seed=seed + 100,  # Offset seed
            )
            for i in range(1)
        ],
        start_method="spawn",
    )
    if is_goal_aux:
        eval_vec_env = FrameStackWrapper(eval_vec_env, n_stack=n_stack, stack_keys=stack_keys)
    eval_vec_env = VecMonitor(eval_vec_env)

    from custom_eval_callback import DeterministicEvalCallback
    eval_callback = DeterministicEvalCallback(
        eval_env=eval_vec_env,
        eval_freq=200000 // cfg.get("train_procs", 1),
        n_eval_episodes=20,
        best_model_save_path=f"{log_dir}/best_checkpoint/"
    )
"""

content = content.replace(
    'checkpoint_callback = CheckpointCallback(save_freq=400000 // cfg.get("train_procs", 1), save_path=log_dir)',
    'checkpoint_callback = CheckpointCallback(save_freq=200000 // cfg.get("train_procs", 1), save_path=log_dir)'
)

content = content.replace(
    'vec_env = VecMonitor(vec_env, log_dir)',
    'vec_env = VecMonitor(vec_env, log_dir)\n' + eval_env_code
)

content = content.replace(
    'callback=[checkpoint_callback, reward_cb],',
    'callback=[checkpoint_callback, reward_cb, eval_callback],'
)

with open('gap_rl/algorithms/scripts/sac_train.py', 'w') as f:
    f.write(content)
