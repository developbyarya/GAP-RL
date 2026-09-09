import re
import sys

def patch():
    with open('gap_rl/algorithms/scripts/sac_train.py', 'r') as f:
        content = f.read()

    # 1. Add to argparse
    old_args = """    parser.add_argument("--timestamp", type=str, default=None, help="exp time stamp.")
    parser.add_argument("--distill", action='store_true')"""
    new_args = """    parser.add_argument("--timestamp", type=str, default=None, help="exp time stamp.")
    parser.add_argument("--distill", action='store_true')
    parser.add_argument("--use-attn-lstm", action='store_true', help="Use Attention before LSTM.")
    parser.add_argument("--d-attn", type=int, default=384, help="Attention embedding dimension.")"""
    content = content.replace(old_args, new_args)

    # 2. Add to log_dir
    old_log_dir = """    exp_suffix = f"YCB{len(model_ids)}_{cfg['num_grasps']}_{cfg['gen_traj_mode']}_{vary_str}_{args.exp_suffix}"
    time_stamp = args.timestamp if args.timestamp is not None else time.strftime("%Y%m%d_%H%M%S", time.localtime())"""
    new_log_dir = """    attn_str = f"_attn{args.d_attn}" if args.use_attn_lstm else ""
    exp_suffix = f"YCB{len(model_ids)}_{cfg['num_grasps']}_{cfg['gen_traj_mode']}_{vary_str}_{args.exp_suffix}{attn_str}"
    time_stamp = args.timestamp if args.timestamp is not None else time.strftime("%Y%m%d_%H%M%S", time.localtime())"""
    content = content.replace(old_log_dir, new_log_dir)

    # 3. Add to CustomSAC policy_kwargs
    old_kwargs = """                extra_pred_dim=9,
                orig_observation_space=orig_obs_space,
                stack_keys=stack_keys,"""
    new_kwargs = """                extra_pred_dim=9,
                orig_observation_space=orig_obs_space,
                stack_keys=stack_keys,
                use_attn_lstm=args.use_attn_lstm,
                d_attn=args.d_attn,"""
    content = content.replace(old_kwargs, new_kwargs)

    with open('gap_rl/algorithms/scripts/sac_train.py', 'w') as f:
        f.write(content)

patch()
print("Patched sac_train.py")
