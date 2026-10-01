def patch_pick_single():
    with open('gap_rl/envs/pick_single.py', 'r') as f:
        content = f.read()
    content = content.replace('info_exist_weight", 3.0', 'info_exist_weight", 0.3')
    with open('gap_rl/envs/pick_single.py', 'w') as f:
        f.write(content)

def patch_sac_train():
    with open('gap_rl/algorithms/scripts/sac_train.py', 'r') as f:
        content = f.read()
    content = content.replace('log_std_init=-3.67', 'log_std_init=-1.5')
    content = content.replace('gradient_steps=128', 'gradient_steps=64')
    content = content.replace('batch_size=128', 'batch_size=512')
    with open('gap_rl/algorithms/scripts/sac_train.py', 'w') as f:
        f.write(content)

def patch_yaml():
    with open('gap_rl/algorithms/config/egopoints_ur85_bezier2d_goalaux.yaml', 'r') as f:
        content = f.read()
    content = content.replace('n_stack: 16', 'n_stack: 4')
    content = content.replace('share_feat: True', 'share_feat: False')
    with open('gap_rl/algorithms/config/egopoints_ur85_bezier2d_goalaux.yaml', 'w') as f:
        f.write(content)

patch_pick_single()
patch_sac_train()
patch_yaml()
print("Patched successfully")
