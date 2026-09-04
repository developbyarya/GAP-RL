import torch
import torch.nn as nn
from gap_rl.algorithms.Networks.pointnet import GraspPointAppGroup

model = GraspPointAppGroup(
    in_ch=3,
    graspgroup_mlp_specs=[16, 32],
    group_mlp_specs=[64, 256],
)
total_params = sum(p.numel() for p in model.parameters())
print(f"GraspPointAppGroup Total params: {total_params}")
