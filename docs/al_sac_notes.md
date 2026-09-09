# AL-SAC Implementation Notes

## Step 1 Findings
I have inspected `gap_rl/algorithms/rl_utils.py` where `CustomGraspPointGroupExtractor` is defined.
The per-frame feature components existing before concatenation are:
1. `pn_feature` (from PointNet/GraspGroupNet) - shape `(N, 256)`
2. `ee_pose_base` (from `tcp_pose`) - shape `(N, 6)`
3. `gripper_pos` - shape `(N, 2)`
4. `action` - shape `(N, 7)`
5. `grasp_exist` - shape `(N, 5)`

These 5 distinct semantic components are the ones present before they are combined. Currently, the code concatenates `ee_pose_base, gripper_pos, action, grasp_exist` into a `state` tensor of shape `(N, 20)`, passes it through a linear `state_map` layer to get a `(N, 128)` tensor, and then concatenates that with `pn_feature (N, 256)` to form a single `(N, 384)` per-frame feature vector.

## Step 2 Design: Token Structure
Since the original paper didn't specify how to convert the flat vector into a sequence of tokens for self-attention, we are defining a deliberate token structure for this codebase:
- We treat each of the 5 identified components (`pn_feature`, `ee_pose_base`, `gripper_pos`, `action`, `grasp_exist`) as a separate token.
- Each component is projected through its own small `Linear` layer into a shared embedding dimension `d_attn`.
- `d_attn` defaults to `features_dim` (which is 384).
- The tokens are stacked into a `(batch, 5, d_attn)` tensor.
- We apply standard single-head scaled dot-product self-attention across these 5 tokens.
- Finally, the attended tokens are concatenated ("flattened") into a single per-frame vector of size `(batch, 5 * d_attn)`.
- This concatenated vector is fed into the LSTM. We adjust the LSTM's `input_size` to `5 * d_attn` while keeping its `hidden_size` at `features_dim` (384), ensuring the rest of the network (MLP heads) requires no resizing.

## Step 5 Remote Commands
To verify shape and integration bugs, run a short sanity-check training run (~20k steps):
```bash
# Temporarily edit gap_rl/algorithms/scripts/sac_train.py to total_timesteps=20_000 before running this if desired,
# or simply cancel the script after a few epochs.
cd gap_rl/algorithms/scripts
python sac_train.py --config-name egopoints_ur85_bezier2d_goalaux --exp-suffix AL_SAC_sanity_check --use-attn-lstm
```

For the full training run matched to the current best run:
```bash
cd gap_rl/algorithms/scripts
python sac_train.py --config-name egopoints_ur85_bezier2d_goalaux --exp-suffix AL_SAC_full_run --use-attn-lstm
```
*(Note: The existing best-known-good run without attention serves as the baseline for this comparison. No need to re-run it.)*
