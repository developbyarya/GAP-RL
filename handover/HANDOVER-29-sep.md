# GAP-RL Project Handover (Sept 29)

## Current Status
The user is currently operating on the `lstm-sac` branch. We have successfully replicated the GAP-RL training pipeline on the YCB dataset and achieved a **50% evaluation success rate** (Experiment 4) by deploying a custom **Unshared PointNet Trunk** architecture. 

The user is now shifting focus to **cross-dataset generalization**: training and evaluating the RL policy on the **GraspNet** dataset (unseen split, 10 objects) instead of YCB.

## Major Accomplishments in this Session
1. **Experiment Compilation**: Compiled all previous `progress.csv` training logs into a master Excel spreadsheet and a Markdown summary artifact for the user's thesis.
2. **GraspNet Integration**: Implemented `PickSingleGraspnetEnv` inside `gap_rl/envs/pick_single.py` and mapped it to `graspnet_train` in `env_settings.yaml`.
3. **Dataset Generation**: Discovered the authors forgot to release the pre-computed LocalGrasp JSON (`info_localgrasp_v0.json`) for the GraspNet dataset. We built and iteratively debugged `gap_rl/examples/gen_LoG_grasps_graspnet.py` to run the authors' C++ PyTorch extension (`LoG.so`) over 72 viewpoints for all 11 GraspNet objects.

## Critical Theories & Discoveries

### 1. The Vast.ai Vulkan Corruption
**RULE:** **Never use SAPIEN camera rendering on the Vast.ai server.**
The Vast.ai instance has a corrupted Vulkan driver. Any attempt to instantiate the SAPIEN renderer (e.g., `gym.make` with cameras attached, or calling `cam.take_picture()`) immediately results in a `Segmentation fault` (Exit Code 139).
* **Workaround:** We train exclusively using mathematically precise point clouds directly from the meshes (`obs_mode="state_egopoints"`) bypassing the renderer entirely. 
* **Note:** Because dataset generation physically required cameras, the user had to pull the repo to their local laptop, run the generation script there, and upload the JSON back to the server.

### 2. The Unshared PointNet Trunk Stability
**RULE:** **Maintain the unshared trunk architecture for SAC.**
Initially, the Actor and Critic shared a single PointNet trunk. This caused severe Q-value divergence because the Actor's policy gradients overwhelmed the Critic's state representations. We separated them (giving the Critic its own isolated PointNet trunk) and concatenated the features. This slashed Critic loss by 4x and boosted success rates from 5% to 50%.

### 3. SAC Auto-Entropy Thrashing
In Experiment 4, the policy reached 50% success at 3.2M steps but subsequently collapsed to 15%. Analysis showed the SAC auto-entropy tuner (`ent_coef`) panicked as the policy became deterministic, aggressively spiking the temperature from 0.024 to 0.053 and injecting massive noise. The checkpoint at 3.2M steps remains the true optimal policy.

## Code Rot & Bugs Fixed (Lessons Learned)
The GAP-RL repository suffers from severe academic code rot. Generation scripts were abandoned after major environment refactors. If you need to touch these scripts, expect the following:

1. **Broken State Methods:** `get_state_objpoints_rt` was refactored by the authors to require an `action` tensor for RL state tracking. Older generation scripts still pass `action=None`, causing immediate crashes. We had to bypass the environment method entirely and inline the point cloud extraction/filtering directly inside the generation script.
2. **Missing Cameras:** The authors forgot to add `data_cam` to the `ur5e_robotiq85_old` camera configurations in the open-source release, breaking any scripts relying on it. We patched `gap_rl/agents/configs/ur5e_robotiq85_old/defaults.py` to formally register `data_cam`.
3. **Hardcoded Model Memory:** The authors hardcoded single objects (`model_ids=["003_cracker_box"]`) in their `gym.make` calls. Because `PickSingleEnv` only allocates memory for the IDs passed at initialization, looping through multiple objects causes `KeyError`s during reset. Always pass the full list of `model_ids` to `gym.make`.
4. **Relative Path Traps:** `sac_train.py` is often executed from `gap_rl/algorithms/scripts/`. If environments use relative paths (e.g., `ASSET_DIR="gap_rl/data"`), they will crash with `FileNotFoundError`. Always use absolute paths via `from gap_rl import format_path`.
5. **Ghost Checkpoints:** The default `--checkpoint-path` (note the dash, not underscore) for the LocalGrasp network is NOT a `.tar` file. It is a directory named `epoch_11_acc_0.915_cover_0.765` located inside `gap_rl/localgrasp/`. 

## Next Steps for the New Agent
1. **Monitor GraspNet Training:** The user is about to begin training the SAC algorithm on GraspNet using:
   `python3 gap_rl/algorithms/scripts/sac_train.py --env-id graspnet_train`
2. **Evaluate Generalization:** Once trained (or using the YCB checkpoint), evaluate performance on the unseen GraspNet split to replicate the paper's claims.
3. **Twin-SAC Plasticity (Future):** The user previously attempted to implement "TwinSAC" to solve plasticity problems. You may be asked to revisit this or other architectural changes if GraspNet performance plateaus.

### 6. IndexError Modulo Bug
In `_compute_near_grasps` (line 551), the `near4` and `near4_filter` modes calculate the indices of the 4 nearest grasping viewpoints out of the 72 possibilities. The original code forgot to apply the `% 12` modulo operator to the first `phi_ids` element. Because the angle mapping could float up to exactly 12.0 due to floating-point representation of `180` degrees, it attempted to access index `72` instead of rolling over to `0`. We patched this by safely enforcing `% 12`.

### 7. Missing Dense Grasps (Dead Auxiliary Loss)
When testing on GraspNet, we discovered the `actor_aux_loss` completely flatlined at `3.002`. This happened because `PickSingleGraspnetEnv` forgot to load `info_grasp_v0.json` into `self.grasps_mat`. Without this dense grasp matrix, the environment fallback defaulted `close_grasp_pose_ee` to `np.zeros(9)`. Because the auxiliary target became an array of zeros, the PointNet geometry extractor lost all geometric supervision and essentially died. We patched `PickSingleGraspnetEnv.__init__` and `_load_model` to properly load and structure `self.grasps_mat`.
