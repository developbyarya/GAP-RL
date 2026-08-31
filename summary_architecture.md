# GAP-RL: Architecture & Code Summary

This document highlights the critical implementation details across the GAP-RL pipeline, focusing on data extraction, architectural design, network structures, and RL integration.

---

## 1. Workspace Filtering (Graspable Region Explorer)
In the simulation environment, point clouds are explicitly filtered to isolate the workspace and target objects. This restricts the feature space to graspable regions, significantly improving the sample efficiency of subsequent components.

```python
# gap_rl/utils/geometry.py - Filtering Logic
def pointcloud_filter(points, *xyz_min_max):
    """Filters point cloud within the bounds of xyz_min_max (Workspace bounding box)"""
    xyz_min_max = xyz_min_max[0]
    mask1 = np.logical_and(points[:, 0] > xyz_min_max[0][0], points[:, 0] < xyz_min_max[0][1])
    mask2 = np.logical_and(points[:, 1] > xyz_min_max[1][0], points[:, 1] < xyz_min_max[1][1])
    mask3 = np.logical_and(points[:, 2] > xyz_min_max[2][0], points[:, 2] < xyz_min_max[2][1])
    mask = np.logical_and(np.logical_and(mask1, mask2), mask3)
    return points[mask], mask

def pc_bbdx_filter(points, bbox_corners):
    """Filters point cloud inside an object's bounding box"""
    min_bound = np.min(bbox_corners, axis=0)
    max_bound = np.max(bbox_corners, axis=0)
    inside_mask = np.all(np.logical_and(min_bound <= points, points <= max_bound), axis=1)
    return points[inside_mask], inside_mask

# gap_rl/envs/pick_single.py - Application
scene_pc, mask = pointcloud_filter(scene_pc, ground_ws)
_, mask = pc_bbdx_filter(scene_pc_obj, obj_bbdx_v)
obj_pc_filter = scene_pc[mask]
```

---

## 2. Grasp Detection
Computes dense candidate grasps locally in simulation. While offline JSON data (`self.grasps_mat_ee`) is typically used during training to load precomputed grasps for efficiency, the framework leverages LocalGrasp (LoG) dynamically during evaluation (`sac_LoG_dynamic_eval.py`) to generate grasps on-the-fly from the point cloud.

Using spatial relations, angles ($\theta$, $\phi$), and distance between the target object and the robot TCP (Tool Center Point), candidate grasps are sampled and filtered within the local observation space.

```python
# gap_rl/algorithms/scripts/sac_LoG_dynamic_eval.py - Dynamic LoG Setup
from gap_rl.localgrasp.LoG import lg_parse, LgNet, GraspGroup

# Initialize LocalGrasp network for dynamic on-the-fly generation
lgNet = LgNet(args)

# gap_rl/envs/pick_single.py - Computing/Filtering the grasps (whether from JSON or LoG)
def _compute_near_grasps(self):
    # Calculates relative pose and geometric angles between TCP and Object
    cam_pose = self._cameras['hand_realsense'].camera.get_model_matrix()
    obj_tcp_vec = cam_pose[:3, 3] - self.obj_pose.p
    dist = np.linalg.norm(obj_tcp_vec)
    
    # Angle calculations for generating/filtering local grasp candidates
    obj_rot_mat = self.obj_pose.to_transformation_matrix()[:3, :3]
    x, y, z = obj_rot_mat[:, 0], obj_rot_mat[:, 1], obj_rot_mat[:, 2]
    vec_z_dot = np.dot(obj_tcp_vec, z)
    theta = np.pi / 2 - np.arccos(vec_z_dot / dist)
    
    # Angular threshold filtering to prune unlikely grasp candidates
    if "filter" in self.grasp_select_mode:
        ## filter the grasp (<direction, grasp direction> > 60 deg)
        filter_idx = grasps_mat_ee[:, 2, 2] > np.cos(np.pi / 3)
        grasp_ids = grasp_ids[filter_idx]
```

---

## 3. Grasp Encoder (Grasps as Points)
To map SE(3) representations into dense spatial features without overfitting to specific gripper geometries, each 6D grasp pose is expanded into a set of 3D Gaussian points relative to the gripper.

```python
# Grasps as Points Initialization
# Generates gaussian distributed points based on gripper width (wg)
wg = self.gripper_w
gripper_pts_init = np.random.normal(0.0, wg / 6, size=(num_grasp_points, 3))

# gap_rl/utils/geometry.py
def transform_points(H, pts):
    """Applies a 4x4 transformation matrix (H) to a point set"""
    pts_pos = pts[:, :3]
    trans_pos = pts_pos @ H[:3, :3].T + H[:3, 3]
    return np.concatenate((trans_pos, pts[:, 3:]), axis=1)

# Application: Transform the gaussian points into the frame of each candidate grasp
pts_ee_all = []
for i in range(num_grasps):
    pts_ee = transform_points(grasps_mat_ee[i], gripper_pts_init)
    pts_ee_all.append(pts_ee)
```

---

## 4. GraspGroupNet
The neural architecture that processes the Gaussian Point Encoded grasps. It uses symmetric 1D Convolution layers to capture intra-grasp relationships (the features of the current grasp) and inter-grasp relationships (spatial contexts comparing the identity grasp against candidate grasps), outputting the compact $O_{grasp}$ feature array.

```python
# gap_rl/algorithms/Networks/pointnet.py
class GraspPointAppGroup(nn.Module):
    def __init__(self, in_ch=3, graspgroup_mlp_specs=[16, 32], group_mlp_specs=[64, 128]):
        super().__init__()
        
        # Intra-grasp Convolution Layers (capturing details of a single grasp point cloud)
        self.grasp_conv1 = torch.nn.Conv1d(in_ch, graspgroup_mlp_specs[0], 1)
        self.grasp_norm1 = nn.LayerNorm(graspgroup_mlp_specs[0], eps=1e-6)
        self.grasp_conv2 = torch.nn.Conv1d(graspgroup_mlp_specs[0], graspgroup_mlp_specs[1], 1)
        self.grasp_norm2 = nn.LayerNorm(graspgroup_mlp_specs[1], eps=1e-6)

        # Inter-grasp Convolution Layers (capturing relationships between grasp poses)
        self.group_conv1 = torch.nn.Conv1d(3 + graspgroup_mlp_specs[-1] * 2, group_mlp_specs[0], 1)
        self.group_norm1 = nn.LayerNorm(group_mlp_specs[0], eps=1e-6)
        self.group_conv2 = torch.nn.Conv1d(group_mlp_specs[0], group_mlp_specs[1], 1)
        self.group_norm2 = nn.LayerNorm(group_mlp_specs[1], eps=1e-6)

    def forward(self, x_init, x):
        # x size: (B, N, K, 3 + C), x_init size: (B, K, 3 + C)
        bat, n_pts, k_pts, feat_num = x.size()
        
        # Concatenate identity grasp points with candidate grasp points
        intra_grasp_feat = torch.cat([x_init.unsqueeze(1), x], dim=1)  # (B, N+1, K, 3+C)
        
        # Extract features
        x = intra_grasp_feat.view(bat, (n_pts+1) * k_pts, feat_num).transpose(2, 1).contiguous()
        x = F.relu(self.grasp_norm1(self.grasp_conv1(x).transpose(2, 1).contiguous()))
        x = x.transpose(2, 1).contiguous()
        x = F.relu(self.grasp_norm2(self.grasp_conv2(x).transpose(2, 1).contiguous()))
        
        # ... additional pooling operations ...
        return feature_vector # The O_grasp abstract guidance tensor
```

---

## 5. RL Policy Learning
The $O_{grasp}$ features from GraspGroupNet are concatenated with traditional RL observations (TCP pose, gripper states, etc.) inside a custom Soft-Actor Critic (SAC) feature extractor.

```python
# gap_rl/algorithms/rl_utils.py
class CustomGraspPointGroupExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, device="cuda:0"):
        super().__init__(observation_space, features_dim=1)

        # GraspGroupNet Initialization
        self.gpcg = GraspPointAppGroup(
            in_ch=3,
            graspgroup_mlp_specs=[16, 32],
            group_mlp_specs=[64, 256],
        ).to(torch.device(device))
        
        # Linear map for standard RL vector state (tcp, gripper pos, etc.)
        self.state_map = torch.nn.Linear(20, 128, bias=True).to(torch.device(device))
        
        # Final RL Agent input dimension (256 from GGN + 128 from state map)
        self._features_dim = 256 + 128

    def forward(self, observations) -> torch.Tensor:
        # 1. Unpack observation dictionary
        gripper_pos = observations["gripper_pos"]            # (N, 2)
        ee_pose_base = observations["tcp_pose"]              # (N, 6)
        action = observations["action"]                      # (N, 7)
        grasp_exist = observations["grasp_exist"]            # (N, 5)
        origin_gripper_pts = observations["origin_gripper_pts"]  # Identity points
        gripper_pts_diff = observations["gripper_pts_diff"]      # Transformed candidates

        # 2. Extract GraspGroupNet Features (O_grasp)
        pn_feature = self.gpcg(origin_gripper_pts, gripper_pts_diff)  # (N, 256)
        
        # 3. Extract other state features
        state = torch.cat([ee_pose_base, gripper_pos, action, grasp_exist], dim=1)
        state_feature = self.state_map(state)

        # 4. Concatenate and return to SAC Policy Network
        return torch.cat((state_feature, pn_feature), dim=1)
```

---

## 6. Policy Architecture (Actor-Critic)
The RL algorithm extends standard Soft Actor-Critic (SAC) using customized Actor and Critic networks that process temporal sequences. The raw RL state arrays along with the abstract $O_{grasp}$ features are tracked across multiple frames and processed by a Long Short-Term Memory (LSTM) network to inject temporal context into the policy and Q-value predictions.

Additionally, auxiliary prediction heads (`extra_pred`) are attached to the Actor and Critic to enforce geometric regularization (e.g., predicting 7D or 9D target grasp pose vectors from the LSTM latent state).

```python
# gap_rl/algorithms/scripts/custom_sac.py - Sequence Processing & LSTM
def _extract_windowed_features(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
    n_stack = obs[self.stack_keys[0]].shape[1]
    frame_feats = []
    
    # Process each frame independently through GraspPointAppGroup (GGN)
    with torch.set_grad_enabled(not self.share_features_extractor):
        for t in range(n_stack):
            obs_t = {k: (v[:, t] if k in self.stack_keys else v) for k, v in obs.items()}
            obs_t = preprocess_obs(obs_t, self.orig_observation_space, normalize_images=self.normalize_images)
            frame_feats.append(self.features_extractor(obs_t))
            
    # Stack temporal features and process through LSTM
    seq = torch.stack(frame_feats, dim=1)
    lstm_out, _ = self.lstm(seq)
    
    # Return the final LSTM hidden state
    return lstm_out[:, -1, :]

# Action / Value Prediction (Actor/Critic)
def forward(self, obs: torch.Tensor, actions: torch.Tensor):
    features = self._extract_windowed_features(obs)
    qvalue_input = torch.cat([features, actions], dim=-1)
    
    # Optional Auxiliary Geometric Prediction
    extra_pred = None
    if self.extra_pred_dim:
        extra_pred = self.extra_pred(features)
        # Normalizing rotation components of the prediction (e.g. quaternions/basis vectors)
        extra_pred = torch.cat(
            (F.normalize(extra_pred[..., :4], p=2, dim=-1), extra_pred[..., 4:]), dim=-1
        )
        
    ... # Compute Q-Values
```

---

## 7. Action Log-Std Clamping (CustomActor)
In `gap_rl/algorithms/scripts/custom_sac.py`, standard deviation clamping is performed during the extraction of action distribution parameters. This keeps the continuous control output distributions stable, preventing excessively wide or narrow standard deviations in SAC.

```python
# gap_rl/algorithms/scripts/custom_sac.py
def get_action_dist_params(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    features = self._extract_windowed_features(obs)
    latent_pi = self.latent_pi(features)
    mean_actions = self.mu(latent_pi)

    if self.use_sde:
        return mean_actions, self.log_std, dict(latent_sde=latent_pi)
        
    log_std = self.log_std(latent_pi)
    
    # Critical step: clamp the standard deviation
    log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
    
    return mean_actions, log_std, {}
```

---

## 8. Dense Reward Computation
The dense reward function constructs shaped rewards that gently guide the RL agent towards the goal. It combines approach terms (translation and rotation distances towards the nearest valid grasp), a grasping term, a lifting/goal-reach term, and a static stability term.

```python
# gap_rl/envs/pick_single.py
def compute_dense_reward(self, info, **kwargs):
    reward = 0.0
    approach_reward = 0.0
    grasp_reward = 0.0
    goal_reward = 0.0
    static_reward = 0.0

    is_robot_static, is_obj_grasp, is_obj_static, is_obj_lift = info["evaluate_info"]
    is_success = info["is_success"]

    info_exist_reward = 3 if info["is_info_exist"] else 0

    if is_success:
        # High sparse reward for full success
        reward = 15.0
    else:
        # 1. Approach Reward (Distance and Rotation to nearest candidate grasp)
        translation_dist, rotation_dist, grasp_id = self.compute_grasps_dist()

        approach_reward = 3 * (1 - np.tanh(5.0 * translation_dist[grasp_id])) + \
                          3 * (1 - np.tanh(5.0 * rotation_dist[grasp_id]))

        # 2. Grasp Reward (Binary bonus when grasping the object)
        grasp_reward = 3.0 if is_obj_grasp else 0.0

        # 3. Reach-Goal Reward (Z-height lifting progression)
        if is_obj_grasp:
            goal_dist_z = np.clip(
                self.goal_thresh - self.obj_pose.p[2], 0, self.goal_thresh
            )  # Maps distance [0, 0.2]
            goal_reward = 2 * (1 - np.tanh(5 * goal_dist_z))

            # 4. Robot Static Reward (Requires both the object and robot to be stable once lifted)
            if is_obj_lift:
                static_reward = 1 if is_obj_static and is_robot_static else 0

        # Compile total stepwise reward
        reward += info_exist_reward + approach_reward + grasp_reward + static_reward + goal_reward

    # Scaled down to prevent overly large Q-values
    reward = reward * 0.5
    
    # ... returns reward and metrics dict
    return reward
```

---

## 9. SAC Log-Std Constants
The bounds for the `log_std` clamping are imported directly from `stable_baselines3` defaults. They prevent numerical instability by ensuring the continuous action distribution standard deviation never explodes or collapses to zero.
```python
# gap_rl/algorithms/scripts/custom_sac.py
from stable_baselines3.sac.policies import Actor, LOG_STD_MAX, LOG_STD_MIN

# Under the hood in Stable Baselines 3:
# LOG_STD_MAX = 2
# LOG_STD_MIN = -20

# Usage in get_action_dist_params
log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
```

---

## 10. Evaluation Metrics (is_obj_grasp & is_info_exist)
The RL environment evaluates success statuses based on simulated physics and geometric vision constraints.

**Object Grasped (`is_obj_grasp`):**
Verified via physics simulation by measuring the collision impulses between the gripper fingers and the object. It ensures the grasp force is strong enough and applied in the correct inward direction.
```python
# gap_rl/agents/robots/ur5e_robotiq85_old.py
def check_grasp(self, actor: sapien.ActorBase, min_impulse=1e-6, max_angle=85): # Evaluated with max_angle=30 in pick_single.py
    contacts = self.scene.get_contacts()

    # Calculate impulse magnitudes and directions
    limpulse = get_pairwise_contact_impulse(contacts, self.finger1_link, actor)
    rimpulse = get_pairwise_contact_impulse(contacts, self.finger2_link, actor)

    # Direction to open the gripper
    ldirection = -self.finger1_link.pose.to_transformation_matrix()[:3, 1]
    rdirection = -self.finger2_link.pose.to_transformation_matrix()[:3, 1]

    # Angle between collision impulse and open direction
    langle = compute_angle_between(ldirection, limpulse)
    rangle = compute_angle_between(rdirection, rimpulse)

    # Verify strong inward force for both fingers
    lflag = np.linalg.norm(limpulse) >= min_impulse and np.rad2deg(langle) <= max_angle
    rflag = np.linalg.norm(rimpulse) >= min_impulse and np.rad2deg(rangle) <= max_angle

    return lflag and rflag
```

**Information Exists (`is_info_exist`):**
Verifies whether the current LocalGrasp candidates are actually visible inside the ego-centric wrist camera frame. It transforms the 3D candidate grasp centers into the 2D camera UV pixel space and checks bounding limits.
```python
# gap_rl/envs/pick_single.py
def _get_grasp_exist_mask(self):
    # Transform grasp candidate centers into the hand camera frame
    trans_ee2world = self.tcp.pose.to_transformation_matrix()
    trans_ee2cam = trans_world2cam @ trans_ee2world
    grasps_centers_cam = transform_points(trans_ee2cam, self.grasps_mat_ee[:, :3, 3])

    # Project 3D camera coordinates to 2D image plane (uvz)
    grasps_centers_uvz = xyz2uvz(grasps_centers_cam, handcam_intrin)

    # Mask validates if the point is within the image dimensions
    grasp_exist_mask = (
        (grasps_centers_uvz[:, 0] > 0)
        & (grasps_centers_uvz[:, 0] < self.cam_paras["hand_realsense_width"])
        & (grasps_centers_uvz[:, 1] > 0)
        & (grasps_centers_uvz[:, 1] < self.cam_paras["hand_realsense_height"])
    )
    return grasp_exist_mask

def evaluate(self, **kwargs):
    # Evaluates true if ANY candidate grasp is visible in the camera frame
    is_info_exist = np.any(self._get_grasp_exist_mask())
    ...
```
