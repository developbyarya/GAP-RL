# T-GAP-RL Reward-Shaping / Exploration-Collapse Diagnosis

## Findings from Code Audit

### `info_exist_reward` weight and location

- **Value**: `3` (integer, hard-coded; not configurable)
- **Set at**: [`gap_rl/envs/pick_single.py:1176`](file:///home/developbyarya/Projects/GAP-RL/gap_rl/envs/pick_single.py#L1176)
  ```python
  info_exist_reward = 3 if info["is_info_exist"] else 0
  ```
- **Added to reward at**: line 1203:
  ```python
  reward += info_exist_reward + approach_reward + grasp_reward + static_reward + goal_reward
  ```
- **Final scaling**: `reward = reward * 0.5` (line 1205), so effective
  per-step contribution when active: **1.5**.

### How `is_info_exist` is computed

`evaluate()` (line 1068) calls `_get_grasp_exist_mask()` (line 724) which:
1. Projects all LocalGrasp candidate centers into the `hand_realsense` camera's
   2D image plane via `xyz2uvz()`.
2. Returns a boolean mask: `True` for each grasp whose UV coordinate falls
   inside the camera resolution bounds.
3. `is_info_exist = np.any(grasp_exist_mask)` — **True if even one grasp
   candidate is visible** in the wrist camera.

Because the wrist camera is mounted on the gripper looking at the nearby
object on a bounded conveyor, this is expected to be True on nearly every
step under a random policy.

### Reward component ceilings (pre-`*0.5` scaling)

| Component            | Max value | Condition                             |
|----------------------|-----------|---------------------------------------|
| `info_exist_reward`  | 3.0       | Any grasp in camera FOV (near-free)   |
| `approach_reward`    | 6.0       | `3*(1-tanh(5*trans)) + 3*(1-tanh(5*rot))` |
| `grasp_reward`       | 3.0       | Object grasped (physics contact)      |
| `goal_reward`        | 2.0       | Object lifted to `goal_thresh` height |
| `static_reward`      | 1.0       | Lifted + robot & object both static   |
| **success bonus**    | 15.0      | All four evaluate flags true          |

### `target_entropy`

- **Value**: `"auto"` in `sac_train.py` (not passed explicitly → defaults to
  `"auto"` in `CustomSAC.__init__`)
- **Resolved in** `CustomSAC._setup_model()` (line 632–633):
  ```python
  if self.target_entropy == "auto":
      self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
  ```
- The action space is `pd_ee_delta_pose_euler` (3 pos + 3 euler) + 1 gripper =
  **7 dims**, so `target_entropy = -7.0`.

### `ent_coef` init value and schedule

- **Passed as**: `ent_coef="auto_0.2"` in `sac_train.py` (lines 138, 167)
- **Resolved in** `CustomSAC._setup_model()` (lines 637–643):
  ```python
  init_value = float(self.ent_coef.split("_")[1])  # = 0.2
  self.log_ent_coef = th.log(th.ones(1) * init_value).requires_grad_(True)
  self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
  ```
  So `ent_coef` starts at **0.2** and is tuned automatically via Adam toward
  the `target_entropy = -7.0` target.

### `learning_starts`

- **Value**: `800` (lines 143, 172 of `sac_train.py`)
- With `train_procs=8` (goalaux config) and `train_freq=64`, this means
  gradient updates begin after `800` total environment steps have been
  collected across all 8 workers (i.e. 100 steps per worker, or 1 episode
  each).

### `LOG_STD_MIN` / `LOG_STD_MAX`

- Imported from `stable_baselines3.sac.policies` (line 28 of `custom_sac.py`).
- SB3 v1.x / v2.x defaults: `LOG_STD_MIN = -20`, `LOG_STD_MAX = 2`.
- However: the training uses `use_sde=True` with `log_std_init=-3.67`, which
  means the actor uses **State-Dependent Exploration** (gSDE) rather than a
  per-action-dim `log_std` network. In gSDE mode, `get_action_dist_params()`
  takes the SDE path (`return mean_actions, self.log_std, dict(latent_sde=...)`)
  and the clamp path is **not reached**. The `log_std` is a learnable parameter
  initialized at `-3.67` (i.e. `std ≈ 0.025`), and the SDE noise matrix is
  reset periodically.

### `train/std` — how it's logged

The `train/std` metric is **not** explicitly logged by `CustomSAC.train()`.
It is logged by SB3's base `SAC._excluded_save_params()` / `collect_rollouts()`
internals when `use_sde=True` — SB3 records the current exploration std
via `self.actor.get_std()`. With `log_std_init=-3.67` ≈ `exp(-3.67) ≈ 0.025`,
this is consistent with the observed 0.025–0.037 range never rising.

### `rollout/success_rate` — how it's logged

SB3's `VecMonitor` (applied at line 122 of `sac_train.py`) automatically
detects the `is_success` key in the `info` dict returned by the env's
`evaluate()` method and tracks it as `rollout/success_rate` in CSV/TB logs.

---

## Step 2: Diagnostic Script

### Command

```bash
cd gap_rl/algorithms/scripts && \
python ../../scripts/debug_info_exist_rate.py --episodes 20
```

Or from repo root:

```bash
python scripts/debug_info_exist_rate.py --episodes 20
```

### Interpreting the output

- **If `is_info_exist` fraction > ~0.8**: Confirms the hypothesis.
  `info_exist_reward` fires on nearly every step regardless of policy quality.
  Its contribution to total reward will be dominant because `approach_reward`
  under a random policy will be moderate (the robot rarely aligns with a
  grasp), and `grasp_reward` / `goal_reward` will be near zero.
- **If `is_info_exist` fraction < ~0.5**: Weakens the hypothesis. The wrist
  camera does lose sight of grasps frequently, so the term isn't trivially
  free.

---

## Step 4: Ablation Runs

### Control run (current behavior)

```bash
cd gap_rl/algorithms/scripts && \
python ../../scripts/train_reward_ablation.py \
    --config-name egopoints_ur85_bezier2d_goalaux \
    --info-exist-weight 3.0 \
    --total-timesteps 300000 \
    --seed 1029
```

### Test run (reduced `info_exist_reward`)

```bash
cd gap_rl/algorithms/scripts && \
python ../../scripts/train_reward_ablation.py \
    --config-name egopoints_ur85_bezier2d_goalaux \
    --info-exist-weight 0.3 \
    --total-timesteps 300000 \
    --seed 1029
```

### Why `0.3` instead of `0.0`

Setting `info_exist_reward` to `0.0` would completely remove the signal that
tells the agent "there are valid grasps in your field of view." While the
hypothesis is that the _magnitude_ is the problem (not the signal itself), the
signal may still carry useful information — it softly encourages the agent to
keep the object in the camera FOV, which is a prerequisite for the
GraspGroupNet features to be meaningful.

`0.3` reduces the effective per-step contribution from `1.5` (after `*0.5`
scaling) down to `0.15` — roughly 10× smaller, which puts it well below the
`approach_reward` ceiling of `3.0` (post-scaling) and makes it unlikely to
dominate the Q-landscape. If the ablation at `0.3` shows improvement, a
follow-up at `0.0` can confirm whether the signal has any value at all.

---

## Run on Remote GPU Machine

### 1. Diagnostic (Step 2) — no GPU model needed, just the simulator

```bash
cd /root/GAP-RL && \
python scripts/debug_info_exist_rate.py --episodes 20
```

Expected runtime: ~2–5 minutes (20 episodes × 100 steps, single env, random
actions, no training).

### 2. Ablation Runs (Step 4) — needs GPU for SAC training

**Control** (existing reward):
```bash
cd /root/GAP-RL/gap_rl/algorithms/scripts && \
python ../../scripts/train_reward_ablation.py \
    --config-name egopoints_ur85_bezier2d_goalaux \
    --info-exist-weight 3.0 \
    --total-timesteps 300000 \
    --seed 1029
```

**Test** (reduced reward):
```bash
cd /root/GAP-RL/gap_rl/algorithms/scripts && \
python ../../scripts/train_reward_ablation.py \
    --config-name egopoints_ur85_bezier2d_goalaux \
    --info-exist-weight 0.3 \
    --total-timesteps 300000 \
    --seed 1029
```

Expected runtime per run: ~30–60 minutes for 300k steps with 8 SubprocVecEnv
workers on a single GPU.

### 3. What to compare

Output directories will be:
- `runs/reward_ablation_<timestamp>/` (each run gets its own timestamped dir)

Open TensorBoard or compare the `progress.csv` files:

```bash
tensorboard --logdir runs/ --port 6007
```

**Key metrics to watch:**

| Metric | What to look for | Supports reducing `info_exist_reward` |
|--------|------------------|---------------------------------------|
| `train/std` (first 50k steps) | Does std stay pinned near 0.025 in control but recover/stay higher in test? | Yes if test std > control std |
| `rollout/success_rate` (full 300k) | Does the test run achieve higher peak success rate? | Yes if test > control |
| `reward/info_exist_reward` | Is this the dominant reward component in the control run? | Yes if it's > 50% of total |
| `reward/approach_reward` | Does the test run show more approach_reward learning signal? | Yes if test develops higher approach signal |
| `train/ent_coef` | Does `ent_coef` collapse to near-zero in control but stay higher in test? | Yes if test ent_coef > control ent_coef |

**Decision criteria:**

- **Permanently reduce `info_exist_reward`** if: (a) the diagnostic confirms
  `is_info_exist` fires on >80% of random-policy steps, AND (b) the test run
  at 0.3 shows meaningfully higher `rollout/success_rate` or prevents
  `train/std` collapse over the first 50k steps.
- **Keep current value** if: the test run shows no improvement or regresses,
  suggesting the exploration collapse has a different root cause (e.g., gSDE
  `log_std_init=-3.67` starting too low, aux loss weights overwhelming the
  RL gradient, or the LSTM architecture itself).
