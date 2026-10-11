"""
Evaluation metrics and diagnostics for GAP-RL policies (LSTM and non-LSTM).

Localizes where episodes fail (stage funnel, failure labels), evaluates control
quality (smoothness, timings), and measures temporal history utilization (counterfactual
window ablation, latent representation diagnostics, aux prediction and critic Q quality).
"""

import os
import json
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from copy import deepcopy

import numpy as np

try:
    import torch
except (ImportError, OSError):
    torch = None


# ==============================================================================
# 1. Statistical Helpers (Wilson Interval, Bootstrap CI, AUROC, Effective Rank)
# ==============================================================================

def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Wilson score interval for binomial proportion with continuity-aware center.
    Returns (ci_lower, ci_upper).
    """
    if n <= 0:
        return 0.0, 0.0
    # Standard normal quantile for 95%: ~1.95996
    z = 1.959963984540054 if abs(confidence - 0.95) < 1e-4 else 1.95996
    p_hat = float(k) / float(n)
    denom = 1.0 + (z ** 2) / n
    center = (p_hat + (z ** 2) / (2.0 * n)) / denom
    margin = (z * np.sqrt((p_hat * (1.0 - p_hat) / n) + (z ** 2) / (4.0 * (n ** 2)))) / denom
    ci_lower = max(0.0, center - margin)
    ci_upper = min(1.0, center + margin)
    return float(ci_lower), float(ci_upper)


def bootstrap_mean_ci(
    data: Sequence[Union[float, int]],
    num_resamples: int = 1000,
    ci: float = 0.95,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[float, float]:
    """
    Nonparametric bootstrap confidence interval for the mean.
    Filters out None and NaN values.
    """
    clean_data = [float(x) for x in data if x is not None and not np.isnan(x)]
    if len(clean_data) == 0:
        return float("nan"), float("nan")
    if len(clean_data) == 1:
        return clean_data[0], clean_data[0]

    if rng is None:
        rng = np.random.RandomState(42)

    arr = np.array(clean_data, dtype=np.float64)
    n = len(arr)
    indices = rng.randint(0, n, size=(num_resamples, n))
    boot_means = np.mean(arr[indices], axis=1)
    alpha = (1.0 - ci) / 2.0
    ci_lower = np.percentile(boot_means, 100.0 * alpha)
    ci_upper = np.percentile(boot_means, 100.0 * (1.0 - alpha))
    return float(ci_lower), float(ci_upper)


def compute_auroc(y_true: Sequence[int], y_score: Sequence[float]) -> Optional[float]:
    """
    Computes AUROC using the Mann-Whitney U rank statistic.
    Avoids heavy external dependencies. Returns None if single-class or no valid samples.
    """
    valid = [
        (bool(yt), float(ys))
        for yt, ys in zip(y_true, y_score)
        if ys is not None and not np.isnan(ys)
    ]
    if not valid:
        return None
    pos_scores = [ys for yt, ys in valid if yt]
    neg_scores = [ys for yt, ys in valid if not yt]
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return None

    pos_arr = np.array(pos_scores, dtype=np.float64)
    neg_arr = np.array(neg_scores, dtype=np.float64)
    diff = pos_arr[:, None] - neg_arr[None, :]
    u = np.sum(diff > 0) + 0.5 * np.sum(diff == 0)
    return float(u / (n_pos * n_neg))


def compute_effective_rank(matrix: np.ndarray) -> float:
    """
    Computes effective rank (Roy & Vetterli, 2007) of a 2D matrix:
    exp(entropy of normalized singular values).
    """
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return 1.0
    try:
        _, s, _ = np.linalg.svd(matrix, full_matrices=False)
    except Exception:
        return 1.0
    s_sum = np.sum(s)
    if s_sum <= 1e-12:
        return 0.0
    p = s / s_sum
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(entropy))


# ==============================================================================
# 2. Geometry & Aux Head Helpers
# ==============================================================================

def compute_rotmat_grasp_error(
    pred_pose: Any,
    goal_pose: Any,
) -> Tuple[float, float, float]:
    """
    Computes translation, rotation, and total error for 9D grasp poses
    [x-axis (3), y-axis (3), translation (3)], adhering to the repo's
    goal_pred_rotmat_loss convention (with 180-deg z-symmetry).
    """
    if torch is not None and isinstance(pred_pose, torch.Tensor):
        pred = pred_pose.detach().cpu().numpy().flatten()
    else:
        pred = np.asarray(pred_pose, dtype=np.float32).flatten()

    if torch is not None and isinstance(goal_pose, torch.Tensor):
        goal = goal_pose.detach().cpu().numpy().flatten()
    else:
        goal = np.asarray(goal_pose, dtype=np.float32).flatten()

    if len(pred) < 9 or len(goal) < 9:
        return float("nan"), float("nan"), float("nan")

    trans_dist = float(np.linalg.norm(pred[6:9] - goal[6:9]))

    u_p = pred[:3] / (np.linalg.norm(pred[:3]) + 1e-8)
    v_p = pred[3:6] / (np.linalg.norm(pred[3:6]) + 1e-8)
    w_p = np.cross(u_p, v_p)
    R_pred = np.stack([u_p, v_p, w_p], axis=1)

    u_g = goal[:3] / (np.linalg.norm(goal[:3]) + 1e-8)
    v_g = goal[3:6] / (np.linalg.norm(goal[3:6]) + 1e-8)
    w_g = np.cross(u_g, v_g)
    R_goal = np.stack([u_g, v_g, w_g], axis=1)

    R_goal_z180 = R_goal.copy()
    R_goal_z180[:, 0] *= -1
    R_goal_z180[:, 1] *= -1

    rot_dist0 = 3.0 - float(np.trace(R_pred @ R_goal.T))
    rot_dist1 = 3.0 - float(np.trace(R_pred @ R_goal_z180.T))
    rot_dist = max(0.0, min(rot_dist0, rot_dist1))

    total_dist = trans_dist + rot_dist
    return float(trans_dist), float(rot_dist), float(total_dist)


# ==============================================================================
# 3. Window Ablation & Re-initialization Helpers
# ==============================================================================

def ablate_observation(
    obs: Dict[str, np.ndarray],
    ablation_type: str,
    stack_keys: Optional[List[str]] = None,
    rng: Optional[np.random.RandomState] = None,
) -> Dict[str, np.ndarray]:
    """
    Creates an ablated copy of the observation window.
    ablation_type: 'none', 'repeat_last', 'shuffle', 'reverse'.
    Leaves unstacked keys untouched.
    """
    if ablation_type in ("none", None):
        return obs

    ablated = {}
    for k, v in obs.items():
        is_stacked = (
            (stack_keys is not None and k in stack_keys)
            or (v.ndim >= 2 and v.shape[0] > 1 and k not in ("close_grasp_pose_ee", "eval_target"))
        )
        if not is_stacked:
            ablated[k] = v
            continue

        n_stack = v.shape[0]
        if n_stack <= 1:
            ablated[k] = v
            continue

        if ablation_type == "repeat_last":
            latest = v[-1]
            ablated[k] = np.repeat(latest[np.newaxis, ...], n_stack, axis=0)
        elif ablation_type == "reverse":
            ablated[k] = v[::-1].copy()
        elif ablation_type == "shuffle":
            if rng is None:
                rng = np.random.RandomState()
            perm = rng.permutation(n_stack)
            if np.array_equal(perm, np.arange(n_stack)):
                perm = np.roll(perm, 1)
            ablated[k] = v[perm].copy()
        else:
            ablated[k] = v

    return ablated


def reinit_window_first_valid(env: Any) -> bool:
    """
    Traverses wrapper chain to find FrameStackObsWrapper and re-seeds its
    history deque with n_stack copies of the latest valid observation.
    Returns True if successfully re-initialized, False otherwise.
    """
    cur = env
    while cur is not None:
        if hasattr(cur, "_history") and hasattr(cur, "stack_keys") and hasattr(cur, "n_stack"):
            history = getattr(cur, "_history")
            stack_keys = getattr(cur, "stack_keys")
            n_stack = getattr(cur, "n_stack")
            for k in stack_keys:
                if k in history and len(history[k]) > 0:
                    latest = history[k][-1]
                    history[k].clear()
                    for _ in range(n_stack):
                        history[k].append(latest)
            return True
        cur = getattr(cur, "env", None)
    return False


def get_valid_frames_count(obs: Dict[str, np.ndarray], n_stack: int) -> int:
    """
    Counts how many frames in the window contain valid grasps based on obs['grasp_exist'].
    """
    if "grasp_exist" not in obs:
        return 0
    grasp_exist = obs["grasp_exist"]
    if grasp_exist.ndim >= 2 and grasp_exist.shape[0] == n_stack:
        return int(sum(np.any(grasp_exist[i] > 0) for i in range(n_stack)))
    else:
        return int(1 if np.any(grasp_exist > 0) else 0)


# ==============================================================================
# 4. Episode Metrics Tracker
# ==============================================================================

class EpisodeMetricsTracker:
    """
    Collects step-by-step metrics for a single episode and computes the final outcome.
    """

    def __init__(
        self,
        max_steps: int = 100,
        success_hold_steps: int = 5,
        near_trans_thresh: float = 0.08,
        near_rot_thresh: float = 0.25,
        gripper_dim: int = 6,
        gripper_close_thresh: float = 0.0,
        n_stack: int = 1,
        stack_keys: Optional[List[str]] = None,
    ):
        self.max_steps = max_steps
        self.success_hold_steps = success_hold_steps
        self.near_trans_thresh = near_trans_thresh
        self.near_rot_thresh = near_rot_thresh
        self.gripper_dim = gripper_dim
        self.gripper_close_thresh = gripper_close_thresh
        self.n_stack = n_stack
        self.stack_keys = stack_keys

        # Outcome tracking
        self.success_any = False
        self.success_final = False
        self.success_once = False
        self.time_to_first_success: Optional[int] = None
        self.consecutive_success = 0
        self.max_consecutive_success = 0

        # Stage funnel tracking (ever reached)
        self.stage_reached = {
            "grasp_visible": False,
            "near_target": False,
            "finger_contact": False,
            "is_obj_grasp": False,
            "is_obj_lift": False,
            "success": False,
        }

        # Step-level records
        self.trans_dist_curve: List[float] = []
        self.rot_dist_curve: List[float] = []
        self.actions: List[np.ndarray] = []
        self.obj_lin_speeds: List[float] = []
        self.obj_ang_speeds: List[float] = []
        self.arm_qvel_maxs: List[float] = []
        self.arm_qacc_maxs: List[float] = []
        self.is_robot_static_list: List[int] = []
        self.valid_frames_curve: List[int] = []

        # Timings
        self.time_grasp_gen_ms: List[float] = []
        self.time_rl_predict_ms: List[float] = []
        self.time_env_step_ms: List[float] = []

        # Counterfactual sensitivity
        self.cf_l2_repeat_last: List[float] = []
        self.cf_l2_shuffle: List[float] = []
        self.cf_l2_reverse: List[float] = []

        # Latent features
        self.latent_ht: List[np.ndarray] = []

        # Aux predictions and critic Q
        self.actor_aux_pose_errs: List[float] = []
        self.critic_aux_pose_errs: List[float] = []
        self.target_pred_accs: List[float] = []
        self.critic_q_step0: Optional[float] = None
        self.critic_q_first_contact: Optional[float] = None

        # Contact and gripper-close snapshot flags
        self.first_contact_step: Optional[int] = None
        self.first_contact_snapshot: Optional[Dict[str, float]] = None
        self.first_gripper_close_step: Optional[int] = None
        self.first_gripper_close_snapshot: Optional[Dict[str, float]] = None

        # Min distances reached
        self.min_trans_dist = float("inf")
        self.min_trans_step = -1
        self.min_rot_dist = float("inf")
        self.min_rot_step = -1

        self.final_info: Dict[str, Any] = {}

    def record_step(
        self,
        step: int,
        action: np.ndarray,
        info: Dict[str, Any],
        timing_ms: Dict[str, float],
        valid_frames_in_window: int = 1,
        cf_diffs: Optional[Dict[str, float]] = None,
        latent_vector: Optional[np.ndarray] = None,
        aux_metrics: Optional[Dict[str, float]] = None,
        critic_q_min: Optional[float] = None,
    ):
        """Records data for a single rollout step."""
        self.actions.append(np.array(action, copy=True))
        self.valid_frames_curve.append(valid_frames_in_window)

        # Timings
        self.time_grasp_gen_ms.append(timing_ms.get("grasp_gen", 0.0))
        self.time_rl_predict_ms.append(timing_ms.get("predict", 0.0))
        self.time_env_step_ms.append(timing_ms.get("env_step", 0.0))

        # Counterfactual diffs
        if cf_diffs is not None:
            if "repeat_last" in cf_diffs:
                self.cf_l2_repeat_last.append(cf_diffs["repeat_last"])
            if "shuffle" in cf_diffs:
                self.cf_l2_shuffle.append(cf_diffs["shuffle"])
            if "reverse" in cf_diffs:
                self.cf_l2_reverse.append(cf_diffs["reverse"])

        # Latent vector
        if latent_vector is not None:
            self.latent_ht.append(np.array(latent_vector, copy=True))

        # Aux metrics
        if aux_metrics is not None:
            if "actor_pose_err" in aux_metrics:
                self.actor_aux_pose_errs.append(aux_metrics["actor_pose_err"])
            if "critic_pose_err" in aux_metrics:
                self.critic_aux_pose_errs.append(aux_metrics["critic_pose_err"])
            if "target_acc" in aux_metrics:
                self.target_pred_accs.append(aux_metrics["target_acc"])

        # Critic Q at step 0
        if step == 0 and critic_q_min is not None:
            self.critic_q_step0 = float(critic_q_min)

        # Extract info signals
        is_success = bool(info.get("is_success", False))
        is_success_once = bool(info.get("is_success_once", False))
        is_info_exist = bool(info.get("is_info_exist", 0) > 0)
        evaluate_info = info.get("evaluate_info", None)

        if is_success:
            self.consecutive_success += 1
            if not self.success_any:
                self.success_any = True
                self.time_to_first_success = step
        else:
            self.consecutive_success = 0
        self.max_consecutive_success = max(self.max_consecutive_success, self.consecutive_success)

        if is_success_once:
            self.success_once = True

        # Distances
        trans_dist_raw = info.get("trans_dist", None)
        rot_dist_raw = info.get("rot_dist", None)
        grasp_id = int(info.get("grasp_id", 0))

        td = float("nan")
        rd = float("nan")
        if trans_dist_raw is not None:
            if isinstance(trans_dist_raw, (np.ndarray, list)):
                td = float(trans_dist_raw[grasp_id]) if len(trans_dist_raw) > grasp_id else float(trans_dist_raw[0])
            else:
                td = float(trans_dist_raw)

        if rot_dist_raw is not None:
            if isinstance(rot_dist_raw, (np.ndarray, list)):
                rd = float(rot_dist_raw[grasp_id]) if len(rot_dist_raw) > grasp_id else float(rot_dist_raw[0])
            else:
                rd = float(rot_dist_raw)

        self.trans_dist_curve.append(td)
        self.rot_dist_curve.append(rd)

        if not np.isnan(td) and td < self.min_trans_dist:
            self.min_trans_dist = td
            self.min_trans_step = step
        if not np.isnan(rd) and rd < self.min_rot_dist:
            self.min_rot_dist = rd
            self.min_rot_step = step

        # Evaluate info: [is_robot_static, is_obj_grasp, is_obj_static, is_obj_lift]
        is_robot_static = 0
        is_obj_grasp = 0
        is_obj_lift = 0
        if evaluate_info is not None and len(evaluate_info) >= 4:
            is_robot_static = int(evaluate_info[0])
            is_obj_grasp = int(evaluate_info[1])
            is_obj_lift = int(evaluate_info[3])
        self.is_robot_static_list.append(is_robot_static)

        # Contact
        finger_contacts = info.get("gripper_finger_contacts", None)
        has_finger_contact = False
        if finger_contacts is not None:
            has_finger_contact = bool(np.any(np.asarray(finger_contacts) > 0))

        # Speeds
        ee_velang = info.get("ee_velang", None)
        obj_angvel = info.get("obj_angvel", None)
        v_ee = np.asarray(ee_velang[:3]) if ee_velang is not None and len(ee_velang) >= 3 else np.zeros(3)
        v_obj = np.asarray(obj_angvel[:3]) if obj_angvel is not None and len(obj_angvel) >= 3 else np.zeros(3)
        w_obj = np.asarray(obj_angvel[3:6]) if obj_angvel is not None and len(obj_angvel) >= 6 else np.zeros(3)

        ee_speed = float(np.linalg.norm(v_ee))
        obj_speed = float(np.linalg.norm(v_obj))
        rel_speed = float(np.linalg.norm(v_ee - v_obj))
        obj_ang_speed = float(np.linalg.norm(w_obj))

        self.obj_lin_speeds.append(obj_speed)
        self.obj_ang_speeds.append(obj_ang_speed)

        # Arm velocities and accelerations
        arm_qvel = info.get("arm_qvel", None)
        arm_qacc = info.get("arm_qacc", None)
        if arm_qvel is not None:
            self.arm_qvel_maxs.append(float(np.max(np.abs(arm_qvel))))
        if arm_qacc is not None:
            self.arm_qacc_maxs.append(float(np.max(np.abs(arm_qacc))))

        # Stage funnel updates
        if is_info_exist:
            self.stage_reached["grasp_visible"] = True
        is_near = (
            not np.isnan(td)
            and not np.isnan(rd)
            and td < self.near_trans_thresh
            and rd < self.near_rot_thresh
        )
        if is_near:
            self.stage_reached["near_target"] = True
        if has_finger_contact:
            self.stage_reached["finger_contact"] = True
        if is_obj_grasp:
            self.stage_reached["is_obj_grasp"] = True
        if is_obj_lift:
            self.stage_reached["is_obj_lift"] = True
        if is_success:
            self.stage_reached["success"] = True

        # First finger contact snapshot
        if has_finger_contact and self.first_contact_step is None:
            self.first_contact_step = step
            self.first_contact_snapshot = {
                "step": step,
                "trans_dist": td,
                "rot_dist": rd,
                "ee_speed": ee_speed,
                "obj_speed": obj_speed,
                "rel_speed": rel_speed,
            }
            if critic_q_min is not None:
                self.critic_q_first_contact = float(critic_q_min)

        # First gripper-close command snapshot
        gripper_dim_idx = self.gripper_dim if self.gripper_dim < len(action) else -1
        is_gripper_closing = bool(action[gripper_dim_idx] > self.gripper_close_thresh)
        if is_gripper_closing and self.first_gripper_close_step is None:
            self.first_gripper_close_step = step
            self.first_gripper_close_snapshot = {
                "step": step,
                "trans_dist": td,
                "rot_dist": rd,
                "ee_speed": ee_speed,
                "obj_speed": obj_speed,
                "rel_speed": rel_speed,
            }

        self.final_info = deepcopy(info)

    def finalize(self) -> Dict[str, Any]:
        """Finalizes and compiles all metrics for the episode."""
        self.success_final = bool(self.final_info.get("is_success", False))

        # Mutually exclusive failure label
        failure_label = "none"
        if not self.success_any:
            if not self.stage_reached["grasp_visible"]:
                failure_label = "never_saw_grasp"
            elif not self.stage_reached["near_target"]:
                failure_label = "never_near"
            elif not self.stage_reached["finger_contact"]:
                failure_label = "near_no_contact"
            elif not self.stage_reached["is_obj_grasp"]:
                failure_label = "contact_no_grasp"
            elif not self.stage_reached["is_obj_lift"]:
                failure_label = "grasp_not_lifted"
            else:
                eval_info = self.final_info.get("evaluate_info", [0, 0, 0, 0])
                is_robot_static = int(eval_info[0]) if len(eval_info) > 0 else 0
                is_obj_grasp = int(eval_info[1]) if len(eval_info) > 1 else 0
                is_obj_static = int(eval_info[2]) if len(eval_info) > 2 else 0
                is_obj_lift = int(eval_info[3]) if len(eval_info) > 3 else 0

                if not is_obj_lift or not is_obj_grasp:
                    failure_label = "lifted_then_dropped"
                elif not is_robot_static or not is_obj_static:
                    failure_label = "lifted_not_static"
                else:
                    failure_label = "other"

        # Action smoothness
        action_diff_mean = float("nan")
        action_jerk_mean = float("nan")
        action_norm_mean = float("nan")
        if len(self.actions) > 1:
            acts = np.array(self.actions)
            diffs = acts[1:] - acts[:-1]
            action_diff_mean = float(np.mean(np.abs(diffs)))
            action_norm_mean = float(np.mean(np.linalg.norm(acts, axis=-1)))
            if len(self.actions) > 2:
                jerks = acts[2:] - 2.0 * acts[1:-1] + acts[:-2]
                action_jerk_mean = float(np.mean(np.abs(jerks)))
        elif len(self.actions) == 1:
            action_norm_mean = float(np.linalg.norm(self.actions[0]))

        # Arm qvel / qacc
        arm_qvel_mean = float(np.mean(self.arm_qvel_maxs)) if self.arm_qvel_maxs else float("nan")
        arm_qvel_peak = float(np.max(self.arm_qvel_maxs)) if self.arm_qvel_maxs else float("nan")
        arm_qacc_mean = float(np.mean(self.arm_qacc_maxs)) if self.arm_qacc_maxs else float("nan")
        arm_qacc_peak = float(np.max(self.arm_qacc_maxs)) if self.arm_qacc_maxs else float("nan")
        robot_static_fraction = float(np.mean(self.is_robot_static_list)) if self.is_robot_static_list else float("nan")

        # Object speeds
        mean_obj_lin_speed = float(np.mean(self.obj_lin_speeds)) if self.obj_lin_speeds else float("nan")
        mean_obj_ang_speed = float(np.mean(self.obj_ang_speeds)) if self.obj_ang_speeds else float("nan")

        # Timing metrics
        def p95(vals: List[float]) -> float:
            return float(np.percentile(vals, 95)) if vals else float("nan")

        def mean_or_nan(vals: List[float]) -> float:
            return float(np.mean(vals)) if vals else float("nan")

        # Counterfactual sensitivities
        cf_diff_repeat_last = mean_or_nan(self.cf_l2_repeat_last)
        cf_diff_shuffle = mean_or_nan(self.cf_l2_shuffle)
        cf_diff_reverse = mean_or_nan(self.cf_l2_reverse)

        # Latent diagnostics
        mean_ht_norm = float("nan")
        mean_ht_cos_sim = float("nan")
        effective_rank = float("nan")
        if len(self.latent_ht) > 0:
            H = np.array(self.latent_ht)
            norms = np.linalg.norm(H, axis=-1)
            mean_ht_norm = float(np.mean(norms))
            effective_rank = compute_effective_rank(H)
            if len(self.latent_ht) > 1:
                cos_sims = []
                for t in range(1, len(self.latent_ht)):
                    n1 = np.linalg.norm(H[t - 1])
                    n2 = np.linalg.norm(H[t])
                    if n1 > 1e-8 and n2 > 1e-8:
                        cos_sims.append(np.dot(H[t - 1], H[t]) / (n1 * n2))
                if cos_sims:
                    mean_ht_cos_sim = float(np.mean(cos_sims))

        # Aux predictions
        actor_aux_pose_err = mean_or_nan(self.actor_aux_pose_errs)
        critic_aux_pose_err = mean_or_nan(self.critic_aux_pose_errs)
        target_pred_acc = mean_or_nan(self.target_pred_accs)

        mean_valid_frames = float(np.mean(self.valid_frames_curve)) if self.valid_frames_curve else float("nan")

        record = {
            "success_any": int(self.success_any),
            "success_final": int(self.success_final),
            "success_once": int(self.success_once),
            "time_to_first_success": (
                self.time_to_first_success if self.time_to_first_success is not None else -1
            ),
            "success_held_ge_N": int(self.max_consecutive_success >= self.success_hold_steps),
            "stage_funnel": {k: int(v) for k, v in self.stage_reached.items()},
            "failure_label": failure_label,
            "min_trans_dist": self.min_trans_dist if self.min_trans_dist != float("inf") else float("nan"),
            "min_trans_step": self.min_trans_step,
            "min_rot_dist": self.min_rot_dist if self.min_rot_dist != float("inf") else float("nan"),
            "min_rot_step": self.min_rot_step,
            "first_contact": self.first_contact_snapshot,
            "first_gripper_close": self.first_gripper_close_snapshot,
            "mean_obj_lin_speed": mean_obj_lin_speed,
            "mean_obj_ang_speed": mean_obj_ang_speed,
            "action_diff_mean": action_diff_mean,
            "action_jerk_mean": action_jerk_mean,
            "action_norm_mean": action_norm_mean,
            "arm_qvel_mean": arm_qvel_mean,
            "arm_qvel_peak": arm_qvel_peak,
            "arm_qacc_mean": arm_qacc_mean,
            "arm_qacc_peak": arm_qacc_peak,
            "robot_static_fraction": robot_static_fraction,
            "timing_ms": {
                "grasp_gen_mean": mean_or_nan(self.time_grasp_gen_ms),
                "grasp_gen_p95": p95(self.time_grasp_gen_ms),
                "predict_mean": mean_or_nan(self.time_rl_predict_ms),
                "predict_p95": p95(self.time_rl_predict_ms),
                "env_step_mean": mean_or_nan(self.time_env_step_ms),
                "env_step_p95": p95(self.time_env_step_ms),
            },
            "temporal": {
                "cf_diff_repeat_last": cf_diff_repeat_last,
                "cf_diff_shuffle": cf_diff_shuffle,
                "cf_diff_reverse": cf_diff_reverse,
                "mean_valid_frames_in_window": mean_valid_frames,
            },
            "latent": {
                "mean_ht_norm": mean_ht_norm,
                "mean_ht_cos_sim": mean_ht_cos_sim,
                "effective_rank": effective_rank,
            },
            "aux_head": {
                "actor_aux_pose_err": actor_aux_pose_err,
                "critic_aux_pose_err": critic_aux_pose_err,
                "target_pred_acc": target_pred_acc,
                "critic_q_step0": self.critic_q_step0,
                "critic_q_first_contact": self.critic_q_first_contact,
            },
        }

        curves_dict = {
            "trans_dist_curve": np.array(self.trans_dist_curve, dtype=np.float32),
            "rot_dist_curve": np.array(self.rot_dist_curve, dtype=np.float32),
            "actions": np.array(self.actions, dtype=np.float32) if self.actions else np.zeros((0, 7)),
            "obj_lin_speeds": np.array(self.obj_lin_speeds, dtype=np.float32),
            "obj_ang_speeds": np.array(self.obj_ang_speeds, dtype=np.float32),
            "valid_frames_curve": np.array(self.valid_frames_curve, dtype=np.int32),
        }

        latent_dict = {
            "latent_ht": np.array(self.latent_ht, dtype=np.float32) if self.latent_ht else np.zeros((0, 0))
        }

        return record, curves_dict, latent_dict


# ==============================================================================
# 5. Run Metrics Aggregator
# ==============================================================================

class RunMetricsAggregator:
    """
    Aggregates per-episode records across model_ids, traj_modes, seeds, and datasets.
    Computes Wilson CIs, bootstrap CIs, speed stratifications, AUROC, and failure summaries.
    """

    def __init__(self, speed_bins: Optional[Sequence[float]] = None):
        self.episodes: List[Dict[str, Any]] = []
        self.speed_bins = list(speed_bins) if speed_bins is not None else None

    def add_episode(self, episode_record: Dict[str, Any]):
        self.episodes.append(episode_record)

    def compute_summary(self, subset_episodes: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        records = self.episodes if subset_episodes is None else subset_episodes
        n = len(records)
        if n == 0:
            return {"num_episodes": 0}

        # Success rates with Wilson CIs
        success_any_count = sum(r["success_any"] for r in records)
        success_final_count = sum(r["success_final"] for r in records)
        success_once_count = sum(r["success_once"] for r in records)
        success_held_count = sum(r["success_held_ge_N"] for r in records)

        sr_any_ci = wilson_score_interval(success_any_count, n)
        sr_final_ci = wilson_score_interval(success_final_count, n)
        sr_once_ci = wilson_score_interval(success_once_count, n)
        sr_held_ci = wilson_score_interval(success_held_count, n)

        # Stage funnel
        funnel_keys = ["grasp_visible", "near_target", "finger_contact", "is_obj_grasp", "is_obj_lift", "success"]
        funnel_counts = {k: sum(r["stage_funnel"].get(k, 0) for r in records) for k in funnel_keys}
        funnel_proportions = {k: float(funnel_counts[k]) / n for k in funnel_keys}
        funnel_cis = {k: wilson_score_interval(funnel_counts[k], n) for k in funnel_keys}

        # Failure modes
        failure_keys = [
            "never_saw_grasp",
            "never_near",
            "near_no_contact",
            "contact_no_grasp",
            "grasp_not_lifted",
            "lifted_then_dropped",
            "lifted_not_static",
            "other",
        ]
        n_failed = n - success_any_count
        failure_counts = {k: sum(1 for r in records if r["failure_label"] == k) for k in failure_keys}
        failure_proportions = {
            k: (float(failure_counts[k]) / n_failed if n_failed > 0 else 0.0) for k in failure_keys
        }

        # Time to first success for successful episodes
        success_times = [
            r["time_to_first_success"] for r in records if r["success_any"] and r["time_to_first_success"] >= 0
        ]
        time_to_success_mean = float(np.mean(success_times)) if success_times else float("nan")
        time_to_success_ci = bootstrap_mean_ci(success_times) if success_times else (float("nan"), float("nan"))

        # Scalar metrics to aggregate with bootstrap CI
        scalar_keys = [
            "action_diff_mean",
            "action_jerk_mean",
            "action_norm_mean",
            "arm_qvel_mean",
            "arm_qvel_peak",
            "arm_qacc_mean",
            "arm_qacc_peak",
            "robot_static_fraction",
            "mean_obj_lin_speed",
            "mean_obj_ang_speed",
            "min_trans_dist",
            "min_rot_dist",
        ]
        scalar_summary = {}
        for sk in scalar_keys:
            vals = [r.get(sk) for r in records if r.get(sk) is not None]
            m = float(np.nanmean(vals)) if vals else float("nan")
            ci = bootstrap_mean_ci(vals) if vals else (float("nan"), float("nan"))
            scalar_summary[sk] = {"mean": m, "ci95": ci}

        # Timings
        timing_summary = {}
        for tk in ["grasp_gen_mean", "grasp_gen_p95", "predict_mean", "predict_p95", "env_step_mean", "env_step_p95"]:
            vals = [r.get("timing_ms", {}).get(tk) for r in records]
            m = float(np.nanmean(vals)) if vals else float("nan")
            ci = bootstrap_mean_ci(vals) if vals else (float("nan"), float("nan"))
            timing_summary[tk] = {"mean": m, "ci95": ci}

        # Temporal sensitivities
        temporal_summary = {}
        for temp_k in ["cf_diff_repeat_last", "cf_diff_shuffle", "cf_diff_reverse", "mean_valid_frames_in_window"]:
            vals = [r.get("temporal", {}).get(temp_k) for r in records]
            m = float(np.nanmean(vals)) if vals else float("nan")
            ci = bootstrap_mean_ci(vals) if vals else (float("nan"), float("nan"))
            temporal_summary[temp_k] = {"mean": m, "ci95": ci}

        # Latent diagnostics
        latent_summary = {}
        for lk in ["mean_ht_norm", "mean_ht_cos_sim", "effective_rank"]:
            vals = [r.get("latent", {}).get(lk) for r in records]
            m = float(np.nanmean(vals)) if vals else float("nan")
            ci = bootstrap_mean_ci(vals) if vals else (float("nan"), float("nan"))
            latent_summary[lk] = {"mean": m, "ci95": ci}

        # Aux predictions and critic Q
        aux_summary = {}
        for ak in ["actor_aux_pose_err", "critic_aux_pose_err", "target_pred_acc"]:
            vals = [r.get("aux_head", {}).get(ak) for r in records]
            m = float(np.nanmean(vals)) if vals else float("nan")
            ci = bootstrap_mean_ci(vals) if vals else (float("nan"), float("nan"))
            aux_summary[ak] = {"mean": m, "ci95": ci}

        # Critic Q splits and AUROC
        q_step0_succ = [
            r["aux_head"]["critic_q_step0"]
            for r in records
            if r["success_any"] and r.get("aux_head", {}).get("critic_q_step0") is not None
        ]
        q_step0_fail = [
            r["aux_head"]["critic_q_step0"]
            for r in records
            if not r["success_any"] and r.get("aux_head", {}).get("critic_q_step0") is not None
        ]
        q_contact_succ = [
            r["aux_head"]["critic_q_first_contact"]
            for r in records
            if r["success_any"] and r.get("aux_head", {}).get("critic_q_first_contact") is not None
        ]
        q_contact_fail = [
            r["aux_head"]["critic_q_first_contact"]
            for r in records
            if not r["success_any"] and r.get("aux_head", {}).get("critic_q_first_contact") is not None
        ]

        auroc_q_step0 = compute_auroc(
            [r["success_any"] for r in records],
            [r.get("aux_head", {}).get("critic_q_step0") for r in records],
        )
        auroc_q_contact = compute_auroc(
            [r["success_any"] for r in records],
            [r.get("aux_head", {}).get("critic_q_first_contact") for r in records],
        )

        critic_q_summary = {
            "q_step0_success_mean": float(np.nanmean(q_step0_succ)) if q_step0_succ else float("nan"),
            "q_step0_fail_mean": float(np.nanmean(q_step0_fail)) if q_step0_fail else float("nan"),
            "q_contact_success_mean": float(np.nanmean(q_contact_succ)) if q_contact_succ else float("nan"),
            "q_contact_fail_mean": float(np.nanmean(q_contact_fail)) if q_contact_fail else float("nan"),
            "auroc_q_step0": auroc_q_step0,
            "auroc_q_contact": auroc_q_contact,
        }

        # Contact & gripper close snapshot aggregates
        snapshot_summary = {}
        for snap_name in ["first_contact", "first_gripper_close"]:
            sub_dict = {}
            for metric_k in ["trans_dist", "rot_dist", "ee_speed", "obj_speed", "rel_speed"]:
                vals = [
                    r.get(snap_name, {}).get(metric_k)
                    for r in records
                    if r.get(snap_name) is not None
                ]
                m = float(np.nanmean(vals)) if vals else float("nan")
                ci = bootstrap_mean_ci(vals) if vals else (float("nan"), float("nan"))
                sub_dict[metric_k] = {"mean": m, "ci95": ci}
            snapshot_summary[snap_name] = sub_dict

        # Object speed stratification
        speed_strat = self._stratify_by_speed(records)

        return {
            "num_episodes": n,
            "success_rates": {
                "success_any": {"rate": float(success_any_count) / n, "ci95": sr_any_ci, "count": success_any_count},
                "success_final": {"rate": float(success_final_count) / n, "ci95": sr_final_ci, "count": success_final_count},
                "success_once": {"rate": float(success_once_count) / n, "ci95": sr_once_ci, "count": success_once_count},
                "success_held_ge_N": {"rate": float(success_held_count) / n, "ci95": sr_held_ci, "count": success_held_count},
                "time_to_first_success": {"mean": time_to_success_mean, "ci95": time_to_success_ci},
            },
            "stage_funnel": {
                "counts": funnel_counts,
                "proportions": funnel_proportions,
                "ci95": funnel_cis,
            },
            "failure_modes": {
                "num_failed": n_failed,
                "counts": failure_counts,
                "proportions": failure_proportions,
            },
            "control_quality": scalar_summary,
            "timing_ms": timing_summary,
            "temporal": temporal_summary,
            "latent": latent_summary,
            "aux_head": aux_summary,
            "critic_q": critic_q_summary,
            "snapshots": snapshot_summary,
            "speed_stratification": speed_strat,
        }

    def _stratify_by_speed(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Bins episodes by mean object linear speed and reports success_any per bin."""
        speeds = [r["mean_obj_lin_speed"] for r in records if not np.isnan(r.get("mean_obj_lin_speed", float("nan")))]
        if not speeds:
            return {}

        if self.speed_bins is not None and len(self.speed_bins) >= 2:
            edges = sorted(self.speed_bins)
            bin_labels = [f"[{edges[i]:.3f}, {edges[i+1]:.3f})" for i in range(len(edges) - 1)]
        else:
            # Terciles
            p33, p67 = np.percentile(speeds, [33.333, 66.667])
            edges = [-np.inf, p33, p67, np.inf]
            bin_labels = ["low (<33%)", "med (33-67%)", "high (>67%)"]

        binned_episodes: Dict[str, List[Dict[str, Any]]] = {lbl: [] for lbl in bin_labels}
        for r in records:
            spd = r.get("mean_obj_lin_speed", float("nan"))
            if np.isnan(spd):
                continue
            for i in range(len(edges) - 1):
                if edges[i] <= spd < edges[i + 1] or (i == len(edges) - 2 and spd >= edges[i + 1]):
                    binned_episodes[bin_labels[i]].append(r)
                    break

        strat_res = {}
        for lbl, bin_recs in binned_episodes.items():
            k = sum(r["success_any"] for r in bin_recs)
            total = len(bin_recs)
            rate = float(k) / total if total > 0 else float("nan")
            ci = wilson_score_interval(k, total) if total > 0 else (float("nan"), float("nan"))
            strat_res[lbl] = {
                "num_episodes": total,
                "success_count": k,
                "success_rate": rate,
                "ci95": ci,
            }
        return strat_res


# ==============================================================================
# 6. File Writers
# ==============================================================================

def write_jsonl_line(file_path: str, data: Dict[str, Any]):
    """Appends a single JSON object as a line in a JSONL file."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")


def write_json_summary(file_path: str, data: Dict[str, Any]):
    """Writes formatted JSON summary file."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_curves_npz(file_path: str, curves: Dict[str, np.ndarray]):
    """Writes step-level curves array to .npz file."""
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    np.savez_compressed(file_path, **curves)
