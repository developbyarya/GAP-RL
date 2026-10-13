#!/usr/bin/env python3
"""
Unit tests for eval_metrics.py and compare_runs.py on synthetic episodes.
Tests math formulas, Wilson intervals, bootstrap, effective rank, AUROC,
failure label priority, stage funnel, window ablation, and compare_runs pairing.
"""

import os
import sys
import json
import tempfile
import numpy as np

# Add gap_rl/algorithms/scripts to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gap_rl", "algorithms", "scripts"))

from eval_metrics import (
    wilson_score_interval,
    bootstrap_mean_ci,
    compute_auroc,
    compute_effective_rank,
    compute_rotmat_grasp_error,
    ablate_observation,
    EpisodeMetricsTracker,
    RunMetricsAggregator,
    write_jsonl_line,
    write_json_summary,
)
from compare_runs import compare, exact_mcnemar_p_value


def test_statistics():
    print("Testing statistics (Wilson, Bootstrap, AUROC, Effective Rank)...")
    # Wilson Score Interval
    low, high = wilson_score_interval(10, 20)
    assert 0.0 <= low <= 0.5 <= high <= 1.0, f"Wilson interval unexpected: {low}, {high}"
    assert abs(wilson_score_interval(0, 10)[0] - 0.0) < 1e-5
    assert abs(wilson_score_interval(10, 10)[1] - 1.0) < 1e-5

    # Bootstrap Mean CI
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    b_low, b_high = bootstrap_mean_ci(data, num_resamples=500)
    assert 1.0 <= b_low <= 3.0 <= b_high <= 5.0, f"Bootstrap CI unexpected: {b_low}, {b_high}"

    # AUROC
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.8, 0.9]
    auroc = compute_auroc(y_true, y_score)
    assert auroc == 1.0, f"AUROC should be 1.0, got {auroc}"

    y_score_rev = [0.9, 0.8, 0.2, 0.1]
    auroc_rev = compute_auroc(y_true, y_score_rev)
    assert auroc_rev == 0.0, f"AUROC should be 0.0, got {auroc_rev}"

    # Effective rank
    # Identity matrix of size 10 -> singular values all equal -> erank == 10
    eye_mat = np.eye(10)
    erank = compute_effective_rank(eye_mat)
    assert abs(erank - 10.0) < 1e-3, f"Effective rank of identity(10) should be 10, got {erank}"

    # Rank 1 matrix -> erank == 1
    rank1_mat = np.ones((10, 10))
    erank1 = compute_effective_rank(rank1_mat)
    assert abs(erank1 - 1.0) < 1e-3, f"Effective rank of rank-1 matrix should be 1, got {erank1}"
    print("  -> Passed!")


def test_geometry_and_rotmat_error():
    print("Testing rotmat grasp error...")
    # Perfect match (identity rotmat + same translation)
    # 9D: [1, 0, 0,  0, 1, 0,  0.1, 0.2, 0.3]
    pose1 = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.1, 0.2, 0.3])
    t_err, r_err, total_err = compute_rotmat_grasp_error(pose1, pose1)
    assert abs(t_err) < 1e-5, f"Translation error should be 0, got {t_err}"
    assert abs(r_err) < 1e-5, f"Rotation error should be 0, got {r_err}"
    assert abs(total_err) < 1e-5, f"Total error should be 0, got {total_err}"

    # 180-deg symmetry check (z-axis flip: x -> -x, y -> -y)
    pose_z180 = np.array([-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.1, 0.2, 0.3])
    t_err, r_err, total_err = compute_rotmat_grasp_error(pose1, pose_z180)
    assert abs(r_err) < 1e-5, f"Rotation error under 180-deg symmetry should be 0, got {r_err}"
    print("  -> Passed!")


def test_window_ablation():
    print("Testing window ablation...")
    # Stacked obs with n_stack=4
    obs = {
        "tcp_pose": np.array([[1], [2], [3], [4]]), # shape (4, 1)
        "gripper_pos": np.array([[10], [20], [30], [40]]), # shape (4, 1)
        "eval_target": np.array([0, 1, 0, 1]), # unstacked
    }
    stack_keys = ["tcp_pose", "gripper_pos"]

    # Repeat last
    abl_rep = ablate_observation(obs, "repeat_last", stack_keys=stack_keys)
    assert np.all(abl_rep["tcp_pose"] == 4), "repeat_last did not replicate latest frame"
    assert np.all(abl_rep["gripper_pos"] == 40), "repeat_last did not replicate latest frame"
    assert np.array_equal(abl_rep["eval_target"], obs["eval_target"]), "unstacked key was modified"

    # Reverse
    abl_rev = ablate_observation(obs, "reverse", stack_keys=stack_keys)
    assert np.array_equal(abl_rev["tcp_pose"].flatten(), [4, 3, 2, 1]), "reverse failed"

    # Shuffle
    abl_shuf = ablate_observation(obs, "shuffle", stack_keys=stack_keys)
    assert not np.array_equal(abl_shuf["tcp_pose"].flatten(), [1, 2, 3, 4]), "shuffle produced identity"
    print("  -> Passed!")


def test_episode_tracking_and_funnel():
    print("Testing EpisodeMetricsTracker (funnel, failure labels, smoothness)...")
    # 1. Successful episode
    tracker_succ = EpisodeMetricsTracker(max_steps=10, success_hold_steps=3)
    for t in range(10):
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0 if t < 5 else 0.8])
        # Success reached from step 6 onwards
        is_succ = int(t >= 6)
        info = {
            "is_success": is_succ,
            "is_success_once": is_succ,
            "is_info_exist": 1,
            "evaluate_info": [1, 1, 1, 1] if t >= 6 else [1, 0, 0, 0],
            "trans_dist": np.array([0.05 if t >= 3 else 0.5]),
            "rot_dist": np.array([0.1 if t >= 3 else 0.5]),
            "grasp_id": 0,
            "gripper_finger_contacts": [1.0, 1.0] if t >= 4 else [0.0, 0.0],
            "ee_velang": np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "obj_angvel": np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "arm_qvel": np.array([0.02, 0.02]),
            "arm_qacc": np.array([0.01, 0.01]),
        }
        tracker_succ.record_step(
            step=t,
            action=action,
            info=info,
            timing_ms={"predict": 1.5, "grasp_gen": 10.0, "env_step": 3.0},
        )
    rec_succ, _, _ = tracker_succ.finalize()
    assert rec_succ["success_any"] == 1
    assert rec_succ["success_final"] == 1
    assert rec_succ["success_held_ge_N"] == 1
    assert rec_succ["time_to_first_success"] == 6
    assert rec_succ["failure_label"] == "none"
    assert rec_succ["first_contact"]["step"] == 4
    assert rec_succ["first_gripper_close"]["step"] == 5

    # 2. Failed episode: never saw grasp
    tracker_fail1 = EpisodeMetricsTracker(max_steps=5)
    for t in range(5):
        tracker_fail1.record_step(
            step=t,
            action=np.zeros(7),
            info={"is_success": 0, "is_info_exist": 0},
            timing_ms={},
        )
    rec1, _, _ = tracker_fail1.finalize()
    assert rec1["failure_label"] == "never_saw_grasp"

    # 3. Failed episode: never near
    tracker_fail2 = EpisodeMetricsTracker(max_steps=5, near_trans_thresh=0.08)
    for t in range(5):
        tracker_fail2.record_step(
            step=t,
            action=np.zeros(7),
            info={"is_success": 0, "is_info_exist": 1, "trans_dist": 0.5, "rot_dist": 0.5},
            timing_ms={},
        )
    rec2, _, _ = tracker_fail2.finalize()
    assert rec2["failure_label"] == "never_near"

    # 4. Failed episode: near no contact
    tracker_fail3 = EpisodeMetricsTracker(max_steps=5, near_trans_thresh=0.08, near_rot_thresh=0.25)
    for t in range(5):
        tracker_fail3.record_step(
            step=t,
            action=np.zeros(7),
            info={"is_success": 0, "is_info_exist": 1, "trans_dist": 0.04, "rot_dist": 0.1, "gripper_finger_contacts": [0, 0]},
            timing_ms={},
        )
    rec3, _, _ = tracker_fail3.finalize()
    assert rec3["failure_label"] == "near_no_contact"

    # 5. Failed episode: contact no grasp
    tracker_fail4 = EpisodeMetricsTracker(max_steps=5, near_trans_thresh=0.08, near_rot_thresh=0.25)
    for t in range(5):
        tracker_fail4.record_step(
            step=t,
            action=np.zeros(7),
            info={
                "is_success": 0,
                "is_info_exist": 1,
                "trans_dist": 0.04,
                "rot_dist": 0.1,
                "gripper_finger_contacts": [1, 1],
                "evaluate_info": [0, 0, 0, 0],
            },
            timing_ms={},
        )
    rec4, _, _ = tracker_fail4.finalize()
    assert rec4["failure_label"] == "contact_no_grasp"

    # 6. Failed episode: grasp not lifted
    tracker_fail5 = EpisodeMetricsTracker(max_steps=5, near_trans_thresh=0.08, near_rot_thresh=0.25)
    for t in range(5):
        tracker_fail5.record_step(
            step=t,
            action=np.zeros(7),
            info={
                "is_success": 0,
                "is_info_exist": 1,
                "trans_dist": 0.04,
                "rot_dist": 0.1,
                "gripper_finger_contacts": [1, 1],
                "evaluate_info": [0, 1, 0, 0], # grasp=1, lift=0
            },
            timing_ms={},
        )
    rec5, _, _ = tracker_fail5.finalize()
    assert rec5["failure_label"] == "grasp_not_lifted"

    # 7. Failed episode: lifted then dropped
    tracker_fail6 = EpisodeMetricsTracker(max_steps=5, near_trans_thresh=0.08, near_rot_thresh=0.25)
    for t in range(5):
        eval_info = [0, 1, 0, 1] if t < 4 else [0, 0, 0, 0] # dropped at last step
        tracker_fail6.record_step(
            step=t,
            action=np.zeros(7),
            info={
                "is_success": 0,
                "is_info_exist": 1,
                "trans_dist": 0.04,
                "rot_dist": 0.1,
                "gripper_finger_contacts": [1, 1],
                "evaluate_info": eval_info,
            },
            timing_ms={},
        )
    rec6, _, _ = tracker_fail6.finalize()
    assert rec6["failure_label"] == "lifted_then_dropped"

    # 8. Failed episode: lifted not static
    tracker_fail7 = EpisodeMetricsTracker(max_steps=5, near_trans_thresh=0.08, near_rot_thresh=0.25)
    for t in range(5):
        tracker_fail7.record_step(
            step=t,
            action=np.zeros(7),
            info={
                "is_success": 0,
                "is_info_exist": 1,
                "trans_dist": 0.04,
                "rot_dist": 0.1,
                "gripper_finger_contacts": [1, 1],
                "evaluate_info": [0, 1, 0, 1], # lifted & grasped, but robot & obj not static
            },
            timing_ms={},
        )
    rec7, _, _ = tracker_fail7.finalize()
    assert rec7["failure_label"] == "lifted_not_static"
    print("  -> Passed!")


def test_compare_runs_mcnemar():
    print("Testing compare_runs and McNemar test...")
    # Exact McNemar test: discordant counts
    p_eq = exact_mcnemar_p_value(10, 10)
    assert abs(p_eq - 1.0) < 1e-3, f"Equal discordant pairs should have p ~ 1.0, got {p_eq}"
    p_skew = exact_mcnemar_p_value(15, 0)
    assert p_skew < 0.001, f"Extremely skewed pairs should have small p-value, got {p_skew}"

    # Test compare function on synthetic files
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = os.path.join(tmpdir, "run1.jsonl")
        file2 = os.path.join(tmpdir, "run2.jsonl")

        # 4 paired episodes
        episodes_run1 = [
            {"gen_traj_mode": "bezier2d", "model_id": "003", "seed": 100, "episode_seed": 1001, "success_any": 1, "failure_label": "none", "stage_funnel": {"grasp_visible": 1, "near_target": 1, "finger_contact": 1, "is_obj_grasp": 1, "is_obj_lift": 1, "success": 1}},
            {"gen_traj_mode": "bezier2d", "model_id": "003", "seed": 100, "episode_seed": 1002, "success_any": 1, "failure_label": "none", "stage_funnel": {"grasp_visible": 1, "near_target": 1, "finger_contact": 1, "is_obj_grasp": 1, "is_obj_lift": 1, "success": 1}},
            {"gen_traj_mode": "bezier2d", "model_id": "003", "seed": 100, "episode_seed": 1003, "success_any": 0, "failure_label": "never_near", "stage_funnel": {"grasp_visible": 1, "near_target": 0, "finger_contact": 0, "is_obj_grasp": 0, "is_obj_lift": 0, "success": 0}},
            {"gen_traj_mode": "bezier2d", "model_id": "003", "seed": 100, "episode_seed": 1004, "success_any": 0, "failure_label": "contact_no_grasp", "stage_funnel": {"grasp_visible": 1, "near_target": 1, "finger_contact": 1, "is_obj_grasp": 0, "is_obj_lift": 0, "success": 0}},
        ]
        episodes_run2 = [
            {"gen_traj_mode": "bezier2d", "model_id": "003", "seed": 100, "episode_seed": 1001, "success_any": 0, "failure_label": "contact_no_grasp", "stage_funnel": {"grasp_visible": 1, "near_target": 1, "finger_contact": 1, "is_obj_grasp": 0, "is_obj_lift": 0, "success": 0}},
            {"gen_traj_mode": "bezier2d", "model_id": "003", "seed": 100, "episode_seed": 1002, "success_any": 1, "failure_label": "none", "stage_funnel": {"grasp_visible": 1, "near_target": 1, "finger_contact": 1, "is_obj_grasp": 1, "is_obj_lift": 1, "success": 1}},
            {"gen_traj_mode": "bezier2d", "model_id": "003", "seed": 100, "episode_seed": 1003, "success_any": 0, "failure_label": "never_near", "stage_funnel": {"grasp_visible": 1, "near_target": 0, "finger_contact": 0, "is_obj_grasp": 0, "is_obj_lift": 0, "success": 0}},
            {"gen_traj_mode": "bezier2d", "model_id": "003", "seed": 100, "episode_seed": 1004, "success_any": 0, "failure_label": "never_saw_grasp", "stage_funnel": {"grasp_visible": 0, "near_target": 0, "finger_contact": 0, "is_obj_grasp": 0, "is_obj_lift": 0, "success": 0}},
        ]

        for ep in episodes_run1:
            write_jsonl_line(file1, ep)
        for ep in episodes_run2:
            write_jsonl_line(file2, ep)

        res = compare(file1, file2, name1="LSTM", name2="Baseline")
        assert res["is_paired"] is True
        assert res["run1"]["success_rate"] == 0.5
        assert res["run2"]["success_rate"] == 0.25
        assert res["success_diff"] == 0.25
    print("  -> Passed!")


def test_miss_grasp_metrics():
    print("Testing closest gripper position on miss grasp tracking...")
    # 1. Failed episode: gripper approaches object but misses grasp
    tracker_fail = EpisodeMetricsTracker(max_steps=5)
    obj_pos = np.array([0.0, 0.0, 0.0])
    # Trajectory of TCP positions: step 2 is closest
    tcp_positions = [
        np.array([0.0, 0.0, 0.5]),
        np.array([0.0, 0.0, 0.2]),
        np.array([0.01, -0.02, 0.03]), # dist = sqrt(0.0001 + 0.0004 + 0.0009) = sqrt(0.0014) ~ 0.03741657
        np.array([0.05, 0.0, 0.1]),
        np.array([0.1, 0.0, 0.2]),
    ]

    for t in range(5):
        tracker_fail.record_step(
            step=t,
            action=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8]), # gripper closing
            info={
                "is_success": 0,
                "is_info_exist": 1,
                "gripper_finger_contacts": [0, 0],
                "evaluate_info": [0, 0, 0, 0], # no grasp
            },
            timing_ms={},
            tcp_pos=tcp_positions[t],
            tcp_rot=[0.0, 0.1, 0.2],
            obj_pos=obj_pos,
            obj_rot=[0.0, 0.0, 0.0],
            gripper_width=0.045,
        )

    rec_fail, _, _ = tracker_fail.finalize()
    assert rec_fail["miss_grasp"]["is_miss_grasp"] == 1
    assert rec_fail["miss_grasp"]["closest_step"] == 2
    expected_dist = float(np.linalg.norm(tcp_positions[2] - obj_pos))
    assert abs(rec_fail["miss_grasp"]["min_dist"] - expected_dist) < 1e-5
    assert abs(rec_fail["miss_grasp_min_dist"] - expected_dist) < 1e-5
    assert rec_fail["miss_grasp_closest_step"] == 2

    # Offset vector: obj_pos - tcp_pos = [-0.01, 0.02, -0.03]
    assert abs(rec_fail["miss_grasp"]["offset_dx"] - (-0.01)) < 1e-5
    assert abs(rec_fail["miss_grasp"]["offset_dy"] - 0.02) < 1e-5
    assert abs(rec_fail["miss_grasp"]["offset_dz"] - (-0.03)) < 1e-5
    assert abs(rec_fail["miss_grasp_offset_dx"] - (-0.01)) < 1e-5
    assert abs(rec_fail["miss_grasp_tcp_x"] - 0.01) < 1e-5
    assert abs(rec_fail["miss_grasp_obj_x"] - 0.0) < 1e-5
    assert abs(rec_fail["miss_grasp"]["gripper_width"] - 0.045) < 1e-5
    assert abs(rec_fail["miss_grasp"]["gripper_action"] - 0.8) < 1e-5
    assert rec_fail["miss_grasp"]["has_contact"] == 0

    # 2. Successful episode: achieved grasp and lifted
    tracker_succ = EpisodeMetricsTracker(max_steps=3)
    for t in range(3):
        tracker_succ.record_step(
            step=t,
            action=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            info={
                "is_success": 1 if t == 2 else 0,
                "is_info_exist": 1,
                "gripper_finger_contacts": [1, 1],
                "evaluate_info": [1, 1, 1, 1], # grasped & lifted
            },
            timing_ms={},
            tcp_pos=np.array([0.0, 0.0, 0.02]),
            obj_pos=np.array([0.0, 0.0, 0.02]),
            gripper_width=0.01,
        )

    rec_succ, _, _ = tracker_succ.finalize()
    assert rec_succ["miss_grasp"]["is_miss_grasp"] == 0
    assert np.isnan(rec_succ["miss_grasp"]["min_dist"])
    assert np.isnan(rec_succ["miss_grasp_min_dist"])
    assert rec_succ["miss_grasp"]["closest_step"] == -1
    # But closest_gripper_snapshot is still recorded!
    assert abs(rec_succ["closest_gripper_snapshot"]["distance"] - 0.0) < 1e-5

    # 3. Fallback when telemetry is in info dict
    tracker_info = EpisodeMetricsTracker(max_steps=2)
    tracker_info.record_step(
        step=0,
        action=np.zeros(7),
        info={
            "is_success": 0,
            "is_info_exist": 1,
            "tcp_pose": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
            "obj_pos": [0.1, 0.2, 0.35],
            "evaluate_info": [0, 0, 0, 0],
        },
        timing_ms={},
    )
    rec_info, _, _ = tracker_info.finalize()
    assert rec_info["miss_grasp"]["is_miss_grasp"] == 1
    assert abs(rec_info["miss_grasp"]["min_dist"] - 0.05) < 1e-5
    assert abs(rec_info["miss_grasp"]["offset_dz"] - 0.05) < 1e-5

    # 4. RunMetricsAggregator
    agg = RunMetricsAggregator()
    agg.add_episode(rec_fail)
    agg.add_episode(rec_succ)
    summary = agg.compute_summary()

    miss_analysis = summary["miss_grasp_analysis"]
    assert miss_analysis["num_miss_grasps"] == 1
    assert abs(miss_analysis["miss_grasp_rate"] - 0.5) < 1e-5
    assert abs(miss_analysis["min_dist"]["mean"] - expected_dist) < 1e-5
    assert abs(miss_analysis["offset_dx"]["mean"] - (-0.01)) < 1e-5
    assert abs(miss_analysis["gripper_width_at_closest"]["mean"] - 0.045) < 1e-5
    assert "miss_grasp_min_dist" in summary["control_quality"]
    print("  -> Passed!")


if __name__ == "__main__":
    test_statistics()
    test_geometry_and_rotmat_error()
    test_window_ablation()
    test_episode_tracking_and_funnel()
    test_compare_runs_mcnemar()
    test_miss_grasp_metrics()
    print("\nALL UNIT TESTS PASSED SUCCESSFULLY!")
