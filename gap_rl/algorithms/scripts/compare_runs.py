#!/usr/bin/env python3
"""
Standalone script to compare two evaluation runs (e.g., LSTM vs Baseline).

Pairs episodes by (traj_mode, model_id, seed, episode_seed).
Reports the paired success difference with exact McNemar test and bootstrap CI.
Prints side-by-side stage funnel and failure-mode tables.
Gracefully falls back to unpaired comparison if episode seeds do not line up.
"""

import os
import sys
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def load_records(file_path: str) -> List[Dict[str, Any]]:
    """Loads episode records from JSONL or JSON."""
    records = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, list):
                    records.extend(data)
                elif isinstance(data, dict):
                    records.append(data)
            except json.JSONDecodeError:
                # If entire file was a single JSON object with "episodes"
                f.seek(0)
                whole = json.load(f)
                if isinstance(whole, list):
                    return whole
                elif isinstance(whole, dict) and "episodes" in whole:
                    return whole["episodes"]
                raise
    return records


def get_episode_key(record: Dict[str, Any]) -> Tuple[Any, Any, Any, Any]:
    traj = record.get("gen_traj_mode") or record.get("traj_mode")
    model_id = record.get("model_id")
    seed = record.get("seed")
    epi_seed = record.get("episode_seed")
    return (traj, model_id, seed, epi_seed)


def exact_mcnemar_p_value(b: int, c: int) -> float:
    """
    Computes exact two-sided p-value for McNemar test.
    b: discordant pairs where run1=1, run2=0
    c: discordant pairs where run1=0, run2=1
    Under H0, b ~ Binomial(b + c, 0.5).
    """
    n = b + c
    if n == 0:
        return 1.0

    k = min(b, c)
    # Sum binomial PMF for i <= k: sum_{i=0}^k (n choose i) * 0.5^n
    try:
        from scipy.stats import binom
        p_val = 2.0 * float(binom.cdf(k, n, 0.5))
    except ImportError:
        import math
        p_one_tail = sum(math.comb(n, i) * (0.5 ** n) for i in range(k + 1))
        p_val = 2.0 * p_one_tail

    return float(min(1.0, max(0.0, p_val)))


def paired_bootstrap_diff_ci(
    y1: np.ndarray, y2: np.ndarray, num_resamples: int = 2000, ci: float = 0.95
) -> Tuple[float, float]:
    """Computes paired bootstrap confidence interval for mean(y1) - mean(y2)."""
    n = len(y1)
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.RandomState(42)
    indices = rng.randint(0, n, size=(num_resamples, n))
    diffs = np.mean(y1[indices], axis=1) - np.mean(y2[indices], axis=1)
    alpha = (1.0 - ci) / 2.0
    return float(np.percentile(diffs, 100.0 * alpha)), float(np.percentile(diffs, 100.0 * (1.0 - alpha)))


def unpaired_bootstrap_diff_ci(
    y1: np.ndarray, y2: np.ndarray, num_resamples: int = 2000, ci: float = 0.95
) -> Tuple[float, float]:
    """Computes independent bootstrap confidence interval for mean(y1) - mean(y2)."""
    n1, n2 = len(y1), len(y2)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    rng = np.random.RandomState(42)
    idx1 = rng.randint(0, n1, size=(num_resamples, n1))
    idx2 = rng.randint(0, n2, size=(num_resamples, n2))
    diffs = np.mean(y1[idx1], axis=1) - np.mean(y2[idx2], axis=1)
    alpha = (1.0 - ci) / 2.0
    return float(np.percentile(diffs, 100.0 * alpha)), float(np.percentile(diffs, 100.0 * (1.0 - alpha)))


def format_table(headers: List[str], rows: List[List[str]]) -> str:
    """Formats markdown / ASCII table cleanly."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(str(val)))

    header_str = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    separator = "-|-".join("-" * col_widths[i] for i in range(len(headers)))
    row_strs = [" | ".join(str(r[i]).ljust(col_widths[i]) for i in range(len(headers))) for r in rows]
    return f"{header_str}\n{separator}\n" + "\n".join(row_strs)


def compare(run1_path: str, run2_path: str, name1: str = "Run 1", name2: str = "Run 2") -> Dict[str, Any]:
    recs1 = load_records(run1_path)
    recs2 = load_records(run2_path)

    # Index by key
    indexed1 = {get_episode_key(r): r for r in recs1}
    indexed2 = {get_episode_key(r): r for r in recs2}

    common_keys = [k for k in indexed1 if k in indexed2 and None not in k]
    is_paired = len(common_keys) > 0 and len(common_keys) == max(len(recs1), len(recs2))
    partial_paired = len(common_keys) > 0 and not is_paired

    print("=" * 80)
    print(f"RUN COMPARISON: {name1} vs {name2}")
    print(f"Total episodes: {name1} = {len(recs1)}, {name2} = {len(recs2)}")
    if is_paired:
        print(f"Comparison Mode: PAIRED (All {len(common_keys)} episodes aligned exactly by seed)")
    elif partial_paired:
        print(f"Comparison Mode: PARTIALLY PAIRED ({len(common_keys)} aligned pairs found)")
    else:
        print("Warning: Episode seeds do NOT line up across runs. Falling back to UNPAIRED comparison.")
    print("=" * 80)

    # Success comparison
    if is_paired or partial_paired:
        pair_list = [(indexed1[k], indexed2[k]) for k in common_keys]
        y1 = np.array([p[0]["success_any"] for p in pair_list], dtype=int)
        y2 = np.array([p[1]["success_any"] for p in pair_list], dtype=int)
        n = len(y1)

        n11 = int(np.sum((y1 == 1) & (y2 == 1)))
        n10 = int(np.sum((y1 == 1) & (y2 == 0)))
        n01 = int(np.sum((y1 == 0) & (y2 == 1)))
        n00 = int(np.sum((y1 == 0) & (y2 == 0)))

        rate1 = float(np.mean(y1))
        rate2 = float(np.mean(y2))
        diff = rate1 - rate2
        mcnemar_p = exact_mcnemar_p_value(n10, n01)
        diff_ci = paired_bootstrap_diff_ci(y1, y2)

        contingency_desc = (
            f"Contingency: Both Success={n11}, {name1} Only={n10}, "
            f"{name2} Only={n01}, Both Fail={n00}"
        )
    else:
        y1 = np.array([r["success_any"] for r in recs1], dtype=int)
        y2 = np.array([r["success_any"] for r in recs2], dtype=int)
        rate1 = float(np.mean(y1)) if len(y1) > 0 else 0.0
        rate2 = float(np.mean(y2)) if len(y2) > 0 else 0.0
        diff = rate1 - rate2
        diff_ci = unpaired_bootstrap_diff_ci(y1, y2)
        mcnemar_p = None
        contingency_desc = "N/A (unpaired episodes)"

    print("\n--- 1. PRIMARY SUCCESS COMPARISON ---")
    headers = ["Metric", name1, name2, "Difference (1 - 2)", "95% Bootstrap CI", "p-value"]
    pval_str = f"{mcnemar_p:.4f} (exact McNemar)" if mcnemar_p is not None else "N/A"
    row = [
        "success_any",
        f"{rate1*100:.1f}%",
        f"{rate2*100:.1f}%",
        f"{diff*100:+.1f}%",
        f"[{diff_ci[0]*100:+.1f}%, {diff_ci[1]*100:+.1f}%]",
        pval_str,
    ]
    print(format_table(headers, [row]))
    print(f"Details: {contingency_desc}")

    # Stage funnel comparison
    print("\n--- 2. STAGE FUNNEL COMPARISON (% of episodes ever reaching stage) ---")
    funnel_keys = ["grasp_visible", "near_target", "finger_contact", "is_obj_grasp", "is_obj_lift", "success"]
    f_headers = ["Funnel Stage", f"{name1} (%)", f"{name2} (%)", "Difference (pts)"]
    f_rows = []
    for fk in funnel_keys:
        p1 = float(np.mean([r["stage_funnel"].get(fk, 0) for r in recs1])) * 100.0
        p2 = float(np.mean([r["stage_funnel"].get(fk, 0) for r in recs2])) * 100.0
        f_diff = p1 - p2
        f_rows.append([fk, f"{p1:.1f}%", f"{p2:.1f}%", f"{f_diff:+.1f}"])
    print(format_table(f_headers, f_rows))

    # Failure modes comparison (mutually exclusive)
    print("\n--- 3. FAILURE MODES BREAKDOWN (counts and % among failed episodes) ---")
    fail_keys = [
        "never_saw_grasp",
        "never_near",
        "near_no_contact",
        "contact_no_grasp",
        "grasp_not_lifted",
        "lifted_then_dropped",
        "lifted_not_static",
        "other",
    ]
    n_failed1 = sum(1 for r in recs1 if not r["success_any"])
    n_failed2 = sum(1 for r in recs2 if not r["success_any"])
    fail_headers = ["Failure Label", f"{name1} (N={n_failed1})", f"{name2} (N={n_failed2})"]
    fail_rows = []
    for fk in fail_keys:
        c1 = sum(1 for r in recs1 if r["failure_label"] == fk)
        c2 = sum(1 for r in recs2 if r["failure_label"] == fk)
        pct1 = (float(c1) / n_failed1 * 100.0) if n_failed1 > 0 else 0.0
        pct2 = (float(c2) / n_failed2 * 100.0) if n_failed2 > 0 else 0.0
        fail_rows.append([fk, f"{c1} ({pct1:.1f}%)", f"{c2} ({pct2:.1f}%)"])
    print(format_table(fail_headers, fail_rows))

    # Control quality & dynamics
    print("\n--- 4. CONTROL QUALITY & DYNAMICS COMPARISON ---")
    cq_keys = [
        ("action_diff_mean", "Action diff mean"),
        ("action_jerk_mean", "Action jerk mean"),
        ("action_norm_mean", "Action norm mean"),
        ("arm_qvel_mean", "Arm qvel mean"),
        ("arm_qvel_peak", "Arm qvel peak"),
        ("robot_static_fraction", "Robot static fraction"),
        ("mean_obj_lin_speed", "Obj speed lin mean"),
        ("min_trans_dist", "Min trans dist"),
        ("min_rot_dist", "Min rot dist"),
    ]
    cq_headers = ["Metric", name1, name2, "Difference"]
    cq_rows = []
    for k, label in cq_keys:
        v1 = [r.get(k) for r in recs1 if r.get(k) is not None and not np.isnan(r.get(k))]
        v2 = [r.get(k) for r in recs2 if r.get(k) is not None and not np.isnan(r.get(k))]
        m1 = float(np.mean(v1)) if v1 else float("nan")
        m2 = float(np.mean(v2)) if v2 else float("nan")
        d = (m1 - m2) if not (np.isnan(m1) or np.isnan(m2)) else float("nan")
        cq_rows.append([label, f"{m1:.4f}", f"{m2:.4f}", f"{d:+.4f}"])
    print(format_table(cq_headers, cq_rows))

    summary_out = {
        "run1": {"name": name1, "path": run1_path, "num_episodes": len(recs1), "success_rate": rate1},
        "run2": {"name": name2, "path": run2_path, "num_episodes": len(recs2), "success_rate": rate2},
        "success_diff": diff,
        "diff_ci95": diff_ci,
        "mcnemar_p_value": mcnemar_p,
        "is_paired": is_paired,
    }
    return summary_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two GAP-RL evaluation runs side-by-side.")
    parser.add_argument("run1", type=str, help="Path to run 1 episodes.jsonl")
    parser.add_argument("run2", type=str, help="Path to run 2 episodes.jsonl")
    parser.add_argument("--name1", type=str, default="Run 1", help="Label for run 1")
    parser.add_argument("--name2", type=str, default="Run 2", help="Label for run 2")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    res = compare(args.run1, args.run2, name1=args.name1, name2=args.name2)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print(f"\nSaved comparison result to: {args.output}")
