#!/usr/bin/env python3
"""Select diverse Allen-key grasp snapshots for physical replay validation."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TIER_NAMES = ("easy", "support", "broad")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rollout-dir", type=Path,
        default=ROOT / "outputs/allen_key_pretrained_turning/20260822_stable_init_headless_12288",
    )
    parser.add_argument(
        "--template-bank", type=Path,
        default=ROOT / "assets/grasp_banks/allen_key_manipulation_v3.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "outputs/allen_key_rollout_bank/provisional_rollout_grasps.json",
    )
    parser.add_argument("--desired-entries", type=int, default=96)
    parser.add_argument(
        "--tier-fractions", type=float, nargs=3, default=(0.60, 0.25, 0.15),
        metavar=("EASY", "SUPPORT", "BROAD"),
    )
    parser.add_argument("--minimum-contact-fingers", type=int, default=2)
    parser.add_argument("--contact-threshold-n", type=float, default=0.05)
    parser.add_argument("--maximum-socket-error-m", type=float, default=0.006)
    return parser.parse_args()


def load_adjustment_utils():
    path = ROOT / "isaacsimenvs/tasks/simtoolreal/utils/adjustment_utils.py"
    spec = importlib.util.spec_from_file_location("allen_rollout_adjustment_utils", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def canonical_quaternion(values: list[float]) -> np.ndarray:
    quaternion = np.asarray(values, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("candidate contains an invalid quaternion")
    norm = np.linalg.norm(quaternion)
    if norm < 1.0e-8:
        raise ValueError("candidate contains a zero quaternion")
    quaternion /= norm
    return quaternion if quaternion[0] >= 0.0 else -quaternion


def candidate_feature(candidate: dict, outcome: dict) -> np.ndarray:
    quaternion = canonical_quaternion(candidate["palm_to_tool_quat_wxyz"])
    yaw = math.radians(float(outcome["yaw_deg"]))
    return np.concatenate((
        np.asarray(candidate["object_pos_local"], dtype=np.float64),
        np.asarray((math.sin(yaw), math.cos(yaw))),
        np.asarray(candidate["palm_to_tool_pos"], dtype=np.float64),
        quaternion,
    ))


def quality(candidate: dict, outcome: dict) -> float:
    return (
        4.0 * int(outcome["all_turns_success"])
        + 1.0 * int(outcome["turn_90_success"])
        + 0.5 * int(outcome["turn_60_success"])
        + 0.25 * int(outcome["turn_30_success"])
        + 0.1 * (int(candidate["turn_stage"]) + 1)
    )


def diverse_select(pool: list[tuple[dict, dict]], count: int) -> list[tuple[dict, dict]]:
    if len(pool) < count:
        raise RuntimeError(f"workspace tier has only {len(pool)}/{count} eligible snapshots")
    features = np.stack([candidate_feature(candidate, outcome) for candidate, outcome in pool])
    scale = features.std(axis=0)
    scale[scale < 1.0e-6] = 1.0
    features = (features - features.mean(axis=0)) / scale
    qualities = np.asarray([quality(candidate, outcome) for candidate, outcome in pool])
    normalized_quality = (qualities - qualities.min()) / max(float(np.ptp(qualities)), 1.0)
    selected = [int(np.argmax(qualities))]
    minimum_distance = np.square(features - features[selected[0]]).sum(axis=1)
    minimum_distance[selected[0]] = -math.inf
    while len(selected) < count:
        score = minimum_distance + 0.25 * normalized_quality
        index = int(np.argmax(score))
        selected.append(index)
        distance = np.square(features - features[index]).sum(axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[selected] = -math.inf
    return [pool[index] for index in selected]


def quota(total: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    raw = np.asarray(fractions, dtype=np.float64) * total
    counts = np.floor(raw).astype(np.int64)
    for index in np.argsort(-(raw - counts))[:total - int(counts.sum())]:
        counts[index] += 1
    return tuple(int(value) for value in counts)


def provisional_entry(candidate: dict, outcome: dict, tier: int,
                      template: dict, args: argparse.Namespace) -> dict:
    forces = np.asarray(candidate["fingertip_force_n"], dtype=np.float64)
    support = int((forces >= float(args.contact_threshold_n)).sum())
    verification = {
        "support_count": support,
        "edge_clearance_m": 0.04,
        "table_force_n": 0.0,
        "stable_steps": 60,
        "hold_steps": 120,
        "hold_drift_m": 0.0,
        "hold_rotation_deg": 0.0,
        "pickup_orientation_error_deg": 0.0,
        "joint_limit_violation_max_rad": 0.0,
        "tactile_finger_count_min": support,
        "tactile_contact_area_mean": 0.0,
        "tactile_depth_mean": 0.0,
        "tactile_depth_max": 0.0,
        "provisional_rollout_snapshot": True,
        "source_env_id": int(candidate["source_env_id"]),
        "source_policy_step": int(candidate["policy_step"]),
        "source_turn_stage": int(candidate["turn_stage"]),
        "source_target_turn_angle_deg": float(candidate["target_turn_angle_deg"]),
        "workspace_tier": TIER_NAMES[tier],
        "workspace_tier_id": tier,
        "rollout_functional_quality": quality(candidate, outcome),
        "rollout_all_turns_success": bool(int(outcome["all_turns_success"])),
        "rollout_turn_30_success": bool(int(outcome["turn_30_success"])),
        "rollout_turn_60_success": bool(int(outcome["turn_60_success"])),
        "rollout_turn_90_success": bool(int(outcome["turn_90_success"])),
        "socket_x_m": float(outcome["socket_x_m"]),
        "socket_y_m": float(outcome["socket_y_m"]),
        "socket_z_m": float(outcome["socket_z_m"]),
        "tool_yaw_deg": float(outcome["yaw_deg"]),
        "handle_center_x_m": float(outcome["handle_center_x_m"]),
        "handle_center_y_m": float(outcome["handle_center_y_m"]),
        "handle_center_z_m": float(outcome["handle_center_z_m"]),
        "source_pose_position_error_m": float(candidate["pose_position_error_m"]),
        "source_pose_rotation_error_deg": float(candidate["pose_rotation_error_deg"]),
        "source_socket_pivot_error_m": float(candidate["socket_pivot_error_m"]),
    }
    object_quat = canonical_quaternion(candidate["object_quat_wxyz"]).tolist()
    return {
        "joint_pos_canonical": candidate["joint_pos_canonical"],
        "joint_vel_canonical": [0.0] * 29,
        "joint_targets_canonical": candidate["joint_targets_canonical"],
        "last_action_canonical": candidate["last_action_canonical"],
        "object_pos_local": candidate["object_pos_local"],
        "object_quat_wxyz": object_quat,
        "object_velocity": [0.0] * 6,
        "palm_to_tool_pos": candidate["palm_to_tool_pos"],
        "palm_to_tool_quat_wxyz": canonical_quaternion(
            candidate["palm_to_tool_quat_wxyz"]
        ).tolist(),
        "reference_contact_quat_wxyz": object_quat,
        "reference_edge_yaw_rad": 0.0,
        "reference_edge_tilt_rad": math.pi / 4.0,
        "verification": verification,
    }


def main() -> None:
    args = parse_args()
    if args.desired_entries <= 0:
        raise ValueError("--desired-entries must be positive")
    fractions = tuple(float(value) for value in args.tier_fractions)
    if any(not math.isfinite(value) or value < 0.0 for value in fractions) or not math.isclose(
        sum(fractions), 1.0, abs_tol=1.0e-6
    ):
        raise ValueError("--tier-fractions must be non-negative and sum to one")
    candidates_path = args.rollout_dir / "grasp_candidates.json"
    outcomes_path = args.rollout_dir / "pose_outcomes.csv"
    if not candidates_path.is_file() or not outcomes_path.is_file():
        raise FileNotFoundError("rollout directory lacks grasp_candidates.json or pose_outcomes.csv")
    candidates_payload = json.loads(candidates_path.read_text())
    template = json.loads(args.template_bank.read_text())
    with outcomes_path.open(newline="") as file:
        outcomes = {int(row["env_id"]): row for row in csv.DictReader(file)}
    utils = load_adjustment_utils()

    pools: list[list[tuple[dict, dict]]] = [[], [], []]
    lower = np.asarray(template["joint_lower_canonical"], dtype=np.float64)
    upper = np.asarray(template["joint_upper_canonical"], dtype=np.float64)
    tolerance = float(template["joint_limit_tolerance_rad"])
    rejected = {"missing": 0, "not_acquired": 0, "contact": 0, "socket": 0, "joint": 0}
    for candidate in candidates_payload.get("entries", []):
        outcome = outcomes.get(int(candidate["source_env_id"]))
        if outcome is None:
            rejected["missing"] += 1
            continue
        if not int(outcome["acquired"]):
            rejected["not_acquired"] += 1
            continue
        forces = np.asarray(candidate["fingertip_force_n"], dtype=np.float64)
        if int((forces >= float(args.contact_threshold_n)).sum()) < int(
            args.minimum_contact_fingers
        ):
            rejected["contact"] += 1
            continue
        if float(candidate["socket_pivot_error_m"]) > float(args.maximum_socket_error_m):
            rejected["socket"] += 1
            continue
        joints = np.asarray(candidate["joint_pos_canonical"], dtype=np.float64)
        if joints.shape != (29,) or not np.isfinite(joints).all() or bool(
            ((joints < lower - tolerance) | (joints > upper + tolerance)).any()
        ):
            rejected["joint"] += 1
            continue
        tier = int(utils.allen_workspace_tier(
            torch.tensor([float(outcome["handle_center_x_m"])]),
            torch.tensor([float(outcome["handle_center_y_m"])]),
            torch.tensor([float(outcome["socket_z_m"])]),
            torch.tensor([float(outcome["yaw_deg"])]),
        )[0].item())
        pools[tier].append((candidate, outcome))

    counts = quota(int(args.desired_entries), fractions)
    selected: list[tuple[dict, dict, int]] = []
    for tier, count in enumerate(counts):
        selected.extend(
            (candidate, outcome, tier)
            for candidate, outcome in diverse_select(pools[tier], count)
        )
    entries = [
        provisional_entry(candidate, outcome, tier, template, args)
        for candidate, outcome, tier in selected
    ]
    payload = {
        key: template[key] for key in (
            "schema_version", "tool_type", "object_name", "asset_sha256",
            "source_checkpoint", "source_checkpoint_sha256", "policy_coefficient_id",
            "tactile_rich_fraction_min", "tactile_min_fingers", "seed", "control_dt_s",
            "joint_lower_canonical", "joint_upper_canonical", "joint_limit_tolerance_rad",
        )
    }
    payload.update({
        "kind": "allen_key_rollout_provisional_grasp_bank",
        "source_rollout_dir": str(args.rollout_dir.resolve()),
        "selection_tier_fractions": fractions,
        "entries": entries,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    selected_quality = [entry["verification"]["rollout_functional_quality"] for entry in entries]
    print(
        f"[pass] wrote {len(entries)} provisional snapshots to {args.output.resolve()} "
        f"tiers={dict(zip(TIER_NAMES, counts, strict=True))} "
        f"quality={min(selected_quality):.2f}..{max(selected_quality):.2f} "
        f"rejected={rejected}"
    )


if __name__ == "__main__":
    main()
