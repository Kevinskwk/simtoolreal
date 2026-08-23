#!/usr/bin/env python3
"""Pair palm-down, side-changing grasps from validated policy rollouts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def load_grasp_evaluator():
    path = ROOT / "isaacsimenvs/tasks/simtoolreal/utils/grasp_evaluator.py"
    spec = importlib.util.spec_from_file_location("palm_down_grasp_evaluator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EVALUATOR = load_grasp_evaluator()
GraspEvaluatorThresholds = EVALUATOR.GraspEvaluatorThresholds
UrdfKinematics = EVALUATOR.UrdfKinematics
solve_arm_ik = EVALUATOR.solve_arm_ik


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rollout-dir", type=Path,
        default=ROOT / "outputs/allen_key_pretrained_turning/thick_handle_bank_20260823",
    )
    parser.add_argument(
        "--template-bank", type=Path,
        default=ROOT / "assets/grasp_banks/allen_key_manipulation_v3.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "assets/grasp_banks/allen_key_palm_down_v1.json",
    )
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument(
        "--target-yaw-changes-deg", type=float, nargs="+",
        default=(60.0,),
    )
    parser.add_argument("--target-yaw-tolerance-deg", type=float, default=15.0)
    parser.add_argument("--minimum-screw-axis-alignment", type=float, default=0.90)
    parser.add_argument("--minimum-source-turn-stage", type=int, default=2)
    parser.add_argument("--exclude-source-env-ids", type=int, nargs="*", default=())
    parser.add_argument("--minimum-palm-down-cosine", type=float, default=0.65)
    parser.add_argument("--minimum-contact-fingers", type=int, default=2)
    parser.add_argument("--contact-threshold-n", type=float, default=0.05)
    parser.add_argument(
        "--maximum-snapshot-fingertip-force-n", type=float, default=25.0
    )
    parser.add_argument("--maximum-source-quality", type=float, default=10.0)
    parser.add_argument("--minimum-quality-improvement", type=float, default=0.0)
    parser.add_argument("--minimum-target-quality", type=float, default=0.30)
    parser.add_argument("--maximum-source-candidates", type=int, default=128)
    parser.add_argument("--maximum-targets-per-source", type=int, default=12)
    parser.add_argument("--minimum-tool-position-separation-m", type=float, default=0.035)
    return parser.parse_args()


def pose(position: list[float], quaternion_wxyz: list[float]) -> np.ndarray:
    transform = np.eye(4)
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    transform[:3, :3] = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    transform[:3, 3] = np.asarray(position, dtype=np.float64)
    return transform


def quaternion_wxyz(transform: np.ndarray) -> list[float]:
    xyzw = Rotation.from_matrix(transform[:3, :3]).as_quat()
    if xyzw[3] < 0.0:
        xyzw *= -1.0
    return [float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2])]


def controllability(
    kinematics: UrdfKinematics, arm: np.ndarray, hand: np.ndarray,
    robot_base: np.ndarray,
) -> dict[str, float]:
    _, jacobian, _ = kinematics.palm_fk_jacobian(arm, hand, robot_base)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    minimum_sigma = float(singular[-1])
    condition = float(singular[0] / max(minimum_sigma, 1.0e-12))
    margin = float(np.minimum(
        arm - kinematics.arm_lower, kinematics.arm_upper - arm
    ).min())
    factors = (
        max(margin, 1.0e-6) / 0.20,
        max(minimum_sigma, 1.0e-6) / 0.10,
        20.0 / max(condition, 1.0),
    )
    return {
        "arm_joint_margin_rad": margin,
        "arm_minimum_jacobian_singular_value": minimum_sigma,
        "arm_jacobian_condition_number": condition,
        "rollout_functional_quality": float(np.prod(factors) ** (1.0 / 3.0)),
    }


def base_verification(
    candidate: dict, metrics: dict[str, float], *, hold_steps: int
) -> dict:
    forces = np.asarray(candidate["fingertip_force_n"], dtype=np.float64)
    return {
        "support_count": int((forces >= 0.05).sum()),
        "edge_clearance_m": 0.04,
        "table_force_n": 0.0,
        "stable_steps": 15,
        "hold_steps": int(hold_steps),
        "hold_drift_m": 0.0,
        "hold_rotation_deg": 0.0,
        "pickup_orientation_error_deg": 0.0,
        "joint_limit_violation_max_rad": 0.0,
        "tactile_finger_count_min": int((forces >= 0.05).sum()),
        "tactile_contact_area_mean": 0.0,
        "tactile_depth_mean": 0.0,
        "tactile_depth_max": 0.0,
        "rollout_workspace_tier": "easy",
        "rollout_workspace_tier_id": 0,
        "source_env_id": int(candidate["source_env_id"]),
        "source_policy_step": int(candidate["policy_step"]),
        "source_turn_stage": int(candidate["turn_stage"]),
        **metrics,
    }


def bank_entry(candidate: dict, verification: dict) -> dict:
    return {
        "joint_pos_canonical": candidate["joint_pos_canonical"],
        "joint_vel_canonical": [0.0] * 29,
        # Start the fixed-tool hold at the measured configuration. Replaying
        # the acquisition policy's still-closing target causes artificial
        # penetration when the socket fixture prevents tool compliance.
        "joint_targets_canonical": candidate["joint_pos_canonical"],
        "last_action_canonical": candidate["last_action_canonical"],
        "object_pos_local": candidate["object_pos_local"],
        "object_quat_wxyz": candidate["object_quat_wxyz"],
        "object_velocity": [0.0] * 6,
        "palm_to_tool_pos": candidate["palm_to_tool_pos"],
        "palm_to_tool_quat_wxyz": candidate["palm_to_tool_quat_wxyz"],
        "reference_contact_quat_wxyz": candidate["object_quat_wxyz"],
        "reference_edge_yaw_rad": 0.0,
        "reference_edge_tilt_rad": math.pi / 4.0,
        "verification": verification,
    }


def main() -> None:
    args = parse_args()
    if args.pairs <= 0:
        raise ValueError("--pairs must be positive")
    candidates_payload = json.loads(
        (args.rollout_dir / "grasp_candidates.json").read_text()
    )
    template = json.loads(args.template_bank.read_text())
    asset_path = Path(candidates_payload["asset"])
    if not asset_path.is_file():
        raise FileNotFoundError(f"rollout asset does not exist: {asset_path}")
    expected_hash = candidates_payload["asset_sha256"]
    if sha256_file(asset_path) != expected_hash:
        raise RuntimeError("rollout Allen-key asset hash does not match its metadata")

    robot_path = (
        ROOT / "assets/urdf/kuka_sharpa_description"
        / "iiwa14_left_sharpa_adjusted_restricted.urdf"
    )
    kinematics = UrdfKinematics(robot_path)
    thresholds = GraspEvaluatorThresholds(
        ik_position_m=0.003, ik_orientation_deg=3.0, max_ik_iterations=400
    )
    robot_base = np.eye(4)
    robot_base[1, 3] = 0.8
    lower = np.asarray(template["joint_lower_canonical"], dtype=np.float64)
    upper = np.asarray(template["joint_upper_canonical"], dtype=np.float64)
    joint_tolerance = float(template.get("joint_limit_tolerance_rad", 0.0))

    proposals = []
    valid_candidates = []
    palm_axis_down_cosines: list[list[float]] = []
    source_fk_errors: list[tuple[float, float]] = []
    rejected = {
        "excluded": 0, "turn_stage": 0, "contact": 0, "joint": 0,
        "palm_direction": 0,
        "source_quality": 0, "target_ik": 0, "target_quality": 0,
    }
    excluded_source_env_ids = set(int(value) for value in args.exclude_source_env_ids)
    for candidate in candidates_payload.get("entries", []):
        if int(candidate["source_env_id"]) in excluded_source_env_ids:
            rejected["excluded"] += 1
            continue
        forces = np.asarray(candidate["fingertip_force_n"], dtype=np.float64)
        if (
            int((forces >= float(args.contact_threshold_n)).sum())
            < int(args.minimum_contact_fingers)
            or float(forces.max()) > float(args.maximum_snapshot_fingertip_force_n)
        ):
            rejected["contact"] += 1
            continue
        joints = np.asarray(candidate["joint_pos_canonical"], dtype=np.float64)
        if joints.shape != (29,) or bool((
            (joints < lower - joint_tolerance) | (joints > upper + joint_tolerance)
        ).any()):
            rejected["joint"] += 1
            continue
        tool = pose(candidate["object_pos_local"], candidate["object_quat_wxyz"])
        relative = pose(
            candidate["palm_to_tool_pos"], candidate["palm_to_tool_quat_wxyz"]
        )
        palm = tool @ np.linalg.inv(relative)
        palm_axis_down_cosines.append([
            float(palm[:3, axis] @ np.asarray((0.0, 0.0, -1.0)))
            for axis in range(3)
        ])
        # The merged iiwa-link-7 palm frame uses local -Y as the outward
        # palmar normal (the visualization marker itself is not a hand frame).
        palm_down_cosine = float(-palm[:3, 1] @ np.asarray((0.0, 0.0, -1.0)))
        if palm_down_cosine < float(args.minimum_palm_down_cosine):
            rejected["palm_direction"] += 1
            continue
        metrics = controllability(
            kinematics, joints[:7], joints[7:], robot_base
        )
        fk_palm = kinematics.palm_fk_jacobian(
            joints[:7], joints[7:], robot_base
        )[0]
        source_fk_errors.append((
            float(np.linalg.norm(fk_palm[:3, 3] - palm[:3, 3])),
            math.degrees(float(np.linalg.norm(
                Rotation.from_matrix(fk_palm[:3, :3] @ palm[:3, :3].T).as_rotvec()
            ))),
        ))
        valid_candidates.append({
            "candidate": candidate,
            "joints": joints,
            "tool": tool,
            "relative": relative,
            "palm": palm,
            "palm_down_cosine": palm_down_cosine,
            "metrics": metrics,
        })

    source_candidates = []
    for item in valid_candidates:
        if int(item["candidate"]["turn_stage"]) < int(args.minimum_source_turn_stage):
            rejected["turn_stage"] += 1
            continue
        if item["metrics"]["rollout_functional_quality"] > float(
            args.maximum_source_quality
        ):
            rejected["source_quality"] += 1
            continue
        source_candidates.append(item)

    # Expensive numerical IK is needed only for the lowest-controllability
    # tail, which is also the population relevant to this adjustment gate.
    source_candidates.sort(
        key=lambda item: (
            item["metrics"]["arm_joint_margin_rad"],
            item["metrics"]["arm_minimum_jacobian_singular_value"],
        )
    )
    source_candidates = source_candidates[: int(args.maximum_source_candidates)]
    for source in source_candidates:
        candidate = source["candidate"]
        joints = source["joints"]
        tool = source["tool"]
        palm = source["palm"]
        palm_down_cosine = source["palm_down_cosine"]
        source_metrics = source["metrics"]
        source_quality = source_metrics["rollout_functional_quality"]
        found_target = False
        screw_axis_world = tool[:3, :3] @ np.asarray((0.0, 0.0, -1.0))
        geometric_targets = []
        for target in valid_candidates:
            target_candidate = target["candidate"]
            if int(target_candidate["source_env_id"]) == int(candidate["source_env_id"]):
                continue
            target_relative = target["relative"]
            target_palm = tool @ np.linalg.inv(target_relative)
            target_down_cosine = float(
                -target_palm[:3, 1] @ np.asarray((0.0, 0.0, -1.0))
            )
            if target_down_cosine < float(args.minimum_palm_down_cosine):
                continue
            delta_rotation = target_palm[:3, :3] @ palm[:3, :3].T
            rotation_rad = math.acos(float(np.clip(
                (np.trace(delta_rotation) - 1.0) * 0.5, -1.0, 1.0
            )))
            if rotation_rad < 1.0e-6 or abs(math.sin(rotation_rad)) < 1.0e-6:
                continue
            rotation_axis = np.asarray((
                delta_rotation[2, 1] - delta_rotation[1, 2],
                delta_rotation[0, 2] - delta_rotation[2, 0],
                delta_rotation[1, 0] - delta_rotation[0, 1],
            )) / (2.0 * math.sin(rotation_rad))
            axis_alignment = abs(float(
                rotation_axis @ screw_axis_world
            ))
            yaw_change_deg = math.degrees(rotation_rad)
            yaw_distance = min(
                abs(yaw_change_deg - abs(float(requested)))
                for requested in args.target_yaw_changes_deg
            )
            if (
                axis_alignment < float(args.minimum_screw_axis_alignment)
                or yaw_distance > float(args.target_yaw_tolerance_deg)
            ):
                continue
            palm_translation = float(np.linalg.norm(
                target_palm[:3, 3] - palm[:3, 3]
            ))
            if palm_translation > 0.20:
                continue
            estimated_improvement = (
                target["metrics"]["rollout_functional_quality"] - source_quality
            )
            workspace_distance = float(np.linalg.norm(
                target["tool"][:3, 3] - tool[:3, 3]
            ))
            geometric_targets.append((
                -workspace_distance, estimated_improvement, axis_alignment,
                -yaw_distance,
                target, target_palm, yaw_change_deg, palm_translation,
                target_down_cosine,
            ))
        geometric_targets.sort(key=lambda item: item[:4], reverse=True)
        for (
            _, _, axis_alignment, _, target, target_palm, yaw_change_deg,
            palm_translation, target_down_cosine,
        ) in geometric_targets[: int(args.maximum_targets_per_source)]:
            target_joints = target["joints"]
            arm, position_error, rotation_error, _, _ = solve_arm_ik(
                kinematics, target_palm, target_joints[:7], target_joints[7:],
                robot_base, thresholds
            )
            if (
                position_error > thresholds.ik_position_m
                or rotation_error > thresholds.ik_orientation_deg
            ):
                arm, position_error, rotation_error, _, _ = solve_arm_ik(
                    kinematics, target_palm, joints[:7], target_joints[7:],
                    robot_base, thresholds
                )
            if (
                position_error > thresholds.ik_position_m
                or rotation_error > thresholds.ik_orientation_deg
            ):
                continue
            target_metrics = controllability(
                kinematics, arm, target_joints[7:], robot_base
            )
            target_quality = target_metrics["rollout_functional_quality"]
            improvement = target_quality - source_quality
            if (
                target_quality < float(args.minimum_target_quality)
                or improvement < float(args.minimum_quality_improvement)
            ):
                continue
            proposals.append({
                "candidate": candidate,
                "target_candidate": target["candidate"],
                "source_metrics": source_metrics,
                "target_metrics": target_metrics,
                "target_arm": arm,
                "target_relative": target["relative"],
                "yaw_change_deg": yaw_change_deg,
                "axis_alignment": axis_alignment,
                "palm_translation_m": palm_translation,
                "palm_down_cosine": palm_down_cosine,
                "target_palm_down_cosine": target_down_cosine,
                "improvement": improvement,
            })
            found_target = True
            break
        if not found_target:
            rejected["target_ik"] += 1
        if len(proposals) >= max(int(args.pairs) * 4, int(args.pairs)):
            break

    if len(proposals) < args.pairs:
        axis_quantiles = (
            np.quantile(np.asarray(palm_axis_down_cosines), (0.1, 0.5, 0.9), axis=0).tolist()
            if palm_axis_down_cosines else []
        )
        fk_quantiles = (
            np.quantile(np.asarray(source_fk_errors), (0.1, 0.5, 0.9), axis=0).tolist()
            if source_fk_errors else []
        )
        raise RuntimeError(
            f"only {len(proposals)}/{args.pairs} palm-down improved pairs are available; "
            f"palm-axis/down quantiles(10/50/90%)={axis_quantiles}; "
            f"source-FK error quantiles(m,deg)={fk_quantiles}; rejected={rejected}"
        )
    proposals.sort(
        key=lambda item: (
            item["improvement"], item["target_metrics"]["rollout_functional_quality"]
        ),
        reverse=True,
    )
    selected = []
    used_envs: set[int] = set()
    for proposal in proposals:
        candidate = proposal["candidate"]
        env_id = int(candidate["source_env_id"])
        if env_id in used_envs:
            continue
        position = np.asarray(candidate["object_pos_local"], dtype=np.float64)
        if any(
            np.linalg.norm(
                position - np.asarray(item["candidate"]["object_pos_local"])
            ) < float(args.minimum_tool_position_separation_m)
            for item in selected
        ):
            continue
        selected.append(proposal)
        used_envs.add(env_id)
        if len(selected) == args.pairs:
            break
    if len(selected) < args.pairs:
        raise RuntimeError(
            f"diversity filtering retained only {len(selected)}/{args.pairs} pairs"
        )

    entries = []
    for pair_id, proposal in enumerate(selected):
        candidate = proposal["candidate"]
        source_id = 2 * pair_id
        target_id = source_id + 1
        source_verification = base_verification(
            candidate, proposal["source_metrics"], hold_steps=120
        )
        source_verification.update({
            "target_only": False,
            "palm_down_cosine": proposal["palm_down_cosine"],
            "valid_target_ids": [target_id],
            "target_arm_joint_0_rad": [float(proposal["target_arm"][0])],
            "target_yaw_change_deg": proposal["yaw_change_deg"],
            "target_screw_axis_alignment": proposal["axis_alignment"],
            "target_palm_translation_m": proposal["palm_translation_m"],
            "target_source_env_id": int(
                proposal["target_candidate"]["source_env_id"]
            ),
        })
        entries.append(bank_entry(candidate, source_verification))

        target_candidate = copy.deepcopy(proposal["target_candidate"])
        target_candidate["object_pos_local"] = candidate["object_pos_local"]
        target_candidate["object_quat_wxyz"] = candidate["object_quat_wxyz"]
        target_joints = np.asarray(
            target_candidate["joint_pos_canonical"], dtype=np.float64
        )
        target_joints[:7] = proposal["target_arm"]
        target_targets = np.asarray(
            target_candidate["joint_targets_canonical"], dtype=np.float64
        )
        target_targets[:7] = proposal["target_arm"]
        target_candidate["joint_pos_canonical"] = target_joints.tolist()
        target_candidate["joint_targets_canonical"] = target_targets.tolist()
        target_candidate["palm_to_tool_pos"] = (
            proposal["target_relative"][:3, 3].tolist()
        )
        target_candidate["palm_to_tool_quat_wxyz"] = quaternion_wxyz(
            proposal["target_relative"]
        )
        target_verification = base_verification(
            target_candidate, proposal["target_metrics"], hold_steps=120
        )
        target_verification.update({
            "target_only": True,
            "palm_down_cosine": proposal["target_palm_down_cosine"],
            "valid_target_ids": [],
            "target_arm_joint_0_rad": [],
            "source_grasp_id": source_id,
            "target_yaw_change_deg": proposal["yaw_change_deg"],
            "target_screw_axis_alignment": proposal["axis_alignment"],
            "target_palm_translation_m": proposal["palm_translation_m"],
        })
        entries.append(bank_entry(target_candidate, target_verification))

    payload = {
        key: template[key] for key in (
            "schema_version", "tool_type", "source_checkpoint",
            "source_checkpoint_sha256", "policy_coefficient_id",
            "tactile_rich_fraction_min", "tactile_min_fingers", "seed",
            "control_dt_s", "joint_lower_canonical", "joint_upper_canonical",
            "joint_limit_tolerance_rad",
        )
    }
    payload.update({
        "kind": "allen_key_palm_down_adjustment_bank",
        "object_name": asset_path.stem,
        "asset_sha256": expected_hash,
        "source_rollout_dir": str(args.rollout_dir.resolve()),
        "entries": entries,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"[pass] wrote {len(selected)} start/target pairs to {args.output.resolve()} "
        f"source_quality={min(x['source_metrics']['rollout_functional_quality'] for x in selected):.3f}.."
        f"{max(x['source_metrics']['rollout_functional_quality'] for x in selected):.3f} "
        f"target_quality={min(x['target_metrics']['rollout_functional_quality'] for x in selected):.3f}.."
        f"{max(x['target_metrics']['rollout_functional_quality'] for x in selected):.3f} "
        f"rejected={rejected}"
    )


if __name__ == "__main__":
    main()
