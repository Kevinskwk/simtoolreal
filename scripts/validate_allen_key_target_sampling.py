#!/usr/bin/env python3
"""Validate Allen-key bank entries and non-trivial start/target pairs offline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grasp-bank", type=Path,
        default=ROOT / "assets/grasp_banks/allen_key_manipulation_v3.json",
    )
    parser.add_argument("--minimum-translation-m", type=float, default=0.012)
    parser.add_argument("--maximum-translation-m", type=float, default=0.090)
    parser.add_argument("--minimum-rotation-deg", type=float, default=18.0)
    parser.add_argument("--maximum-rotation-deg", type=float, default=100.0)
    parser.add_argument("--maximum-reset-yaw-deg", type=float, default=30.0)
    parser.add_argument("--position-tolerance-m", type=float, default=0.003)
    parser.add_argument("--rotation-tolerance-deg", type=float, default=3.0)
    return parser.parse_args()


def pose_from_wxyz(position, quaternion) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    pose = np.eye(4)
    pose[:3, 3] = np.asarray(position, dtype=np.float64)
    pose[:3, :3] = Rotation.from_quat(
        (quaternion[1], quaternion[2], quaternion[3], quaternion[0])
    ).as_matrix()
    return pose


def main() -> None:
    args = parse_args()
    if not args.grasp_bank.is_file():
        raise FileNotFoundError(args.grasp_bank)
    payload = json.loads(args.grasp_bank.read_text())
    entries = payload.get("entries", [])
    if payload.get("tool_type") != "allen_key" or len(entries) < 2:
        raise ValueError("target validation requires at least two Allen-key grasps")

    evaluator = load_module(
        "allen_target_grasp_evaluator",
        ROOT / "isaacsimenvs/tasks/simtoolreal/utils/grasp_evaluator.py",
    )
    robot_urdf = (
        ROOT / "assets/urdf/kuka_sharpa_description"
        / "iiwa14_left_sharpa_adjusted_restricted.urdf"
    )
    kinematics = evaluator.UrdfKinematics(robot_urdf)
    thresholds = evaluator.GraspEvaluatorThresholds(
        ik_position_m=float(args.position_tolerance_m),
        ik_orientation_deg=float(args.rotation_tolerance_deg),
        max_ik_iterations=300,
    )
    robot_base = np.eye(4)
    robot_base[1, 3] = 0.8
    reset_yaw = math.radians(float(args.maximum_reset_yaw_deg))
    centers: list[np.ndarray] = []
    rotations: list[Rotation] = []
    worst_position = 0.0
    worst_rotation = 0.0
    minimum_joint_margin = math.inf

    for index, entry in enumerate(entries):
        palm_to_tool = pose_from_wxyz(
            entry["palm_to_tool_pos"], entry["palm_to_tool_quat_wxyz"]
        )
        centers.append(np.linalg.inv(palm_to_tool)[:3, 3])
        rotations.append(Rotation.from_matrix(palm_to_tool[:3, :3]))
        tool_pose = pose_from_wxyz(entry["object_pos_local"], entry["object_quat_wxyz"])
        target_palm = tool_pose @ np.linalg.inv(palm_to_tool)
        original = np.asarray(entry["joint_pos_canonical"], dtype=np.float64)
        arm, position_error, rotation_error, _, _ = evaluator.solve_arm_ik(
            kinematics, target_palm, original[:7], original[7:], robot_base, thresholds
        )
        rotated_arm = arm.copy()
        rotated_arm[0] += reset_yaw
        joint_margin = np.minimum(
            rotated_arm - kinematics.arm_lower,
            kinematics.arm_upper - rotated_arm,
        ).min()
        worst_position = max(worst_position, float(position_error))
        worst_rotation = max(worst_rotation, float(rotation_error))
        minimum_joint_margin = min(minimum_joint_margin, float(joint_margin))
        if (
            position_error > thresholds.ik_position_m
            or rotation_error > thresholds.ik_orientation_deg
            or joint_margin < 0.0
        ):
            raise RuntimeError(
                f"bank entry {index} is not reachable across reset yaw: "
                f"position={position_error:.6f}m rotation={rotation_error:.3f}deg "
                f"joint_margin={joint_margin:.6f}rad"
            )

    adjacency = np.zeros((len(entries), len(entries)), dtype=bool)
    translations: list[float] = []
    angles: list[float] = []
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            translation = float(np.linalg.norm(centers[i] - centers[j]))
            angle = math.degrees(float((rotations[i].inv() * rotations[j]).magnitude()))
            valid = (
                translation <= args.maximum_translation_m
                and angle <= args.maximum_rotation_deg
                and (
                    translation >= args.minimum_translation_m
                    or angle >= args.minimum_rotation_deg
                )
            )
            if valid:
                adjacency[i, j] = adjacency[j, i] = True
                translations.append(translation)
                angles.append(angle)
    isolated = np.flatnonzero(adjacency.sum(axis=1) == 0)
    if isolated.size:
        raise RuntimeError(
            f"Allen-key bank has targetless start entries: {isolated.tolist()}"
        )
    directed_pair_count = 0
    for source_id, entry in enumerate(entries):
        verification = entry.get("verification", {})
        target_ids = verification.get("valid_target_ids")
        target_joint_0 = verification.get("target_arm_joint_0_rad")
        if target_ids is None or target_joint_0 is None:
            raise RuntimeError(
                f"bank entry {source_id} is missing its physical target graph"
            )
        if verification.get("target_only", False):
            if target_ids or target_joint_0:
                raise RuntimeError(
                    f"target-only bank entry {source_id} must be a graph leaf"
                )
            continue
        if len(target_ids) != len(target_joint_0) or not target_ids:
            raise RuntimeError(
                f"bank entry {source_id} has an empty or malformed target graph"
            )
        for target_id, joint_0 in zip(target_ids, target_joint_0, strict=True):
            target_id = int(target_id)
            if (
                not 0 <= target_id < len(entries)
                or not adjacency[source_id, target_id]
                or not math.isfinite(float(joint_0))
            ):
                raise RuntimeError(
                    f"bank entry {source_id} declares invalid target {target_id}"
                )
            directed_pair_count += 1
    print(
        f"[pass] {len(entries)} reachable grasps, {directed_pair_count} screened "
        f"directed targets ({len(translations)} geometric pairs); "
        f"translation={min(translations):.4f}..{max(translations):.4f}m "
        f"rotation={min(angles):.1f}..{max(angles):.1f}deg "
        f"worst_ik={worst_position:.6f}m/{worst_rotation:.2f}deg "
        f"joint_margin={minimum_joint_margin:.3f}rad"
    )


if __name__ == "__main__":
    main()
