#!/usr/bin/env python3
"""Fail unless every Allen-key target in the training range is arm-reachable."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
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
        default=ROOT / "assets/grasp_banks/allen_key_canonical_v1.json",
    )
    parser.add_argument("--minimum-angle-deg", type=float, default=15.0)
    parser.add_argument("--maximum-angle-deg", type=float, default=60.0)
    parser.add_argument("--angle-step-deg", type=float, default=0.5)
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
    if not 0.0 < args.minimum_angle_deg <= args.maximum_angle_deg:
        raise ValueError("target angle interval is invalid")
    if args.angle_step_deg <= 0.0 or args.maximum_reset_yaw_deg < 0.0:
        raise ValueError("angle step and reset yaw must be non-negative")

    adjustment = load_module(
        "allen_target_adjustment_utils",
        ROOT / "isaacsimenvs/tasks/simtoolreal/utils/adjustment_utils.py",
    )
    evaluator = load_module(
        "allen_target_grasp_evaluator",
        ROOT / "isaacsimenvs/tasks/simtoolreal/utils/grasp_evaluator.py",
    )
    payload = json.loads(args.grasp_bank.read_text())
    entries = payload.get("entries", [])
    if payload.get("tool_type") != "allen_key" or not entries:
        raise ValueError("target validation requires a non-empty Allen-key grasp bank")

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
    angles = np.arange(
        args.minimum_angle_deg,
        args.maximum_angle_deg + 0.5 * args.angle_step_deg,
        args.angle_step_deg,
    )
    failures: list[str] = []
    worst_position = 0.0
    worst_rotation = 0.0
    minimum_joint_margin = math.inf
    reset_yaw = math.radians(float(args.maximum_reset_yaw_deg))

    for entry_index, entry in enumerate(entries):
        palm_to_tool_pos = torch.tensor(
            [entry["palm_to_tool_pos"]], dtype=torch.float64
        )
        palm_to_tool_quat = torch.tensor(
            [entry["palm_to_tool_quat_wxyz"]], dtype=torch.float64
        )
        tool_pose = pose_from_wxyz(
            entry["object_pos_local"], entry["object_quat_wxyz"]
        )
        original = np.asarray(entry["joint_pos_canonical"], dtype=np.float64)
        for angle_deg in angles:
            target_pos, target_quat = adjustment.orbit_palm_tool_about_screw_axis(
                palm_to_tool_pos,
                palm_to_tool_quat,
                torch.tensor([math.radians(float(angle_deg))], dtype=torch.float64),
                torch.tensor((0.0, 0.0, -1.0), dtype=torch.float64),
                torch.tensor((0.192, 0.0, -0.03), dtype=torch.float64),
            )
            palm_to_tool = pose_from_wxyz(
                target_pos[0].numpy(), target_quat[0].numpy()
            )
            target_palm = tool_pose @ np.linalg.inv(palm_to_tool)
            arm, position_error, rotation_error, _, _ = evaluator.solve_arm_ik(
                kinematics, target_palm, original[:7], original[7:],
                robot_base, thresholds,
            )
            rotated_arm = arm.copy()
            rotated_arm[0] += reset_yaw
            joint_margin = np.minimum(
                rotated_arm - kinematics.arm_lower,
                kinematics.arm_upper - rotated_arm,
            ).min()
            worst_position = max(worst_position, position_error)
            worst_rotation = max(worst_rotation, rotation_error)
            minimum_joint_margin = min(minimum_joint_margin, float(joint_margin))
            if (
                position_error > thresholds.ik_position_m
                or rotation_error > thresholds.ik_orientation_deg
                or joint_margin < 0.0
            ):
                failures.append(
                    f"entry={entry_index} angle={angle_deg:.2f}deg "
                    f"position={position_error:.6f}m rotation={rotation_error:.3f}deg "
                    f"joint_margin={joint_margin:.6f}rad"
                )

    if failures:
        preview = "\n".join(failures[:12])
        raise RuntimeError(
            f"{len(failures)} Allen-key palm targets are invalid:\n{preview}"
        )
    print(
        f"[pass] {len(entries) * len(angles)} Allen-key palm targets are reachable; "
        f"worst_position={worst_position:.6f}m "
        f"worst_rotation={worst_rotation:.3f}deg "
        f"minimum_joint_margin={minimum_joint_margin:.4f}rad"
    )


if __name__ == "__main__":
    main()
