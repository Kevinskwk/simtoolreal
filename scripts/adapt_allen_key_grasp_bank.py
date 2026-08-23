#!/usr/bin/env python3
"""Build a physically validated Allen-key manipulation-grasp bank."""

from __future__ import annotations

import argparse
import copy
import json
import math
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-bank", type=Path,
        default=ROOT / "assets/grasp_banks/allen_key_manipulation_v2.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "assets/grasp_banks/allen_key_manipulation_v3.json",
    )
    parser.add_argument(
        "--object-urdf", type=Path,
        default=ROOT / "assets/urdf/objects/allen_key_canonical.urdf",
    )
    parser.add_argument("--source-assets", type=int, default=256)
    parser.add_argument("--max-source-entries", type=int, default=24)
    parser.add_argument(
        "--direct-source-entries", action="store_true",
        help=(
            "Replay source entries at their recorded poses instead of generating "
            "synthetic palm/tool transforms. Intended for rollout snapshot banks."
        ),
    )
    parser.add_argument(
        "--source-entry-indices", type=int, nargs="+", default=(0, 3),
        help="Optional explicit source grasps; otherwise sample diverse robust entries.",
    )
    parser.add_argument(
        "--palm-axial-shifts-m", type=float, nargs="+",
        default=(0.0, -0.06),
        help="Palm-to-tool shifts along the shaft; negative values move the palm toward the bend.",
    )
    parser.add_argument(
        "--palm-shift-radii-m", type=float, nargs="+",
        default=(0.0,),
        help="Tool-frame radial shifts searched around the long handle axis.",
    )
    parser.add_argument(
        "--tool-yaw-angles-deg", type=float, nargs="+",
        default=(-100.0, -50.0, 0.0),
        help="Engaged Allen-key world orientations represented in the bank.",
    )
    parser.add_argument(
        "--palm-shift-angles-deg", type=float, nargs="+",
        default=(210.0, 225.0, 240.0, 255.0, 270.0, 285.0, 300.0, 315.0, 330.0),
        help="Tool-frame radial shift angles searched around the handle.",
    )
    parser.add_argument(
        "--palm-rotation-angles-deg", type=float, nargs="+",
        default=(0.0, 30.0),
        help="Physically validate palm orientations about the engaged screw axis.",
    )
    parser.add_argument(
        "--palm-handle-roll-angles-deg", type=float, nargs="+",
        default=(180.0, 200.0, 220.0),
        help="Orbit the hand around the Allen key's long handle axis.",
    )
    parser.add_argument(
        "--palm-down-opposite-side", action="store_true",
        help=(
            "Generate palm-down grasps whose finger approach directions orbit "
            "around the engaged world/tool Z axis."
        ),
    )
    parser.add_argument(
        "--palm-side-angles-deg", type=float, nargs="+",
        default=(-105.0, -90.0, -75.0, 75.0, 90.0, 105.0),
        help="Finger approach angles relative to the horizontal handle direction.",
    )
    parser.add_argument(
        "--palm-normal-offsets-m", type=float, nargs="+", default=(0.045, 0.055),
        help="Vertical handle offset from the palm frame for palm-down candidates.",
    )
    parser.add_argument(
        "--palm-reach-offsets-m", type=float, nargs="+", default=(0.065, 0.080),
        help="Handle reach along the palm-to-finger direction.",
    )
    parser.add_argument(
        "--palm-lateral-offsets-m", type=float, nargs="+", default=(0.0,),
        help="Handle offsets across the palm in the palm-local Y direction.",
    )
    parser.add_argument(
        "--handle-grasp-x-m", type=float, nargs="+", default=(0.02, 0.07),
        help="Candidate grasp locations along the Allen key's long local-X handle.",
    )
    parser.add_argument(
        "--tightening-levels", type=float, nargs="+",
        default=TIGHTENING_LEVELS if "TIGHTENING_LEVELS" in globals() else (
            0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90
        ),
        help="Finger residual fractions; negative values open the source grasp.",
    )
    parser.add_argument("--pair-min-translation-m", type=float, default=0.012)
    parser.add_argument("--pair-max-translation-m", type=float, default=0.090)
    parser.add_argument("--pair-min-rotation-deg", type=float, default=18.0)
    parser.add_argument("--pair-max-rotation-deg", type=float, default=100.0)
    parser.add_argument("--settle-steps", type=int, default=60)
    parser.add_argument("--hold-steps", type=int, default=120)
    parser.add_argument("--max-fingertip-force-n", type=float, default=12.0)
    parser.add_argument("--max-palm-force-n", type=float, default=20.0)
    parser.add_argument("--minimum-actual-contact-ratio", type=float, default=0.80)
    parser.add_argument(
        "--max-hold-drift-m", type=float, default=0.001,
        help="Maximum palm-tool drift admitted before bank diversity selection.",
    )
    parser.add_argument("--desired-entries", type=int, default=10)
    parser.add_argument(
        "--tool-position", type=float, nargs=3, default=(0.0, 0.08, 0.45),
        metavar=("X", "Y", "Z"),
        help="Engaged Allen-key position in the environment-local frame.",
    )
    parser.add_argument(
        "--tool-position-offsets", type=float, nargs="+",
        default=(
            -0.10, -0.04, -0.03,
            -0.05, 0.04, 0.00,
            0.00, 0.00, 0.03,
            0.05, -0.02, 0.00,
            0.10, 0.04, -0.02,
        ),
        help="Flat sequence of XYZ offsets added to --tool-position.",
    )
    parser.add_argument("--wrist-above-margin-m", type=float, default=0.03)
    parser.add_argument("--wrist-above-fraction", type=float, default=1.0)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    return parser.parse_args()


ARGS = parse_args()
APP = AppLauncher(ARGS).app


import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import isaacsimenvs  # noqa: E402,F401
from isaacsimenvs.tasks.simtoolreal.simtoolreal_tacmap_env_cfg import (  # noqa: E402
    SimToolRealAllenKeyAdjustmentEnvCfg,
)
from isaacsimenvs.tasks.simtoolreal.utils.grasp_evaluator import (  # noqa: E402
    GraspEvaluatorThresholds,
    UrdfKinematics,
    pose_matrix,
    solve_arm_ik,
)
from isaacsimenvs.tasks.simtoolreal.utils.inhand_grasp_bank import (  # noqa: E402
    sha256_file,
    validate_grasp_bank,
)
from isaacsimenvs.tasks.simtoolreal.utils.scene_utils import (  # noqa: E402
    JOINT_NAMES_CANONICAL,
)


TIGHTENING_LEVELS = (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90)
FLEXION_INDICES = tuple(
    index for index, name in enumerate(JOINT_NAMES_CANONICAL)
    if index >= 7 and (
        name.endswith("_FE") or name.endswith("_PIP") or name.endswith("_DIP")
        or name.endswith("_IP") or name == "left_pinky_CMC"
    )
)


def quaternion_wxyz(matrix: np.ndarray) -> list[float]:
    xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return [float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2])]


def arm_controllability(
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
        np.clip(margin / 0.20, 0.0, 1.0),
        np.clip(minimum_sigma / 0.10, 0.0, 1.0),
        np.clip(20.0 / max(condition, 1.0), 0.0, 1.0),
    )
    score = float(np.prod(factors) ** (1.0 / len(factors)))
    return {
        "arm_joint_margin_rad": margin,
        "arm_minimum_jacobian_singular_value": minimum_sigma,
        "arm_jacobian_condition_number": condition,
        "rollout_functional_quality": score,
    }


def palm_down_entries(
    source: dict,
    source_entry_indices: list[int] | None,
    max_source_entries: int,
    desired_tool_positions: tuple[tuple[float, float, float], ...],
    tool_yaw_angles_deg: tuple[float, ...],
    palm_side_angles_deg: tuple[float, ...],
    normal_offsets_m: tuple[float, ...],
    reach_offsets_m: tuple[float, ...],
    lateral_offsets_m: tuple[float, ...],
    handle_grasp_x_m: tuple[float, ...],
) -> list[tuple[dict, dict]]:
    """Generate palm-down, opposite-side grasps around a horizontal handle."""
    robot = (
        ROOT / "assets/urdf/kuka_sharpa_description"
        / "iiwa14_left_sharpa_adjusted_restricted.urdf"
    )
    kinematics = UrdfKinematics(robot)
    thresholds = GraspEvaluatorThresholds(
        ik_position_m=0.003, ik_orientation_deg=3.0, max_ik_iterations=400
    )
    entries = source.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("palm-down generation requires a single-asset source bank")
    indexed = list(enumerate(entries))
    if source_entry_indices is not None:
        requested = set(int(value) for value in source_entry_indices)
        indexed = [item for item in indexed if item[0] in requested]
        missing = requested - {item[0] for item in indexed}
        if missing:
            raise ValueError(f"source entry indices are out of range: {sorted(missing)}")
    indexed = indexed[: int(max_source_entries)]
    source_lower = np.asarray(source["joint_lower_canonical"], dtype=np.float64)[:7]
    source_upper = np.asarray(source["joint_upper_canonical"], dtype=np.float64)[:7]
    robot_base = np.eye(4)
    robot_base[1, 3] = 0.8
    candidates: list[tuple[dict, dict]] = []
    for source_index, source_entry in indexed:
        original = np.asarray(source_entry["joint_pos_canonical"], dtype=np.float64)
        for tool_position in desired_tool_positions:
            for tool_yaw_deg in tool_yaw_angles_deg:
                tool_yaw = math.radians(float(tool_yaw_deg))
                desired_tool = np.eye(4)
                desired_tool[:3, :3] = Rotation.from_euler(
                    "z", tool_yaw_deg, degrees=True
                ).as_matrix()
                desired_tool[:3, 3] = np.asarray(tool_position, dtype=np.float64)
                for side_angle_deg in palm_side_angles_deg:
                    approach = tool_yaw + math.radians(float(side_angle_deg))
                    # Palm local +X is its palmar normal and points down. Local
                    # +Z points from wrist toward the fingers in the XY plane.
                    palm_rotation = np.column_stack((
                        np.asarray((0.0, 0.0, -1.0)),
                        np.asarray((-math.sin(approach), math.cos(approach), 0.0)),
                        np.asarray((math.cos(approach), math.sin(approach), 0.0)),
                    ))
                    for normal_offset in normal_offsets_m:
                        for reach_offset in reach_offsets_m:
                            for lateral_offset in lateral_offsets_m:
                                tool_in_palm = np.asarray((
                                    float(normal_offset), float(lateral_offset),
                                    float(reach_offset),
                                ))
                                for grasp_x in handle_grasp_x_m:
                                    grasp_world = desired_tool @ np.asarray((
                                        float(grasp_x), 0.0, 0.0, 1.0
                                    ))
                                    target_palm = np.eye(4)
                                    target_palm[:3, :3] = palm_rotation
                                    target_palm[:3, 3] = (
                                        grasp_world[:3] - palm_rotation @ tool_in_palm
                                    )
                                    arm, pos_error, rot_error, _, _ = solve_arm_ik(
                                        kinematics, target_palm, original[:7], original[7:],
                                        robot_base, thresholds,
                                    )
                                    if (
                                        pos_error > thresholds.ik_position_m
                                        or rot_error > thresholds.ik_orientation_deg
                                    ):
                                        continue
                                    arm = np.clip(arm, source_lower, source_upper)
                                    palm_to_tool = np.linalg.inv(target_palm) @ desired_tool
                                    entry = copy.deepcopy(source_entry)
                                    joints = np.concatenate((arm, original[7:]))
                                    targets = np.asarray(
                                        source_entry["joint_targets_canonical"], dtype=np.float64
                                    ).copy()
                                    targets[:7] = arm
                                    entry.update({
                                    "joint_pos_canonical": joints.tolist(),
                                    "joint_vel_canonical": [0.0] * 29,
                                    "joint_targets_canonical": targets.tolist(),
                                    "object_pos_local": desired_tool[:3, 3].tolist(),
                                    "object_quat_wxyz": quaternion_wxyz(desired_tool),
                                    "object_velocity": [0.0] * 6,
                                    "palm_to_tool_pos": palm_to_tool[:3, 3].tolist(),
                                    "palm_to_tool_quat_wxyz": quaternion_wxyz(palm_to_tool),
                                    "reference_contact_quat_wxyz": quaternion_wxyz(desired_tool),
                                    "reference_edge_yaw_rad": 0.0,
                                    "reference_edge_tilt_rad": math.radians(45.0),
                                    })
                                    verification = dict(entry["verification"])
                                    verification.update({
                                    "edge_clearance_m": 0.04,
                                    "table_force_n": 0.0,
                                    "pickup_orientation_error_deg": 0.0,
                                    "tactile_finger_count_min": 0,
                                    "tactile_contact_area_mean": 0.0,
                                    "tactile_depth_mean": 0.0,
                                    "tactile_depth_max": 0.0,
                                    })
                                    entry["verification"] = verification
                                    metrics = arm_controllability(
                                        kinematics, arm, original[7:], robot_base
                                    )
                                    candidates.append((entry, {
                                    "source_asset_index": 0,
                                    "source_entry_index": int(source_index),
                                    "palm_shift_tool_m": [0.0, 0.0, 0.0],
                                    "palm_rotation_about_screw_deg": float(side_angle_deg),
                                    "palm_handle_roll_deg": float(side_angle_deg),
                                    "palm_side_angle_deg": float(side_angle_deg),
                                    "palm_normal_offset_m": float(normal_offset),
                                    "palm_reach_offset_m": float(reach_offset),
                                    "palm_lateral_offset_m": float(lateral_offset),
                                    "handle_grasp_x_m": float(grasp_x),
                                    "tool_yaw_deg": float(tool_yaw_deg),
                                    "tool_position_local": list(tool_position),
                                    "wrist_above_tool_m": float(
                                        target_palm[2, 3] - desired_tool[2, 3]
                                    ),
                                    "ik_position_error_m": float(pos_error),
                                    "ik_rotation_error_deg": float(rot_error),
                                    "rollout_workspace_tier": "easy",
                                    "rollout_workspace_tier_id": 0,
                                    **metrics,
                                    }))
    if not candidates:
        raise RuntimeError("no palm-down Allen-key candidate is arm-reachable")
    return candidates


def provisional_entries(
    source: dict, count_assets: int, source_entry_indices: list[int] | None,
    palm_axial_shifts_m: tuple[float, ...],
    palm_shift_radii_m: tuple[float, ...], palm_shift_angles_deg: tuple[float, ...],
    palm_rotation_angles_deg: tuple[float, ...], tool_yaw_angles_deg: tuple[float, ...],
    palm_handle_roll_angles_deg: tuple[float, ...], max_source_entries: int,
    desired_tool_positions: tuple[tuple[float, float, float], ...],
) -> list[tuple[dict, dict]]:
    robot = ROOT / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
    kinematics = UrdfKinematics(robot)
    thresholds = GraspEvaluatorThresholds(
        ik_position_m=0.003, ik_orientation_deg=3.0, max_ik_iterations=300
    )
    tool_positions = np.asarray(desired_tool_positions, dtype=np.float64)
    if tool_positions.ndim != 2 or tool_positions.shape[1] != 3:
        raise ValueError("tool positions must have shape (N, 3)")
    if not np.isfinite(tool_positions).all():
        raise ValueError("tool positions must be finite")
    robot_base = np.eye(4)
    robot_base[1, 3] = 0.8
    if "assets" in source:
        assets = sorted(
            source["assets"],
            key=lambda asset: abs(float(asset["object_scale"][1]) * 0.04 - 0.02),
        )[:count_assets]
        assets = [
            {**asset, "indexed_entries": list(enumerate(asset["entries"]))}
            for asset in assets
        ]
    else:
        indexed = list(enumerate(source["entries"]))
        if source_entry_indices is not None:
            requested = set(source_entry_indices)
            indexed = [item for item in indexed if item[0] in requested]
            missing = requested - {item[0] for item in indexed}
            if missing:
                raise ValueError(f"source entry indices are out of range: {sorted(missing)}")
        else:
            indexed = indexed[:count_assets]
            if len(indexed) > max_source_entries:
                selected = np.linspace(
                    0, len(indexed) - 1, max_source_entries, dtype=np.int64
                )
                indexed = [indexed[int(index)] for index in selected]
        assets = [{"asset_index": 0, "indexed_entries": indexed}]
    shifts_tool = []
    for axial in palm_axial_shifts_m:
        if not math.isfinite(float(axial)):
            raise ValueError("palm axial shifts must be finite")
        shifts_tool.append(np.asarray((float(axial), 0.0, 0.0)))
    for axial in palm_axial_shifts_m:
        for radius in palm_shift_radii_m:
            if not math.isfinite(float(radius)) or float(radius) < 0.0:
                raise ValueError("palm shift radii must be finite and non-negative")
            if float(radius) == 0.0:
                continue
            for angle_deg in palm_shift_angles_deg:
                if not math.isfinite(float(angle_deg)):
                    raise ValueError("palm shift angles must be finite")
                angle = math.radians(float(angle_deg))
                shifts_tool.append(np.asarray((
                    float(axial), float(radius) * math.cos(angle),
                    float(radius) * math.sin(angle),
                )))
    candidates: list[tuple[dict, dict]] = []
    for asset in assets:
        for source_index, source_entry in asset["indexed_entries"]:
            source_palm_to_tool = pose_matrix(
                np.asarray(source_entry["palm_to_tool_pos"]),
                np.asarray(source_entry["palm_to_tool_quat_wxyz"]),
            )
            for shift_tool in shifts_tool:
                shifted = source_palm_to_tool.copy()
                shifted[:3, 3] += shifted[:3, :3] @ shift_tool
                base_tool_to_palm = np.linalg.inv(shifted)
                for handle_roll_deg in palm_handle_roll_angles_deg:
                    if not math.isfinite(float(handle_roll_deg)):
                        raise ValueError("palm handle roll angles must be finite")
                    handle_rotation = Rotation.from_rotvec(
                        np.asarray((1.0, 0.0, 0.0))
                        * math.radians(float(handle_roll_deg))
                    ).as_matrix()
                    rolled_tool_to_palm = base_tool_to_palm.copy()
                    rolled_tool_to_palm[:3, :3] = (
                        handle_rotation @ base_tool_to_palm[:3, :3]
                    )
                    rolled_tool_to_palm[:3, 3] = (
                        handle_rotation @ base_tool_to_palm[:3, 3]
                    )
                    for palm_rotation_deg in palm_rotation_angles_deg:
                        tool_to_palm = rolled_tool_to_palm.copy()
                        screw_rotation = Rotation.from_rotvec(
                            np.asarray((0.0, 0.0, -1.0))
                            * math.radians(float(palm_rotation_deg))
                        ).as_matrix()
                        tool_to_palm[:3, :3] = (
                            screw_rotation @ tool_to_palm[:3, :3]
                        )
                        palm_to_tool = np.linalg.inv(tool_to_palm)
                        for desired_tool_position in tool_positions:
                            for tool_yaw_deg in tool_yaw_angles_deg:
                                desired_tool = np.eye(4)
                                desired_tool[:3, :3] = Rotation.from_euler(
                                    "z", float(tool_yaw_deg), degrees=True
                                ).as_matrix()
                                desired_tool[:3, 3] = desired_tool_position
                                target_palm = desired_tool @ tool_to_palm
                                original = np.asarray(
                                    source_entry["joint_pos_canonical"], dtype=np.float64
                                )
                                arm, pos_error, rot_error, _, _ = solve_arm_ik(
                                    kinematics, target_palm, original[:7], original[7:],
                                    robot_base, thresholds,
                                )
                                if (
                                    pos_error > thresholds.ik_position_m
                                    or rot_error > thresholds.ik_orientation_deg
                                ):
                                    continue
                                entry = copy.deepcopy(source_entry)
                                joints = np.concatenate((arm, original[7:]))
                                targets = np.asarray(
                                    source_entry["joint_targets_canonical"], dtype=np.float64
                                ).copy()
                                targets[:7] = arm
                                entry.update({
                                    "joint_pos_canonical": joints.tolist(),
                                    "joint_vel_canonical": [0.0] * 29,
                                    "joint_targets_canonical": targets.tolist(),
                                    "object_pos_local": desired_tool[:3, 3].tolist(),
                                    "object_quat_wxyz": quaternion_wxyz(desired_tool),
                                    "object_velocity": [0.0] * 6,
                                    "palm_to_tool_pos": palm_to_tool[:3, 3].tolist(),
                                    "palm_to_tool_quat_wxyz": quaternion_wxyz(palm_to_tool),
                                    "reference_contact_quat_wxyz": quaternion_wxyz(desired_tool),
                                    "reference_edge_yaw_rad": 0.0,
                                    "reference_edge_tilt_rad": math.radians(45.0),
                                })
                                verification = dict(entry["verification"])
                                verification.update({
                                    "edge_clearance_m": 0.04,
                                    "table_force_n": 0.0,
                                    "pickup_orientation_error_deg": 0.0,
                                    "tactile_finger_count_min": 0,
                                    "tactile_contact_area_mean": 0.0,
                                    "tactile_depth_mean": 0.0,
                                    "tactile_depth_max": 0.0,
                                })
                                entry["verification"] = verification
                                candidates.append((entry, {
                                    "source_asset_index": int(asset["asset_index"]),
                                    "source_entry_index": source_index,
                                    "palm_shift_tool_m": shift_tool.tolist(),
                                    "palm_rotation_about_screw_deg": float(palm_rotation_deg),
                                    "palm_handle_roll_deg": float(handle_roll_deg),
                                    "tool_yaw_deg": float(tool_yaw_deg),
                                    "tool_position_local": desired_tool_position.tolist(),
                                    "wrist_above_tool_m": float(
                                        target_palm[2, 3] - desired_tool_position[2]
                                    ),
                                    "ik_position_error_m": pos_error,
                                    "ik_rotation_error_deg": rot_error,
                                }))
    if not candidates:
        raise RuntimeError("no grasp candidate has a reachable engaged Allen-key pose")
    return candidates


def direct_provisional_entries(
    source: dict, max_source_entries: int,
) -> list[tuple[dict, dict]]:
    """Preserve live rollout snapshots for strict simulator replay validation."""
    entries = source.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("direct source bank has no entries")
    if max_source_entries <= 0:
        raise ValueError("--max-source-entries must be positive")
    entries = entries[:max_source_entries]
    candidates: list[tuple[dict, dict]] = []
    for source_index, source_entry in enumerate(entries):
        verification = source_entry.get("verification", {})
        if not verification.get("provisional_rollout_snapshot", False):
            raise ValueError(
                "--direct-source-entries requires provisional rollout snapshots"
            )
        palm_to_tool = pose_matrix(
            np.asarray(source_entry["palm_to_tool_pos"], dtype=np.float64),
            np.asarray(source_entry["palm_to_tool_quat_wxyz"], dtype=np.float64),
        )
        tool_to_palm = np.linalg.inv(palm_to_tool)
        tool_pose = pose_matrix(
            np.asarray(source_entry["object_pos_local"], dtype=np.float64),
            np.asarray(source_entry["object_quat_wxyz"], dtype=np.float64),
        )
        target_palm = tool_pose @ tool_to_palm
        handle_roll = math.degrees(math.atan2(
            float(tool_to_palm[2, 3]), float(tool_to_palm[1, 3])
        ))
        # Quantization creates meaningful approach-side groups without treating
        # small live-policy variations as distinct grasp families.
        handle_roll = 45.0 * round(handle_roll / 45.0)
        tool_yaw = math.degrees(math.atan2(
            float(tool_pose[1, 0]), float(tool_pose[0, 0])
        ))
        meta = {
            "source_asset_index": 0,
            "source_entry_index": source_index,
            "palm_shift_tool_m": [0.0, 0.0, 0.0],
            "palm_rotation_about_screw_deg": 0.0,
            "palm_handle_roll_deg": handle_roll,
            "tool_yaw_deg": tool_yaw,
            "tool_position_local": tool_pose[:3, 3].tolist(),
            "wrist_above_tool_m": float(target_palm[2, 3] - tool_pose[2, 3]),
            "ik_position_error_m": 0.0,
            "ik_rotation_error_deg": 0.0,
            "rollout_workspace_tier": verification["workspace_tier"],
            "rollout_workspace_tier_id": int(verification["workspace_tier_id"]),
            "rollout_functional_quality": float(
                verification["rollout_functional_quality"]
            ),
            "rollout_all_turns_success": bool(
                verification["rollout_all_turns_success"]
            ),
            "rollout_source_env_id": int(verification["source_env_id"]),
            "rollout_source_turn_stage": int(verification["source_turn_stage"]),
        }
        candidates.append((copy.deepcopy(source_entry), meta))
    return candidates


def bank_payload(source: dict, entries: list[dict], object_urdf: Path) -> dict:
    checkpoint = ROOT / "pretrained_policy/model.pth"
    return {
        "schema_version": 2,
        "tool_type": "allen_key",
        "object_name": object_urdf.stem,
        "asset_sha256": sha256_file(object_urdf),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "policy_coefficient_id": 0.0,
        "tactile_rich_fraction_min": 0.0,
        "tactile_min_fingers": 1,
        "seed": 42,
        "control_dt_s": 1.0 / 60.0,
        "joint_lower_canonical": source["joint_lower_canonical"],
        "joint_upper_canonical": source["joint_upper_canonical"],
        "joint_limit_tolerance_rad": source["joint_limit_tolerance_rad"],
        "entries": entries,
    }


def main() -> None:
    if not ARGS.source_bank.is_file():
        raise FileNotFoundError(f"source grasp bank does not exist: {ARGS.source_bank}")
    source = json.loads(ARGS.source_bank.read_text())
    if not ARGS.object_urdf.is_file():
        raise FileNotFoundError(f"Allen-key object URDF does not exist: {ARGS.object_urdf}")
    if not 0.0 <= float(ARGS.minimum_actual_contact_ratio) <= 1.0:
        raise ValueError("--minimum-actual-contact-ratio must be in [0, 1]")
    position_offsets = tuple(float(value) for value in ARGS.tool_position_offsets)
    if len(position_offsets) == 0 or len(position_offsets) % 3 != 0:
        raise ValueError("--tool-position-offsets must contain one or more XYZ triples")
    base_tool_position = np.asarray(ARGS.tool_position, dtype=np.float64)
    desired_tool_positions = tuple(
        tuple((base_tool_position + np.asarray(position_offsets[index:index + 3])).tolist())
        for index in range(0, len(position_offsets), 3)
    )
    wrist_above_fraction = float(ARGS.wrist_above_fraction)
    if not 0.0 <= wrist_above_fraction <= 1.0:
        raise ValueError("--wrist-above-fraction must be in [0, 1]")
    if ARGS.direct_source_entries:
        candidates = direct_provisional_entries(source, int(ARGS.max_source_entries))
    elif ARGS.palm_down_opposite_side:
        candidates = palm_down_entries(
            source,
            ARGS.source_entry_indices,
            int(ARGS.max_source_entries),
            desired_tool_positions,
            tuple(float(value) for value in ARGS.tool_yaw_angles_deg),
            tuple(float(value) for value in ARGS.palm_side_angles_deg),
            tuple(float(value) for value in ARGS.palm_normal_offsets_m),
            tuple(float(value) for value in ARGS.palm_reach_offsets_m),
            tuple(float(value) for value in ARGS.palm_lateral_offsets_m),
            tuple(float(value) for value in ARGS.handle_grasp_x_m),
        )
    else:
        candidates = provisional_entries(
            source,
            int(ARGS.source_assets),
            ARGS.source_entry_indices,
            tuple(float(value) for value in ARGS.palm_axial_shifts_m),
            tuple(float(value) for value in ARGS.palm_shift_radii_m),
            tuple(float(value) for value in ARGS.palm_shift_angles_deg),
            tuple(float(value) for value in ARGS.palm_rotation_angles_deg),
            tuple(float(value) for value in ARGS.tool_yaw_angles_deg),
            tuple(float(value) for value in ARGS.palm_handle_roll_angles_deg),
            int(ARGS.max_source_entries),
            desired_tool_positions,
        )
    expanded_entries: list[dict] = []
    metadata: list[dict] = []
    levels: list[float] = []
    lower = torch.tensor(source["joint_lower_canonical"])
    upper = torch.tensor(source["joint_upper_canonical"])
    tightening_levels = tuple(float(value) for value in ARGS.tightening_levels)
    if not tightening_levels or any(
        not math.isfinite(value) or not -1.0 <= value <= 1.0
        for value in tightening_levels
    ):
        raise ValueError("--tightening-levels must be finite and in [-1, 1]")
    for entry, meta in candidates:
        base = torch.tensor(entry["joint_targets_canonical"])
        for level in tightening_levels:
            tightened = base.clone()
            ids = list(FLEXION_INDICES)
            if level >= 0.0:
                tightened[ids] += float(level) * (upper[ids] - tightened[ids])
            else:
                tightened[ids] += float(-level) * (lower[ids] - tightened[ids])
            candidate = copy.deepcopy(entry)
            candidate["joint_pos_canonical"] = tightened.tolist()
            candidate["joint_targets_canonical"] = tightened.tolist()
            hand_action = 2.0 * (tightened[7:] - lower[7:]) / (upper[7:] - lower[7:]) - 1.0
            candidate["last_action_canonical"] = [0.0] * 7 + hand_action.clamp(-1, 1).tolist()
            expanded_entries.append(candidate)
            metadata.append(meta)
            levels.append(level)

    provisional_path = ARGS.output.with_suffix(".provisional.json")
    provisional_path.parent.mkdir(parents=True, exist_ok=True)
    provisional_path.write_text(json.dumps(
        bank_payload(source, expanded_entries, ARGS.object_urdf), indent=2
    ))
    print(f"[adapt] prepared {len(expanded_entries)} tightened candidates", flush=True)
    cfg = SimToolRealAllenKeyAdjustmentEnvCfg()
    cfg.seed = int(source.get("seed", 42))
    cfg.grasp_bank_path = str(provisional_path)
    cfg.assets.object_urdf = str(ARGS.object_urdf.resolve())
    cfg.assets.object_scale = (3.5, 1.5, 1.5) if ARGS.palm_down_opposite_side else (
        3.5, 0.5, 1.5
    )
    cfg.grasp_bank_min_entries = 1
    cfg.allen_require_valid_target_pairs = False
    cfg.allen_target_pair_translation_range_m = (
        float(ARGS.pair_min_translation_m), float(ARGS.pair_max_translation_m)
    )
    cfg.allen_target_pair_rotation_range_deg = (
        float(ARGS.pair_min_rotation_deg), float(ARGS.pair_max_rotation_deg)
    )
    cfg.allen_reset_yaw_range_stages_deg = (0.0,) * len(
        cfg.adjustment_target_rotation_deg
    )
    cfg.scene.num_envs = len(expanded_entries)
    cfg.episode_length_s = max(8.0, (ARGS.settle_steps + ARGS.hold_steps + 30) / 60.0)
    cfg.adjustment_curriculum_min_eligible_count = 1_000_000
    env = gym.make("Isaacsimenvs-SimToolReal-AllenKey-Adjustment-Direct-v0", cfg=cfg)
    print("[adapt] environment initialized", flush=True)
    try:
        inner = env.unwrapped
        env_ids = torch.arange(inner.num_envs, device=inner.device)
        inner._restore_inhand_state(env_ids, env_ids)
        print("[adapt] candidate states restored", flush=True)
        targets = inner._inhand_bank_joint_targets[:, inner._perm_canon_to_lab]
        inner._replay_target_lab_order = targets
        action = inner._inhand_bank_last_action.clone()
        restored_pos, restored_quat = inner._palm_tool_relative()
        hold_reference_pos = restored_pos.clone()
        hold_reference_quat = restored_quat.clone()
        min_support = torch.full((inner.num_envs,), 99, device=inner.device, dtype=torch.long)
        palm_contact_steps = torch.zeros(inner.num_envs, device=inner.device)
        actual_contact_steps = torch.zeros_like(palm_contact_steps)
        socket_valid_steps = torch.zeros_like(palm_contact_steps)
        max_drift = torch.zeros_like(palm_contact_steps)
        max_rotation = torch.zeros_like(palm_contact_steps)
        max_fingertip_force = torch.zeros_like(palm_contact_steps)
        max_palm_force = torch.zeros_like(palm_contact_steps)
        total = int(ARGS.settle_steps) + int(ARGS.hold_steps)
        for step in range(total):
            env.step(action)
            if step == 0:
                current_pos, current_quat = inner._palm_tool_relative()
                initial_position_error = torch.linalg.vector_norm(
                    current_pos - restored_pos, dim=-1
                )
                initial_alignment = torch.abs(
                    (current_quat * restored_quat).sum(-1)
                ).clamp(0, 1)
                print(
                    "[adapt] first-step diagnostics: "
                    f"fingertip_distance_min="
                    f"{float(inner._curr_fingertip_distances.min().item()):.4f}m "
                    f"relative_position_jump_max="
                    f"{float(initial_position_error.max().item()):.4f}m "
                    f"relative_rotation_jump_max="
                    f"{float(torch.rad2deg(2.0 * torch.acos(initial_alignment)).max().item()):.2f}deg",
                    flush=True,
                )
            if (step + 1) % 30 == 0:
                print(f"[adapt] validation step {step + 1}/{total}", flush=True)
            if step == int(ARGS.settle_steps) - 1:
                hold_reference_pos, hold_reference_quat = inner._palm_tool_relative()
            if step < int(ARGS.settle_steps):
                if step >= min(10, int(ARGS.settle_steps) - 1):
                    max_fingertip_force.copy_(torch.maximum(
                        max_fingertip_force,
                        inner._allen_fingertip_force_n.max(dim=-1).values,
                    ))
                    max_palm_force.copy_(torch.maximum(
                        max_palm_force, inner._allen_palm_force_n
                    ))
                continue
            current_pos, current_quat = inner._palm_tool_relative()
            drift = torch.linalg.vector_norm(
                current_pos - hold_reference_pos, dim=-1
            )
            alignment = torch.abs(
                (current_quat * hold_reference_quat).sum(-1)
            ).clamp(0, 1)
            rotation = torch.rad2deg(2.0 * torch.acos(alignment))
            max_drift.copy_(torch.maximum(max_drift, drift))
            max_rotation.copy_(torch.maximum(max_rotation, rotation))
            min_support.copy_(torch.minimum(min_support, inner._stable_support_count))
            palm_contact_steps += inner._allen_palm_contact.float()
            actual_contact_steps += (
                inner._allen_palm_contact
                | (inner._allen_fingertip_contact_count >= 2)
            ).float()
            socket_valid_steps += inner._allen_socket_valid.float()

        hold = float(ARGS.hold_steps)
        fixture_passing = (
            (min_support >= 2)
            & (socket_valid_steps / hold >= 0.95)
            & (max_drift <= float(ARGS.max_hold_drift_m))
            & (max_rotation <= 2.0)
            & (max_fingertip_force <= float(ARGS.max_fingertip_force_n))
            & (max_palm_force <= float(ARGS.max_palm_force_n))
            & (
                actual_contact_steps / hold
                >= float(ARGS.minimum_actual_contact_ratio)
            )
        )
        # A fixture-supported settle is not enough. Close around the settled
        # relationship, release the fixture, and apply the same wrench challenge
        # used by the RL endpoint gate.
        settled_pos, settled_quat = inner._palm_tool_relative()
        inner._adjustment_target_relative_pos.copy_(settled_pos)
        inner._adjustment_target_relative_quat.copy_(settled_quat)
        inner._adjustment_initial_tool_pos.copy_(inner.object.data.root_pos_w)
        inner._adjustment_initial_tool_quat.copy_(inner.object.data.root_quat_w)
        inner.episode_length_buf[:] = int(cfg.allen_adjustment_steps)
        release_validation_steps = (
            int(cfg.allen_closure_steps) + int(cfg.allen_release_steps) - 1
        )
        for step in range(release_validation_steps):
            env.step(action)
            if (step + 1) % 50 == 0:
                print(
                    f"[adapt] release validation step {step + 1}/"
                    f"{release_validation_steps}", flush=True
                )
        release_passing = (
            inner._allen_combined_valid
            & (inner._allen_hold_count >= int(cfg.allen_success_hold_steps))
        )
        passing = fixture_passing & release_passing
        print("[adapt] validation by palm handle roll:", flush=True)
        for handle_roll in sorted({
            float(meta["palm_handle_roll_deg"]) for meta in metadata
        }):
            group = torch.tensor(
                [
                    float(meta["palm_handle_roll_deg"]) == handle_roll
                    for meta in metadata
                ],
                device=inner.device,
                dtype=torch.bool,
            )
            print(
                f"  roll={handle_roll:+.1f}deg candidates={int(group.sum().item())} "
                f"fixture_pass={int((fixture_passing & group).sum().item())} "
                f"release_pass={int((release_passing & group).sum().item())} "
                f"combined_pass={int((passing & group).sum().item())} "
                f"palm_ratio_mean={float((palm_contact_steps[group] / hold).mean().item()):.3f} "
                f"support_mean={float(min_support[group].float().mean().item()):.2f}",
                flush=True,
            )
        diagnostic_rows = []
        for index in range(inner.num_envs):
            actual_contact_ratio = float(
                (actual_contact_steps[index] / hold).item()
            )
            force_ok = bool(
                max_fingertip_force[index] <= float(ARGS.max_fingertip_force_n)
                and max_palm_force[index] <= float(ARGS.max_palm_force_n)
            )
            diagnostic_rows.append((
                int(passing[index].item()),
                int(actual_contact_ratio >= float(ARGS.minimum_actual_contact_ratio)),
                int(force_ok),
                actual_contact_ratio,
                -float(max(max_fingertip_force[index], max_palm_force[index]).item()),
                float((palm_contact_steps[index] / hold).item()),
                float((socket_valid_steps[index] / hold).item()),
                int(min_support[index].item()),
                -float(max_drift[index].item()),
                -float(max_rotation[index].item()),
                index,
            ))
        diagnostic_rows.sort(reverse=True)
        print("[adapt] best physical-validation candidates:", flush=True)
        for (
            _, _, _, actual_contact_ratio, _, palm_ratio, socket_ratio,
            support, neg_drift, neg_rotation, index,
        ) in diagnostic_rows[:10]:
            print(
                f"  candidate={index:03d} source="
                f"{metadata[index]['source_asset_index']}/{metadata[index]['source_entry_index']} "
                f"tighten={levels[index]:.2f} support={support} "
                f"palm={palm_ratio:.3f} socket={socket_ratio:.3f} "
                f"drift={-neg_drift:.4f}m rotation={-neg_rotation:.2f}deg "
                f"finger_force={float(max_fingertip_force[index].item()):.2f}N "
                f"palm_force={float(max_palm_force[index].item()):.2f}N "
                f"actual_contact={actual_contact_ratio:.2f} "
                f"release_hold={int(inner._allen_hold_count[index].item())} "
                f"pass={bool(passing[index])}",
                flush=True,
            )
        # Keep the strongest tightening level for each distinct palm transform,
        # then select connected start/target pairs under the runtime bounds.
        best_by_transform: dict[tuple, tuple[tuple, int]] = {}
        for row in diagnostic_rows:
            index = row[-1]
            if not bool(passing[index]):
                continue
            transform_id = (
                metadata[index]["source_asset_index"],
                metadata[index]["source_entry_index"],
                *tuple(round(float(v), 6) for v in metadata[index]["palm_shift_tool_m"]),
                round(float(metadata[index]["palm_rotation_about_screw_deg"]), 3),
                round(float(metadata[index]["palm_handle_roll_deg"]), 3),
                round(float(metadata[index]["tool_yaw_deg"]), 3),
                *tuple(
                    round(float(v), 4)
                    for v in metadata[index]["tool_position_local"]
                ),
            )
            if transform_id not in best_by_transform:
                best_by_transform[transform_id] = (row[:-1], index)
        ranked = [value[1] for value in sorted(
            best_by_transform.values(), key=lambda value: value[0], reverse=True
        )]
        if len(ranked) < int(ARGS.desired_entries):
            raise RuntimeError(
                f"only {len(ranked)} distinct Allen-key transforms passed physical validation"
            )

        centers = []
        quaternions = []
        for index in ranked:
            entry = expanded_entries[index]
            palm_to_tool = pose_matrix(
                np.asarray(entry["palm_to_tool_pos"]),
                np.asarray(entry["palm_to_tool_quat_wxyz"]),
            )
            tool_to_palm = np.linalg.inv(palm_to_tool)
            centers.append(tool_to_palm[:3, 3])
            quaternions.append(Rotation.from_matrix(palm_to_tool[:3, :3]))
        adjacency = np.zeros((len(ranked), len(ranked)), dtype=bool)
        for i in range(len(ranked)):
            for j in range(i + 1, len(ranked)):
                translation = float(np.linalg.norm(centers[i] - centers[j]))
                rotation = float((quaternions[i].inv() * quaternions[j]).magnitude())
                rotation = math.degrees(rotation)
                valid = (
                    translation <= float(ARGS.pair_max_translation_m)
                    and rotation <= float(ARGS.pair_max_rotation_deg) + 1.0e-6
                    and (
                        translation >= float(ARGS.pair_min_translation_m)
                        or rotation >= float(ARGS.pair_min_rotation_deg)
                    )
                )
                if ARGS.palm_down_opposite_side:
                    side_i = float(metadata[ranked[i]]["palm_side_angle_deg"])
                    side_j = float(metadata[ranked[j]]["palm_side_angle_deg"])
                    valid = valid and side_i * side_j < 0.0
                adjacency[i, j] = adjacency[j, i] = valid
        if not bool(adjacency.any()):
            raise RuntimeError(
                "physically stable grasps contain no non-trivial manipulation pair"
            )
        eligible = [index for index in range(len(ranked)) if adjacency[index].any()]

        def select_connected(pool: list[int], count: int) -> list[int]:
            if count == 0:
                return []
            if count > len(pool):
                raise RuntimeError(f"only {len(pool)}/{count} grasp candidates are available")

            def bucket(index: int) -> tuple[
                float, tuple[float, float, float], float
            ]:
                meta = metadata[ranked[index]]
                return (
                    round(float(meta["tool_yaw_deg"]), 3),
                    tuple(round(float(v), 3) for v in meta["tool_position_local"]),
                    round(float(meta["palm_handle_roll_deg"]), 3),
                )

            selected: list[int] = []
            yaw_counts: dict[float, int] = {}
            position_counts: dict[tuple[float, float, float], int] = {}
            roll_counts: dict[float, int] = {}
            unused = set(pool)
            while len(selected) + 2 <= count:
                edges = [
                    (first, second)
                    for first in unused
                    for second in unused
                    if first < second and bool(adjacency[first, second])
                ]
                if not edges:
                    break

                def edge_score(edge: tuple[int, int]) -> tuple:
                    yaw_a, position_a, roll_a = bucket(edge[0])
                    yaw_b, position_b, roll_b = bucket(edge[1])
                    next_yaw = dict(yaw_counts)
                    next_position = dict(position_counts)
                    next_roll = dict(roll_counts)
                    next_yaw[yaw_a] = next_yaw.get(yaw_a, 0) + 1
                    next_yaw[yaw_b] = next_yaw.get(yaw_b, 0) + 1
                    next_position[position_a] = next_position.get(position_a, 0) + 1
                    next_position[position_b] = next_position.get(position_b, 0) + 1
                    next_roll[roll_a] = next_roll.get(roll_a, 0) + 1
                    next_roll[roll_b] = next_roll.get(roll_b, 0) + 1
                    return (
                        sum(value * value for value in next_roll.values()),
                        sum(value * value for value in next_yaw.values()),
                        sum(value * value for value in next_position.values()),
                        int(yaw_a == yaw_b),
                        int(position_a == position_b),
                        -abs(yaw_a - yaw_b),
                        edge,
                    )

                edge = min(edges, key=edge_score)
                for candidate in edge:
                    yaw, position, roll = bucket(candidate)
                    yaw_counts[yaw] = yaw_counts.get(yaw, 0) + 1
                    position_counts[position] = position_counts.get(position, 0) + 1
                    roll_counts[roll] = roll_counts.get(roll, 0) + 1
                    selected.append(candidate)
                    unused.remove(candidate)
            if len(selected) < count:
                candidates = [
                    candidate for candidate in unused
                    if bool(adjacency[candidate, selected].any())
                ]
                if candidates:
                    selected.append(min(
                        candidates,
                        key=lambda candidate: (
                            roll_counts.get(bucket(candidate)[2], 0),
                            yaw_counts.get(bucket(candidate)[0], 0),
                            position_counts.get(bucket(candidate)[1], 0),
                            candidate,
                        ),
                    ))
            if len(selected) < count:
                raise RuntimeError(
                    f"only {len(selected)}/{count} connected grasps could be selected "
                    "for a required wrist-side group"
                )
            return selected

        above_margin = float(ARGS.wrist_above_margin_m)
        above = [
            index for index in eligible
            if float(metadata[ranked[index]]["wrist_above_tool_m"]) >= above_margin
        ]
        not_above = [index for index in eligible if index not in set(above)]
        desired_count = int(ARGS.desired_entries)
        required_above = int(math.ceil(desired_count * wrist_above_fraction))
        desired_other = desired_count - required_above

        def select_across_handle_rolls(pool: list[int], count: int) -> list[int]:
            rolls = {
                round(float(metadata[ranked[index]]["palm_handle_roll_deg"]), 3)
                for index in pool
            }
            if count >= 4 and len(rolls) < 2:
                raise RuntimeError(
                    "fewer than two handle-side groups passed physical validation"
                )
            selected = select_connected(pool, count)
            selected_rolls = {
                round(float(metadata[ranked[index]]["palm_handle_roll_deg"]), 3)
                for index in selected
            }
            if count >= 4 and len(selected_rolls) < 2:
                raise RuntimeError(
                    "connected selection collapsed to one handle-side group"
                )
            return selected

        chosen = select_across_handle_rolls(above, required_above)
        if desired_other:
            try:
                chosen.extend(select_connected(not_above, desired_other))
            except RuntimeError:
                chosen.extend(select_across_handle_rolls(
                    [index for index in above if index not in chosen], desired_other
                ))
        chosen_adjacency = adjacency[np.ix_(chosen, chosen)]
        if len(chosen) < int(ARGS.desired_entries) or bool(
            (chosen_adjacency.sum(axis=1) == 0).any()
        ):
            raise RuntimeError(
                f"could not select {ARGS.desired_entries} mutually useful manipulation grasps"
            )
        selected_rolls = {
            round(float(metadata[ranked[index]]["palm_handle_roll_deg"]), 3)
            for index in chosen
        }
        if len(selected_rolls) < 2:
            raise RuntimeError(
                "selected bank uses only one side of the Allen-key handle; expand handle rolls"
            )

        selected: list[dict] = []
        for ranked_index in chosen:
            index = ranked[ranked_index]
            entry = copy.deepcopy(expanded_entries[index])
            verification = entry["verification"]
            verification.pop("provisional_rollout_snapshot", None)
            verification.update({
                "physical_replay_validated": True,
                "support_count": int(min_support[index].item()),
                "stable_steps": int(ARGS.settle_steps),
                "hold_steps": int(ARGS.hold_steps),
                "hold_drift_m": float(max_drift[index].item()),
                "hold_rotation_deg": float(max_rotation[index].item()),
                "joint_limit_violation_max_rad": 0.0,
                "palm_contact_ratio": float((palm_contact_steps[index] / hold).item()),
                "actual_contact_ratio": float((actual_contact_steps[index] / hold).item()),
                "socket_valid_ratio": float((socket_valid_steps[index] / hold).item()),
                "tightening_residual_fraction": float(levels[index]),
                "settled_fingertip_force_max_n": float(
                    max_fingertip_force[index].item()
                ),
                "settled_palm_force_max_n": float(max_palm_force[index].item()),
                **metadata[index],
            })
            selected.append(entry)
        # Relative-pose bounds do not guarantee that the arm can realize a
        # target relationship at a particular start tool pose. Materialize a
        # directed IK-valid target graph and store the corresponding target
        # first-joint angle for reset-yaw intersection at runtime.
        robot_urdf = (
            ROOT / "assets/urdf/kuka_sharpa_description"
            / "iiwa14_left_sharpa_adjusted_restricted.urdf"
        )
        target_kinematics = UrdfKinematics(robot_urdf)
        target_thresholds = GraspEvaluatorThresholds(
            ik_position_m=0.0005, ik_orientation_deg=1.0, max_ik_iterations=500
        )
        robot_base = np.eye(4)
        robot_base[1, 3] = 0.8
        reachable = np.zeros((len(selected), len(selected)), dtype=bool)
        target_arm_joint_0 = np.full(
            (len(selected), len(selected)), np.nan, dtype=np.float64
        )
        for source_id, source_entry in enumerate(selected):
            tool_pose = pose_matrix(
                np.asarray(source_entry["object_pos_local"]),
                np.asarray(source_entry["object_quat_wxyz"]),
            )
            source_joints = np.asarray(
                source_entry["joint_pos_canonical"], dtype=np.float64
            )
            for target_id, target_entry in enumerate(selected):
                if source_id == target_id or not chosen_adjacency[source_id, target_id]:
                    continue
                palm_to_tool = pose_matrix(
                    np.asarray(target_entry["palm_to_tool_pos"]),
                    np.asarray(target_entry["palm_to_tool_quat_wxyz"]),
                )
                target_palm = tool_pose @ np.linalg.inv(palm_to_tool)
                arm, position_error, rotation_error, _, _ = solve_arm_ik(
                    target_kinematics,
                    target_palm,
                    source_joints[:7],
                    source_joints[7:],
                    robot_base,
                    target_thresholds,
                )
                if (
                    position_error <= target_thresholds.ik_position_m
                    and rotation_error <= target_thresholds.ik_orientation_deg
                ):
                    reachable[source_id, target_id] = True
                    target_arm_joint_0[source_id, target_id] = float(arm[0])
        isolated = np.flatnonzero(reachable.sum(axis=1) == 0)
        if isolated.size:
            raise RuntimeError(
                "selected Allen-key starts have no arm-reachable targets: "
                f"{isolated.tolist()}"
            )
        for source_id, entry in enumerate(selected):
            target_ids = np.flatnonzero(reachable[source_id]).tolist()
            entry["verification"]["valid_target_ids"] = target_ids
            entry["verification"]["target_arm_joint_0_rad"] = [
                float(target_arm_joint_0[source_id, target_id])
                for target_id in target_ids
            ]
        payload = bank_payload(source, selected, ARGS.object_urdf)
        validate_grasp_bank(payload, minimum_entries=int(ARGS.desired_entries))
        ARGS.output.write_text(json.dumps(payload, indent=2))
        print(f"[pass] wrote {len(selected)} validated Allen-key grasps to {ARGS.output.resolve()}")
    except Exception:
        traceback.print_exc()
        raise
    finally:
        env.close()
        APP.close()


if __name__ == "__main__":
    main()
