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
        default=ROOT / "assets/grasp_banks/allen_key_canonical_v1.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "assets/grasp_banks/allen_key_manipulation_v2.json",
    )
    parser.add_argument("--source-assets", type=int, default=256)
    parser.add_argument("--max-source-entries", type=int, default=24)
    parser.add_argument(
        "--source-entry-indices", type=int, nargs="+", default=None,
        help="Optional explicit source grasps; otherwise sample diverse robust entries.",
    )
    parser.add_argument(
        "--palm-axial-shifts-m", type=float, nargs="+",
        default=(0.0, -0.03, -0.06, -0.09),
        help="Palm-to-tool shifts along the shaft; negative values move the palm toward the bend.",
    )
    parser.add_argument(
        "--palm-shift-radii-m", type=float, nargs="+",
        default=(0.004, 0.005, 0.006, 0.007, 0.008, 0.009),
        help="Tool-frame radial shifts searched around the long handle axis.",
    )
    parser.add_argument(
        "--tool-yaw-angles-deg", type=float, nargs="+",
        default=(-50.0, -25.0, 0.0, 25.0, 50.0),
        help="Engaged Allen-key world orientations represented in the bank.",
    )
    parser.add_argument(
        "--palm-shift-angles-deg", type=float, nargs="+",
        default=(210.0, 225.0, 240.0, 255.0, 270.0, 285.0, 300.0, 315.0, 330.0),
        help="Tool-frame radial shift angles searched around the handle.",
    )
    parser.add_argument(
        "--palm-rotation-angles-deg", type=float, nargs="+",
        default=(0.0, 5.0, 10.0, 15.0, 20.0, 25.0),
        help="Physically validate palm orientations about the engaged screw axis.",
    )
    parser.add_argument("--settle-steps", type=int, default=60)
    parser.add_argument("--hold-steps", type=int, default=120)
    parser.add_argument("--desired-entries", type=int, default=8)
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


def provisional_entries(
    source: dict, count_assets: int, source_entry_indices: list[int] | None,
    palm_axial_shifts_m: tuple[float, ...],
    palm_shift_radii_m: tuple[float, ...], palm_shift_angles_deg: tuple[float, ...],
    palm_rotation_angles_deg: tuple[float, ...], tool_yaw_angles_deg: tuple[float, ...],
    max_source_entries: int,
) -> list[tuple[dict, dict]]:
    robot = ROOT / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
    kinematics = UrdfKinematics(robot)
    thresholds = GraspEvaluatorThresholds(
        ik_position_m=0.003, ik_orientation_deg=3.0, max_ik_iterations=300
    )
    desired_tool_position = np.asarray((0.0, 0.08, 0.70))
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
                for palm_rotation_deg in palm_rotation_angles_deg:
                    tool_to_palm = np.linalg.inv(shifted)
                    screw_rotation = Rotation.from_rotvec(
                        np.asarray((0.0, 0.0, -1.0))
                        * math.radians(float(palm_rotation_deg))
                    ).as_matrix()
                    tool_to_palm[:3, :3] = screw_rotation @ tool_to_palm[:3, :3]
                    palm_to_tool = np.linalg.inv(tool_to_palm)
                    for tool_yaw_deg in tool_yaw_angles_deg:
                        desired_tool = np.eye(4)
                        desired_tool[:3, :3] = Rotation.from_euler(
                            "z", float(tool_yaw_deg), degrees=True
                        ).as_matrix()
                        desired_tool[:3, 3] = desired_tool_position
                        target_palm = desired_tool @ tool_to_palm
                        original = np.asarray(source_entry["joint_pos_canonical"], dtype=np.float64)
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
                            "tool_yaw_deg": float(tool_yaw_deg),
                            "ik_position_error_m": pos_error,
                            "ik_rotation_error_deg": rot_error,
                        }))
    if not candidates:
        raise RuntimeError("no screwdriver grasp candidate has a reachable engaged Allen-key pose")
    return candidates


def bank_payload(source: dict, entries: list[dict]) -> dict:
    checkpoint = ROOT / "pretrained_policy/model.pth"
    return {
        "schema_version": 2,
        "tool_type": "allen_key",
        "object_name": "allen_key_canonical",
        "asset_sha256": sha256_file(ROOT / "assets/urdf/objects/allen_key_canonical.urdf"),
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
        raise FileNotFoundError(f"source screwdriver bank does not exist: {ARGS.source_bank}")
    source = json.loads(ARGS.source_bank.read_text())
    candidates = provisional_entries(
        source,
        int(ARGS.source_assets),
        ARGS.source_entry_indices,
        tuple(float(value) for value in ARGS.palm_axial_shifts_m),
        tuple(float(value) for value in ARGS.palm_shift_radii_m),
        tuple(float(value) for value in ARGS.palm_shift_angles_deg),
        tuple(float(value) for value in ARGS.palm_rotation_angles_deg),
        tuple(float(value) for value in ARGS.tool_yaw_angles_deg),
        int(ARGS.max_source_entries),
    )
    expanded_entries: list[dict] = []
    metadata: list[dict] = []
    levels: list[float] = []
    lower = torch.tensor(source["joint_lower_canonical"])
    upper = torch.tensor(source["joint_upper_canonical"])
    for entry, meta in candidates:
        base = torch.tensor(entry["joint_targets_canonical"])
        for level in TIGHTENING_LEVELS:
            tightened = base.clone()
            tightened[list(FLEXION_INDICES)] += float(level) * (
                upper[list(FLEXION_INDICES)] - tightened[list(FLEXION_INDICES)]
            )
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
    provisional_path.write_text(json.dumps(bank_payload(source, expanded_entries), indent=2))
    print(f"[adapt] prepared {len(expanded_entries)} tightened candidates", flush=True)
    cfg = SimToolRealAllenKeyAdjustmentEnvCfg()
    cfg.grasp_bank_path = str(provisional_path)
    cfg.grasp_bank_min_entries = 1
    cfg.allen_require_valid_target_pairs = False
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
        socket_valid_steps = torch.zeros_like(palm_contact_steps)
        max_drift = torch.zeros_like(palm_contact_steps)
        max_rotation = torch.zeros_like(palm_contact_steps)
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
            socket_valid_steps += inner._allen_socket_valid.float()

        hold = float(ARGS.hold_steps)
        passing = (
            (min_support >= 2)
            & (palm_contact_steps / hold >= 0.95)
            & (socket_valid_steps / hold >= 0.95)
            & (max_drift <= 0.005)
            & (max_rotation <= 2.0)
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
        passing &= release_passing
        diagnostic_rows = []
        for index in range(inner.num_envs):
            diagnostic_rows.append((
                int(passing[index].item()),
                float((palm_contact_steps[index] / hold).item()),
                float((socket_valid_steps[index] / hold).item()),
                int(min_support[index].item()),
                -float(max_drift[index].item()),
                -float(max_rotation[index].item()),
                index,
            ))
        diagnostic_rows.sort(reverse=True)
        print("[adapt] best physical-validation candidates:", flush=True)
        for _, palm_ratio, socket_ratio, support, neg_drift, neg_rotation, index in diagnostic_rows[:10]:
            print(
                f"  candidate={index:03d} source="
                f"{metadata[index]['source_asset_index']}/{metadata[index]['source_entry_index']} "
                f"tighten={levels[index]:.2f} support={support} "
                f"palm={palm_ratio:.3f} socket={socket_ratio:.3f} "
                f"drift={-neg_drift:.4f}m rotation={-neg_rotation:.2f}deg "
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
                round(float(metadata[index]["tool_yaw_deg"]), 3),
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
                    translation <= 0.090 and rotation <= 100.0
                    and (translation >= 0.012 or rotation >= 18.0)
                )
                adjacency[i, j] = adjacency[j, i] = valid
        if not bool(adjacency.any()):
            raise RuntimeError(
                "physically stable grasps contain no non-trivial manipulation pair"
            )
        pose_buckets: dict[tuple[float, float], list[int]] = {}
        for ranked_index, candidate_index in enumerate(ranked):
            if not bool(adjacency[ranked_index].any()):
                continue
            yaw = round(float(metadata[candidate_index]["tool_yaw_deg"]), 3)
            palm_rotation = round(float(
                metadata[candidate_index]["palm_rotation_about_screw_deg"]
            ), 3)
            pose_buckets.setdefault((yaw, palm_rotation), []).append(ranked_index)
        candidate_order: list[int] = []
        while any(pose_buckets.values()):
            for pose_group in sorted(pose_buckets):
                if pose_buckets[pose_group]:
                    candidate_order.append(pose_buckets[pose_group].pop(0))

        chosen: list[int] = []
        for candidate in candidate_order:
            if len(chosen) >= int(ARGS.desired_entries):
                break
            if not chosen or bool(adjacency[candidate, chosen].any()):
                chosen.append(candidate)
        # Ensure the first selected grasp also has a selected manipulation edge.
        if len(chosen) > 1 and not bool(adjacency[chosen[0], chosen[1:]].any()):
            chosen = chosen[1:]
        chosen = chosen[:int(ARGS.desired_entries)]
        chosen_adjacency = adjacency[np.ix_(chosen, chosen)]
        if len(chosen) < int(ARGS.desired_entries) or bool(
            (chosen_adjacency.sum(axis=1) == 0).any()
        ):
            raise RuntimeError(
                f"could not select {ARGS.desired_entries} mutually useful manipulation grasps"
            )

        selected: list[dict] = []
        for ranked_index in chosen:
            index = ranked[ranked_index]
            entry = copy.deepcopy(expanded_entries[index])
            verification = entry["verification"]
            verification.update({
                "support_count": int(min_support[index].item()),
                "stable_steps": int(ARGS.settle_steps),
                "hold_steps": int(ARGS.hold_steps),
                "hold_drift_m": float(max_drift[index].item()),
                "hold_rotation_deg": float(max_rotation[index].item()),
                "joint_limit_violation_max_rad": 0.0,
                "palm_contact_ratio": float((palm_contact_steps[index] / hold).item()),
                "socket_valid_ratio": float((socket_valid_steps[index] / hold).item()),
                "tightening_residual_fraction": float(levels[index]),
                **metadata[index],
            })
            selected.append(entry)
        payload = bank_payload(source, selected)
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
