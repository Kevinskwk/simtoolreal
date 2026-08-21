#!/usr/bin/env python3
"""Adapt screwdriver grasps to a tightened, socket-engaged Allen-key bank."""

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
        default=ROOT
        / "outputs/multitool_grasp_banks/20260818_123726"
        / "long_screwdriver_seed0/robust.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "assets/grasp_banks/allen_key_canonical_v1.json",
    )
    parser.add_argument("--source-assets", type=int, default=256)
    parser.add_argument("--settle-steps", type=int, default=60)
    parser.add_argument("--hold-steps", type=int, default=120)
    parser.add_argument("--desired-entries", type=int, default=1)
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


TIGHTENING_LEVELS = (0.0, 0.05, 0.10, 0.15, 0.20)
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


def provisional_entries(source: dict, count_assets: int) -> list[tuple[dict, dict]]:
    robot = ROOT / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
    kinematics = UrdfKinematics(robot)
    thresholds = GraspEvaluatorThresholds(
        ik_position_m=0.003, ik_orientation_deg=3.0, max_ik_iterations=300
    )
    desired_tool = pose_matrix(
        np.asarray((0.0, 0.08, 0.70)), np.asarray((1.0, 0.0, 0.0, 0.0))
    )
    robot_base = np.eye(4)
    robot_base[1, 3] = 0.8
    if "assets" in source:
        assets = sorted(
            source["assets"],
            key=lambda asset: abs(float(asset["object_scale"][1]) * 0.04 - 0.02),
        )[:count_assets]
    else:
        assets = [{"asset_index": 0, "entries": source["entries"][:count_assets]}]
    candidates: list[tuple[dict, dict]] = []
    for asset in assets:
        for source_index, source_entry in enumerate(asset["entries"]):
            palm_to_tool = pose_matrix(
                np.asarray(source_entry["palm_to_tool_pos"]),
                np.asarray(source_entry["palm_to_tool_quat_wxyz"]),
            )
            target_palm = desired_tool @ np.linalg.inv(palm_to_tool)
            original = np.asarray(source_entry["joint_pos_canonical"], dtype=np.float64)
            arm, pos_error, rot_error, _, _ = solve_arm_ik(
                kinematics, target_palm, original[:7], original[7:],
                robot_base, thresholds,
            )
            if pos_error > thresholds.ik_position_m or rot_error > thresholds.ik_orientation_deg:
                continue
            entry = copy.deepcopy(source_entry)
            joints = np.concatenate((arm, original[7:]))
            targets = np.asarray(source_entry["joint_targets_canonical"], dtype=np.float64)
            targets[:7] = arm
            entry.update({
                "joint_pos_canonical": joints.tolist(),
                "joint_vel_canonical": [0.0] * 29,
                "joint_targets_canonical": targets.tolist(),
                "object_pos_local": desired_tool[:3, 3].tolist(),
                "object_quat_wxyz": quaternion_wxyz(desired_tool),
                "object_velocity": [0.0] * 6,
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
    candidates = provisional_entries(source, int(ARGS.source_assets))
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
        initial_pos, initial_quat = inner._palm_tool_relative()
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
                    current_pos - initial_pos, dim=-1
                )
                initial_alignment = torch.abs(
                    (current_quat * initial_quat).sum(-1)
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
            if step < int(ARGS.settle_steps):
                continue
            current_pos, current_quat = inner._palm_tool_relative()
            drift = torch.linalg.vector_norm(current_pos - initial_pos, dim=-1)
            alignment = torch.abs((current_quat * initial_quat).sum(-1)).clamp(0, 1)
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
                f"pass={bool(passing[index])}",
                flush=True,
            )
        selected: list[dict] = []
        used_sources: set[tuple[int, int]] = set()
        for index in range(inner.num_envs):
            source_id = (
                metadata[index]["source_asset_index"],
                metadata[index]["source_entry_index"],
            )
            if source_id in used_sources or not bool(passing[index]):
                continue
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
            actual = inner.robot.data.joint_pos[index, inner._perm_lab_to_canon]
            current_pos, current_quat = inner._palm_tool_relative()
            entry["joint_pos_canonical"] = actual.tolist()
            entry["joint_targets_canonical"] = actual.tolist()
            hand_action = 2.0 * (actual[7:] - lower[7:].to(inner.device)) / (
                upper[7:].to(inner.device) - lower[7:].to(inner.device)
            ) - 1.0
            entry["last_action_canonical"] = [0.0] * 7 + hand_action.clamp(-1, 1).tolist()
            object_pos_local = (
                inner.object.data.root_pos_w[index] - inner.scene.env_origins[index]
            )
            entry["object_pos_local"] = object_pos_local.tolist()
            entry["object_quat_wxyz"] = inner.object.data.root_quat_w[index].tolist()
            entry["reference_contact_quat_wxyz"] = entry["object_quat_wxyz"]
            entry["palm_to_tool_pos"] = current_pos[index].tolist()
            entry["palm_to_tool_quat_wxyz"] = current_quat[index].tolist()
            selected.append(entry)
            used_sources.add(source_id)
            if len(selected) >= int(ARGS.desired_entries):
                break
        if len(selected) < int(ARGS.desired_entries):
            raise RuntimeError(
                f"only {len(selected)}/{ARGS.desired_entries} Allen-key grasps passed "
                "tightening and physical hold validation"
            )
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
