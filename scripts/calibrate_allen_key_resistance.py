#!/usr/bin/env python3
"""Calibrate a robust static resistance torque from validated Allen-key grasps."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grasp-bank",
        type=Path,
        default=ROOT / "assets/grasp_banks/allen_key_palm_down_v1.json",
    )
    parser.add_argument(
        "--torque-levels-nm",
        type=float,
        nargs="+",
        default=(0.0, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0, 1.25, 1.50, 2.0),
    )
    parser.add_argument("--settle-steps", type=int, default=120)
    parser.add_argument("--load-steps", type=int, default=120)
    parser.add_argument("--closure-fraction", type=float, default=0.08)
    parser.add_argument("--maximum-drift-m", type=float, default=0.015)
    parser.add_argument("--maximum-drift-deg", type=float, default=10.0)
    parser.add_argument("--minimum-stable-ratio", type=float, default=0.80)
    parser.add_argument("--maximum-saturation-ratio", type=float, default=0.20)
    parser.add_argument("--output-dir", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    return parser.parse_args()


ARGS = parse_args()
APP = AppLauncher(ARGS).app


import gymnasium as gym  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    quat_apply,
    quat_from_angle_axis,
    subtract_frame_transforms,
)

import isaacsimenvs  # noqa: E402,F401
from isaacsimenvs.tasks.simtoolreal.simtoolreal_tacmap_env_cfg import (  # noqa: E402
    SimToolRealAllenKeyTurningEnvCfg,
)


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


TASK_ID = "Isaacsimenvs-SimToolReal-AllenKey-Turning-Direct-v0"


def quaternion_distance_deg(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    alignment = torch.abs((first * second).sum(-1)).clamp(0.0, 1.0)
    return torch.rad2deg(2.0 * torch.acos(alignment))


def validate_args(entries: list[dict]) -> list[float]:
    levels = [float(value) for value in ARGS.torque_levels_nm]
    if not ARGS.grasp_bank.is_file():
        raise FileNotFoundError(ARGS.grasp_bank)
    if len(entries) < 8:
        raise RuntimeError("resistance calibration requires at least eight validated grasps")
    if (
        not levels or levels[0] != 0.0
        or any(not math.isfinite(value) or value < 0.0 for value in levels)
        or any(later <= earlier for earlier, later in zip(levels, levels[1:]))
    ):
        raise ValueError("torque levels must start at zero and strictly increase")
    if int(ARGS.settle_steps) <= 0 or int(ARGS.load_steps) <= 0:
        raise ValueError("settle and load steps must be positive")
    if not 0.0 <= float(ARGS.closure_fraction) <= 0.25:
        raise ValueError("closure fraction must lie in [0, 0.25]")
    return levels


def load_entries() -> list[dict]:
    payload = json.loads(ARGS.grasp_bank.read_text())
    entries = [
        entry for entry in payload.get("entries", [])
        if int(entry.get("verification", {}).get("hold_steps", 0)) >= 120
        and not bool(entry.get("verification", {}).get("target_only", False))
    ]
    required = (
        "joint_pos_canonical", "joint_targets_canonical",
        "object_pos_local", "object_quat_wxyz",
    )
    for index, entry in enumerate(entries):
        missing = [name for name in required if entry.get(name) is None]
        if missing:
            raise RuntimeError(f"grasp entry {index} is missing {missing}")
    return entries


def install_grasps(inner, entries: list[dict]) -> torch.Tensor:
    base_count = len(entries)
    env_ids = torch.arange(2 * base_count, device=inner.device)
    entry_ids = torch.arange(2 * base_count, device=inner.device) % base_count
    direction = torch.where(
        env_ids < base_count,
        torch.ones_like(env_ids, dtype=torch.float32),
        -torch.ones_like(env_ids, dtype=torch.float32),
    )
    canonical_position = torch.tensor(
        [entry["joint_pos_canonical"] for entry in entries],
        device=inner.device,
    )[entry_ids]
    canonical_target = torch.tensor(
        [entry["joint_targets_canonical"] for entry in entries],
        device=inner.device,
    )[entry_ids]
    joint_pos = canonical_position[:, inner._perm_canon_to_lab]
    joint_target = canonical_target[:, inner._perm_canon_to_lab]
    flexion_ids = inner._allen_flexion_joint_ids if hasattr(
        inner, "_allen_flexion_joint_ids"
    ) else torch.tensor([
        index for index, name in enumerate(inner.robot.data.joint_names)
        if name.endswith(("_FE", "_PIP", "_DIP", "_IP"))
        or name == "left_pinky_CMC"
    ], device=inner.device, dtype=torch.long)
    if flexion_ids.numel() == 0:
        raise RuntimeError("calibration could not identify hand flexion joints")
    upper = inner.robot.data.joint_pos_limits[:, flexion_ids, 1]
    joint_target[:, flexion_ids] += float(ARGS.closure_fraction) * (
        upper - joint_target[:, flexion_ids]
    )
    joint_target = torch.maximum(
        torch.minimum(joint_target, inner.robot.data.joint_pos_limits[:, :, 1]),
        inner.robot.data.joint_pos_limits[:, :, 0],
    )
    inner.robot.write_joint_state_to_sim(
        joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids
    )
    inner._cur_targets.copy_(joint_target)
    inner._prev_targets.copy_(joint_target)
    inner._replay_target_lab_order = joint_target

    local_position = torch.tensor(
        [entry["object_pos_local"] for entry in entries], device=inner.device
    )[entry_ids]
    source_quaternion = torch.tensor(
        [entry["object_quat_wxyz"] for entry in entries], device=inner.device
    )[entry_ids]
    local_x = torch.zeros(2 * base_count, 3, device=inner.device)
    local_x[:, 0] = 1.0
    world_x = quat_apply(source_quaternion, local_x)
    yaw = torch.atan2(world_x[:, 1], world_x[:, 0])
    z_axis = torch.zeros_like(local_x)
    z_axis[:, 2] = 1.0
    quaternion = quat_from_angle_axis(yaw, z_axis)
    raw_position = local_position + inner.scene.env_origins
    pivot_tool = torch.tensor(
        inner.cfg.allen_turn_screw_pivot_tool_m, device=inner.device
    ).expand(2 * base_count, -1)
    pivot = raw_position + quat_apply(source_quaternion, pivot_tool)
    position = pivot - quat_apply(quaternion, pivot_tool)
    inner.object.write_root_pose_to_sim(
        torch.cat((position, quaternion), -1), env_ids=env_ids
    )
    inner.object.write_root_velocity_to_sim(
        torch.zeros(2 * base_count, 6, device=inner.device), env_ids=env_ids
    )
    socket_position = pivot + torch.tensor(
        inner.cfg.allen_turn_socket_root_from_pivot_m, device=inner.device
    )
    inner.workpiece.write_root_pose_to_sim(
        torch.cat((socket_position, quaternion), -1), env_ids=env_ids
    )
    inner.workpiece.write_root_velocity_to_sim(
        torch.zeros(2 * base_count, 6, device=inner.device), env_ids=env_ids
    )
    inner._turn_phase.fill_(1)
    inner._turn_direction.copy_(direction)
    inner._turn_acquired.fill_(True)
    inner._turn_initial_tool_pos.copy_(position)
    inner._turn_initial_tool_quat.copy_(quaternion)
    inner._turn_pivot_w.copy_(pivot)
    inner._turn_initial_yaw.copy_(yaw)
    inner._turn_previous_yaw.copy_(yaw)
    inner._turn_cumulative_angle.zero_()
    inner._turn_target_angle.copy_(direction * math.radians(30.0))
    inner._turn_subgoal_index.zero_()
    inner._turn_subgoal_hold.zero_()
    inner._turn_final_hold.zero_()
    inner.episode_length_buf.zero_()
    inner._write_turn_goal(env_ids, inner._turn_target_angle)
    return env_ids


def palm_tool_transform(inner) -> tuple[torch.Tensor, torch.Tensor]:
    palm_position = inner.robot.data.body_link_pos_w[:, inner._palm_body_id]
    palm_quaternion = inner.robot.data.body_link_quat_w[:, inner._palm_body_id]
    return subtract_frame_transforms(
        palm_position,
        palm_quaternion,
        inner.object.data.root_pos_w,
        inner.object.data.root_quat_w,
    )


def main() -> None:
    entries = load_entries()
    levels = validate_args(entries)
    output_dir = ARGS.output_dir or (
        ROOT / "outputs" / "allen_key_resistance_calibration"
        / time.strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    cfg = SimToolRealAllenKeyTurningEnvCfg()
    cfg.scene.num_envs = 2 * len(entries)
    cfg.assets.allen_key_lengths_m = (0.264,)
    cfg.allen_turn_require_calibrated_load = False
    cfg.allen_turn_calibrated_torque_nm = 0.0
    cfg.allen_turn_resistance_fractions = (1.0,) * len(
        cfg.allen_turn_resistance_fractions
    )
    cfg.allen_turn_acquisition_timeout_steps = 1_000_000
    cfg.termination.episode_length = 1_000_000
    cfg.episode_length_s = 20_000.0
    cfg.allen_turn_curriculum_min_episodes = 1_000_000
    env = gym.make(TASK_ID, cfg=cfg)
    inner = env.unwrapped
    zero_action = torch.zeros(
        inner.num_envs, inner.cfg.action_space, device=inner.device
    )
    rows = []
    pass_matrix = []
    try:
        for level in levels:
            inner._reset_idx(torch.arange(inner.num_envs, device=inner.device))
            install_grasps(inner, entries)
            inner.cfg.allen_turn_calibrated_torque_nm = 0.0
            inner._turn_external_torque_override_nm = torch.zeros(
                inner.num_envs, device=inner.device
            )
            for _ in range(int(ARGS.settle_steps)):
                _, _, terminated, truncated, _ = env.step(zero_action)
                if bool(terminated.any()) or bool(truncated.any()):
                    raise RuntimeError("calibration environment ended during settle")
            baseline_position, baseline_quaternion = palm_tool_transform(inner)
            stable_steps = torch.zeros(inner.num_envs, device=inner.device)
            support_steps = torch.zeros_like(stable_steps)
            saturation_sum = torch.zeros_like(stable_steps)
            maximum_translation_drift = torch.zeros_like(stable_steps)
            maximum_rotation_drift = torch.zeros_like(stable_steps)
            inner._turn_external_torque_override_nm = (
                inner._turn_direction * float(level)
            )
            for _ in range(int(ARGS.load_steps)):
                _, _, terminated, truncated, _ = env.step(zero_action)
                if bool(terminated.any()) or bool(truncated.any()):
                    raise RuntimeError("calibration environment ended under load")
                stable_steps += inner._turn_stable_grasp.float()
                support = inner._turn_palm_contact | (
                    inner._turn_fingertip_contact.sum(-1)
                    >= int(inner.cfg.allen_turn_minimum_contact_fingers)
                )
                support_steps += support.float()
                saturation_sum += inner._turn_effort_saturation
                current_position, current_quaternion = palm_tool_transform(inner)
                maximum_translation_drift = torch.maximum(
                    maximum_translation_drift,
                    torch.linalg.vector_norm(
                        current_position - baseline_position, dim=-1
                    ),
                )
                maximum_rotation_drift = torch.maximum(
                    maximum_rotation_drift,
                    quaternion_distance_deg(current_quaternion, baseline_quaternion),
                )
            final_position, final_quaternion = palm_tool_transform(inner)
            translation_drift = torch.linalg.vector_norm(
                final_position - baseline_position, dim=-1
            )
            rotation_drift = quaternion_distance_deg(
                final_quaternion, baseline_quaternion
            )
            stable_ratio = stable_steps / float(ARGS.load_steps)
            support_ratio = support_steps / float(ARGS.load_steps)
            saturation_ratio = saturation_sum / float(ARGS.load_steps)
            passed = (
                (maximum_translation_drift <= float(ARGS.maximum_drift_m))
                & (maximum_rotation_drift <= float(ARGS.maximum_drift_deg))
                & (support_ratio >= float(ARGS.minimum_stable_ratio))
                & (saturation_ratio <= float(ARGS.maximum_saturation_ratio))
            )
            pass_matrix.append(passed.detach().cpu())
            rows.append({
                "torque_nm": float(level),
                "pass_rate": float(passed.float().mean()),
                "translation_drift_mean_m": float(translation_drift.mean()),
                "rotation_drift_mean_deg": float(rotation_drift.mean()),
                "maximum_translation_drift_mean_m": float(
                    maximum_translation_drift.mean()
                ),
                "maximum_rotation_drift_mean_deg": float(
                    maximum_rotation_drift.mean()
                ),
                "stable_ratio_mean": float(stable_ratio.mean()),
                "support_ratio_mean": float(support_ratio.mean()),
                "fingertip_contact_count_mean": float(
                    inner._turn_fingertip_contact.float().sum(-1).mean()
                ),
                "palm_contact_ratio": float(inner._turn_palm_contact.float().mean()),
                "relative_linear_speed_mean_mps": float(
                    inner._turn_relative_linear_speed.mean()
                ),
                "relative_angular_speed_mean_radps": float(
                    inner._turn_relative_angular_speed.mean()
                ),
                "effort_saturation_mean": float(saturation_ratio.mean()),
            })
            print(f"[calibration] {json.dumps(rows[-1], sort_keys=True)}", flush=True)
        passed = torch.stack(pass_matrix)
        contiguous = torch.cumprod(passed.long(), dim=0).bool()
        level_tensor = torch.tensor(levels)[:, None].expand_as(contiguous)
        capacity = torch.where(contiguous, level_tensor, torch.zeros_like(level_tensor)).amax(0)
        robust_capacity = float(torch.quantile(
            capacity, 0.25, interpolation="lower"
        ))
        recommended = 0.8 * robust_capacity
        if recommended <= 0.0:
            raise RuntimeError(
                "calibration found no positive robust retained torque; training is blocked"
            )
        summary = {
            "schema_version": 1,
            "grasp_bank": str(ARGS.grasp_bank.resolve()),
            "num_grasps": len(entries),
            "num_directional_trials": inner.num_envs,
            "torque_levels_nm": levels,
            "rows": rows,
            "capacity_nm": capacity.tolist(),
            "lower_quartile_capacity_nm": robust_capacity,
            "recommended_training_torque_nm": recommended,
            "recommendation_fraction": 0.8,
        }
        (output_dir / "calibration.json").write_text(json.dumps(summary, indent=2) + "\n")
        figure, axis = plt.subplots(figsize=(6.4, 4.0))
        axis.plot(levels, [row["pass_rate"] for row in rows], marker="o", linewidth=2)
        axis.axvline(recommended, color="#c44e52", linestyle="--", label="training maximum")
        axis.set(xlabel="Opposing torque (N m)", ylabel="Retained-grasp pass rate", ylim=(-0.03, 1.03))
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
        figure.tight_layout()
        figure.savefig(output_dir / "calibration.png", dpi=180)
        plt.close(figure)
        print(f"[pass] recommended_training_torque_nm={recommended:.6f}", flush=True)
        print(f"[output] {output_dir.resolve()}", flush=True)
    finally:
        if hasattr(inner, "_turn_external_torque_override_nm"):
            del inner._turn_external_torque_override_nm
        if hasattr(inner, "_replay_target_lab_order"):
            del inner._replay_target_lab_order
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    else:
        APP.close()
