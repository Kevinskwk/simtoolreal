#!/usr/bin/env python3
"""Evaluate virtual tool-pose penetration as a contact-force command.

The policy first acquires and lifts a real grasp. A loading/unloading sweep at
the first scrape pose calibrates a virtual table-normal offset near 4 N. The
same offset is then retained while the policy moves through consecutive scrape
poses without resetting, exposing contact loss and force transients in motion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINTS = {
    "vanilla": REPO_ROOT / "pretrained_policy/model.pth",
    "tactile": REPO_ROOT
    / "outputs/2026-07-03/22-42-37/0_simtoolreal_sapg/nn/"
    "last_0_simtoolreal_sapg_ep_75000_rew_27025.139.pth",
    "no-tactile": REPO_ROOT
    / "outputs/2026-07-07/10-00-40/0_simtoolreal_sapg/nn/"
    "last_0_simtoolreal_sapg_ep_3000_rew_27526.992.pth",
}
DEFAULT_POLICY_CONFIG = REPO_ROOT / "pretrained_policy/config.yaml"
VANILLA_TABLE_HEIGHT_OFFSET_M = 0.10
VANILLA_TARGET_Z_RANGE_M = (0.68, 1.05)
ACQUISITION_LIFT_OFFSET_M = 0.12
POST_RAISE_CLEARANCE_M = 0.05
ACQUISITION_MAX_LINEAR_SPEED_MPS = 0.08
ACQUISITION_MAX_ANGULAR_SPEED_RADPS = 0.75
CLEARANCE_STABILITY_WINDOW_STEPS = 5
BASE_OBS = (
    "joint_pos",
    "joint_vel",
    "prev_action_targets",
    "palm_pos",
    "palm_rot",
    "object_rot",
    "fingertip_pos_rel_palm",
    "keypoints_rel_palm",
    "keypoints_rel_goal",
    "object_scales",
)
LOADING_OFFSETS_M = (0.003, 0.001, 0.0, -0.00025, -0.0005, -0.001, -0.0015, -0.0025)
SWEEP_OFFSETS_M = LOADING_OFFSETS_M + (
    -0.0015,
    -0.001,
    -0.0005,
    -0.00025,
    0.0,
    0.001,
    0.003,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=("vanilla", "tactile", "no-tactile"), default="tactile"
    )
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--policy-config", default=str(DEFAULT_POLICY_CONFIG))
    parser.add_argument(
        "--output-root", default=str(REPO_ROOT / "outputs/virtual_pose_offset")
    )
    parser.add_argument("--tool-type", default="spatula")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy-coef-id", type=float, default=0.0)
    parser.add_argument("--sequences", type=int, default=3)
    parser.add_argument("--poses-per-sequence", type=int, default=5)
    parser.add_argument("--acquisition-steps", type=int, default=3600)
    parser.add_argument("--clearance-lift-timeout-steps", type=int, default=600)
    parser.add_argument("--stage-timeout-steps", type=int, default=240)
    parser.add_argument("--minimum-stage-steps", type=int, default=30)
    parser.add_argument("--stable-window-steps", type=int, default=30)
    parser.add_argument("--minimum-pose-translation-m", type=float, default=0.04)
    parser.add_argument("--target-force-n", type=float, default=4.0)
    parser.add_argument("--contact-threshold-n", type=float, default=0.1)
    parser.add_argument("--maximum-force-n", type=float, default=20.0)
    parser.add_argument("--safety-force-consecutive-steps", type=int, default=3)
    parser.add_argument("--immediate-force-limit-n", type=float, default=100.0)
    parser.add_argument("--grasp-warning-drift-m", type=float, default=0.03)
    parser.add_argument("--grasp-loss-drift-m", type=float, default=0.06)
    parser.add_argument("--table-angle-range-deg", type=float, default=8.0)
    parser.add_argument("--table-height-range-m", type=float, default=0.01)
    parser.add_argument(
        "--table-height-offset-m",
        type=float,
        default=None,
        help=(
            "Offset added to the task's nominal table height. Defaults to "
            f"{VANILLA_TABLE_HEIGHT_OFFSET_M:.2f} m for --variant vanilla and "
            "0 m otherwise."
        ),
    )
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--camera-width", type=int, default=960)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument(
        "--require-pass",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Return exit code 2 when the scientific acceptance criteria are not met.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = bool(args.video)
    return args


ARGS = parse_args()
APP = AppLauncher(ARGS).app


import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402

import isaacsimenvs  # noqa: E402,F401
from deployment.rl_player import RlPlayer  # noqa: E402
from isaacsimenvs.tasks.simtoolreal.simtoolreal_tacmap_env_cfg import (  # noqa: E402
    SimToolRealTacMapScrapePoseEnvCfg,
)
from isaacsimenvs.tasks.simtoolreal.utils import scene_utils  # noqa: E402
from isaacsimenvs.tasks.simtoolreal.utils.scrape_pose_utils import (  # noqa: E402
    edge_contact_points_w,
    sample_edge_contact_goal_pose,
    table_top_state,
)
from isaacsimenvs.tasks.simtoolreal.utils.virtual_pose_offset_eval import (  # noqa: E402
    StabilityLimits,
    pose_window_is_stable,
    quaternion_angle_deg,
    select_force_calibration,
    spearman_correlation,
    transition_force_metrics,
    update_persistent_violation,
)


class EvaluationFailure(RuntimeError):
    """A failure that invalidates one sequence or the complete evaluation."""


def validate_args() -> None:
    if ARGS.sequences <= 0:
        raise ValueError("--sequences must be positive")
    if ARGS.poses_per_sequence < 2:
        raise ValueError("--poses-per-sequence must be at least 2")
    if (
        ARGS.acquisition_steps <= 0
        or ARGS.clearance_lift_timeout_steps <= 0
        or ARGS.stage_timeout_steps <= 0
    ):
        raise ValueError("step limits must be positive")
    if not 1 <= ARGS.minimum_stage_steps <= ARGS.stage_timeout_steps:
        raise ValueError("--minimum-stage-steps must be within the stage timeout")
    if not 2 <= ARGS.stable_window_steps <= ARGS.stage_timeout_steps:
        raise ValueError("--stable-window-steps must be between 2 and the stage timeout")
    if ARGS.clearance_lift_timeout_steps < ARGS.stable_window_steps:
        raise ValueError(
            "--clearance-lift-timeout-steps must be at least --stable-window-steps"
        )
    if ARGS.target_force_n <= 0.0 or ARGS.maximum_force_n <= ARGS.target_force_n:
        raise ValueError("force target/limit are invalid")
    if ARGS.safety_force_consecutive_steps <= 0:
        raise ValueError("--safety-force-consecutive-steps must be positive")
    if ARGS.immediate_force_limit_n <= ARGS.maximum_force_n:
        raise ValueError("--immediate-force-limit-n must exceed --maximum-force-n")
    if not 0.0 < ARGS.grasp_warning_drift_m < ARGS.grasp_loss_drift_m:
        raise ValueError("grasp warning drift must be positive and below loss drift")
    if ARGS.contact_threshold_n < 0.0:
        raise ValueError("--contact-threshold-n must be non-negative")
    if ARGS.video_fps <= 0 or ARGS.camera_width <= 0 or ARGS.camera_height <= 0:
        raise ValueError("video dimensions and frame rate must be positive")
    if ARGS.table_height_offset_m is not None and not math.isfinite(
        ARGS.table_height_offset_m
    ):
        raise ValueError("--table-height-offset-m must be finite")


def resolved_table_height_offset_m() -> float:
    if ARGS.table_height_offset_m is not None:
        return float(ARGS.table_height_offset_m)
    if ARGS.variant == "vanilla":
        return VANILLA_TABLE_HEIGHT_OFFSET_M
    return 0.0


def git_text(*args: str) -> str:
    result = subprocess.run(
        ("git", *args), cwd=REPO_ROOT, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise EvaluationFailure("refusing to write an empty samples.csv")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_cfg() -> SimToolRealTacMapScrapePoseEnvCfg:
    cfg = SimToolRealTacMapScrapePoseEnvCfg()
    cfg.seed = ARGS.seed
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 300.0
    cfg.assets.handle_head_types = (ARGS.tool_type,)
    cfg.assets.num_assets_per_type = 1
    cfg.assets.shuffle_assets = False

    tactile = ARGS.variant == "tactile"
    cfg.use_tacmap = tactile
    cfg.enable_vbts = tactile
    cfg.enable_tactile = tactile
    cfg.include_tacmap_in_policy = tactile
    cfg.tacmap_history_len = 5
    cfg.tacmap_policy_include_depth = False
    cfg.obs.obs_list = BASE_OBS + (("tacmap",) if tactile else ())

    cfg.table_pitch_roll_range_deg = float(ARGS.table_angle_range_deg)
    cfg.reset.table_reset_pitch_roll_range_deg = float(ARGS.table_angle_range_deg)
    cfg.reset.table_reset_z_range = float(ARGS.table_height_range_m)
    cfg.reset.reset_position_noise_x = 0.0
    cfg.reset.reset_position_noise_y = 0.0
    cfg.reset.reset_position_noise_z = 0.0
    cfg.reset.reset_dof_pos_random_interval_arm = 0.0
    cfg.reset.reset_dof_pos_random_interval_fingers = 0.0
    cfg.reset.reset_dof_vel_random_interval = 0.0

    dr = cfg.domain_randomization
    dr.use_obs_delay = False
    dr.use_action_delay = False
    dr.use_object_state_delay_noise = False
    dr.joint_velocity_obs_noise_std = 0.0
    dr.force_scale = 0.0
    dr.torque_scale = 0.0
    dr.force_prob_range = (1.0e-12, 1.0e-12)
    dr.torque_prob_range = (1.0e-12, 1.0e-12)

    cfg.enable_tool_table_contact_force_reward = True
    cfg.target_contact_normal_force = float(ARGS.target_force_n)
    cfg.contact_force_use_control_interval_average = True
    cfg.tool_table_contact_sensor_update_period = 0.0
    cfg.tool_table_contact_sensor_history_len = int(cfg.decimation)
    cfg.tool_table_contact_sensor_force_threshold = 0.0
    cfg.termination.max_consecutive_successes = 0
    return cfg


def make_player(inner, checkpoint: Path, policy_config: Path) -> RlPlayer:
    if not checkpoint.is_file():
        raise EvaluationFailure(f"checkpoint does not exist: {checkpoint}")
    if not policy_config.is_file():
        raise EvaluationFailure(f"policy config does not exist: {policy_config}")
    return RlPlayer(
        num_observations=inner.cfg.observation_space,
        num_actions=inner.cfg.action_space,
        config_path=str(policy_config),
        checkpoint_path=str(checkpoint),
        device=str(inner.device),
        num_envs=1,
        coefficient_id=ARGS.policy_coef_id,
    )


def verify_joint_impedance(inner) -> dict:
    actuator = inner.robot.actuators.get("arm")
    if actuator is None:
        raise EvaluationFailure("robot has no 'arm' actuator")
    actuator_type = type(actuator).__name__
    if "ImplicitActuator" not in actuator_type:
        raise EvaluationFailure(
            f"arm actuator must be an implicit joint-position actuator, got {actuator_type}"
        )
    names = tuple(actuator.joint_names)
    if len(names) != 7:
        raise EvaluationFailure(f"arm actuator covers {len(names)} joints instead of 7")
    stiffness = actuator.stiffness[0].detach().cpu().numpy()
    damping = actuator.damping[0].detach().cpu().numpy()
    if (
        stiffness.shape != (7,)
        or damping.shape != (7,)
        or not np.isfinite(stiffness).all()
        or not np.isfinite(damping).all()
        or np.any(stiffness <= 0.0)
        or np.any(damping <= 0.0)
    ):
        raise EvaluationFailure("arm impedance stiffness/damping are not finite and positive")
    for index, name in enumerate(names):
        expected_stiffness = float(scene_utils.ARM_JOINT_STIFFNESS[name])
        expected_damping = float(scene_utils.ARM_JOINT_DAMPING[name])
        if not np.isclose(stiffness[index], expected_stiffness, atol=1.0e-3):
            raise EvaluationFailure(
                f"{name} stiffness {stiffness[index]} != {expected_stiffness}"
            )
        if not np.isclose(damping[index], expected_damping, atol=1.0e-3):
            raise EvaluationFailure(
                f"{name} damping {damping[index]} != {expected_damping}"
            )
    if inner._cur_targets.shape != (1, inner.cfg.action_space):
        raise EvaluationFailure(
            f"joint-position target buffer has invalid shape {tuple(inner._cur_targets.shape)}"
        )
    if not torch.isfinite(inner._cur_targets[:, inner._arm_joint_ids]).all():
        raise EvaluationFailure("arm joint-position targets contain NaN or Inf")
    return {
        "type": actuator_type,
        "command": "joint_position_target",
        "policy_action": "velocity_delta_accumulated_to_joint_position",
        "joint_names": list(names),
        "stiffness": stiffness.tolist(),
        "damping": damping.tolist(),
        "dof_speed_scale": float(inner.cfg.action.dof_speed_scale),
        "arm_moving_average": float(inner.cfg.action.arm_moving_average),
        "control_hz": 1.0 / float(inner.step_dt),
        "physics_hz": 1.0 / float(inner.physics_dt),
    }


def table_state(inner) -> tuple[torch.Tensor, torch.Tensor]:
    quaternion = getattr(inner, "_table_quat_wxyz_per_env", inner.table.data.root_quat_w)
    return table_top_state(inner.table.data.root_pos_w, quaternion)


def raise_table_after_acquisition(inner) -> None:
    """Raise the table only after the tool is held safely above it."""
    height_offset = resolved_table_height_offset_m()
    if height_offset == 0.0:
        return
    pose = torch.cat(
        (inner.table.data.root_pos_w.clone(), inner.table.data.root_quat_w.clone()),
        dim=-1,
    )
    pose[:, 2] += height_offset
    inner.table.write_root_pose_to_sim(pose)
    inner.table.write_root_velocity_to_sim(
        torch.zeros(inner.num_envs, 6, device=inner.device)
    )
    inner._table_z_per_env += height_offset
    print(
        f"[table] raised {height_offset:.3f} m after verified grasp; "
        f"contact table z={pose[0, 2].item():.3f} m"
    )


def pickup_goal_at_initial_table(
    inner, future_goal: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Translate a raised-table contact goal back to the reset table height."""
    shift = torch.zeros_like(inner.table.data.root_pos_w)
    shift[:, 2] = resolved_table_height_offset_m()
    return {
        "pose": future_goal["pose"] - torch.cat(
            (shift, torch.zeros_like(future_goal["pose"][:, 3:])), dim=-1
        ),
        "anchor": future_goal["anchor"] - shift,
        "edge_yaw": future_goal["edge_yaw"],
    }


def set_goal(
    inner,
    nominal_pose: torch.Tensor,
    nominal_anchor: torch.Tensor,
    normal: torch.Tensor,
    offset_m: float,
) -> torch.Tensor:
    pose = nominal_pose.clone()
    pose[:, :3] += float(offset_m) * normal
    anchor = nominal_anchor + float(offset_m) * normal
    inner.goal_viz.write_root_pose_to_sim(pose)
    inner.goal_viz.write_root_velocity_to_sim(
        torch.zeros(1, 6, device=inner.device)
    )
    inner._scrape_edge_anchor_w.copy_(anchor)
    inner._scrape_target_contact_normal_force.fill_(float(ARGS.target_force_n))
    return pose


def sample_pose_sequence(inner) -> list[dict[str, torch.Tensor]]:
    table_quaternion = getattr(
        inner, "_table_quat_wxyz_per_env", inner.table.data.root_quat_w
    )
    common = {
        "table_pos_w": inner.table.data.root_pos_w,
        "table_quat_wxyz": table_quaternion,
        "x_tip": inner._scrape_x_tip_per_env,
        "y_center": inner._scrape_y_center_per_env,
        "z_contact": inner._scrape_z_contact_per_env,
        "xy_half_range": tuple(float(v) for v in inner.cfg.edge_contact_xy_range_m),
        "edge_yaw_range_rad": math.radians(float(inner.cfg.edge_contact_yaw_range_deg)),
        "tilt_range_rad": tuple(
            math.radians(float(v)) for v in inner.cfg.edge_tilt_range_deg
        ),
        "device": inner.device,
    }
    _, table_normal = table_state(inner)
    env_origin_z = inner.scene.env_origins[:, 2]
    future_table_shift = torch.zeros_like(inner.table.data.root_pos_w)
    future_table_shift[:, 2] = resolved_table_height_offset_m()
    enforce_vanilla_range = ARGS.variant == "vanilla"
    poses: list[dict[str, torch.Tensor]] = []
    edge_yaw = None
    for pose_index in range(ARGS.poses_per_sequence):
        rejected_z_ranges: list[tuple[float, float]] = []
        for attempt in range(100):
            pos, quaternion, anchor, sampled_yaw = sample_edge_contact_goal_pose(
                **common, edge_yaw=edge_yaw
            )
            # Sample against the collision-free reset table, then translate the
            # goals to the table pose that will be applied after grasp acquisition.
            pos = pos + future_table_shift
            anchor = anchor + future_table_shift
            local_z = pos[:, 2] - env_origin_z
            min_virtual_z = local_z + min(SWEEP_OFFSETS_M) * table_normal[:, 2]
            pickup_target_z = (
                local_z
                - resolved_table_height_offset_m()
                + ACQUISITION_LIFT_OFFSET_M * table_normal[:, 2]
            )
            min_virtual_z = torch.minimum(min_virtual_z, pickup_target_z)
            max_offset = (
                ACQUISITION_LIFT_OFFSET_M
                if pose_index == 0
                else max(SWEEP_OFFSETS_M)
            )
            max_virtual_z = local_z + max_offset * table_normal[:, 2]
            max_virtual_z = torch.maximum(max_virtual_z, pickup_target_z)
            z_in_range = bool(
                (
                    (min_virtual_z >= VANILLA_TARGET_Z_RANGE_M[0])
                    & (max_virtual_z <= VANILLA_TARGET_Z_RANGE_M[1])
                ).all()
            )
            if enforce_vanilla_range and not z_in_range:
                rejected_z_ranges.append(
                    (float(min_virtual_z[0].item()), float(max_virtual_z[0].item()))
                )
                continue
            if pose_index == 0:
                accepted = True
            else:
                displacement = torch.linalg.vector_norm(anchor - poses[-1]["anchor"])
                accepted = float(displacement.item()) >= ARGS.minimum_pose_translation_m
            if accepted:
                edge_yaw = sampled_yaw
                poses.append(
                    {
                        "pose": torch.cat((pos, quaternion), dim=-1).clone(),
                        "anchor": anchor.clone(),
                        "edge_yaw": sampled_yaw.clone(),
                    }
                )
                break
        else:
            if enforce_vanilla_range and rejected_z_ranges:
                observed_min = min(value[0] for value in rejected_z_ranges)
                observed_max = max(value[1] for value in rejected_z_ranges)
                raise EvaluationFailure(
                    "could not sample a target inside the vanilla policy z range "
                    f"{VANILLA_TARGET_Z_RANGE_M} m after 100 attempts; candidate "
                    f"virtual-target envelope was {observed_min:.3f}–"
                    f"{observed_max:.3f} m. Increase --table-height-offset-m."
                )
            raise EvaluationFailure(
                "could not sample a sufficiently separated feasible target pose "
                f"after 100 attempts (minimum {ARGS.minimum_pose_translation_m:.3f} m)"
            )
    if enforce_vanilla_range:
        local_z = torch.cat([item["pose"][:, 2] - env_origin_z for item in poses])
        pickup_target_z = float(
            local_z[0].item()
            - resolved_table_height_offset_m()
            + ACQUISITION_LIFT_OFFSET_M * table_normal[0, 2].item()
        )
        minimum_z = min(
            float(
                (local_z + min(SWEEP_OFFSETS_M) * table_normal[0, 2])
                .min()
                .item()
            ),
            pickup_target_z,
        )
        maximum_z = max(
            float(
                local_z[0].item()
                + ACQUISITION_LIFT_OFFSET_M * table_normal[0, 2].item()
            ),
            float(
                (
                    local_z[1:] + max(SWEEP_OFFSETS_M) * table_normal[0, 2]
                )
                .max()
                .item()
            ),
            pickup_target_z,
        )
        print(
            "[targets] vanilla policy virtual-target z envelope "
            f"{minimum_z:.3f}–{maximum_z:.3f} m "
            f"(trained range {VANILLA_TARGET_Z_RANGE_M[0]:.2f}–"
            f"{VANILLA_TARGET_Z_RANGE_M[1]:.2f} m)"
        )
    return poses


def quaternion_error_deg(current: torch.Tensor, target: torch.Tensor) -> float:
    dot = torch.abs((current * target).sum(dim=-1)).clamp(0.0, 1.0)
    return float(torch.rad2deg(2.0 * torch.acos(dot))[0].item())


def termination_description(inner, terminated: bool, truncated: bool) -> str:
    labels = []
    for name, mask in getattr(inner, "_termination_reasons", {}).items():
        try:
            if bool(mask[0]):
                labels.append(str(name))
        except (IndexError, TypeError):
            continue
    if labels:
        return ",".join(labels)
    if terminated:
        return "terminated"
    if truncated:
        return "truncated"
    return "unknown"


def measure_row(
    inner,
    *,
    global_step: int,
    sequence: int,
    phase: str,
    stage: str,
    stage_step: int,
    pose_index: int,
    transition_index: int,
    offset_m: float,
    target_pose: torch.Tensor,
    target_anchor: torch.Tensor,
    baseline_relative: torch.Tensor | None,
) -> dict:
    sensor = getattr(inner, "_tool_table_contact_sensor", None)
    if sensor is None:
        raise EvaluationFailure("tool-table ContactSensor is unavailable")
    if getattr(inner, "_tool_table_contact_sensor_failed", False):
        raise EvaluationFailure(
            "tool-table ContactSensor failed: "
            f"{getattr(inner, '_tool_table_contact_sensor_error', '')}"
        )
    history = getattr(sensor.data, "force_matrix_w_history", None)
    if history is None or history.shape[1] < int(inner.cfg.decimation):
        raise EvaluationFailure(
            "pair-filtered contact-force history is unavailable or shorter than decimation"
        )

    top, normal = table_state(inner)
    object_pos = inner.object.data.root_pos_w
    object_quaternion = inner.object.data.root_quat_w
    edge_points = edge_contact_points_w(
        object_pos,
        object_quaternion,
        inner._scrape_x_tip_per_env,
        inner._scrape_y_min_per_env,
        inner._scrape_y_max_per_env,
        inner._scrape_z_contact_per_env,
    )
    signed_edge_distances = ((edge_points - top.unsqueeze(1)) * normal.unsqueeze(1)).sum(
        dim=-1
    )
    edge_midpoint = edge_points[:, 1]
    target_tangent_error = edge_midpoint - target_anchor
    target_tangent_error -= (
        target_tangent_error * normal
    ).sum(dim=-1, keepdim=True) * normal
    linear_velocity = inner.object.data.root_lin_vel_w
    angular_velocity = inner.object.data.root_ang_vel_w
    normal_speed = (linear_velocity * normal).sum(dim=-1, keepdim=True)
    tangent_velocity = linear_velocity - normal_speed * normal
    palm = inner.robot.data.body_link_pos_w[:, inner._palm_body_id]
    palm_quaternion = inner.robot.data.body_link_quat_w[:, inner._palm_body_id]
    relative, _ = subtract_frame_transforms(
        palm, palm_quaternion, object_pos, object_quaternion
    )
    grasp_drift = (
        float(torch.linalg.vector_norm(relative - baseline_relative, dim=-1)[0].item())
        if baseline_relative is not None
        else 0.0
    )
    fingertip_count = int((inner._curr_fingertip_distances[0] < 0.12).sum().item())
    arm_ids = inner._arm_joint_ids
    joint_positions = inner.robot.data.joint_pos[0, arm_ids]
    joint_targets = inner._cur_targets[0, arm_ids]
    joint_errors = joint_targets - joint_positions
    table_quaternion = getattr(
        inner, "_table_quat_wxyz_per_env", inner.table.data.root_quat_w
    )

    row = {
        "global_step": global_step,
        "time_s": global_step * float(inner.step_dt),
        "sequence": sequence,
        "phase": phase,
        "stage": stage,
        "stage_step": stage_step,
        "pose_index": pose_index,
        "transition_index": transition_index,
        "stage_result": "running",
        "failure_message": "",
        "force_safety_violation_steps": 0,
        "grasp_warning": 0,
        "commanded_offset_m": float(offset_m),
        "virtual_depth_m": float(-offset_m),
        "target_force_n": float(ARGS.target_force_n),
        "force_raw_n": float(inner._scrape_table_normal_force_raw[0].item()),
        "force_interval_n": float(inner._scrape_table_normal_force_interval[0].item()),
        "force_ema_n": float(inner._scrape_table_normal_force[0].item()),
        "contact": int(
            float(inner._scrape_table_normal_force_interval[0].item())
            >= ARGS.contact_threshold_n
        ),
        "actual_edge_signed_distance_m": float(signed_edge_distances[0, 1].item()),
        "actual_edge_max_abs_distance_m": float(
            torch.abs(signed_edge_distances[0]).max().item()
        ),
        "edge_target_tangent_error_m": float(
            torch.linalg.vector_norm(target_tangent_error, dim=-1)[0].item()
        ),
        "keypoint_pose_error_m": float(inner._keypoints_max_dist[0].item()),
        "root_position_error_m": float(
            torch.linalg.vector_norm(object_pos - target_pose[:, :3], dim=-1)[0].item()
        ),
        "root_orientation_error_deg": quaternion_error_deg(
            object_quaternion, target_pose[:, 3:]
        ),
        "tool_x_m": float(object_pos[0, 0].item()),
        "tool_y_m": float(object_pos[0, 1].item()),
        "tool_z_m": float(object_pos[0, 2].item()),
        "tool_qw": float(object_quaternion[0, 0].item()),
        "tool_qx": float(object_quaternion[0, 1].item()),
        "tool_qy": float(object_quaternion[0, 2].item()),
        "tool_qz": float(object_quaternion[0, 3].item()),
        "tool_linear_speed_mps": float(
            torch.linalg.vector_norm(linear_velocity, dim=-1)[0].item()
        ),
        "tool_angular_speed_radps": float(
            torch.linalg.vector_norm(angular_velocity, dim=-1)[0].item()
        ),
        "tool_tangential_speed_mps": float(
            torch.linalg.vector_norm(tangent_velocity, dim=-1)[0].item()
        ),
        "fingertip_count": fingertip_count,
        "palm_tool_drift_m": grasp_drift,
        "arm_joint_error_l2_rad": float(torch.linalg.vector_norm(joint_errors).item()),
        "arm_joint_error_max_rad": float(torch.abs(joint_errors).max().item()),
        "table_x_m": float(inner.table.data.root_pos_w[0, 0].item()),
        "table_y_m": float(inner.table.data.root_pos_w[0, 1].item()),
        "table_z_m": float(inner.table.data.root_pos_w[0, 2].item()),
        "table_qw": float(table_quaternion[0, 0].item()),
        "table_qx": float(table_quaternion[0, 1].item()),
        "table_qy": float(table_quaternion[0, 2].item()),
        "table_qz": float(table_quaternion[0, 3].item()),
        "table_normal_x": float(normal[0, 0].item()),
        "table_normal_y": float(normal[0, 1].item()),
        "table_normal_z": float(normal[0, 2].item()),
    }
    for index in range(7):
        row[f"arm_joint_{index + 1}_position_rad"] = float(joint_positions[index].item())
        row[f"arm_joint_{index + 1}_target_rad"] = float(joint_targets[index].item())
        row[f"arm_joint_{index + 1}_error_rad"] = float(joint_errors[index].item())
    numeric = [
        value
        for value in row.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if not np.isfinite(np.asarray(numeric, dtype=np.float64)).all():
        raise EvaluationFailure("recorded state contains NaN or Inf")
    return row


def stage_stability(
    rows: list[dict],
    limits: StabilityLimits | None = None,
    window_steps: int | None = None,
) -> tuple[bool, dict[str, float]]:
    required_steps = (
        ARGS.stable_window_steps if window_steps is None else int(window_steps)
    )
    window = rows[-required_steps:]
    if len(window) < required_steps:
        return False, {}
    positions = np.asarray(
        [[row["tool_x_m"], row["tool_y_m"], row["tool_z_m"]] for row in window]
    )
    quaternions = np.asarray(
        [[row["tool_qw"], row["tool_qx"], row["tool_qy"], row["tool_qz"]] for row in window]
    )
    linear = np.asarray([row["tool_linear_speed_mps"] for row in window])
    angular = np.asarray([row["tool_angular_speed_radps"] for row in window])
    return pose_window_is_stable(
        positions,
        quaternions,
        linear,
        angular,
        StabilityLimits() if limits is None else limits,
    )


class VideoRecorder:
    def __init__(self, inner, output_dir: Path):
        self.inner = inner
        self.enabled = bool(ARGS.video)
        self.camera: Camera | None = None
        self.raw_path = output_dir / "_camera_raw.mp4"
        self.output_path = output_dir / f"virtual_pose_offset_{ARGS.variant}.mp4"
        self.writer = None
        self.row_indices: list[int] = []
        self.capture_every = max(
            1, round((1.0 / ARGS.video_fps) / float(inner.step_dt))
        )
        self.policy_steps = 0
        if not self.enabled:
            return
        camera_cfg = CameraCfg(
            prim_path="/World/VirtualOffsetCamera",
            update_period=0,
            height=ARGS.camera_height,
            width=ARGS.camera_width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 100.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 10.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                convention="opengl",
            ),
        )
        self.camera = Camera(cfg=camera_cfg)
        inner.sim.reset()
        origin = inner.scene.env_origins[0]
        eye = origin + torch.tensor([0.5, -1.5, 1.2], device=inner.device)
        target = origin + torch.tensor([0.0, 0.4, 0.5], device=inner.device)
        self.camera.set_world_poses_from_view(eye.unsqueeze(0), target.unsqueeze(0))
        self.writer = imageio.get_writer(
            self.raw_path,
            fps=ARGS.video_fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )

    def capture(self, row_index: int) -> None:
        if not self.enabled:
            return
        self.policy_steps += 1
        if self.policy_steps % self.capture_every != 0:
            return
        assert self.camera is not None and self.writer is not None
        self.camera.update(self.capture_every * float(self.inner.step_dt))
        rgb = self.camera.data.output.get("rgb")
        if rgb is None or rgb.shape[0] == 0:
            raise EvaluationFailure("record camera returned no RGB image")
        frame = rgb[0].detach().cpu().numpy()[:, :, :3]
        if frame.shape[:2] != (ARGS.camera_height, ARGS.camera_width):
            raise EvaluationFailure(f"record camera returned unexpected shape {frame.shape}")
        if not np.isfinite(frame).all() or float(frame.mean()) <= 0.0:
            raise EvaluationFailure("record camera frame is blank or non-finite")
        self.writer.append_data(frame.astype(np.uint8))
        self.row_indices.append(row_index)

    def close_raw(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def compose(self, rows: list[dict]) -> None:
        if not self.enabled:
            return
        self.close_raw()
        if not self.row_indices or not self.raw_path.is_file():
            raise EvaluationFailure("no video frames were captured")
        reader = imageio.get_reader(self.raw_path)
        writer = imageio.get_writer(
            self.output_path,
            fps=ARGS.video_fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )
        count = 0
        try:
            for frame, row_index in zip(reader, self.row_indices):
                camera_frame = cv2.resize(
                    frame[:, :, :3],
                    (ARGS.camera_width, ARGS.camera_height),
                    interpolation=cv2.INTER_AREA,
                )
                panel = render_telemetry_panel(rows, row_index)
                writer.append_data(np.concatenate((camera_frame, panel), axis=1))
                count += 1
        finally:
            reader.close()
            writer.close()
        if count != len(self.row_indices):
            raise EvaluationFailure(
                f"video frame/telemetry mismatch: {count} != {len(self.row_indices)}"
            )
        self.raw_path.unlink()


def draw_series(
    canvas: np.ndarray,
    rect: tuple[int, int, int, int],
    x: np.ndarray,
    series: list[tuple[np.ndarray, tuple[int, int, int], int]],
    y_limits: tuple[float, float],
    title: str,
    ylabel: str,
    markers_x: list[float] | None = None,
) -> None:
    left, top, right, bottom = rect
    cv2.rectangle(canvas, (left, top), (right, bottom), (210, 215, 220), 1)
    cv2.putText(canvas, title, (left, top - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (25, 30, 35), 2)
    cv2.putText(canvas, ylabel, (left + 8, top + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 75, 80), 1)
    if x.size < 2:
        return
    x_min, x_max = float(x[0]), float(x[-1])
    if x_max <= x_min:
        x_max = x_min + 1.0
    y_min, y_max = y_limits
    if y_max <= y_min:
        y_max = y_min + 1.0
    for marker in markers_x or []:
        px = int(left + (float(marker) - x_min) / (x_max - x_min) * (right - left))
        if left <= px <= right:
            cv2.line(canvas, (px, top), (px, bottom), (215, 120, 35), 1, cv2.LINE_AA)
    for values, color, thickness in series:
        finite = np.isfinite(values)
        points = []
        for xi, yi, valid in zip(x, values, finite):
            if not valid:
                if len(points) >= 2:
                    cv2.polylines(canvas, [np.asarray(points)], False, color, thickness, cv2.LINE_AA)
                points = []
                continue
            px = int(left + (float(xi) - x_min) / (x_max - x_min) * (right - left))
            py = int(bottom - (float(yi) - y_min) / (y_max - y_min) * (bottom - top))
            points.append((px, int(np.clip(py, top, bottom))))
        if len(points) >= 2:
            cv2.polylines(canvas, [np.asarray(points)], False, color, thickness, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"{y_min:.1f}",
        (left + 4, bottom - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (80, 85, 90),
        1,
    )
    cv2.putText(
        canvas,
        f"{y_max:.1f}",
        (left + 4, top + 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (80, 85, 90),
        1,
    )


def render_telemetry_panel(rows: list[dict], row_index: int) -> np.ndarray:
    width, height = 640, ARGS.camera_height
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    current = rows[row_index]
    first = max(0, row_index - int(round(10.0 / float(current["time_s"] / max(current["global_step"], 1)))))
    history = rows[first : row_index + 1]
    time = np.asarray([row["time_s"] for row in history])
    depth = np.asarray([1000.0 * row["virtual_depth_m"] for row in history])
    actual_depth = np.asarray(
        [-1000.0 * row["actual_edge_signed_distance_m"] for row in history]
    )
    raw_force = np.asarray([row["force_raw_n"] for row in history])
    interval_force = np.asarray([row["force_interval_n"] for row in history])
    ema_force = np.asarray([row["force_ema_n"] for row in history])
    target = np.full(time.shape, float(ARGS.target_force_n))
    markers = [
        float(history[index]["time_s"])
        for index in range(1, len(history))
        if (
            history[index]["phase"],
            history[index]["stage"],
            history[index]["pose_index"],
        )
        != (
            history[index - 1]["phase"],
            history[index - 1]["stage"],
            history[index - 1]["pose_index"],
        )
    ]

    cv2.putText(
        canvas,
        f"Sequence {current['sequence'] + 1}  Pose {current['pose_index'] + 1}",
        (30, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.88,
        (20, 25, 30),
        2,
    )
    cv2.putText(
        canvas,
        f"{current['phase']} / {current['stage']}",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (60, 65, 70),
        2,
    )
    status_color = (25, 135, 65) if current["contact"] else (210, 45, 45)
    cv2.putText(
        canvas,
        f"Contact: {'YES' if current['contact'] else 'NO'}    "
        f"Force: {current['force_interval_n']:.2f} N",
        (30, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        status_color,
        2,
    )
    draw_series(
        canvas,
        (45, 155, 610, 360),
        time,
        [
            (depth, (20, 95, 190), 3),
            (actual_depth, (30, 155, 90), 2),
        ],
        (-4.0, 4.0),
        "Virtual and actual edge depth",
        "mm   blue=command, green=actual",
        markers,
    )
    force_max = max(8.0, min(ARGS.maximum_force_n, float(np.nanmax(raw_force)) * 1.1))
    draw_series(
        canvas,
        (45, 430, 610, 665),
        time,
        [
            (raw_force, (185, 185, 185), 1),
            (interval_force, (20, 95, 190), 3),
            (ema_force, (30, 155, 90), 2),
            (target, (210, 70, 45), 1),
        ],
        (0.0, force_max),
        "Tool-table normal force",
        "N   gray=raw, blue=interval, green=EMA, red=4 N",
        markers,
    )
    if current["stage_result"] == "failure":
        cv2.rectangle(canvas, (0, height - 46), (width, height), (205, 35, 35), -1)
        message = str(current["failure_message"])[:72]
        cv2.putText(
            canvas,
            f"FAILED: {message}",
            (18, height - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
    return canvas


def policy_step(
    env,
    inner,
    player: RlPlayer,
    *,
    nominal_pose: torch.Tensor,
    nominal_anchor: torch.Tensor,
    normal: torch.Tensor,
    offset_m: float,
) -> tuple[torch.Tensor, bool, bool, torch.Tensor]:
    target_pose = set_goal(inner, nominal_pose, nominal_anchor, normal, offset_m)
    observation = inner._get_observations()
    action = player.get_normalized_action(
        observation["policy"], deterministic_actions=True
    )
    observation, _, terminated, truncated, _ = env.step(action.to(inner.device))
    return observation, bool(terminated[0]), bool(truncated[0]), target_pose


def current_palm_tool_relative(inner) -> torch.Tensor:
    palm = inner.robot.data.body_link_pos_w[:, inner._palm_body_id]
    palm_quaternion = inner.robot.data.body_link_quat_w[:, inner._palm_body_id]
    relative, _ = subtract_frame_transforms(
        palm,
        palm_quaternion,
        inner.object.data.root_pos_w,
        inner.object.data.root_quat_w,
    )
    return relative.detach().clone()


def acquire_grasp(
    env,
    inner,
    player: RlPlayer,
    first_pose: dict[str, torch.Tensor],
    rows: list[dict],
    recorder: VideoRecorder,
    sequence: int,
    global_step: int,
) -> tuple[torch.Tensor, int, dict]:
    _, normal = table_state(inner)
    relative_window: deque[torch.Tensor] = deque(maxlen=ARGS.stable_window_steps)
    over_force_steps = 0
    best = {"lift_m": -float("inf"), "fingertips": 0, "relative_drift_m": float("inf")}
    lifted_offset = ACQUISITION_LIFT_OFFSET_M
    minimum_lift = POST_RAISE_CLEARANCE_M
    for step in range(ARGS.acquisition_steps):
        _, terminated, truncated, target_pose = policy_step(
            env,
            inner,
            player,
            nominal_pose=first_pose["pose"],
            nominal_anchor=first_pose["anchor"],
            normal=normal,
            offset_m=lifted_offset,
        )
        global_step += 1
        if terminated or truncated:
            raise EvaluationFailure(
                f"environment ended during grasp acquisition at step {step + 1}: "
                f"{termination_description(inner, terminated, truncated)}"
            )
        row = measure_row(
            inner,
            global_step=global_step,
            sequence=sequence,
            phase="acquisition",
            stage="lifted_pose_1",
            stage_step=step,
            pose_index=0,
            transition_index=-1,
            offset_m=lifted_offset,
            target_pose=target_pose,
            target_anchor=first_pose["anchor"] + lifted_offset * normal,
            baseline_relative=None,
        )
        rows.append(row)
        recorder.capture(len(rows) - 1)
        force = float(row["force_interval_n"])
        over_force_steps, unsafe_force = update_persistent_violation(
            force,
            threshold=ARGS.maximum_force_n,
            previous_steps=over_force_steps,
            required_steps=ARGS.safety_force_consecutive_steps,
            immediate_threshold=ARGS.immediate_force_limit_n,
        )
        row["force_safety_violation_steps"] = over_force_steps
        if unsafe_force:
            raise EvaluationFailure(
                f"force {force:.3f} N violates safety gate during acquisition "
                f"({over_force_steps} consecutive steps above "
                f"{ARGS.maximum_force_n:.1f} N; immediate limit "
                f"{ARGS.immediate_force_limit_n:.1f} N)"
            )
        lift = float(row["actual_edge_signed_distance_m"])
        fingertips = int(row["fingertip_count"])
        relative = current_palm_tool_relative(inner)
        best["lift_m"] = max(float(best["lift_m"]), lift)
        best["fingertips"] = max(int(best["fingertips"]), fingertips)
        if lift >= minimum_lift and fingertips >= 2:
            relative_window.append(relative[0])
            if len(relative_window) == ARGS.stable_window_steps:
                stack = torch.stack(tuple(relative_window))
                drift = float(
                    torch.linalg.vector_norm(stack - stack.mean(dim=0), dim=-1)
                    .max()
                    .item()
                )
                best["relative_drift_m"] = min(float(best["relative_drift_m"]), drift)
                if drift <= 0.005:
                    row["stage_result"] = "verified"
                    baseline = stack.mean(dim=0, keepdim=True)
                    return baseline, global_step, {
                        "steps": step + 1,
                        "edge_lift_m": lift,
                        "minimum_edge_lift_m": minimum_lift,
                        "fingertip_count": fingertips,
                        "relative_position_variation_m": drift,
                        "tool_linear_speed_mps": float(row["tool_linear_speed_mps"]),
                        "tool_angular_speed_radps": float(
                            row["tool_angular_speed_radps"]
                        ),
                        "maximum_interval_force_n": max(
                            float(item["force_interval_n"])
                            for item in rows
                            if int(item["sequence"]) == sequence
                            and item["phase"] == "acquisition"
                        ),
                    }
        else:
            relative_window.clear()
    raise EvaluationFailure(
        "policy did not produce a mechanically verified grasp: "
        f"best edge lift={best['lift_m']:.4f} m, fingertips={best['fingertips']}, "
        f"relative variation={best['relative_drift_m']:.4f} m"
    )


def run_stage(
    env,
    inner,
    player: RlPlayer,
    *,
    rows: list[dict],
    recorder: VideoRecorder,
    sequence: int,
    phase: str,
    stage: str,
    pose_index: int,
    transition_index: int,
    nominal_pose: torch.Tensor,
    nominal_anchor: torch.Tensor,
    normal: torch.Tensor,
    offset_m: float,
    baseline_relative: torch.Tensor,
    global_step: int,
    timeout_steps: int | None = None,
    stability_limits: StabilityLimits | None = None,
    stability_window_steps: int | None = None,
) -> tuple[dict, int]:
    stage_rows: list[dict] = []
    fingertip_loss_steps = 0
    over_force_steps = 0
    stability_metrics: dict[str, float] = {}
    stage_timeout_steps = (
        ARGS.stage_timeout_steps if timeout_steps is None else int(timeout_steps)
    )
    for step in range(stage_timeout_steps):
        _, terminated, truncated, target_pose = policy_step(
            env,
            inner,
            player,
            nominal_pose=nominal_pose,
            nominal_anchor=nominal_anchor,
            normal=normal,
            offset_m=offset_m,
        )
        global_step += 1
        if terminated or truncated:
            raise EvaluationFailure(
                f"environment ended during {phase}/{stage} at step {step + 1}: "
                f"{termination_description(inner, terminated, truncated)}"
            )
        target_anchor = nominal_anchor + float(offset_m) * normal
        row = measure_row(
            inner,
            global_step=global_step,
            sequence=sequence,
            phase=phase,
            stage=stage,
            stage_step=step,
            pose_index=pose_index,
            transition_index=transition_index,
            offset_m=offset_m,
            target_pose=target_pose,
            target_anchor=target_anchor,
            baseline_relative=baseline_relative,
        )
        rows.append(row)
        stage_rows.append(row)
        recorder.capture(len(rows) - 1)

        over_force_steps, unsafe_force = update_persistent_violation(
            float(row["force_interval_n"]),
            threshold=ARGS.maximum_force_n,
            previous_steps=over_force_steps,
            required_steps=ARGS.safety_force_consecutive_steps,
            immediate_threshold=ARGS.immediate_force_limit_n,
        )
        row["force_safety_violation_steps"] = over_force_steps
        row["grasp_warning"] = int(
            row["palm_tool_drift_m"] > ARGS.grasp_warning_drift_m
        )
        if unsafe_force:
            raise EvaluationFailure(
                f"force {row['force_interval_n']:.3f} N violates safety gate "
                f"during {phase}/{stage} ({over_force_steps} consecutive steps "
                f"above {ARGS.maximum_force_n:.1f} N; immediate limit "
                f"{ARGS.immediate_force_limit_n:.1f} N)"
            )
        if row["palm_tool_drift_m"] > ARGS.grasp_loss_drift_m:
            raise EvaluationFailure(
                f"palm-tool drift {row['palm_tool_drift_m']:.4f} m indicates grasp loss "
                f"during {phase}/{stage} (limit {ARGS.grasp_loss_drift_m:.4f} m)"
            )
        fingertip_loss_steps = (
            fingertip_loss_steps + 1 if row["fingertip_count"] < 2 else 0
        )
        if fingertip_loss_steps >= 15:
            raise EvaluationFailure(
                f"fewer than two fingertips remained near the tool for "
                f"{fingertip_loss_steps} steps during {phase}/{stage}"
            )
        if step + 1 >= ARGS.minimum_stage_steps:
            stable, stability_metrics = stage_stability(
                stage_rows,
                limits=stability_limits,
                window_steps=stability_window_steps,
            )
            if stable:
                row["stage_result"] = "stable"
                break
    else:
        stable = False
        stage_rows[-1]["stage_result"] = "timeout"

    steady_rows = stage_rows[-min(ARGS.stable_window_steps, len(stage_rows)) :]
    return {
        "phase": phase,
        "stage": stage,
        "pose_index": pose_index,
        "transition_index": transition_index,
        "offset_m": float(offset_m),
        "virtual_depth_m": float(-offset_m),
        "stable": bool(stable),
        "timed_out": not bool(stable),
        "steps": len(stage_rows),
        "duration_s": len(stage_rows) * float(inner.step_dt),
        "steady_force_interval_n": float(
            np.mean([row["force_interval_n"] for row in steady_rows])
        ),
        "steady_force_ema_n": float(np.mean([row["force_ema_n"] for row in steady_rows])),
        "maximum_interval_force_n": float(
            max(row["force_interval_n"] for row in stage_rows)
        ),
        "maximum_palm_tool_drift_m": float(
            max(row["palm_tool_drift_m"] for row in stage_rows)
        ),
        "grasp_warning_ratio": float(
            np.mean([row["grasp_warning"] for row in stage_rows])
        ),
        "final_keypoint_pose_error_m": float(stage_rows[-1]["keypoint_pose_error_m"]),
        "final_root_position_error_m": float(stage_rows[-1]["root_position_error_m"]),
        "final_root_orientation_error_deg": float(
            stage_rows[-1]["root_orientation_error_deg"]
        ),
        "final_edge_signed_distance_m": float(
            stage_rows[-1]["actual_edge_signed_distance_m"]
        ),
        "stability": stability_metrics,
        "row_start": len(rows) - len(stage_rows),
        "row_end": len(rows),
    }, global_step


def estimate_depth_at_force(depth_m: np.ndarray, force_n: np.ndarray, target_n: float) -> float | None:
    order = np.argsort(force_n)
    force_sorted = force_n[order]
    depth_sorted = depth_m[order]
    if force_sorted[0] > target_n or force_sorted[-1] < target_n:
        return None
    unique_force, unique_indices = np.unique(force_sorted, return_index=True)
    if unique_force.size < 2:
        return None
    return float(np.interp(target_n, unique_force, depth_sorted[unique_indices]))


def summarize_sweep(stages: list[dict]) -> tuple[dict, dict]:
    loading = stages[: len(LOADING_OFFSETS_M)]
    offsets = np.asarray([stage["offset_m"] for stage in loading])
    forces = np.asarray([stage["steady_force_interval_n"] for stage in loading])
    stable = np.asarray([stage["stable"] for stage in loading])
    try:
        calibration = select_force_calibration(
            offsets,
            forces,
            stable,
            target_force_n=ARGS.target_force_n,
            maximum_force_n=ARGS.maximum_force_n,
        )
    except ValueError as exc:
        if str(exc) != "no stable, finite, safe force-offset samples are available":
            raise
        raise EvaluationFailure(
            "offset sweep produced no stable, finite, safe samples for force "
            "calibration"
        ) from exc
    valid = stable & np.isfinite(forces)
    rho = (
        spearman_correlation(-offsets[valid], forces[valid])
        if int(valid.sum()) >= 2
        else float("nan")
    )
    span = float(forces[valid].max() - forces[valid].min()) if np.any(valid) else 0.0

    unloading_by_offset = {
        round(float(stage["offset_m"]), 8): stage for stage in stages[len(loading) :]
    }
    hysteresis = []
    for stage in loading:
        other = unloading_by_offset.get(round(float(stage["offset_m"]), 8))
        if stage["stable"] and other is not None and other["stable"]:
            hysteresis.append(
                abs(stage["steady_force_interval_n"] - other["steady_force_interval_n"])
            )
    depth = -offsets[valid]
    valid_forces = forces[valid]
    summary = {
        "loading_stages": loading,
        "unloading_stages": stages[len(loading) :],
        "spearman_rho": rho,
        "force_span_n": span,
        "hysteresis_mae_n": float(np.mean(hysteresis)) if hysteresis else float("nan"),
        "estimated_depth_m": {
            str(force): estimate_depth_at_force(depth, valid_forces, force)
            for force in (2.0, 4.0, 6.0)
        },
        "passed": bool(
            calibration["bracketed"]
            and math.isfinite(rho)
            and rho >= 0.8
            and span >= 4.0
        ),
    }
    return summary, calibration


def run_sequence(
    env,
    inner,
    player: RlPlayer,
    recorder: VideoRecorder,
    rows: list[dict],
    sequence: int,
    global_step: int,
) -> tuple[dict, int]:
    print(f"[sequence {sequence + 1}/{ARGS.sequences}] reset and policy grasp")
    player.reset()
    _, _ = env.reset()
    inner._replay_target_lab_order = None
    inner.cfg.termination.eval_success_tolerance = 0.0
    inner._current_success_tolerance = 0.0
    inner._near_goal_steps.zero_()
    inner._is_success.zero_()
    poses = sample_pose_sequence(inner)
    _, normal = table_state(inner)
    pickup_goal = pickup_goal_at_initial_table(inner, poses[0])
    baseline_relative, global_step, acquisition = acquire_grasp(
        env, inner, player, pickup_goal, rows, recorder, sequence, global_step
    )
    print(f"[sequence {sequence + 1}] post-grasp clearance lift")
    clearance_lift, global_step = run_stage(
        env,
        inner,
        player,
        rows=rows,
        recorder=recorder,
        sequence=sequence,
        phase="clearance_lift",
        stage="raise_tool_before_table",
        pose_index=0,
        transition_index=-1,
        nominal_pose=poses[0]["pose"],
        nominal_anchor=poses[0]["anchor"],
        normal=normal,
        offset_m=ACQUISITION_LIFT_OFFSET_M,
        baseline_relative=baseline_relative,
        global_step=global_step,
        timeout_steps=ARGS.clearance_lift_timeout_steps,
        stability_limits=StabilityLimits(
            position_m=0.01,
            orientation_deg=10.0,
            linear_speed_mps=ACQUISITION_MAX_LINEAR_SPEED_MPS,
            angular_speed_radps=ACQUISITION_MAX_ANGULAR_SPEED_RADPS,
        ),
        stability_window_steps=CLEARANCE_STABILITY_WINDOW_STEPS,
    )
    minimum_pre_raise_lift = (
        resolved_table_height_offset_m() + POST_RAISE_CLEARANCE_M
    )
    if not clearance_lift["stable"]:
        raise EvaluationFailure(
            "tool did not stabilize during the post-grasp clearance lift "
            f"within {ARGS.clearance_lift_timeout_steps} steps"
        )
    if clearance_lift["final_edge_signed_distance_m"] < minimum_pre_raise_lift:
        raise EvaluationFailure(
            "post-grasp clearance lift is too low to raise the table safely: "
            f"{clearance_lift['final_edge_signed_distance_m']:.4f} m < "
            f"{minimum_pre_raise_lift:.4f} m"
        )
    settled_relative = current_palm_tool_relative(inner)
    rebaseline_shift = float(
        torch.linalg.vector_norm(
            settled_relative - baseline_relative, dim=-1
        )[0].item()
    )
    clearance_lift["post_clearance_rebaseline_shift_m"] = rebaseline_shift
    baseline_relative = settled_relative
    print(
        f"[sequence {sequence + 1}] grasp rebaselined after clearance lift "
        f"(settling shift={rebaseline_shift:.4f} m)"
    )
    raise_table_after_acquisition(inner)
    _, normal = table_state(inner)

    print(f"[sequence {sequence + 1}] loading/unloading offset sweep")
    sweep_stages = []
    for stage_index, offset in enumerate(SWEEP_OFFSETS_M):
        stage, global_step = run_stage(
            env,
            inner,
            player,
            rows=rows,
            recorder=recorder,
            sequence=sequence,
            phase="offset_sweep",
            stage=f"sweep_{stage_index:02d}",
            pose_index=0,
            transition_index=-1,
            nominal_pose=poses[0]["pose"],
            nominal_anchor=poses[0]["anchor"],
            normal=normal,
            offset_m=offset,
            baseline_relative=baseline_relative,
            global_step=global_step,
        )
        sweep_stages.append(stage)
    sweep_summary, calibration = summarize_sweep(sweep_stages)
    calibrated_offset = float(calibration["offset_m"])
    print(
        f"[sequence {sequence + 1}] calibrated offset={calibrated_offset * 1000:.3f} mm, "
        f"force={calibration['steady_force_n']:.3f} N, "
        f"bracketed={calibration['bracketed']}"
    )

    hold, global_step = run_stage(
        env,
        inner,
        player,
        rows=rows,
        recorder=recorder,
        sequence=sequence,
        phase="calibrated_hold",
        stage="pose_1_hold",
        pose_index=0,
        transition_index=0,
        nominal_pose=poses[0]["pose"],
        nominal_anchor=poses[0]["anchor"],
        normal=normal,
        offset_m=calibrated_offset,
        baseline_relative=baseline_relative,
        global_step=global_step,
    )

    transitions = []
    print(f"[sequence {sequence + 1}] consecutive pose transitions")
    for pose_index in range(1, len(poses)):
        stage, global_step = run_stage(
            env,
            inner,
            player,
            rows=rows,
            recorder=recorder,
            sequence=sequence,
            phase="pose_transition",
            stage=f"pose_{pose_index}_to_{pose_index + 1}",
            pose_index=pose_index,
            transition_index=pose_index,
            nominal_pose=poses[pose_index]["pose"],
            nominal_anchor=poses[pose_index]["anchor"],
            normal=normal,
            offset_m=calibrated_offset,
            baseline_relative=baseline_relative,
            global_step=global_step,
        )
        segment = rows[stage["row_start"] : stage["row_end"]]
        force_metrics = transition_force_metrics(
            np.asarray([row["force_interval_n"] for row in segment]),
            np.asarray([bool(row["contact"]) for row in segment]),
            control_dt_s=float(inner.step_dt),
            target_force_n=ARGS.target_force_n,
            settle_tolerance_n=1.0,
            settle_window_steps=ARGS.stable_window_steps,
            steady_window_steps=ARGS.stable_window_steps,
        )
        target_displacement = poses[pose_index]["anchor"] - poses[pose_index - 1]["anchor"]
        target_displacement -= (
            target_displacement * normal
        ).sum(dim=-1, keepdim=True) * normal
        stage.update(force_metrics)
        stage["target_tangential_displacement_m"] = float(
            torch.linalg.vector_norm(target_displacement, dim=-1)[0].item()
        )
        stage["mean_tool_tangential_speed_mps"] = float(
            np.mean([row["tool_tangential_speed_mps"] for row in segment])
        )
        transitions.append(stage)

    contact_ratio = float(
        np.mean([transition["contact_maintenance_ratio"] for transition in transitions])
    )
    steady_mae = float(np.mean([transition["steady_force_mae_n"] for transition in transitions]))
    timeout_rate = float(np.mean([transition["timed_out"] for transition in transitions]))
    transition_passed = bool(
        contact_ratio >= 0.9 and steady_mae <= 1.0 and timeout_rate <= 0.25
    )
    table_top, table_normal = table_state(inner)
    sampled_goals = [
        {
            "pose_wxyz": item["pose"][0].detach().cpu().tolist(),
            "edge_anchor_w": item["anchor"][0].detach().cpu().tolist(),
            "edge_yaw_rad": float(item["edge_yaw"][0].item()),
        }
        for item in poses
    ]
    return {
        "sequence": sequence,
        "passed": bool(sweep_summary["passed"] and transition_passed),
        "acquisition": acquisition,
        "clearance_lift": clearance_lift,
        "table": {
            "position_w": inner.table.data.root_pos_w[0].detach().cpu().tolist(),
            "quaternion_wxyz": getattr(
                inner, "_table_quat_wxyz_per_env", inner.table.data.root_quat_w
            )[0]
            .detach()
            .cpu()
            .tolist(),
            "top_w": table_top[0].detach().cpu().tolist(),
            "normal_w": table_normal[0].detach().cpu().tolist(),
        },
        "sampled_goals": sampled_goals,
        "sweep": sweep_summary,
        "calibration": calibration,
        "calibrated_hold": hold,
        "transitions": transitions,
        "transition_summary": {
            "passed": transition_passed,
            "contact_maintenance_ratio": contact_ratio,
            "steady_force_mae_n": steady_mae,
            "timeout_rate": timeout_rate,
        },
    }, global_step


def save_plots(output_dir: Path, sequences: list[dict]) -> None:
    completed = [item for item in sequences if "sweep" in item]
    if not completed:
        return
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(3, len(completed))))
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for color, sequence in zip(colors, completed):
        loading = sequence["sweep"]["loading_stages"]
        depth = [1000.0 * stage["virtual_depth_m"] for stage in loading]
        force = [stage["steady_force_interval_n"] for stage in loading]
        ax.plot(depth, force, marker="o", linewidth=2.0, color=color, label=f"Sequence {sequence['sequence'] + 1}")
    ax.axhline(ARGS.target_force_n, color="black", linestyle="--", linewidth=1.5, label="4 N target")
    ax.set(xlabel="Virtual penetration depth (mm)", ylabel="Steady interval force (N)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"force_depth_response.{suffix}", dpi=200)
    plt.close(fig)

    labels = []
    contact = []
    mae = []
    for sequence in completed:
        for transition in sequence["transitions"]:
            labels.append(f"S{sequence['sequence'] + 1}\nT{transition['transition_index']}")
            contact.append(100.0 * transition["contact_maintenance_ratio"])
            mae.append(transition["steady_force_mae_n"])
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.6), sharex=True)
    axes[0].bar(x, contact, color="#2774AE")
    axes[0].axhline(90.0, color="black", linestyle="--", linewidth=1.2)
    axes[0].set_ylabel("Contact retained (%)")
    axes[0].set_ylim(0.0, 105.0)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[1].bar(x, mae, color="#3A923A")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.2)
    axes[1].set(ylabel="Steady force MAE (N)", xticks=x, xticklabels=labels)
    axes[1].grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"transition_contact_force.{suffix}", dpi=200)
    plt.close(fig)


def main() -> int:
    validate_args()
    torch.manual_seed(ARGS.seed)
    np.random.seed(ARGS.seed)
    checkpoint = (
        Path(ARGS.checkpoint).expanduser().resolve()
        if ARGS.checkpoint
        else DEFAULT_CHECKPOINTS[ARGS.variant].resolve()
    )
    policy_config = Path(ARGS.policy_config).expanduser().resolve()
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{ARGS.variant}"
    output_dir = Path(ARGS.output_root).expanduser().resolve() / run_name
    output_dir.mkdir(parents=True, exist_ok=False)

    cfg = make_cfg()
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "git_commit": git_text("rev-parse", "HEAD"),
        "git_dirty": bool(git_text("status", "--porcelain")),
        "variant": ARGS.variant,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "policy_config": str(policy_config),
        "policy_coefficient_id": ARGS.policy_coef_id,
        "seed": ARGS.seed,
        "tool_type": ARGS.tool_type,
        "observation_fields": list(cfg.obs.obs_list),
        "tacmap_policy_include_depth": bool(cfg.tacmap_policy_include_depth),
        "sequences": ARGS.sequences,
        "poses_per_sequence": ARGS.poses_per_sequence,
        "sweep_offsets_m": list(SWEEP_OFFSETS_M),
        "target_force_n": ARGS.target_force_n,
        "contact_threshold_n": ARGS.contact_threshold_n,
        "maximum_force_n": ARGS.maximum_force_n,
        "safety_force_consecutive_steps": ARGS.safety_force_consecutive_steps,
        "immediate_force_limit_n": ARGS.immediate_force_limit_n,
        "grasp_warning_drift_m": ARGS.grasp_warning_drift_m,
        "grasp_loss_drift_m": ARGS.grasp_loss_drift_m,
        "table_angle_range_deg": ARGS.table_angle_range_deg,
        "initial_table_nominal_height_m": cfg.reset.table_reset_z,
        "contact_table_nominal_height_m": (
            cfg.reset.table_reset_z + resolved_table_height_offset_m()
        ),
        "table_height_offset_m": resolved_table_height_offset_m(),
        "table_height_range_m": ARGS.table_height_range_m,
        "enforced_policy_target_z_range_m": (
            list(VANILLA_TARGET_Z_RANGE_M) if ARGS.variant == "vanilla" else None
        ),
        "stability_limits": {
            "window_steps": ARGS.stable_window_steps,
            "position_m": 0.001,
            "orientation_deg": 2.0,
            "linear_speed_mps": 0.005,
            "angular_speed_radps": 0.1,
        },
        "acquisition_safety": {
            "post_raise_clearance_m": POST_RAISE_CLEARANCE_M,
            "pickup_minimum_edge_lift_m": POST_RAISE_CLEARANCE_M,
            "pre_raise_minimum_edge_lift_m": (
                resolved_table_height_offset_m() + POST_RAISE_CLEARANCE_M
            ),
            "clearance_maximum_linear_speed_mps": (
                ACQUISITION_MAX_LINEAR_SPEED_MPS
            ),
            "clearance_maximum_angular_speed_radps": (
                ACQUISITION_MAX_ANGULAR_SPEED_RADPS
            ),
            "clearance_lift_timeout_steps": ARGS.clearance_lift_timeout_steps,
            "clearance_stability_window_steps": (
                CLEARANCE_STABILITY_WINDOW_STEPS
            ),
        },
        "disturbances_disabled": [
            "observation_delay",
            "action_delay",
            "object_pose_noise",
            "joint_velocity_noise",
            "external_force",
            "external_torque",
        ],
    }
    write_json(output_dir / "metadata.json", metadata)

    env = None
    recorder = None
    rows: list[dict] = []
    sequence_results: list[dict] = []
    global_step = 0
    try:
        env = gym.make("Isaacsimenvs-SimToolReal-TacMap-Scrape-Direct-v0", cfg=cfg)
        inner = env.unwrapped
        recorder = VideoRecorder(inner, output_dir)
        _, _ = env.reset()
        impedance = verify_joint_impedance(inner)
        metadata["impedance_controller"] = impedance
        write_json(output_dir / "metadata.json", metadata)
        player = make_player(inner, checkpoint, policy_config)

        for sequence in range(ARGS.sequences):
            try:
                result, global_step = run_sequence(
                    env,
                    inner,
                    player,
                    recorder,
                    rows,
                    sequence,
                    global_step,
                )
            except EvaluationFailure as exc:
                if rows:
                    global_step = max(global_step, int(rows[-1]["global_step"]))
                    rows[-1]["stage_result"] = "failure"
                    rows[-1]["failure_message"] = str(exc)
                result = {
                    "sequence": sequence,
                    "passed": False,
                    "failure": str(exc),
                }
                print(f"[sequence {sequence + 1}] FAILED: {exc}", file=sys.stderr)
            sequence_results.append(result)

        if rows:
            write_csv(output_dir / "samples.csv", rows)
        save_plots(output_dir, sequence_results)
        recorder.compose(rows)
        supported = sum(bool(result.get("passed", False)) for result in sequence_results)
        summary = {
            "passed": supported >= 2,
            "supported_sequences": supported,
            "required_supported_sequences": min(2, ARGS.sequences),
            "sequences": sequence_results,
            "criteria": {
                "minimum_static_spearman_rho": 0.8,
                "minimum_static_force_span_n": 4.0,
                "target_force_must_be_bracketed": True,
                "minimum_transition_contact_ratio": 0.9,
                "maximum_transition_steady_force_mae_n": 1.0,
                "maximum_transition_timeout_rate": 0.25,
            },
            "comparison_caveat": (
                "The tactile and no-tactile defaults are not matched in training duration; "
                "results are descriptive mechanism checks, not statistical evidence."
            ),
        }
        if ARGS.sequences == 1:
            summary["passed"] = bool(sequence_results[0].get("passed", False))
        write_json(output_dir / "summary.json", {"metadata": metadata, "results": summary})
        print(f"[output] {output_dir}")
        return 0 if summary["passed"] or not ARGS.require_pass else 2
    except Exception as exc:
        traceback.print_exc()
        if recorder is not None:
            recorder.close_raw()
        if rows:
            write_csv(output_dir / "samples.csv", rows)
        write_json(
            output_dir / "summary.json",
            {
                "metadata": metadata,
                "results": {
                    "passed": False,
                    "infrastructure_failure": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                    "sequences": sequence_results,
                },
            },
        )
        print(f"[output] {output_dir}", file=sys.stderr)
        return 1
    finally:
        if recorder is not None:
            recorder.close_raw()
        if env is not None:
            env.close()


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        APP._app.post_quit(exit_code)
        APP.close()
    raise SystemExit(exit_code)
