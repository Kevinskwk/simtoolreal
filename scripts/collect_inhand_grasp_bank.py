#!/usr/bin/env python3
"""Collect mechanically verified compliant grasps for in-hand scrape resets."""

from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_OBS = (
    "joint_pos", "joint_vel", "prev_action_targets", "palm_pos", "palm_rot",
    "object_rot", "fingertip_pos_rel_palm", "keypoints_rel_palm",
    "keypoints_rel_goal", "object_scales",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=REPO_ROOT / "pretrained_policy/model.pth"
    )
    parser.add_argument(
        "--policy-config", type=Path, default=REPO_ROOT / "pretrained_policy/config.yaml"
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT / "assets/grasp_banks/eraser_canonical_v2.json",
    )
    parser.add_argument("--entries", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--acquisition-steps", type=int, default=1800)
    parser.add_argument("--stable-steps", type=int, default=15)
    parser.add_argument("--hold-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy-coef-id", type=float, default=0.0)
    parser.add_argument("--pickup-height-m", type=float, default=0.12)
    parser.add_argument("--max-pickup-orientation-error-deg", type=float, default=15.0)
    parser.add_argument("--tactile-entry-fraction", type=float, default=0.75)
    parser.add_argument("--min-tactile-fingers", type=int, default=1)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    args = parser.parse_args()
    args.enable_cameras = False
    return args


ARGS = parse_args()
APP = AppLauncher(ARGS).app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402

import isaacsimenvs  # noqa: E402,F401
from deployment.rl_player import RlPlayer  # noqa: E402
from isaacsimenvs.tasks.simtoolreal.simtoolreal_tacmap_env_cfg import (  # noqa: E402
    SimToolRealTacMapScrapePoseEnvCfg,
)
from isaacsimenvs.tasks.simtoolreal.utils.inhand_grasp_bank import (  # noqa: E402
    SCHEMA_VERSION,
    sha256_file,
    validate_grasp_bank,
)
from isaacsimenvs.tasks.simtoolreal.utils.scrape_pose_utils import (  # noqa: E402
    edge_contact_points_w,
    edge_tilt_from_pose,
    table_top_state,
)


class CollectionFailure(RuntimeError):
    pass


def validate_args() -> None:
    for name in ("entries", "num_envs", "acquisition_steps", "stable_steps", "hold_steps"):
        if int(getattr(ARGS, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if ARGS.entries > ARGS.num_envs:
        raise ValueError("--entries cannot exceed --num-envs in the one-pass collector")
    if float(ARGS.policy_coef_id) != 0.0:
        raise ValueError("V1 grasp collection requires --policy-coef-id=0.0")
    if float(ARGS.pickup_height_m) <= 0.0:
        raise ValueError("--pickup-height-m must be positive")
    if not 0.0 < float(ARGS.max_pickup_orientation_error_deg) <= 15.0:
        raise ValueError("--max-pickup-orientation-error-deg must be in (0, 15]")
    if not 0.0 <= float(ARGS.tactile_entry_fraction) <= 1.0:
        raise ValueError("--tactile-entry-fraction must be in [0, 1]")
    if not 1 <= int(ARGS.min_tactile_fingers) <= 5:
        raise ValueError("--min-tactile-fingers must be in [1, 5]")
    if not ARGS.checkpoint.is_file() or not ARGS.policy_config.is_file():
        raise FileNotFoundError("checkpoint and policy config must both exist")


def make_cfg() -> SimToolRealTacMapScrapePoseEnvCfg:
    cfg = SimToolRealTacMapScrapePoseEnvCfg()
    cfg.seed = int(ARGS.seed)
    cfg.scene.num_envs = int(ARGS.num_envs)
    cfg.episode_length_s = max(60.0, (ARGS.acquisition_steps + ARGS.hold_steps + 60) / 60.0)
    cfg.assets.handle_head_types = ("eraser",)
    cfg.assets.object_urdf = str(
        REPO_ROOT / "assets/urdf/objects/eraser_tactile_canonical.urdf"
    )
    cfg.assets.object_scale = (
        2.9373215824170767,
        0.5126639800346792,
        1.2951200580119278,
    )
    cfg.use_tacmap = True
    cfg.enable_vbts = True
    cfg.enable_tactile = True
    cfg.include_tacmap_in_policy = False
    cfg.obs.obs_list = BASE_OBS
    cfg.obs.state_list = BASE_OBS
    cfg.enable_tool_table_contact_force_reward = False
    cfg.enable_tool_table_contact_sensor = True
    cfg.contact_force_use_control_interval_average = True
    cfg.tool_table_contact_sensor_update_period = 0.0
    cfg.tool_table_contact_sensor_history_len = int(cfg.decimation)
    cfg.tool_table_contact_sensor_force_threshold = 0.0
    cfg.table_pitch_roll_range_deg = 0.0
    cfg.reset.table_reset_z_range = 0.0
    cfg.reset.table_reset_pitch_roll_range_deg = 0.0
    cfg.termination.success_steps = 1_000_000
    cfg.termination.max_consecutive_successes = 0
    cfg.domain_randomization.force_scale = 0.0
    cfg.domain_randomization.torque_scale = 0.0
    cfg.domain_randomization.force_prob_range = (1.0e-12, 1.0e-12)
    cfg.domain_randomization.torque_prob_range = (1.0e-12, 1.0e-12)
    return cfg


def quaternion_error_deg(actual: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if actual.shape != target.shape or actual.shape[-1] != 4:
        raise ValueError("quaternion tensors must have matching (..., 4) shapes")
    alignment = torch.abs((actual * target).sum(dim=-1)).clamp(0.0, 1.0)
    return torch.rad2deg(2.0 * torch.acos(alignment))


def tactile_metrics(inner) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not bool(inner.cfg.enable_vbts) or len(inner._vbts_sensor) != 5:
        raise RuntimeError("tactile-rich collection requires all five VBTS sensors")
    scale = float(inner.cfg.tacmap_obs_normalization)
    if scale <= 0.0:
        raise RuntimeError("tacmap_obs_normalization must be positive")
    tactile = inner.vbts_deform.to(torch.float32) / scale
    if tactile.shape[:2] != (inner.num_envs, 5) or not bool(torch.isfinite(tactile).all()):
        raise RuntimeError(
            f"invalid tactile tensor during grasp collection: {tuple(tactile.shape)}"
        )
    mask = tactile > float(inner.cfg.contact_threshold)
    active_fingers = mask.flatten(start_dim=2).any(dim=-1).sum(dim=-1)
    area = mask.to(torch.float32).mean(dim=(-1, -2, -3))
    active_count = mask.sum(dim=(-1, -2, -3))
    active_depth = torch.where(mask, tactile, torch.zeros_like(tactile))
    depth_mean = active_depth.sum(dim=(-1, -2, -3)) / active_count.clamp_min(1)
    depth_max = active_depth.amax(dim=(-1, -2, -3))
    return active_fingers, area, depth_mean, depth_max


def fingertip_support(inner) -> tuple[torch.Tensor, torch.Tensor]:
    """Return center-distance support using the eraser bounding sphere plus 12 mm."""
    object_size = inner._object_scale_per_env * 0.04
    object_radius = 0.5 * torch.linalg.vector_norm(object_size, dim=-1)
    max_center_distance = object_radius + 0.012
    support = (
        inner._curr_fingertip_distances < max_center_distance.unsqueeze(-1)
    ).sum(dim=-1)
    return support, max_center_distance


def palm_tool_relative(inner) -> tuple[torch.Tensor, torch.Tensor]:
    return subtract_frame_transforms(
        inner.robot.data.body_link_pos_w[:, inner._palm_body_id],
        inner.robot.data.body_link_quat_w[:, inner._palm_body_id],
        inner.object.data.root_pos_w,
        inner.object.data.root_quat_w,
    )


def edge_clearance(inner) -> torch.Tensor:
    points = edge_contact_points_w(
        inner.object.data.root_pos_w,
        inner.object.data.root_quat_w,
        inner._scrape_x_tip_per_env,
        inner._scrape_y_min_per_env,
        inner._scrape_y_max_per_env,
        inner._scrape_z_contact_per_env,
    )
    table_quat = getattr(inner, "_table_quat_wxyz_per_env", inner.table.data.root_quat_w)
    top, normal = table_top_state(inner.table.data.root_pos_w, table_quat)
    return ((points[:, 1] - top) * normal).sum(-1)


def hold_action(inner) -> torch.Tensor:
    targets = inner._cur_targets[:, inner._perm_lab_to_canon]
    action = torch.zeros(inner.num_envs, inner.cfg.action_space, device=inner.device)
    action[:, 7:] = 2.0 * (
        targets[:, 7:] - inner._joint_lower_canon[7:]
    ) / (
        inner._joint_upper_canon[7:] - inner._joint_lower_canon[7:]
    ) - 1.0
    return action.clamp(-1.0, 1.0)


def snapshot(
    inner,
    env_id: int,
    action: torch.Tensor,
    verification: dict,
    reference_contact_quat: torch.Tensor,
    reference_edge_yaw: torch.Tensor,
    reference_edge_tilt: torch.Tensor,
) -> dict:
    perm = inner._perm_lab_to_canon
    relative_pos, relative_quat = palm_tool_relative(inner)
    return {
        "joint_pos_canonical": inner.robot.data.joint_pos[env_id, perm].tolist(),
        "joint_vel_canonical": inner.robot.data.joint_vel[env_id, perm].tolist(),
        "joint_targets_canonical": inner._cur_targets[env_id, perm].tolist(),
        "last_action_canonical": action[env_id].tolist(),
        "object_pos_local": (
            inner.object.data.root_pos_w[env_id] - inner.scene.env_origins[env_id]
        ).tolist(),
        "object_quat_wxyz": inner.object.data.root_quat_w[env_id].tolist(),
        "object_velocity": torch.cat((
            inner.object.data.root_lin_vel_w[env_id],
            inner.object.data.root_ang_vel_w[env_id],
        )).tolist(),
        "palm_to_tool_pos": relative_pos[env_id].tolist(),
        "palm_to_tool_quat_wxyz": relative_quat[env_id].tolist(),
        "reference_contact_quat_wxyz": reference_contact_quat[env_id].tolist(),
        "reference_edge_yaw_rad": float(reference_edge_yaw[env_id].item()),
        "reference_edge_tilt_rad": float(reference_edge_tilt[env_id].item()),
        "verification": verification,
    }


def collect() -> dict:
    cfg = make_cfg()
    env = gym.make(
        "Isaacsimenvs-SimToolReal-TacMap-Scrape-Direct-v0", cfg=cfg
    )
    inner = env.unwrapped
    player = RlPlayer(
        num_observations=inner.cfg.observation_space,
        num_actions=inner.cfg.action_space,
        config_path=str(ARGS.policy_config),
        checkpoint_path=str(ARGS.checkpoint),
        device=str(inner.device),
        num_envs=inner.num_envs,
        coefficient_id=float(ARGS.policy_coef_id),
    )
    player.reset()
    observation, _ = env.reset()
    _, normal = table_top_state(
        inner.table.data.root_pos_w,
        getattr(inner, "_table_quat_wxyz_per_env", inner.table.data.root_quat_w),
    )
    contact_pose = torch.cat((
        inner.goal_viz.data.root_pos_w.clone(),
        inner.goal_viz.data.root_quat_w.clone(),
    ), dim=-1)
    contact_anchor = inner._scrape_edge_anchor_w.clone()
    reference_edge_yaw = inner._scrape_edge_yaw.clone()
    reference_edge_tilt = edge_tilt_from_pose(
        contact_pose[:, 3:], normal, reference_edge_yaw
    )
    pickup_pose = contact_pose.clone()
    pickup_pose[:, :3] += float(ARGS.pickup_height_m) * normal
    inner.goal_viz.write_root_pose_to_sim(pickup_pose)
    inner.goal_viz.write_root_velocity_to_sim(
        torch.zeros(inner.num_envs, 6, device=inner.device)
    )
    inner._scrape_edge_anchor_w.copy_(
        contact_anchor + float(ARGS.pickup_height_m) * normal
    )
    observation = inner._get_observations()

    stable_count = torch.zeros(inner.num_envs, dtype=torch.long, device=inner.device)
    hold_count = torch.zeros_like(stable_count)
    holding = torch.zeros(inner.num_envs, dtype=torch.bool, device=inner.device)
    complete = torch.zeros_like(holding)
    failed = torch.zeros_like(holding)
    reference_pos = torch.zeros(inner.num_envs, 3, device=inner.device)
    reference_quat = torch.zeros(inner.num_envs, 4, device=inner.device)
    reference_quat[:, 0] = 1.0
    maximum_drift = torch.zeros(inner.num_envs, device=inner.device)
    maximum_rotation = torch.zeros(inner.num_envs, device=inner.device)
    minimum_tactile_fingers = torch.full(
        (inner.num_envs,), 5, dtype=torch.long, device=inner.device
    )
    tactile_area_sum = torch.zeros(inner.num_envs, device=inner.device)
    tactile_depth_sum = torch.zeros(inner.num_envs, device=inner.device)
    tactile_depth_max = torch.zeros(inner.num_envs, device=inner.device)
    last_action = torch.zeros(inner.num_envs, inner.cfg.action_space, device=inner.device)
    tactile_entries: list[dict] = []
    fallback_entries: list[dict] = []
    required_tactile = math.ceil(float(ARGS.tactile_entry_fraction) * int(ARGS.entries))
    observed_tactile_depth_max = 0.0
    observed_tactile_fingers_max = 0
    observed_min_fingertip_distance = float("inf")

    for step in range(int(ARGS.acquisition_steps)):
        policy_action = player.get_normalized_action(
            observation["policy"], deterministic_actions=True
        ).to(inner.device)
        scripted_hold = hold_action(inner)
        action = torch.where(holding.unsqueeze(-1), scripted_hold, policy_action)
        last_action.copy_(action)
        observation, _, terminated, truncated, _ = env.step(action)
        relative_pos, relative_quat = palm_tool_relative(inner)
        support, support_distance = fingertip_support(inner)
        observed_min_fingertip_distance = min(
            observed_min_fingertip_distance,
            float(inner._curr_fingertip_distances.min().item()),
        )
        tactile_fingers, tactile_area, tactile_depth_mean, tactile_max = tactile_metrics(inner)
        observed_tactile_depth_max = max(
            observed_tactile_depth_max, float(tactile_max.max().item())
        )
        observed_tactile_fingers_max = max(
            observed_tactile_fingers_max, int(tactile_fingers.max().item())
        )
        clearance = edge_clearance(inner)
        force = inner._scrape_table_normal_force_interval
        pickup_orientation_error = quaternion_error_deg(
            inner.object.data.root_quat_w, pickup_pose[:, 3:]
        )
        done = terminated | truncated

        candidate = (
            ~holding & ~complete & ~failed & ~done
            & (support >= 2) & (clearance >= 0.03) & (force < 0.1)
            & (pickup_orientation_error <= float(ARGS.max_pickup_orientation_error_deg))
        )
        starting = candidate & (stable_count == 0)
        reference_pos[starting] = relative_pos[starting]
        reference_quat[starting] = relative_quat[starting]
        drift = torch.linalg.vector_norm(relative_pos - reference_pos, dim=-1)
        rotation = torch.rad2deg(2.0 * torch.acos(
            torch.abs((relative_quat * reference_quat).sum(-1)).clamp(0.0, 1.0)
        ))
        stable = candidate & (drift <= 0.005) & (rotation <= 2.0)
        stable_count = torch.where(stable, stable_count + 1, torch.zeros_like(stable_count))
        restarting = candidate & ~stable
        reference_pos[restarting] = relative_pos[restarting]
        reference_quat[restarting] = relative_quat[restarting]
        newly_holding = stable_count >= int(ARGS.stable_steps)
        holding |= newly_holding
        hold_count[newly_holding] = 0
        reference_pos[newly_holding] = relative_pos[newly_holding]
        reference_quat[newly_holding] = relative_quat[newly_holding]
        minimum_tactile_fingers[newly_holding] = tactile_fingers[newly_holding]
        tactile_area_sum[newly_holding] = 0.0
        tactile_depth_sum[newly_holding] = 0.0
        tactile_depth_max[newly_holding] = 0.0

        hold_drift = torch.linalg.vector_norm(relative_pos - reference_pos, dim=-1)
        hold_rotation = torch.rad2deg(2.0 * torch.acos(
            torch.abs((relative_quat * reference_quat).sum(-1)).clamp(0.0, 1.0)
        ))
        maximum_drift = torch.where(holding, torch.maximum(maximum_drift, hold_drift), maximum_drift)
        maximum_rotation = torch.where(
            holding, torch.maximum(maximum_rotation, hold_rotation), maximum_rotation
        )
        minimum_tactile_fingers = torch.where(
            holding,
            torch.minimum(minimum_tactile_fingers, tactile_fingers),
            minimum_tactile_fingers,
        )
        tactile_area_sum = torch.where(
            holding, tactile_area_sum + tactile_area, tactile_area_sum
        )
        tactile_depth_sum = torch.where(
            holding, tactile_depth_sum + tactile_depth_mean, tactile_depth_sum
        )
        tactile_depth_max = torch.where(
            holding, torch.maximum(tactile_depth_max, tactile_max), tactile_depth_max
        )
        valid_hold = (
            holding & ~done & (support >= 2) & (clearance >= 0.03)
            & (force < 0.1) & (maximum_drift <= 0.005) & (maximum_rotation <= 2.0)
            & (pickup_orientation_error <= float(ARGS.max_pickup_orientation_error_deg))
        )
        failed_now = holding & ~valid_hold
        failed |= failed_now
        holding &= ~failed_now
        hold_count = torch.where(valid_hold, hold_count + 1, hold_count)
        newly_complete = valid_hold & (hold_count >= int(ARGS.hold_steps)) & ~complete
        for env_id in newly_complete.nonzero(as_tuple=False).squeeze(-1).tolist():
            verification = {
                "support_count": int(support[env_id].item()),
                "edge_clearance_m": float(clearance[env_id].item()),
                "table_force_n": float(force[env_id].item()),
                "stable_steps": int(ARGS.stable_steps),
                "hold_steps": int(ARGS.hold_steps),
                "hold_drift_m": float(maximum_drift[env_id].item()),
                "hold_rotation_deg": float(maximum_rotation[env_id].item()),
                "pickup_orientation_error_deg": float(
                    pickup_orientation_error[env_id].item()
                ),
                "tactile_finger_count_min": int(
                    minimum_tactile_fingers[env_id].item()
                ),
                "tactile_contact_area_mean": float(
                    tactile_area_sum[env_id].item() / max(int(hold_count[env_id].item()), 1)
                ),
                "tactile_depth_mean": float(
                    tactile_depth_sum[env_id].item() / max(int(hold_count[env_id].item()), 1)
                ),
                "tactile_depth_max": float(tactile_depth_max[env_id].item()),
            }
            entry = snapshot(
                inner,
                env_id,
                last_action,
                verification,
                contact_pose[:, 3:],
                reference_edge_yaw,
                reference_edge_tilt,
            )
            if verification["tactile_finger_count_min"] >= int(ARGS.min_tactile_fingers):
                tactile_entries.append(entry)
            else:
                fallback_entries.append(entry)
            complete[env_id] = True
            holding[env_id] = False
            if (
                len(tactile_entries) >= required_tactile
                and len(tactile_entries) + len(fallback_entries) >= int(ARGS.entries)
            ):
                break
        if (
            len(tactile_entries) >= required_tactile
            and len(tactile_entries) + len(fallback_entries) >= int(ARGS.entries)
        ):
            break
        if (step + 1) % 300 == 0:
            print(
                f"[collect] step={step + 1} stable={int(holding.sum())} "
                f"tactile={len(tactile_entries)} fallback={len(fallback_entries)} "
                f"failed={int(failed.sum())} "
                f"tactile_depth_max={observed_tactile_depth_max:.4f} "
                f"tactile_fingers_max={observed_tactile_fingers_max} "
                f"fingertip_distance_min={observed_min_fingertip_distance:.4f} "
                f"support_distance_mean={float(support_distance.mean().item()):.4f}",
                flush=True,
            )

    asset_path = Path(inner._object_urdf_paths[0])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "tool_type": "eraser",
        "asset_sha256": sha256_file(asset_path),
        "source_checkpoint": str(ARGS.checkpoint.resolve()),
        "source_checkpoint_sha256": sha256_file(ARGS.checkpoint),
        "policy_coefficient_id": float(ARGS.policy_coef_id),
        "tactile_rich_fraction_min": float(ARGS.tactile_entry_fraction),
        "tactile_min_fingers": int(ARGS.min_tactile_fingers),
        "seed": int(ARGS.seed),
        "control_dt_s": float(inner.step_dt),
        "entries": (
            tactile_entries[: int(ARGS.entries)]
            + fallback_entries[: max(0, int(ARGS.entries) - len(tactile_entries))]
        )[: int(ARGS.entries)],
    }
    env.close()
    if len(payload["entries"]) < int(ARGS.entries) or len(tactile_entries) < required_tactile:
        raise CollectionFailure(
            "failed grasp-bank quota: "
            f"total={len(payload['entries'])}/{ARGS.entries}, "
            f"tactile={len(tactile_entries)}/{required_tactile}"
        )
    return validate_grasp_bank(payload, minimum_entries=int(ARGS.entries))


def main() -> None:
    validate_args()
    payload = collect()
    ARGS.output.parent.mkdir(parents=True, exist_ok=True)
    ARGS.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"[output] {ARGS.output.resolve()} ({len(payload['entries'])} grasps)")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        APP.close()
