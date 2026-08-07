#!/usr/bin/env python3
"""Collect mechanically verified compliant grasps for in-hand scrape resets."""

from __future__ import annotations

import argparse
import json
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
        default=REPO_ROOT / "assets/grasp_banks/spatula_canonical_v1.json",
    )
    parser.add_argument("--entries", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--acquisition-steps", type=int, default=1800)
    parser.add_argument("--stable-steps", type=int, default=15)
    parser.add_argument("--hold-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy-coef-id", type=float, default=0.0)
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
    if not ARGS.checkpoint.is_file() or not ARGS.policy_config.is_file():
        raise FileNotFoundError("checkpoint and policy config must both exist")


def make_cfg() -> SimToolRealTacMapScrapePoseEnvCfg:
    cfg = SimToolRealTacMapScrapePoseEnvCfg()
    cfg.seed = int(ARGS.seed)
    cfg.scene.num_envs = int(ARGS.num_envs)
    cfg.episode_length_s = max(60.0, (ARGS.acquisition_steps + ARGS.hold_steps + 60) / 60.0)
    cfg.assets.handle_head_types = ("spatula",)
    cfg.assets.num_assets_per_type = 1
    cfg.assets.shuffle_assets = False
    cfg.assets.object_pool_limit = 1
    cfg.use_tacmap = False
    cfg.enable_vbts = False
    cfg.enable_tactile = False
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


def snapshot(inner, env_id: int, action: torch.Tensor, verification: dict) -> dict:
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
    pickup_pose = contact_pose.clone()
    pickup_pose[:, :3] += 0.12 * normal
    inner.goal_viz.write_root_pose_to_sim(pickup_pose)
    inner.goal_viz.write_root_velocity_to_sim(
        torch.zeros(inner.num_envs, 6, device=inner.device)
    )
    inner._scrape_edge_anchor_w.copy_(contact_anchor + 0.12 * normal)
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
    last_action = torch.zeros(inner.num_envs, inner.cfg.action_space, device=inner.device)
    entries: list[dict] = []

    for step in range(int(ARGS.acquisition_steps)):
        policy_action = player.get_normalized_action(
            observation["policy"], deterministic_actions=True
        ).to(inner.device)
        scripted_hold = hold_action(inner)
        action = torch.where(holding.unsqueeze(-1), scripted_hold, policy_action)
        last_action.copy_(action)
        observation, _, terminated, truncated, _ = env.step(action)
        relative_pos, relative_quat = palm_tool_relative(inner)
        support = (inner._curr_fingertip_distances < 0.12).sum(-1)
        clearance = edge_clearance(inner)
        force = inner._scrape_table_normal_force_interval
        done = terminated | truncated

        candidate = (
            ~holding & ~complete & ~failed & ~done
            & (support >= 2) & (clearance >= 0.03) & (force < 0.1)
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

        hold_drift = torch.linalg.vector_norm(relative_pos - reference_pos, dim=-1)
        hold_rotation = torch.rad2deg(2.0 * torch.acos(
            torch.abs((relative_quat * reference_quat).sum(-1)).clamp(0.0, 1.0)
        ))
        maximum_drift = torch.where(holding, torch.maximum(maximum_drift, hold_drift), maximum_drift)
        maximum_rotation = torch.where(
            holding, torch.maximum(maximum_rotation, hold_rotation), maximum_rotation
        )
        valid_hold = (
            holding & ~done & (support >= 2) & (clearance >= 0.03)
            & (force < 0.1) & (maximum_drift <= 0.005) & (maximum_rotation <= 2.0)
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
            }
            entries.append(snapshot(inner, env_id, last_action, verification))
            complete[env_id] = True
            holding[env_id] = False
            if len(entries) >= int(ARGS.entries):
                break
        if len(entries) >= int(ARGS.entries):
            break
        if (step + 1) % 300 == 0:
            print(
                f"[collect] step={step + 1} stable={int(holding.sum())} "
                f"complete={len(entries)} failed={int(failed.sum())}",
                flush=True,
            )

    asset_path = Path(inner._object_urdf_paths[0])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "tool_type": "spatula",
        "asset_sha256": sha256_file(asset_path),
        "source_checkpoint": str(ARGS.checkpoint.resolve()),
        "source_checkpoint_sha256": sha256_file(ARGS.checkpoint),
        "policy_coefficient_id": float(ARGS.policy_coef_id),
        "seed": int(ARGS.seed),
        "control_dt_s": float(inner.step_dt),
        "entries": entries[: int(ARGS.entries)],
    }
    env.close()
    if len(payload["entries"]) < int(ARGS.entries):
        raise CollectionFailure(
            f"collected only {len(payload['entries'])}/{ARGS.entries} verified grasps"
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
    finally:
        APP.close()
