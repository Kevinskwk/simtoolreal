#!/usr/bin/env python3
"""Collect post-grasp tactile observability episodes in Isaac Sim."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_OBS = (
    "joint_pos", "joint_vel", "prev_action_targets", "palm_pos", "palm_rot",
    "object_rot", "fingertip_pos_rel_palm", "keypoints_rel_palm",
    "keypoints_rel_goal", "object_scales",
)
DEFAULT_CHECKPOINTS = {
    "tactile": REPO_ROOT / "outputs/2026-07-27/12-51-36/0_simtoolreal_sapg/last/model.pth",
    "no-tactile": REPO_ROOT / "outputs/2026-07-27/21-20-02/0_simtoolreal_sapg/last/model.pth",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-source", choices=("tactile", "no-tactile"), required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--policy-config", type=Path, default=REPO_ROOT / "pretrained_policy/config.yaml")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs/tactile_observability/data")
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--assets-per-type", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=3500)
    parser.add_argument("--episodes-per-shard", type=int, default=64)
    parser.add_argument("--acquisition-steps", type=int, default=900)
    parser.add_argument("--collection-steps", type=int, default=300)
    parser.add_argument("--grasp-stability-steps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-coef-id", type=float, default=0.0)
    parser.add_argument("--force-scale", type=float, default=8.0)
    parser.add_argument("--torque-scale", type=float, default=0.5)
    parser.add_argument("--max-rounds", type=int, default=20)
    AppLauncher.add_app_launcher_args(parser)
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
from isaacsimenvs.tasks.simtoolreal.utils.scrape_pose_utils import (  # noqa: E402
    edge_contact_points_w,
    table_top_state,
)
from isaacsimenvs.tasks.simtoolreal.utils.tactile_observability import (  # noqa: E402
    LabelConfig,
    derive_labels,
    label_config_dict,
    validate_episode,
)


class CollectionFailure(RuntimeError):
    pass


def validate_args() -> None:
    positive_names = (
        "num_envs", "assets_per_type", "episodes", "episodes_per_shard",
        "acquisition_steps", "collection_steps", "grasp_stability_steps",
        "max_rounds",
    )
    for name in positive_names:
        if int(getattr(ARGS, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if ARGS.collection_steps <= 20:
        raise ValueError("--collection-steps must exceed all future label horizons")
    if ARGS.force_scale < 0.0 or ARGS.torque_scale < 0.0:
        raise ValueError("disturbance scales must be non-negative")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_text(*args: str) -> str:
    return subprocess.run(("git", *args), cwd=REPO_ROOT, check=True, text=True, capture_output=True).stdout.strip()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def make_cfg() -> SimToolRealTacMapScrapePoseEnvCfg:
    cfg = SimToolRealTacMapScrapePoseEnvCfg()
    cfg.seed = ARGS.seed
    cfg.scene.num_envs = ARGS.num_envs
    cfg.episode_length_s = max(60.0, (ARGS.acquisition_steps + ARGS.collection_steps + 60) / 60.0)
    cfg.assets.handle_head_types = ("spatula", "eraser", "brush", "marker")
    cfg.assets.num_assets_per_type = ARGS.assets_per_type
    cfg.assets.shuffle_assets = True
    cfg.use_tacmap = True
    cfg.enable_vbts = True
    cfg.enable_tactile = True
    cfg.include_tacmap_in_policy = ARGS.policy_source == "tactile"
    cfg.tacmap_history_len = 5
    cfg.tacmap_policy_include_depth = True
    cfg.contact_latency = 0.0
    cfg.contact_sensor_noise = 0.0
    cfg.obs.obs_list = BASE_OBS + (
        (("tacmap",) if ARGS.policy_source == "tactile" else ())
        + ("scrape_target_contact_normal_force",)
    )
    cfg.obs.state_list = cfg.obs.obs_list
    cfg.enable_tool_table_contact_force_reward = True
    cfg.contact_force_use_control_interval_average = True
    cfg.tool_table_contact_sensor_update_period = 0.0
    cfg.tool_table_contact_sensor_history_len = int(cfg.decimation)
    cfg.tool_table_contact_sensor_force_threshold = 0.0
    cfg.table_pitch_roll_range_deg = 8.0
    cfg.reset.table_reset_z_range = 0.01
    cfg.reset.table_reset_pitch_roll_range_deg = 8.0
    cfg.termination.success_steps = 1_000_000
    cfg.termination.max_consecutive_successes = 0
    dr = cfg.domain_randomization
    dr.force_scale = ARGS.force_scale
    dr.torque_scale = ARGS.torque_scale
    dr.force_prob_range = (0.003, 0.02)
    dr.torque_prob_range = (0.003, 0.02)
    dr.force_decay = 0.5
    dr.torque_decay = 0.5
    return cfg


def make_player(inner, checkpoint: Path) -> RlPlayer:
    if not checkpoint.is_file():
        raise CollectionFailure(f"checkpoint does not exist: {checkpoint}")
    if not ARGS.policy_config.is_file():
        raise CollectionFailure(f"policy config does not exist: {ARGS.policy_config}")
    return RlPlayer(
        num_observations=inner.cfg.observation_space,
        num_actions=inner.cfg.action_space,
        config_path=str(ARGS.policy_config),
        checkpoint_path=str(checkpoint),
        device=str(inner.device),
        num_envs=inner.num_envs,
        coefficient_id=ARGS.policy_coef_id,
    )


def table_state(inner) -> tuple[torch.Tensor, torch.Tensor]:
    quaternion = getattr(inner, "_table_quat_wxyz_per_env", inner.table.data.root_quat_w)
    return table_top_state(inner.table.data.root_pos_w, quaternion)


def palm_tool_relative(inner) -> tuple[torch.Tensor, torch.Tensor]:
    palm_pos = inner.robot.data.body_link_pos_w[:, inner._palm_body_id]
    palm_quat = inner.robot.data.body_link_quat_w[:, inner._palm_body_id]
    return subtract_frame_transforms(
        palm_pos, palm_quat, inner.object.data.root_pos_w, inner.object.data.root_quat_w
    )


def edge_state(inner) -> tuple[torch.Tensor, torch.Tensor]:
    top, normal = table_state(inner)
    points = edge_contact_points_w(
        inner.object.data.root_pos_w,
        inner.object.data.root_quat_w,
        inner._scrape_x_tip_per_env,
        inner._scrape_y_min_per_env,
        inner._scrape_y_max_per_env,
        inner._scrape_z_contact_per_env,
    )
    signed = ((points - top.unsqueeze(1)) * normal.unsqueeze(1)).sum(-1)
    return signed.abs().amax(-1), signed[:, 1]


def set_goal_pose(inner, pose: torch.Tensor, anchor: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
        env_ids = torch.arange(inner.num_envs, device=inner.device)
    inner.goal_viz.write_root_pose_to_sim(pose[env_ids], env_ids=env_ids)
    inner.goal_viz.write_root_velocity_to_sim(torch.zeros(env_ids.numel(), 6, device=inner.device), env_ids=env_ids)
    inner._scrape_edge_anchor_w[env_ids] = anchor[env_ids]


def compact_frame(inner) -> torch.Tensor:
    frame = inner.get_last_tacmap_policy_frame()
    if ARGS.policy_source != "tactile":
        # The no-tactile actor does not trigger compact feature construction.
        frame = inner._get_tacmap_policy_obs_frame()
    if tuple(frame.shape) != (inner.num_envs, 25):
        raise CollectionFailure(f"compact TacMap shape is {tuple(frame.shape)}, expected {(inner.num_envs, 25)}")
    return frame.reshape(inner.num_envs, 5, 5)


def allocate_buffers(inner) -> dict[str, torch.Tensor]:
    steps, envs = ARGS.collection_steps, inner.num_envs
    cpu = torch.device("cpu")
    return {
        "actor_state": torch.empty(steps, envs, 140, device=cpu),
        "deployable_geometry": torch.empty(steps, envs, 13, device=cpu),
        "oracle_geometry": torch.empty(steps, envs, 3, device=cpu),
        "tactile_compact": torch.empty(steps, envs, 5, 5, device=cpu),
        "tactile_raw": torch.empty(steps, envs, 5, 12, 12, dtype=torch.uint8, device=cpu),
        "force_interval_n": torch.empty(steps, envs, device=cpu),
        "edge_error_m": torch.empty(steps, envs, device=cpu),
        "palm_tool_pos": torch.empty(steps, envs, 3, device=cpu),
        "palm_tool_quat": torch.empty(steps, envs, 4, device=cpu),
        "fingertip_count": torch.empty(steps, envs, dtype=torch.int8, device=cpu),
        "object_fallen": torch.empty(steps, envs, dtype=torch.bool, device=cpu),
        "action": torch.empty(steps, envs, 29, device=cpu),
        "intervention": torch.empty(steps, envs, dtype=torch.int8, device=cpu),
        "valid": torch.zeros(steps, envs, dtype=torch.bool, device=cpu),
    }


def record_step(
    inner,
    observation: dict,
    action: torch.Tensor,
    buffers: dict,
    step: int,
    active: torch.Tensor,
    intervention: int,
) -> None:
    top, normal = table_state(inner)
    edge_error, edge_signed = edge_state(inner)
    relative_pos, relative_quat = palm_tool_relative(inner)
    target_anchor = inner._scrape_edge_anchor_w
    geometry = torch.cat(
        (
            inner.object.data.root_pos_w - top,
            inner.object.data.root_quat_w,
            normal,
            target_anchor - top,
        ), dim=-1,
    )
    oracle = torch.stack((edge_error, edge_signed, inner._keypoints_max_dist), dim=-1)
    local_z = inner.object.data.root_pos_w[:, 2] - inner.scene.env_origins[:, 2]
    values = {
        "actor_state": observation["policy"][:, :140],
        "deployable_geometry": geometry,
        "oracle_geometry": oracle,
        "tactile_compact": compact_frame(inner),
        "tactile_raw": inner.vbts_deform,
        "force_interval_n": inner._scrape_table_normal_force_interval,
        "edge_error_m": edge_error,
        "palm_tool_pos": relative_pos,
        "palm_tool_quat": relative_quat,
        "fingertip_count": (inner._curr_fingertip_distances < 0.12).sum(-1).to(torch.int8),
        "object_fallen": local_z < 0.1,
        "action": action,
    }
    for key, value in values.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise CollectionFailure(f"non-finite {key} at collection step {step}")
        buffers[key][step].copy_(value.detach().cpu())
    buffers["intervention"][step].fill_(intervention)
    buffers["valid"][step].copy_(active.detach().cpu())


def tool_id_for_env(inner, env_id: int) -> str:
    asset_index = int(inner._object_asset_index_per_env[env_id].item())
    return Path(inner._object_urdf_paths[asset_index]).name


def extract_episodes(
    inner,
    buffers: dict,
    verified: torch.Tensor,
    reference_pos: torch.Tensor,
    reference_quat: torch.Tensor,
    round_index: int,
    accepted_offset: int,
) -> list[dict]:
    episodes = []
    for env_id in verified.nonzero(as_tuple=False).squeeze(-1).tolist():
        valid = buffers["valid"][:, env_id]
        length = int(valid.sum().item())
        if length <= LabelConfig().instability_horizon_steps + 5:
            continue
        episode = {
            key: value[:length, env_id].clone()
            for key, value in buffers.items()
            if key != "valid"
        }
        episode["grasp_reference_pos"] = reference_pos[env_id].detach().cpu().clone()
        episode["grasp_reference_quat"] = reference_quat[env_id].detach().cpu().clone()
        episode["metadata"] = {
            "episode_id": (
                f"{ARGS.policy_source}-s{ARGS.seed}-r{round_index:03d}-"
                f"e{env_id:05d}-n{accepted_offset + len(episodes):06d}"
            ),
            "tool_id": tool_id_for_env(inner, env_id),
            "policy_source": ARGS.policy_source,
            "seed": ARGS.seed,
            "round": round_index,
            "env_id": env_id,
            "control_dt_s": float(inner.step_dt),
        }
        validate_episode(episode)
        episode["label_counts"] = {
            key: int(value.sum().item())
            for key, value in derive_labels(episode).items()
            if key in {"onset", "loss", "slip", "instability", "hard_loss"}
        }
        episodes.append(episode)
    return episodes


def save_shards(output_dir: Path, pending: list[dict], shard_index: int) -> tuple[list[dict], int, list[dict]]:
    summaries = []
    while len(pending) >= ARGS.episodes_per_shard:
        selected = pending[: ARGS.episodes_per_shard]
        pending = pending[ARGS.episodes_per_shard :]
        path = output_dir / f"shard_{shard_index:05d}.pt"
        torch.save({"schema_version": 1, "episodes": selected}, path)
        summaries.append(summarize_shard(path, selected))
        print(f"[shard] {path} ({len(selected)} episodes)")
        shard_index += 1
    return pending, shard_index, summaries


def summarize_shard(path: Path, episodes: list[dict]) -> dict:
    frames = sum(int(episode["actor_state"].shape[0]) for episode in episodes)
    raw_active = sum(
        int(episode["tactile_raw"].reshape(episode["tactile_raw"].shape[0], -1).any(-1).sum())
        for episode in episodes
    )
    compact_active = sum(
        int(
            episode["tactile_compact"]
            .abs()
            .reshape(episode["tactile_compact"].shape[0], -1)
            .any(-1)
            .sum()
        )
        for episode in episodes
    )
    label_counts = {
        name: sum(int(episode["label_counts"][name]) for episode in episodes)
        for name in ("onset", "loss", "slip", "instability", "hard_loss")
    }
    return {
        "path": path.name,
        "sha256": sha256(path),
        "episodes": len(episodes),
        "frames": frames,
        "raw_tactile_active_frames": raw_active,
        "compact_tactile_active_frames": compact_active,
        "label_counts": label_counts,
    }


def collect_round(env, inner, player: RlPlayer, round_index: int, accepted_offset: int) -> list[dict]:
    player.reset()
    observation, _ = env.reset()
    all_ids = torch.arange(inner.num_envs, device=inner.device)
    _, normal = table_state(inner)
    contact_pose = torch.cat((inner.goal_viz.data.root_pos_w.clone(), inner.goal_viz.data.root_quat_w.clone()), dim=-1)
    contact_anchor = inner._scrape_edge_anchor_w.clone()
    pickup_pose = contact_pose.clone()
    pickup_pose[:, :3] += 0.12 * normal
    pickup_anchor = contact_anchor + 0.12 * normal
    set_goal_pose(inner, pickup_pose, pickup_anchor)
    observation = inner._get_observations()

    verified = torch.zeros(inner.num_envs, dtype=torch.bool, device=inner.device)
    stable_count = torch.zeros(inner.num_envs, dtype=torch.long, device=inner.device)
    candidate_pos = torch.zeros(inner.num_envs, 3, device=inner.device)
    candidate_quat = torch.zeros(inner.num_envs, 4, device=inner.device)
    reference_pos = torch.zeros(inner.num_envs, 3, device=inner.device)
    reference_quat = torch.zeros(inner.num_envs, 4, device=inner.device)
    for _ in range(ARGS.acquisition_steps):
        action = player.get_normalized_action(observation["policy"], deterministic_actions=True)
        observation, _, terminated, truncated, _ = env.step(action.to(inner.device))
        relative_pos, relative_quat = palm_tool_relative(inner)
        edge_error, signed_mid = edge_state(inner)
        fingertips = (inner._curr_fingertip_distances < 0.12).sum(-1)
        force = inner._scrape_table_normal_force_interval
        base_candidate = (signed_mid > 0.03) & (fingertips >= 2) & (force < 0.1)
        starting = base_candidate & (stable_count == 0)
        candidate_pos[starting] = relative_pos[starting]
        candidate_quat[starting] = relative_quat[starting]
        orientation_delta = 2.0 * torch.acos(
            torch.abs((relative_quat * candidate_quat).sum(-1)).clamp(0.0, 1.0)
        )
        stable = (
            base_candidate
            & (torch.linalg.vector_norm(relative_pos - candidate_pos, dim=-1) <= 0.005)
            & (torch.rad2deg(orientation_delta) <= 2.0)
        )
        stable_count = torch.where(stable, stable_count + 1, torch.zeros_like(stable_count))
        restarting = base_candidate & ~stable
        candidate_pos[restarting] = relative_pos[restarting]
        candidate_quat[restarting] = relative_quat[restarting]
        newly_verified = (~verified) & (stable_count >= ARGS.grasp_stability_steps)
        reference_pos[newly_verified] = relative_pos[newly_verified]
        reference_quat[newly_verified] = relative_quat[newly_verified]
        verified |= newly_verified
        done = terminated | truncated
        verified &= ~done
        stable_count[done] = 0
        if bool(verified.float().mean() >= 0.8):
            break
    if not verified.any():
        print(f"[round {round_index}] no mechanically verified grasps")
        return []
    valid_ids = verified.nonzero(as_tuple=False).squeeze(-1)
    set_goal_pose(inner, contact_pose, contact_anchor, valid_ids)
    observation = inner._get_observations()
    buffers = allocate_buffers(inner)
    active = verified.clone()
    saved_pose = None
    saved_anchor = None
    for step in range(ARGS.collection_steps):
        intervention = 0
        if step in (90, 210):
            inner._reset_scrape_goal_pose(valid_ids, resample_edge=False)
            intervention = 1
        if step == 145:
            saved_pose = torch.cat(
                (
                    inner.goal_viz.data.root_pos_w.clone(),
                    inner.goal_viz.data.root_quat_w.clone(),
                ),
                dim=-1,
            )
            saved_anchor = inner._scrape_edge_anchor_w.clone()
            retract_pose = saved_pose.clone()
            retract_pose[:, :3] += 0.008 * normal
            set_goal_pose(inner, retract_pose, saved_anchor + 0.008 * normal, valid_ids)
            intervention = 2
        elif step == 160 and saved_pose is not None and saved_anchor is not None:
            set_goal_pose(inner, saved_pose, saved_anchor, valid_ids)
            intervention = 3
        action = player.get_normalized_action(observation["policy"], deterministic_actions=True)
        observation, _, terminated, truncated, _ = env.step(action.to(inner.device))
        done = terminated | truncated
        active &= ~done
        record_step(inner, observation, action, buffers, step, active, intervention)
        if not active.any():
            break
    episodes = extract_episodes(inner, buffers, verified, reference_pos, reference_quat, round_index, accepted_offset)
    print(f"[round {round_index}] verified={int(verified.sum())}, accepted={len(episodes)}")
    return episodes


def main() -> int:
    validate_args()
    checkpoint = (ARGS.checkpoint or DEFAULT_CHECKPOINTS[ARGS.policy_source]).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ARGS.output_root / f"{stamp}_{ARGS.policy_source}_seed{ARGS.seed}"
    output_dir.mkdir(parents=True, exist_ok=False)
    cfg = make_cfg()
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "policy_source": ARGS.policy_source,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "git_commit": git_text("rev-parse", "HEAD"),
        "git_dirty": bool(git_text("status", "--porcelain")),
        "args": vars(ARGS),
        "label_config": label_config_dict(LabelConfig()),
        "shards": [],
        "accepted_episodes": 0,
    }
    write_json(output_dir / "manifest.json", manifest)
    env = None
    pending: list[dict] = []
    shard_index = 0
    try:
        env = gym.make("Isaacsimenvs-SimToolReal-TacMap-Scrape-Direct-v0", cfg=cfg)
        inner = env.unwrapped
        if inner.cfg.observation_space not in (141, 266):
            raise CollectionFailure(f"policy observation dimension {inner.cfg.observation_space} is not 141 or 266")
        if inner.vbts_deform.shape[1:] != (5, 12, 12):
            raise CollectionFailure(f"unexpected raw TacMap shape {tuple(inner.vbts_deform.shape)}")
        if inner._tool_table_contact_sensor is None or inner._tool_table_contact_sensor_failed:
            raise CollectionFailure(f"tool-table contact sensor unavailable: {inner._tool_table_contact_sensor_error}")
        player = make_player(inner, checkpoint)
        for round_index in range(ARGS.max_rounds):
            episodes = collect_round(env, inner, player, round_index, manifest["accepted_episodes"])
            remaining = ARGS.episodes - manifest["accepted_episodes"]
            pending.extend(episodes[:remaining])
            manifest["accepted_episodes"] += min(len(episodes), remaining)
            pending, shard_index, summaries = save_shards(output_dir, pending, shard_index)
            manifest["shards"].extend(summaries)
            write_json(output_dir / "manifest.json", manifest)
            if manifest["accepted_episodes"] >= ARGS.episodes:
                break
        if pending:
            path = output_dir / f"shard_{shard_index:05d}.pt"
            torch.save({"schema_version": 1, "episodes": pending}, path)
            manifest["shards"].append(summarize_shard(path, pending))
        if manifest["accepted_episodes"] < ARGS.episodes:
            raise CollectionFailure(
                f"collected {manifest['accepted_episodes']} episodes, "
                f"below requested {ARGS.episodes}"
            )
        write_json(output_dir / "manifest.json", manifest)
        print(f"[output] {output_dir}")
        return 0
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    finally:
        APP.close()
    raise SystemExit(exit_code)
