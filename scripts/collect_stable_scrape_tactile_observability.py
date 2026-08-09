#!/usr/bin/env python3
"""Collect matched tactile-observability data from the frozen stable-scrape policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "outputs/2026-08-07/17-06-31/0_simtoolreal_sapg/last/model.pth"
)
DEFAULT_RUN_NAME = "eraser_inhand_stable_scrape_20260807_170626_20260807_170917"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--policy-config",
        type=Path,
        default=REPO_ROOT / "pretrained_policy/config.yaml",
    )
    parser.add_argument("--source-run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs/tactile_observability/stable_scrape_data",
    )
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--episodes", type=int, default=4096)
    parser.add_argument("--episodes-per-shard", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=90)
    parser.add_argument("--collection-steps", type=int, default=360)
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-coef-id", type=float, default=0.0)
    parser.add_argument("--curriculum-stage", type=int, default=-1)
    parser.add_argument("--retraction-min-m", type=float, default=0.004)
    parser.add_argument("--retraction-max-m", type=float, default=0.010)
    parser.add_argument("--retraction-duration-min", type=int, default=12)
    parser.add_argument("--retraction-duration-max", type=int, default=24)
    parser.add_argument("--external-force-scale", type=float, default=1.5)
    parser.add_argument("--external-torque-scale", type=float, default=0.1)
    parser.add_argument("--object-friction-scale-min", type=float, default=0.8)
    parser.add_argument("--object-friction-scale-max", type=float, default=1.2)
    parser.add_argument("--fingertip-friction-scale-min", type=float, default=0.9)
    parser.add_argument("--fingertip-friction-scale-max", type=float, default=1.1)
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
    SimToolRealInHandStableScrapeEnvCfg,
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
    for name in (
        "num_envs",
        "episodes",
        "episodes_per_shard",
        "warmup_steps",
        "collection_steps",
        "max_rounds",
        "retraction_duration_min",
        "retraction_duration_max",
    ):
        if int(getattr(ARGS, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if ARGS.collection_steps <= LabelConfig().instability_horizon_steps + 5:
        raise ValueError("--collection-steps is too short for future-event labels")
    if not 0.0 < ARGS.retraction_min_m <= ARGS.retraction_max_m:
        raise ValueError("retraction range must be positive and ordered")
    if ARGS.retraction_duration_min > ARGS.retraction_duration_max:
        raise ValueError("retraction duration range must be ordered")
    for prefix in ("object", "fingertip"):
        low = float(getattr(ARGS, f"{prefix}_friction_scale_min"))
        high = float(getattr(ARGS, f"{prefix}_friction_scale_max"))
        if low <= 0.0 or high < low:
            raise ValueError(f"{prefix} friction scale range must be positive and ordered")
    if ARGS.external_force_scale < 0.0 or ARGS.external_torque_scale < 0.0:
        raise ValueError("external disturbance scales must be non-negative")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_text(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=REPO_ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def make_cfg() -> SimToolRealInHandStableScrapeEnvCfg:
    cfg = SimToolRealInHandStableScrapeEnvCfg()
    cfg.seed = ARGS.seed
    cfg.scene.num_envs = ARGS.num_envs
    cfg.assets.num_assets_per_type = 1
    cfg.assets.object_pool_limit = 1
    cfg.assets.shuffle_assets = False
    cfg.episode_length_s = max(
        20.0, (ARGS.warmup_steps + ARGS.collection_steps + 60) * float(cfg.decimation) * cfg.sim.dt
    )

    # Tactile is measured but deliberately excluded from the frozen actor.
    cfg.use_tacmap = True
    cfg.enable_vbts = True
    cfg.enable_tactile = True
    cfg.include_tacmap_in_policy = False
    cfg.tacmap_history_len = 5
    cfg.tacmap_policy_include_depth = True
    cfg.contact_latency = 0.0
    cfg.contact_sensor_noise = 0.0

    cfg.enable_tool_table_contact_sensor = True
    cfg.contact_force_use_control_interval_average = True
    cfg.tool_table_contact_sensor_update_period = 0.0
    cfg.tool_table_contact_sensor_history_len = int(cfg.decimation)
    cfg.tool_table_contact_sensor_force_threshold = 0.0

    dr = cfg.domain_randomization
    dr.object_friction_scale_range = (
        ARGS.object_friction_scale_min,
        ARGS.object_friction_scale_max,
    )
    dr.fingertip_friction_scale_range = (
        ARGS.fingertip_friction_scale_min,
        ARGS.fingertip_friction_scale_max,
    )
    dr.force_scale = ARGS.external_force_scale
    dr.torque_scale = ARGS.external_torque_scale
    dr.force_prob_range = (0.003, 0.015)
    dr.torque_prob_range = (0.003, 0.015)
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
    return subtract_frame_transforms(
        inner.robot.data.body_link_pos_w[:, inner._palm_body_id],
        inner.robot.data.body_link_quat_w[:, inner._palm_body_id],
        inner.object.data.root_pos_w,
        inner.object.data.root_quat_w,
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


def compact_frame(inner) -> torch.Tensor:
    frame = inner._get_tacmap_policy_obs_frame()
    expected = (inner.num_envs, 25)
    if tuple(frame.shape) != expected:
        raise CollectionFailure(
            f"compact TacMap shape is {tuple(frame.shape)}, expected {expected}"
        )
    return frame.reshape(inner.num_envs, 5, 5)


def allocate_buffers(inner) -> dict[str, torch.Tensor]:
    steps, envs = ARGS.collection_steps, inner.num_envs
    actor_dim = int(inner.cfg.observation_space)
    return {
        "actor_state": torch.empty(steps, envs, actor_dim),
        "deployable_geometry": torch.empty(steps, envs, 13),
        "oracle_geometry": torch.empty(steps, envs, 3),
        "tactile_compact": torch.empty(steps, envs, 5, 5),
        "tactile_raw": torch.empty(steps, envs, 5, 12, 12, dtype=torch.uint8),
        "force_interval_n": torch.empty(steps, envs),
        "edge_error_m": torch.empty(steps, envs),
        "palm_tool_pos": torch.empty(steps, envs, 3),
        "palm_tool_quat": torch.empty(steps, envs, 4),
        "fingertip_count": torch.empty(steps, envs, dtype=torch.int8),
        "object_fallen": torch.empty(steps, envs, dtype=torch.bool),
        "action": torch.empty(steps, envs, 29),
        "intervention": torch.empty(steps, envs, dtype=torch.int8),
        "valid": torch.zeros(steps, envs, dtype=torch.bool),
    }


def record_step(
    inner,
    observation: dict,
    action: torch.Tensor,
    buffers: dict[str, torch.Tensor],
    step: int,
    active: torch.Tensor,
    intervention: torch.Tensor,
) -> None:
    top, normal = table_state(inner)
    edge_error, edge_signed = edge_state(inner)
    relative_pos, relative_quat = palm_tool_relative(inner)
    geometry = torch.cat(
        (
            inner.object.data.root_pos_w - top,
            inner.object.data.root_quat_w,
            normal,
            inner._scrape_edge_anchor_w - top,
        ),
        dim=-1,
    )
    oracle = torch.stack((edge_error, edge_signed, inner._keypoints_max_dist), dim=-1)
    local_z = inner.object.data.root_pos_w[:, 2] - inner.scene.env_origins[:, 2]
    values = {
        "actor_state": observation["policy"],
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
    buffers["intervention"][step].copy_(intervention.detach().cpu())
    buffers["valid"][step].copy_(active.detach().cpu())


def shift_contact_target(inner, env_ids: torch.Tensor, offset_m: torch.Tensor) -> None:
    if env_ids.numel() == 0:
        return
    _, normal = table_state(inner)
    delta = normal[env_ids] * offset_m.unsqueeze(-1)
    inner._stable_contact_goal_pos_w[env_ids] += delta
    inner._stable_contact_anchor_w[env_ids] += delta
    path_delta = (
        inner._stable_path_direction_w[env_ids]
        * inner._stable_path_offset[env_ids].unsqueeze(-1)
    )
    inner._write_goal(
        env_ids,
        inner._stable_contact_goal_pos_w[env_ids] + path_delta,
        inner._stable_contact_goal_quat_w[env_ids],
        inner._stable_contact_anchor_w[env_ids] + path_delta,
    )


def make_intervention_schedule(inner) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    envs = inner.num_envs
    device = inner.device
    first = torch.randint(60, 121, (envs,), device=device)
    second = torch.randint(210, 271, (envs,), device=device)
    durations = torch.randint(
        ARGS.retraction_duration_min,
        ARGS.retraction_duration_max + 1,
        (envs, 2),
        device=device,
    )
    offsets = torch.empty(envs, 2, device=device).uniform_(
        ARGS.retraction_min_m, ARGS.retraction_max_m
    )
    starts = torch.stack((first, second), dim=-1)
    if int((starts + durations).max().item()) >= ARGS.collection_steps:
        raise CollectionFailure("retraction schedule exceeds collection horizon")
    return starts, durations, offsets


def object_friction(inner) -> torch.Tensor:
    materials = inner.object.root_physx_view.get_material_properties()
    if materials.ndim != 3 or materials.shape[0] != inner.num_envs:
        raise CollectionFailure(
            f"unexpected object material tensor shape {tuple(materials.shape)}"
        )
    return materials[:, 0, 0].detach().cpu()


def extract_episodes(
    inner,
    buffers: dict[str, torch.Tensor],
    references: tuple[torch.Tensor, torch.Tensor],
    conditions: dict[str, torch.Tensor],
    round_index: int,
    accepted_offset: int,
) -> list[dict]:
    reference_pos, reference_quat = references
    episodes = []
    for env_id in range(inner.num_envs):
        length = int(buffers["valid"][:, env_id].sum().item())
        if length <= LabelConfig().instability_horizon_steps + 5:
            continue
        episode = {
            key: value[:length, env_id].clone()
            for key, value in buffers.items()
            if key != "valid"
        }
        episode["grasp_reference_pos"] = reference_pos[env_id].cpu().clone()
        episode["grasp_reference_quat"] = reference_quat[env_id].cpu().clone()
        bank_index = int(conditions["bank_index"][env_id].item())
        episode["metadata"] = {
            "episode_id": (
                f"stable-scrape-s{ARGS.seed}-r{round_index:03d}-"
                f"e{env_id:05d}-n{accepted_offset + len(episodes):06d}"
            ),
            "tool_id": "eraser_tactile_canonical",
            "split_group": f"grasp-bank-{bank_index}",
            "policy_source": "stable-scrape-frozen-no-tactile",
            "source_run_name": ARGS.source_run_name,
            "seed": ARGS.seed,
            "round": round_index,
            "env_id": env_id,
            "grasp_bank_index": bank_index,
            "table_height_local_m": float(conditions["table_height"][env_id]),
            "table_normal": conditions["table_normal"][env_id].tolist(),
            "object_friction": float(conditions["object_friction"][env_id]),
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


def summarize_shard(path: Path, episodes: list[dict]) -> dict:
    frames = sum(int(episode["actor_state"].shape[0]) for episode in episodes)
    raw_active = sum(
        int(episode["tactile_raw"].reshape(episode["tactile_raw"].shape[0], -1).any(-1).sum())
        for episode in episodes
    )
    compact_active = sum(
        int(episode["tactile_compact"].abs().reshape(episode["tactile_compact"].shape[0], -1).any(-1).sum())
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


def save_shards(
    output_dir: Path, pending: list[dict], shard_index: int
) -> tuple[list[dict], int, list[dict]]:
    summaries = []
    while len(pending) >= ARGS.episodes_per_shard:
        selected = pending[: ARGS.episodes_per_shard]
        pending = pending[ARGS.episodes_per_shard :]
        path = output_dir / f"shard_{shard_index:05d}.pt"
        torch.save({"schema_version": 2, "episodes": selected}, path)
        summaries.append(summarize_shard(path, selected))
        print(f"[shard] {path} ({len(selected)} episodes)", flush=True)
        shard_index += 1
    return pending, shard_index, summaries


def collect_round(env, inner, player: RlPlayer, round_index: int, accepted_offset: int) -> list[dict]:
    player.reset()
    observation, _ = env.reset()
    active = torch.ones(inner.num_envs, dtype=torch.bool, device=inner.device)
    reference_pos = inner._stable_grasp_reference_pos.clone()
    reference_quat = inner._stable_grasp_reference_quat.clone()
    bank_index = inner._inhand_reset_bank_index.clone().cpu()
    table_top, table_normal = table_state(inner)
    conditions = {
        "bank_index": bank_index,
        "table_height": (table_top[:, 2] - inner.scene.env_origins[:, 2]).cpu(),
        "table_normal": table_normal.cpu(),
        "object_friction": object_friction(inner),
    }

    for _ in range(ARGS.warmup_steps):
        action = player.get_normalized_action(
            observation["policy"], deterministic_actions=True
        )
        observation, _, terminated, truncated, _ = env.step(action.to(inner.device))
        active &= ~(terminated | truncated)
    if not bool(active.any()):
        print(f"[round {round_index}] all environments ended during warmup", flush=True)
        return []

    starts, durations, offsets = make_intervention_schedule(inner)
    buffers = allocate_buffers(inner)
    for step in range(ARGS.collection_steps):
        intervention = torch.zeros(inner.num_envs, dtype=torch.int8, device=inner.device)
        for event in range(2):
            start = starts[:, event]
            stop = start + durations[:, event]
            starting = active & (step == start)
            stopping = active & (step == stop)
            held = active & (step > start) & (step < stop)
            if bool(starting.any()):
                ids = starting.nonzero(as_tuple=False).squeeze(-1)
                shift_contact_target(inner, ids, offsets[ids, event])
                intervention[starting] = 1
            intervention[held] = 1
            if bool(stopping.any()):
                ids = stopping.nonzero(as_tuple=False).squeeze(-1)
                shift_contact_target(inner, ids, -offsets[ids, event])
                intervention[stopping] = 2

        action = player.get_normalized_action(
            observation["policy"], deterministic_actions=True
        )
        observation, _, terminated, truncated, _ = env.step(action.to(inner.device))
        active &= ~(terminated | truncated)
        record_step(inner, observation, action, buffers, step, active, intervention)
        if not bool(active.any()):
            break

    episodes = extract_episodes(
        inner,
        buffers,
        (reference_pos, reference_quat),
        conditions,
        round_index,
        accepted_offset,
    )
    print(
        f"[round {round_index}] surviving={int(active.sum())}, accepted={len(episodes)}",
        flush=True,
    )
    return episodes


def validate_manifest_data(manifest: dict) -> None:
    if not manifest["shards"]:
        raise CollectionFailure("collection produced no data shards")
    raw_active = sum(item["raw_tactile_active_frames"] for item in manifest["shards"])
    compact_active = sum(
        item["compact_tactile_active_frames"] for item in manifest["shards"]
    )
    if raw_active == 0 or compact_active == 0:
        raise CollectionFailure(
            "tactile sensors produced no active frames; refusing to save a silent dataset"
        )
    counts = {
        name: sum(item["label_counts"][name] for item in manifest["shards"])
        for name in ("onset", "loss", "slip", "instability", "hard_loss")
    }
    missing = [name for name, count in counts.items() if count == 0]
    if missing:
        raise CollectionFailure(f"dataset has no positive labels for {missing}")
    manifest["totals"] = {
        "raw_tactile_active_frames": raw_active,
        "compact_tactile_active_frames": compact_active,
        "label_counts": counts,
    }


def main() -> int:
    validate_args()
    checkpoint = ARGS.checkpoint.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ARGS.output_root / f"{stamp}_seed{ARGS.seed}"
    output_dir.mkdir(parents=True, exist_ok=False)
    cfg = make_cfg()
    manifest = {
        "schema_version": 2,
        "created_at": datetime.now().astimezone().isoformat(),
        "policy_source": "stable-scrape-frozen-no-tactile",
        "source_run_name": ARGS.source_run_name,
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
        env = gym.make(
            "Isaacsimenvs-SimToolReal-Stable-Scrape-InHand-Direct-v0", cfg=cfg
        )
        inner = env.unwrapped
        if int(inner.cfg.observation_space) != 146:
            raise CollectionFailure(
                f"frozen policy observation dimension is {inner.cfg.observation_space}, expected 146"
            )
        if tuple(inner.vbts_deform.shape[1:]) != (5, 12, 12):
            raise CollectionFailure(
                f"unexpected raw TacMap shape {tuple(inner.vbts_deform.shape)}"
            )
        if inner._tool_table_contact_sensor is None or inner._tool_table_contact_sensor_failed:
            raise CollectionFailure(
                f"tool-table contact sensor unavailable: {inner._tool_table_contact_sensor_error}"
            )
        last_stage = len(inner.cfg.inhand_table_angle_stages_deg) - 1
        stage = last_stage if ARGS.curriculum_stage < 0 else ARGS.curriculum_stage
        if stage < 0 or stage > last_stage:
            raise CollectionFailure(
                f"curriculum stage {stage} is outside [0, {last_stage}]"
            )
        inner._inhand_curriculum_stage = stage
        player = make_player(inner, checkpoint)

        for round_index in range(ARGS.max_rounds):
            episodes = collect_round(
                env, inner, player, round_index, manifest["accepted_episodes"]
            )
            remaining = ARGS.episodes - manifest["accepted_episodes"]
            accepted = episodes[:remaining]
            pending.extend(accepted)
            manifest["accepted_episodes"] += len(accepted)
            pending, shard_index, summaries = save_shards(
                output_dir, pending, shard_index
            )
            manifest["shards"].extend(summaries)
            write_json(output_dir / "manifest.json", manifest)
            if manifest["accepted_episodes"] >= ARGS.episodes:
                break

        if pending:
            path = output_dir / f"shard_{shard_index:05d}.pt"
            torch.save({"schema_version": 2, "episodes": pending}, path)
            manifest["shards"].append(summarize_shard(path, pending))
        if manifest["accepted_episodes"] < ARGS.episodes:
            raise CollectionFailure(
                f"collected {manifest['accepted_episodes']} episodes, "
                f"below requested {ARGS.episodes}"
            )
        validate_manifest_data(manifest)
        write_json(output_dir / "manifest.json", manifest)
        print(f"[output] {output_dir}", flush=True)
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
