#!/usr/bin/env python3
"""Collect policy-independent TacMap responses to commanded tool wrenches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path,
        default=REPO_ROOT / "outputs/wrench_tactile_observability/data",
    )
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=1024)
    parser.add_argument("--episodes-per-shard", type=int, default=64)
    parser.add_argument("--settle-steps", type=int, default=30)
    parser.add_argument("--zero-steps", type=int, default=8)
    parser.add_argument("--ramp-steps", type=int, default=4)
    parser.add_argument("--hold-steps", type=int, default=20)
    parser.add_argument("--force-levels-n", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    parser.add_argument("--torque-levels-nm", nargs="+", type=float, default=[0.01, 0.025, 0.05])
    parser.add_argument("--max-translation-drift-m", type=float, default=0.02)
    parser.add_argument("--max-rotation-drift-deg", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=0)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = False
    return args


ARGS = parse_args()
APP = AppLauncher(ARGS).app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    quat_apply,
    quat_apply_inverse,
    subtract_frame_transforms,
)

import isaacsimenvs  # noqa: E402,F401
from isaacsimenvs.tasks.simtoolreal.simtoolreal_tacmap_env_cfg import (  # noqa: E402
    SimToolRealInHandStableScrapeEnvCfg,
)
from isaacsimenvs.tasks.simtoolreal.utils.wrench_tactile_observability import (  # noqa: E402
    WRENCH_MODE_NAMES,
    validate_impulse_episode,
)
from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import (  # noqa: E402
    compute_intermediate_values,
)


TASK_ID = "Isaacsimenvs-SimToolReal-Stable-Scrape-InHand-Direct-v0"


def check_args() -> None:
    for name in (
        "num_envs", "episodes", "episodes_per_shard", "settle_steps",
        "zero_steps", "ramp_steps", "hold_steps",
    ):
        if int(getattr(ARGS, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not ARGS.force_levels_n or min(ARGS.force_levels_n) <= 0.0:
        raise ValueError("--force-levels-n must contain positive values")
    if not ARGS.torque_levels_nm or min(ARGS.torque_levels_nm) <= 0.0:
        raise ValueError("--torque-levels-nm must contain positive values")
    if ARGS.max_translation_drift_m <= 0.0 or ARGS.max_rotation_drift_deg <= 0.0:
        raise ValueError("grasp drift limits must be positive")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_cfg() -> SimToolRealInHandStableScrapeEnvCfg:
    cfg = SimToolRealInHandStableScrapeEnvCfg()
    cfg.seed = ARGS.seed
    cfg.scene.num_envs = min(ARGS.num_envs, ARGS.episodes)
    cfg.assets.num_assets_per_type = 1
    cfg.assets.object_pool_limit = 1
    cfg.assets.shuffle_assets = False
    cfg.use_tacmap = True
    cfg.enable_vbts = True
    cfg.enable_tactile = True
    cfg.include_tacmap_in_policy = False
    cfg.tacmap_history_len = 1
    cfg.tacmap_policy_include_depth = True
    cfg.contact_latency = 0.0
    cfg.contact_sensor_noise = 0.0
    cfg.enable_fingertip_tool_contact_sensors = True
    cfg.fingertip_tool_contact_sensor_update_period = 0.0
    dr = cfg.domain_randomization
    dr.force_scale = 0.0
    dr.torque_scale = 0.0
    dr.force_prob_range = (1.0e-12, 1.0e-12)
    dr.torque_prob_range = (1.0e-12, 1.0e-12)
    dr.force_decay = 0.0
    dr.torque_decay = 0.0
    dr.use_action_delay = False
    return cfg


def palm_tool_relative(inner) -> tuple[torch.Tensor, torch.Tensor]:
    return subtract_frame_transforms(
        inner.robot.data.body_link_pos_w[:, inner._palm_body_id],
        inner.robot.data.body_link_quat_w[:, inner._palm_body_id],
        inner.object.data.root_pos_w,
        inner.object.data.root_quat_w,
    )


def quaternion_error_deg(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    dot = torch.abs((current * reference).sum(-1)).clamp(0.0, 1.0)
    return torch.rad2deg(2.0 * torch.acos(dot))


def proprio_state(inner) -> torch.Tensor:
    joint_pos = inner.robot.data.joint_pos[:, inner._perm_lab_to_canon]
    joint_vel = inner.robot.data.joint_vel[:, inner._perm_lab_to_canon]
    relative_pos, relative_quat = palm_tool_relative(inner)
    object_velocity = torch.cat(
        (inner.object.data.root_lin_vel_w, inner.object.data.root_ang_vel_w), dim=-1
    )
    result = torch.cat((joint_pos, joint_vel, relative_pos, relative_quat, object_velocity), dim=-1)
    if tuple(result.shape) != (inner.num_envs, 71) or not torch.isfinite(result).all():
        raise RuntimeError(f"expected finite proprioceptive state (N, 71), got {result.shape}")
    return result


def compact_tactile(inner) -> torch.Tensor:
    inner.get_tacmap_obs()
    frame = inner._get_tacmap_policy_obs_frame()
    if tuple(frame.shape) != (inner.num_envs, 25):
        raise RuntimeError(f"unexpected compact TacMap shape {tuple(frame.shape)}")
    return frame.reshape(inner.num_envs, 5, 5)


def move_table_away(inner) -> None:
    pose = torch.cat((inner.table.data.root_pos_w.clone(), inner.table.data.root_quat_w.clone()), dim=-1)
    pose[:, 2] = inner.scene.env_origins[:, 2] - 1.5
    inner.table.write_root_pose_to_sim(pose)
    inner.table.write_root_velocity_to_sim(torch.zeros(inner.num_envs, 6, device=inner.device))
    inner._table_z_per_env.fill_(-1.5)
    inner._table_quat_wxyz_per_env.copy_(pose[:, 3:])


def physics_step(
    inner, force_palm: torch.Tensor, torque_palm: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    sensors = getattr(inner, "_fingertip_tool_contact_sensors", None)
    if not isinstance(sensors, list) or len(sensors) != 5:
        raise RuntimeError(
            "expected five initialized pair-filtered finger-tool contact sensors"
        )
    normal_impulse_palm = torch.zeros(
        inner.num_envs, 5, 3, device=inner.device
    )
    tangential_impulse_palm = torch.zeros_like(normal_impulse_palm)
    for _ in range(int(inner.cfg.decimation)):
        palm_quat = inner.robot.data.body_link_quat_w[:, inner._palm_body_id]
        force_w = quat_apply(palm_quat, force_palm)
        torque_w = quat_apply(palm_quat, torque_palm)
        if not torch.isfinite(force_w).all() or not torch.isfinite(torque_w).all():
            raise RuntimeError("commanded external wrench became non-finite")
        inner.object.set_external_force_and_torque(
            force_w[:, None, :], torque_w[:, None, :], is_global=True
        )
        inner.robot.set_joint_position_target(inner._cur_targets)
        inner.scene.write_data_to_sim()
        inner.sim.step(render=False)
        inner.scene.update(dt=inner.physics_dt)
        palm_quat = inner.robot.data.body_link_quat_w[:, inner._palm_body_id]
        for finger_id, sensor in enumerate(sensors):
            normal_w = sensor.data.force_matrix_w
            tangential_w = sensor.data.friction_forces_w
            expected = (inner.num_envs, 1, 1, 3)
            if normal_w is None or tuple(normal_w.shape) != expected:
                raise RuntimeError(
                    f"finger sensor {finger_id} normal-force shape is "
                    f"{getattr(normal_w, 'shape', None)}, expected {expected}"
                )
            if tangential_w is None or tuple(tangential_w.shape) != expected:
                raise RuntimeError(
                    f"finger sensor {finger_id} friction-force shape is "
                    f"{getattr(tangential_w, 'shape', None)}, expected {expected}"
                )
            normal_w = normal_w[:, 0, 0]
            tangential_w = tangential_w[:, 0, 0]
            if not torch.isfinite(normal_w).all() or not torch.isfinite(tangential_w).all():
                raise RuntimeError(f"finger sensor {finger_id} returned NaN or Inf")
            normal_impulse_palm[:, finger_id] += quat_apply_inverse(
                palm_quat, normal_w * inner.physics_dt
            )
            tangential_impulse_palm[:, finger_id] += quat_apply_inverse(
                palm_quat, tangential_w * inner.physics_dt
            )
    return normal_impulse_palm, tangential_impulse_palm


def build_protocol(inner) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build balanced, independently shuffled axis/sign segments for every grasp."""
    envs = inner.num_envs
    segment = ARGS.zero_steps + ARGS.ramp_steps + ARGS.hold_steps
    total = len(WRENCH_MODE_NAMES[1:]) * segment + ARGS.zero_steps
    force = torch.zeros(total, envs, 3, device=inner.device)
    torque = torch.zeros_like(force)
    mode = torch.zeros(total, envs, dtype=torch.long, device=inner.device)
    force_levels = torch.tensor(ARGS.force_levels_n, device=inner.device)
    torque_levels = torch.tensor(ARGS.torque_levels_nm, device=inner.device)
    for env_id in range(envs):
        order = torch.randperm(12, device=inner.device) + 1
        cursor = ARGS.zero_steps
        for mode_id in order.tolist():
            cursor += ARGS.zero_steps
            is_force = mode_id <= 6
            local = mode_id - 1 if is_force else mode_id - 7
            axis, sign_index = divmod(local, 2)
            sign = 1.0 if sign_index == 0 else -1.0
            levels = force_levels if is_force else torque_levels
            magnitude = float(levels[torch.randint(len(levels), (1,), device=inner.device)])
            target = torch.zeros(3, device=inner.device)
            target[axis] = sign * magnitude
            for ramp_index in range(ARGS.ramp_steps):
                alpha = float(ramp_index + 1) / ARGS.ramp_steps
                at = cursor + ramp_index
                (force if is_force else torque)[at, env_id] = alpha * target
                mode[at, env_id] = mode_id
            start = cursor + ARGS.ramp_steps
            stop = start + ARGS.hold_steps
            (force if is_force else torque)[start:stop, env_id] = target
            mode[start:stop, env_id] = mode_id
            cursor = stop
    return force, torque, mode


def collect_round(inner, round_index: int) -> list[dict]:
    env_ids = torch.arange(inner.num_envs, device=inner.device)
    bank_ids = (env_ids + round_index * inner.num_envs) % inner._inhand_bank_size
    inner._restore_inhand_state(env_ids, bank_ids)
    move_table_away(inner)
    zero = torch.zeros(inner.num_envs, 3, device=inner.device)
    for _ in range(ARGS.settle_steps):
        physics_step(inner, zero, zero)
    reference_pos, reference_quat = palm_tool_relative(inner)
    force, torque, mode = build_protocol(inner)
    total = force.shape[0]
    state = torch.empty(total, inner.num_envs, 71)
    compact = torch.empty(total, inner.num_envs, 5, 5)
    normal_impulse = torch.empty(total, inner.num_envs, 5, 3)
    tangential_impulse = torch.empty_like(normal_impulse)
    valid = torch.ones(total, inner.num_envs, dtype=torch.bool)
    active = torch.ones(inner.num_envs, dtype=torch.bool, device=inner.device)
    maximum_drift = torch.zeros(inner.num_envs, device=inner.device)
    maximum_rotation = torch.zeros_like(maximum_drift)
    for step in range(total):
        normal_step, tangential_step = physics_step(inner, force[step], torque[step])
        compute_intermediate_values(inner)
        relative_pos, relative_quat = palm_tool_relative(inner)
        drift = torch.linalg.vector_norm(relative_pos - reference_pos, dim=-1)
        rotation = quaternion_error_deg(relative_quat, reference_quat)
        fingertips = (inner._curr_fingertip_distances < 0.12).sum(-1)
        active &= drift <= ARGS.max_translation_drift_m
        active &= rotation <= ARGS.max_rotation_drift_deg
        active &= fingertips >= 2
        maximum_drift = torch.maximum(maximum_drift, drift)
        maximum_rotation = torch.maximum(maximum_rotation, rotation)
        state[step].copy_(proprio_state(inner).detach().cpu())
        compact[step].copy_(compact_tactile(inner).detach().cpu())
        normal_impulse[step].copy_(normal_step.detach().cpu())
        tangential_impulse[step].copy_(tangential_step.detach().cpu())
        valid[step].copy_(active.cpu())
    physics_step(inner, zero, zero)

    episodes = []
    tactile_std = compact.std(0).mean(dim=(-1, -2))
    tactile_sensor_count = (compact[:, :, :, 0].amax(0) > 0.0).sum(-1)
    impulse_contact_threshold_ns = 1.0e-6
    impulse_sensor_count = (
        torch.linalg.vector_norm(normal_impulse, dim=-1).amax(0)
        > impulse_contact_threshold_ns
    ).sum(-1)
    for env_id in range(inner.num_envs):
        # A full stable sequence is required so wrench classes cannot correlate with failure time.
        if not bool(valid[:, env_id].all()) or int(tactile_sensor_count[env_id]) == 0:
            continue
        episode = {
            "proprio_state": state[:, env_id].clone(),
            "tactile_compact": compact[:, env_id].clone(),
            "finger_normal_impulse_palm_ns": normal_impulse[:, env_id].clone(),
            "finger_tangential_impulse_palm_ns": tangential_impulse[:, env_id].clone(),
            "commanded_force_palm_n": force[:, env_id].cpu().clone(),
            "commanded_torque_palm_nm": torque[:, env_id].cpu().clone(),
            "wrench_mode": mode[:, env_id].cpu().clone(),
            "valid": valid[:, env_id].clone(),
            "metadata": {
                "episode_id": f"wrench-s{ARGS.seed}-r{round_index:03d}-e{env_id:04d}",
                "split_group": f"grasp-bank-{int(bank_ids[env_id].item())}",
                "grasp_bank_index": int(bank_ids[env_id].item()),
                "tool_id": "eraser_tactile_canonical",
                "grasp_mode": "compliant-held-targets",
                "table_contact": False,
                "control_dt_s": float(inner.step_dt),
                "maximum_translation_drift_m": float(maximum_drift[env_id]),
                "maximum_rotation_drift_deg": float(maximum_rotation[env_id]),
                "compact_tactile_temporal_std_mean": float(
                    compact[:, env_id].std(0).mean()
                ),
                "compact_tactile_temporal_std_max": float(
                    compact[:, env_id].std(0).max()
                ),
                "active_tactile_sensor_count": int(tactile_sensor_count[env_id]),
                "active_normal_impulse_sensor_count": int(
                    impulse_sensor_count[env_id]
                ),
                "impulse_contact_threshold_ns": impulse_contact_threshold_ns,
                "finger_order": ["thumb", "index", "middle", "ring", "pinky"],
                "impulse_frame": "palm",
                "impulse_sign": "force acting on sensed finger body",
            },
        }
        validate_impulse_episode(episode)
        episodes.append(episode)
    print(
        f"[round {round_index}] accepted={len(episodes)}/{inner.num_envs} "
        f"max_drift={float(maximum_drift.max()):.4f}m "
        f"max_rotation={float(maximum_rotation.max()):.2f}deg "
        f"tactile_temporal_std_mean={float(tactile_std.mean()):.3e} "
        f"physical_contact_envs={int((impulse_sensor_count > 0).sum())}/{inner.num_envs}",
        flush=True,
    )
    return episodes


def main() -> None:
    check_args()
    torch.manual_seed(ARGS.seed)
    output_dir = ARGS.output_root.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    cfg = make_cfg()
    env = gym.make(TASK_ID, cfg=cfg)
    inner = env.unwrapped
    pending: list[dict] = []
    shards: list[dict] = []
    accepted = 0
    shard_index = 0
    rounds = math.ceil(ARGS.episodes / inner.num_envs) * 4
    try:
        env.reset()
        for round_index in range(rounds):
            pending.extend(collect_round(inner, round_index))
            while len(pending) >= ARGS.episodes_per_shard and accepted < ARGS.episodes:
                count = min(ARGS.episodes_per_shard, ARGS.episodes - accepted)
                selected, pending = pending[:count], pending[count:]
                path = output_dir / f"shard_{shard_index:05d}.pt"
                torch.save({"schema_version": 2, "episodes": selected}, path)
                shards.append({
                    "path": path.name,
                    "sha256": sha256(path),
                    "episodes": len(selected),
                    "frames": sum(item["proprio_state"].shape[0] for item in selected),
                })
                accepted += len(selected)
                shard_index += 1
                print(f"[shard] accepted={accepted}/{ARGS.episodes} path={path.name}", flush=True)
            if accepted >= ARGS.episodes:
                break
        if accepted < ARGS.episodes:
            raise RuntimeError(
                f"only {accepted}/{ARGS.episodes} complete stable episodes were collected"
            )
    finally:
        env.close()

    manifest = {
        "schema_version": 2,
        "created_at": datetime.now().astimezone().isoformat(),
        "experiment": "policy-independent-compliant-grasp-wrench-tactile-observability",
        "git_commit": subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=REPO_ROOT, check=True,
            text=True, capture_output=True,
        ).stdout.strip(),
        "arguments": vars(ARGS),
        "wrench_modes": list(WRENCH_MODE_NAMES),
        "finger_order": ["thumb", "index", "middle", "ring", "pinky"],
        "proprio_state_fields": [
            "joint_pos_canonical[29]", "joint_vel_canonical[29]",
            "palm_to_tool_position[3]", "palm_to_tool_quaternion_wxyz[4]",
            "tool_linear_velocity_world[3]", "tool_angular_velocity_world[3]",
        ],
        "episodes": accepted,
        "shards": shards,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str, allow_nan=False) + "\n"
    )
    print(f"[output] {output_dir}", flush=True)
    APP.close()


if __name__ == "__main__":
    main()
