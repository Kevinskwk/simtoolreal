#!/usr/bin/env python3
"""Evaluate a trained Allen-key workspace-adjustment policy across bank pairs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            ROOT / "outputs/2026-08-23/00-35-17/0_simtoolreal_sapg/last/model.pth"
        ),
    )
    parser.add_argument(
        "--policy-config", type=Path, default=ROOT / "pretrained_policy/config.yaml"
    )
    parser.add_argument(
        "--grasp-bank",
        type=Path,
        default=ROOT / "assets/grasp_banks/allen_key_rollout_adjustment_v1.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--num-replays", type=int, default=6)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--policy-coef-id", type=float, default=0.0)
    parser.add_argument(
        "--video", action=argparse.BooleanOptionalAction, default=True
    )
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    args = parser.parse_args()
    args.enable_cameras = bool(args.video)
    return args


ARGS = parse_args()
APP = AppLauncher(ARGS).app


import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab import sim as sim_utils  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    combine_frame_transforms,
    quat_apply,
    quat_inv,
)

import isaacsimenvs  # noqa: E402,F401
from deployment.rl_player import RlPlayer  # noqa: E402
from isaacsimenvs.tasks.simtoolreal.pose_viewer import (  # noqa: E402
    build_pose_viewer_html,
    capture_pose_viewer_frame,
    object_urdf_for_env,
    table_urdf_for_env,
    workpiece_urdf_for_env,
)
from isaacsimenvs.tasks.simtoolreal.simtoolreal_tacmap_env_cfg import (  # noqa: E402
    SimToolRealAllenKeyWorkspaceAdjustmentEnvCfg,
)
from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import (  # noqa: E402
    compute_intermediate_values,
)


TASK_ID = "Isaacsimenvs-SimToolReal-AllenKey-Workspace-Adjustment-Direct-v0"
TIERS = ("easy", "support", "broad")


class EvaluationFailure(RuntimeError):
    pass


def validate_args() -> None:
    for path in (ARGS.checkpoint, ARGS.policy_config, ARGS.grasp_bank):
        if not path.is_file():
            raise FileNotFoundError(path)
    if int(ARGS.frame_stride) <= 0:
        raise ValueError("--frame-stride must be positive")
    if not 1 <= int(ARGS.num_replays) <= 12:
        raise ValueError("--num-replays must be in [1, 12]")


def candidate_pairs(inner) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return all improved valid pairs and identify stage-zero in-distribution pairs."""
    quality = inner._allen_bank_functional_quality
    improvement = quality.unsqueeze(0) - quality.unsqueeze(1)
    improved = inner._allen_target_pair_valid & (
        improvement >= float(inner.cfg.allen_target_min_quality_improvement)
    )
    source, target = torch.nonzero(improved, as_tuple=True)
    if source.numel() == 0:
        raise EvaluationFailure("grasp bank contains no improved valid target pairs")
    stage_zero = (
        (inner._allen_bank_workspace_tier[source] != 2)
        & (
            inner._allen_pair_translation_m[source, target]
            <= float(inner.cfg.allen_pair_max_translation_stages_m[0])
        )
        & (
            inner._allen_pair_rotation_deg[source, target]
            <= float(inner.cfg.allen_pair_max_rotation_stages_deg[0])
        )
    )
    return source, target, stage_zero


def install_pairs(inner, source: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """Restore exact source grasps and install exact bank targets without resampling."""
    count = source.numel()
    env_ids = torch.arange(count, device=inner.device)
    # Some stress-test sources are intentionally outside stage zero and have no
    # active curriculum target. Permit the reset's temporary self-target, then
    # replace it immediately with the exact physically screened pair below.
    require_pairs = bool(inner.cfg.allen_require_valid_target_pairs)
    randomize_yaw = inner._randomize_engaged_yaw
    inner.cfg.allen_require_valid_target_pairs = False
    inner._randomize_engaged_yaw = lambda ids: inner._allen_reset_yaw_rad[ids].zero_()
    try:
        inner._restore_inhand_state(env_ids, source)
    finally:
        inner.cfg.allen_require_valid_target_pairs = require_pairs
        inner._randomize_engaged_yaw = randomize_yaw
    inner._allen_target_bank_index[env_ids] = target
    target_pos = inner._inhand_bank_relative_pos[target]
    target_quat = inner._inhand_bank_relative_quat[target]
    inner._adjustment_target_relative_pos[env_ids] = target_pos
    inner._adjustment_target_relative_quat[env_ids] = target_quat

    tool_to_palm_quat = quat_inv(target_quat)
    tool_to_palm_pos = quat_apply(tool_to_palm_quat, -target_pos)
    target_palm_pos, target_palm_quat = combine_frame_transforms(
        inner.object.data.root_pos_w[env_ids],
        inner.object.data.root_quat_w[env_ids],
        tool_to_palm_pos,
        tool_to_palm_quat,
    )
    inner._adjustment_target_palm_pos_w[env_ids] = target_palm_pos
    inner._adjustment_target_palm_quat_w[env_ids] = target_palm_quat
    inner._allen_source_workspace_tier[env_ids] = inner._allen_bank_workspace_tier[source]
    inner._allen_target_quality_improvement[env_ids] = (
        inner._allen_bank_functional_quality[target]
        - inner._allen_bank_functional_quality[source]
    )
    inner._allen_sampled_pair_translation_m[env_ids] = (
        inner._allen_pair_translation_m[source, target]
    )
    inner._allen_sampled_pair_rotation_deg[env_ids] = (
        inner._allen_pair_rotation_deg[source, target]
    )
    inner.episode_length_buf[env_ids] = 0
    inner._allen_hold_count[env_ids] = 0
    inner._allen_succeeded[env_ids] = False
    inner._allen_just_succeeded[env_ids] = False
    inner._allen_previous_pose_potential[env_ids] = 0.0

    compute_intermediate_values(inner)
    inner._update_relative_motion()
    inner._update_adjustment_metrics()
    inner._update_allen_metrics()
    return inner._get_observations()


def run_rollout(
    env,
    inner,
    player,
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    capture_ids: set[int] | None = None,
    recorder=None,
) -> tuple[list[dict], dict[int, list[dict]]]:
    observation = install_pairs(inner, source, target)
    player.reset()
    count = source.numel()
    start_palm = inner.robot.data.body_link_pos_w[:count, inner._palm_body_id].clone()
    target_palm = inner._adjustment_target_palm_pos_w[:count].clone()
    previous_palm = start_palm.clone()
    path_length = torch.zeros(count, device=inner.device)
    peak_speed = torch.zeros(count, device=inner.device)
    peak_lateral = torch.zeros(count, device=inner.device)
    peak_overshoot = torch.zeros(count, device=inner.device)
    peak_height = torch.zeros(count, device=inner.device)
    contact_steps = torch.zeros(count, device=inner.device)
    valid_steps = torch.zeros(count, device=inner.device)
    frames = {env_id: [] for env_id in (capture_ids or set())}
    displacement = target_palm - start_palm
    distance_sq = (displacement * displacement).sum(-1).clamp_min(1.0e-8)
    reference_height = torch.maximum(start_palm[:, 2], target_palm[:, 2])

    # Stop one step before timeout so DirectRLEnv cannot replace terminal state
    # with a newly reset episode before per-environment diagnostics are read.
    steps = int(inner.max_episode_length) - 1
    for step in range(steps):
        action = player.get_normalized_action(
            observation["policy"], deterministic_actions=True
        ).to(inner.device)
        observation, reward, terminated, truncated, _ = env.step(action)
        if not bool(torch.isfinite(reward).all()):
            raise EvaluationFailure(f"non-finite reward at step {step}")
        if step < steps - 1 and bool((terminated[:count] | truncated[:count]).any()):
            raise EvaluationFailure(f"evaluation episode ended early at step {step}")

        palm = inner.robot.data.body_link_pos_w[:count, inner._palm_body_id]
        delta = palm - previous_palm
        step_distance = torch.linalg.vector_norm(delta, dim=-1)
        path_length += step_distance
        peak_speed = torch.maximum(peak_speed, step_distance / float(inner.step_dt))
        previous_palm = palm.clone()
        progress = ((palm - start_palm) * displacement).sum(-1) / distance_sq
        closest = start_palm + progress.clamp(0.0, 1.0).unsqueeze(-1) * displacement
        peak_lateral = torch.maximum(
            peak_lateral, torch.linalg.vector_norm(palm - closest, dim=-1)
        )
        peak_overshoot = torch.maximum(
            peak_overshoot, (progress - 1.0).clamp_min(0.0)
        )
        peak_height = torch.maximum(
            peak_height, (palm[:, 2] - reference_height).clamp_min(0.0)
        )
        contact_steps += inner._allen_final_grasp_valid[:count].float()
        valid_steps += inner._allen_combined_valid[:count].float()

        if capture_ids and step % int(ARGS.frame_stride) == 0:
            for env_id in capture_ids:
                frames[env_id].append(capture_pose_viewer_frame(inner, env_id))
        if recorder is not None:
            recorder.capture()

    # Terminal success is equivalent to the environment endpoint test, but is
    # computed from the pre-reset terminal state retained above.
    success = (
        inner._allen_combined_valid[:count]
        & (inner._allen_hold_count[:count] >= int(inner.cfg.allen_success_hold_steps))
    )
    straight_distance = torch.sqrt(distance_sq)
    final_error = inner._allen_palm_keypoint_error[:count]
    records = []
    for index in range(count):
        source_id = int(source[index])
        target_id = int(target[index])
        tier_id = int(inner._allen_bank_workspace_tier[source_id])
        records.append({
            "pair_index": index,
            "source_id": source_id,
            "target_id": target_id,
            "source_tier": TIERS[tier_id],
            "target_quality_improvement": float(
                inner._allen_bank_functional_quality[target_id]
                - inner._allen_bank_functional_quality[source_id]
            ),
            "pair_translation_m": float(inner._allen_pair_translation_m[source_id, target_id]),
            "pair_rotation_deg": float(inner._allen_pair_rotation_deg[source_id, target_id]),
            "success": bool(success[index]),
            "final_palm_keypoint_error_m": float(final_error[index]),
            "hold_steps": int(inner._allen_hold_count[index]),
            "grasp_valid_fraction": float(contact_steps[index] / steps),
            "combined_valid_fraction": float(valid_steps[index] / steps),
            "straight_palm_displacement_m": float(straight_distance[index]),
            "palm_path_length_m": float(path_length[index]),
            "path_ratio": float(path_length[index] / straight_distance[index].clamp_min(1.0e-5)),
            "peak_palm_speed_mps": float(peak_speed[index]),
            "peak_lateral_excursion_m": float(peak_lateral[index]),
            "peak_overshoot_fraction": float(peak_overshoot[index]),
            "peak_height_excursion_m": float(peak_height[index]),
        })
    return records, frames


def select_replays(records: list[dict], in_distribution: list[bool]) -> list[tuple[str, int]]:
    ids = list(range(len(records)))
    train_ids = [i for i in ids if in_distribution[i]]
    successes = [i for i in train_ids if records[i]["success"]]
    if not train_ids:
        raise EvaluationFailure("no stage-zero in-distribution pairs were evaluated")
    selected: list[tuple[str, int]] = []

    def add(label: str, candidates: list[int], key, reverse: bool = False) -> None:
        ranked = sorted(candidates, key=lambda i: key(records[i]), reverse=reverse)
        for index in ranked:
            if all(existing != index for _, existing in selected):
                selected.append((label, index))
                return

    add("best_success", successes, lambda r: r["final_palm_keypoint_error_m"])
    median = sorted(train_ids, key=lambda i: records[i]["final_palm_keypoint_error_m"])
    if median:
        candidate = median[len(median) // 2]
        if all(existing != candidate for _, existing in selected):
            selected.append(("typical", candidate))
    add("largest_target_move", train_ids, lambda r: r["straight_palm_displacement_m"], True)
    add(
        "largest_flyover_excursion",
        train_ids,
        lambda r: r["peak_height_excursion_m"] + r["peak_lateral_excursion_m"],
        True,
    )
    weakest_retention = min(records[i]["grasp_valid_fraction"] for i in train_ids)
    if weakest_retention < 0.999:
        add("weakest_grasp_retention", train_ids, lambda r: r["grasp_valid_fraction"])
    else:
        support = [i for i in train_ids if records[i]["source_tier"] == "support"]
        add(
            "support_tier",
            support,
            lambda r: r["final_palm_keypoint_error_m"],
            True,
        )
    broad = [i for i in ids if records[i]["source_tier"] == "broad"]
    add("broad_tier_stress", broad, lambda r: r["final_palm_keypoint_error_m"], True)
    failures = [i for i in train_ids if not records[i]["success"]]
    add("worst_in_distribution_failure", failures, lambda r: r["final_palm_keypoint_error_m"], True)
    return selected[: int(ARGS.num_replays)]


class VideoRecorder:
    def __init__(self, inner, output_dir: Path, labels: list[str]):
        self.inner = inner
        self.labels = labels
        self.capture_every = max(1, round(60 / int(ARGS.video_fps)))
        self.step = 0
        self.cameras: list[Camera] = []
        self.writers = []
        if not ARGS.video:
            return
        for env_id, label in enumerate(labels):
            camera = Camera(CameraCfg(
                prim_path=f"/World/AllenAdjustmentEvalCamera_{env_id}",
                update_period=0,
                height=int(ARGS.camera_height),
                width=int(ARGS.camera_width),
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=32.0,
                    focus_distance=400.0,
                    horizontal_aperture=24.0,
                    clipping_range=(0.1, 10.0),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=(0.0, 0.0, 10.0), rot=(1.0, 0.0, 0.0, 0.0),
                    convention="opengl",
                ),
            ))
            self.cameras.append(camera)
            self.writers.append(imageio.get_writer(
                output_dir / f"{label}.mp4", fps=int(ARGS.video_fps),
                codec="libx264", quality=8, macro_block_size=None,
            ))
        inner.sim.reset()

    def set_views(self) -> None:
        for env_id, camera in enumerate(self.cameras):
            pivot = self.inner.object.data.root_pos_w[env_id]
            eye = pivot + torch.tensor(
                (0.36, -0.50, 0.28), device=self.inner.device
            )
            target = pivot + torch.tensor(
                (0.0, 0.0, 0.05), device=self.inner.device
            )
            camera.set_world_poses_from_view(eye.unsqueeze(0), target.unsqueeze(0))

    def capture(self) -> None:
        if not self.cameras:
            return
        self.step += 1
        if self.step % self.capture_every:
            return
        self.inner.sim.render()
        for camera, writer in zip(self.cameras, self.writers, strict=True):
            camera.update(self.capture_every * float(self.inner.step_dt))
            rgb = camera.data.output.get("rgb")
            if rgb is None or rgb.shape[0] != 1:
                raise EvaluationFailure("evaluation camera returned no RGB frame")
            frame = rgb[0, :, :, :3].detach().cpu().numpy()
            if not np.isfinite(frame).all() or float(frame.mean()) <= 0.0:
                raise EvaluationFailure("evaluation camera returned a blank frame")
            writer.append_data(frame.astype(np.uint8))

    def close(self) -> None:
        for writer in self.writers:
            writer.close()
        self.writers.clear()


def write_outputs(
    output_dir: Path,
    records: list[dict],
    selected: list[tuple[str, int]],
    replay_records: list[dict],
    frames: dict[int, list[dict]],
    inner,
) -> None:
    fieldnames = list(records[0])
    with (output_dir / "all_pair_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    object_text, object_path = object_urdf_for_env(inner, 0)
    table_text, table_path = table_urdf_for_env(inner, 0)
    workpiece_text, workpiece_path = workpiece_urdf_for_env(inner)
    manifest = []
    for env_id, ((label, original_index), record) in enumerate(zip(selected, replay_records, strict=True)):
        html = build_pose_viewer_html(
            frames=frames[env_id],
            object_urdf_text=object_text,
            table_urdf_text=table_text,
            workpiece_urdf_text=workpiece_text,
            object_urdf_path=object_path,
            table_urdf_path=table_path,
            workpiece_urdf_path=workpiece_path,
        )
        (output_dir / f"{label}.html").write_text(html, encoding="utf-8")
        manifest.append({"label": label, "screening_pair_index": original_index, **record})
    (output_dir / "replay_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    summary = {
        "checkpoint": str(ARGS.checkpoint.resolve()),
        "evaluated_pairs": len(records),
        "stage_zero_pairs": sum(bool(r["stage_zero_in_distribution"]) for r in records),
        "stage_zero_success_rate": float(np.mean([
            r["success"] for r in records if r["stage_zero_in_distribution"]
        ])),
        "all_improved_pair_success_rate": float(np.mean([r["success"] for r in records])),
        "selected_replays": manifest,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    validate_args()
    output_dir = ARGS.output_dir or (
        ROOT / "outputs/allen_key_workspace_adjustment_eval" / time.strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    cfg = SimToolRealAllenKeyWorkspaceAdjustmentEnvCfg()
    cfg.seed = int(ARGS.seed)
    cfg.grasp_bank_path = str(ARGS.grasp_bank.resolve())
    cfg.adjustment_curriculum_min_eligible_count = 1_000_000_000

    # The environment count is set after materializing pair metadata in a small
    # temporary payload-independent calculation: 130 declared edges is the bank
    # upper bound, and unused environments are harmless during screening.
    payload = json.loads(ARGS.grasp_bank.read_text(encoding="utf-8"))
    declared_count = sum(
        len(entry.get("verification", {}).get("valid_target_ids", []))
        for entry in payload["entries"]
    )
    cfg.scene.num_envs = max(1, declared_count)
    env = gym.make(TASK_ID, cfg=cfg)
    recorder = None
    try:
        inner = env.unwrapped
        source, target, stage_zero = candidate_pairs(inner)
        pair_count = source.numel()
        source = source.to(inner.device)
        target = target.to(inner.device)
        stage_zero = stage_zero.to(inner.device)
        print(
            f"[screen] improved pairs={pair_count} "
            f"stage-zero={int(stage_zero.sum())}", flush=True,
        )
        player = RlPlayer(
            num_observations=inner.cfg.observation_space,
            num_actions=inner.cfg.action_space,
            config_path=str(ARGS.policy_config),
            checkpoint_path=str(ARGS.checkpoint),
            device=str(inner.device),
            num_envs=inner.num_envs,
            coefficient_id=float(ARGS.policy_coef_id),
        )
        screen_records, _ = run_rollout(env, inner, player, source, target)
        stage_zero_list = stage_zero.detach().cpu().tolist()
        for record, is_stage_zero in zip(screen_records, stage_zero_list, strict=True):
            record["stage_zero_in_distribution"] = bool(is_stage_zero)
        selected = select_replays(screen_records, stage_zero_list)
        print("[selected] " + ", ".join(
            f"{label}=({screen_records[index]['source_id']}->{screen_records[index]['target_id']})"
            for label, index in selected
        ), flush=True)

        selected_source = torch.tensor(
            [screen_records[index]["source_id"] for _, index in selected],
            device=inner.device, dtype=torch.long,
        )
        selected_target = torch.tensor(
            [screen_records[index]["target_id"] for _, index in selected],
            device=inner.device, dtype=torch.long,
        )
        labels = [label for label, _ in selected]
        recorder = VideoRecorder(inner, output_dir, labels)
        install_pairs(inner, selected_source, selected_target)
        recorder.set_views()
        replay_records, frames = run_rollout(
            env, inner, player, selected_source, selected_target,
            capture_ids=set(range(len(selected))), recorder=recorder,
        )
        recorder.close()
        recorder = None
        write_outputs(
            output_dir, screen_records, selected, replay_records, frames, inner
        )
        print(f"[output] {output_dir}", flush=True)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if recorder is not None:
            recorder.close()
        env.close()
        APP.close()


if __name__ == "__main__":
    main()
