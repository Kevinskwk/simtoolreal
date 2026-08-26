#!/usr/bin/env python3
"""Evaluate a trained Allen-key turning policy and flag apparent regrasp events."""

from __future__ import annotations

import argparse
import csv
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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--policy-config", type=Path, default=ROOT / "pretrained_policy/config.yaml"
    )
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--curriculum-stage", type=int, default=6)
    parser.add_argument("--maximum-steps", type=int, default=1800)
    parser.add_argument("--viewer-env-ids", type=int, nargs="*", default=(0, 1, 2, 3))
    parser.add_argument("--viewer-stride", type=int, default=15)
    parser.add_argument(
        "--trace-env-id", type=int,
        help="Record per-step rewards and grasp geometry for one environment.",
    )
    parser.add_argument(
        "--compact-gpu-buffers", action="store_true",
        help="Use contact-buffer capacities suitable for small diagnostic batches.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--policy-coef-id", type=float, default=0.0)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    return parser.parse_args()


ARGS = parse_args()
APP = AppLauncher(ARGS).app


import gymnasium as gym  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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
    SimToolRealAllenKeyTurningEnvCfg,
)


TASK_ID = "Isaacsimenvs-SimToolReal-AllenKey-Turning-Direct-v0"


def write_rollout_trace(rows: list[dict[str, float]], output_dir: Path) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with (output_dir / "rollout_trace.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    values = {name: np.asarray([row[name] for row in rows]) for name in fields}
    steps = values["step"]
    figure, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    axes[0].plot(steps, values["palm_z_m"], label="Palm center")
    axes[0].plot(steps, values["handle_center_z_m"], label="Handle center")
    axes[0].plot(steps, values["target_handle_center_z_m"], "--", label="Target handle")
    axes[0].set_ylabel("World z (m)")
    axes[0].legend(ncol=3)

    axes[1].plot(steps, values["palm_minus_handle_z_m"], label="Palm z - handle z")
    axes[1].plot(steps, values["palm_handle_distance_m"], label="3D distance")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Distance (m)")
    axes[1].legend(ncol=2)

    reward_fields = [
        "total_reward", "handle_approach_rew", "keypoint_rew",
        "grasp_maintenance_rew", "kuka_actions_penalty", "hand_actions_penalty",
    ]
    for name in reward_fields:
        axes[2].plot(steps, values[name], label=name)
    axes[2].set_ylabel("Reward / step")
    axes[2].legend(ncol=3, fontsize=8)

    axes[3].plot(steps, values["grasp_quality"], label="Grasp quality")
    axes[3].plot(steps, values["finger_contact_count"] / 5.0, label="Finger contacts / 5")
    axes[3].plot(steps, values["palm_contact"], label="Palm contact")
    axes[3].set_ylabel("Contact state")
    axes[3].set_xlabel("Environment step")
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].legend(ncol=3)
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "rollout_trace.png", dpi=180)
    plt.close(figure)


def validate_inputs() -> None:
    for path in (ARGS.checkpoint, ARGS.policy_config):
        if not path.is_file():
            raise FileNotFoundError(path)


def main() -> None:
    cfg = SimToolRealAllenKeyTurningEnvCfg()
    validate_inputs()
    cfg.sim.device = str(ARGS.device)
    if ARGS.compact_gpu_buffers:
        if int(ARGS.num_envs) > 16:
            raise ValueError("--compact-gpu-buffers requires --num-envs <= 16")
        cfg.sim.physx.gpu_max_rigid_contact_count = 2**16
        cfg.sim.physx.gpu_max_rigid_patch_count = 2**15
    if not 1 <= int(ARGS.num_envs) <= 4096:
        raise ValueError("--num-envs must lie in [1, 4096]")
    if int(ARGS.viewer_stride) <= 0:
        raise ValueError("--viewer-stride must be positive")
    if not 0 <= int(ARGS.curriculum_stage) < len(
        cfg.allen_turn_friction_ranges_nm
    ):
        raise ValueError("--curriculum-stage is outside the configured curriculum")
    cfg.scene.num_envs = int(ARGS.num_envs)
    cfg.allen_turn_curriculum_min_episodes = 1_000_000_000
    cfg.termination.episode_length = int(ARGS.maximum_steps)
    cfg.episode_length_s = int(ARGS.maximum_steps) * 0.02
    env = gym.make(TASK_ID, cfg=cfg)
    inner = env.unwrapped
    inner._turn_curriculum_stage = int(ARGS.curriculum_stage)
    observation, _ = env.reset()
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
    selected = sorted(set(int(value) for value in ARGS.viewer_env_ids))
    if any(not 0 <= value < inner.num_envs for value in selected):
        raise ValueError("viewer environment ID is outside the environment batch")
    trace_env_id = None if ARGS.trace_env_id is None else int(ARGS.trace_env_id)
    if trace_env_id is not None and not 0 <= trace_env_id < inner.num_envs:
        raise ValueError("trace environment ID is outside the environment batch")
    trace_rows: list[dict[str, float]] = []
    frames = {env_id: [] for env_id in selected}
    finished = torch.zeros(inner.num_envs, dtype=torch.bool, device=inner.device)
    final_fields = (
        "allen_turn_subgoals_completed",
        "allen_turn_full_success",
        "allen_turn_ever_loaded_grasp",
        "allen_turn_loaded_grasp_at_end",
        "allen_turn_max_grasp_quality",
        "allen_turn_ever_palm_contact",
        "allen_turn_min_palm_handle_distance_m",
        "allen_turn_max_finger_contact_count",
        "allen_turn_signed_progress_deg",
        "allen_turn_effort_max_ratio",
        "allen_turn_palm_tool_translation_m",
        "allen_turn_palm_tool_rotation_deg",
        "allen_turn_contact_topology_changes",
        "allen_turn_contact_losses",
        "allen_turn_contact_reacquisitions",
        "allen_turn_arm_joint_margin_rad",
    )
    results = {
        name: torch.full((inner.num_envs,), float("nan"), device=inner.device)
        for name in final_fields
    }
    completion_step = torch.full(
        (inner.num_envs,), -1, dtype=torch.long, device=inner.device
    )
    try:
        for step in range(int(ARGS.maximum_steps)):
            action = player.get_normalized_action(
                observation["policy"], deterministic_actions=True
            ).to(inner.device)
            observation, reward, terminated, truncated, infos = env.step(action)
            if not bool(torch.isfinite(reward).all()):
                raise RuntimeError(f"evaluation reward became non-finite at step {step}")
            trace_done = terminated | truncated
            if (
                trace_env_id is not None
                and not bool(finished[trace_env_id])
                and not bool(trace_done[trace_env_id])
            ):
                palm_position, _ = inner._current_palm_center_pose_w()
                handle_position, _ = inner._current_handle_center_pose_w()
                handle_center_local = torch.zeros(
                    inner.num_envs, 3, device=inner.device
                )
                handle_center_local[:, 0] = (
                    float(inner.cfg.assets.allen_key_elbow_x_m)
                    - 0.5 * inner._turn_handle_length
                    + 0.25 * float(inner.cfg.assets.allen_key_handle_across_flats_m)
                )
                target_handle_position = inner.goal_viz.data.root_pos_w + quat_apply(
                    inner.goal_viz.data.root_quat_w, handle_center_local
                )
                reward_terms = inner._reward_terms
                required_terms = (
                    "total_reward", "handle_approach_rew", "keypoint_rew",
                    "grasp_maintenance_rew", "kuka_actions_penalty",
                    "hand_actions_penalty",
                )
                missing = [name for name in required_terms if name not in reward_terms]
                if missing:
                    raise RuntimeError(f"rollout trace reward terms are missing: {missing}")
                env_id = trace_env_id
                trace_rows.append({
                    "step": float(step),
                    "palm_z_m": float(palm_position[env_id, 2]),
                    "handle_center_z_m": float(handle_position[env_id, 2]),
                    "target_handle_center_z_m": float(target_handle_position[env_id, 2]),
                    "palm_minus_handle_z_m": float(
                        palm_position[env_id, 2] - handle_position[env_id, 2]
                    ),
                    "palm_handle_distance_m": float(
                        inner._turn_palm_handle_distance[env_id]
                    ),
                    "tool_pose_error_m": float(inner._keypoints_max_dist[env_id]),
                    "grasp_quality": float(inner._turn_grasp_quality[env_id]),
                    "finger_contact_count": float(
                        inner._turn_finger_contact[env_id].sum()
                    ),
                    "palm_contact": float(inner._turn_palm_contact[env_id]),
                    **{
                        name: float(reward_terms[name][env_id])
                        for name in required_terms
                    },
                })
            if step % int(ARGS.viewer_stride) == 0:
                for env_id in selected:
                    if not bool(finished[env_id]):
                        frames[env_id].append(capture_pose_viewer_frame(inner, env_id))
            done = (terminated | truncated) & ~finished
            if bool(done.any()):
                episode_final = infos.get("episode_final")
                if not isinstance(episode_final, dict):
                    raise RuntimeError("evaluation did not receive episode_final metrics")
                for name in final_fields:
                    value = episode_final.get(name)
                    if value is None or value.shape != (inner.num_envs,):
                        raise RuntimeError(f"episode_final metric {name!r} is missing")
                    results[name][done] = value[done]
                completion_step[done] = step + 1
                finished |= done
            if bool(finished.all()):
                break
        if not bool(finished.all()):
            raise RuntimeError(
                f"evaluation ended before {int((~finished).sum())} environments completed"
            )
        arrays = {name: value.detach().cpu().numpy() for name, value in results.items()}
        regrasp = (
            (
                (arrays["allen_turn_palm_tool_translation_m"] >= 0.03)
                | (arrays["allen_turn_palm_tool_rotation_deg"] >= 30.0)
            )
            & (arrays["allen_turn_contact_topology_changes"] >= 2)
            & (arrays["allen_turn_contact_reacquisitions"] >= 1)
            & (arrays["allen_turn_subgoals_completed"] >= 1)
            & (arrays["allen_turn_ever_loaded_grasp"] > 0.5)
        )
        output_dir = ARGS.output_dir or (
            ROOT / "outputs" / "allen_key_turning_evaluation"
            / time.strftime("%Y%m%d_%H%M%S")
        )
        output_dir.mkdir(parents=True, exist_ok=False)
        write_rollout_trace(trace_rows, output_dir)
        fieldnames = ["env_id", "completion_step", *final_fields, "apparent_regrasp"]
        with (output_dir / "outcomes.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for env_id in range(inner.num_envs):
                writer.writerow({
                    "env_id": env_id,
                    "completion_step": int(completion_step[env_id]),
                    **{name: float(arrays[name][env_id]) for name in final_fields},
                    "apparent_regrasp": int(regrasp[env_id]),
                })
        table_text, table_path = table_urdf_for_env(inner, 0)
        workpiece_text, workpiece_path = workpiece_urdf_for_env(inner)
        for env_id, env_frames in frames.items():
            if not env_frames:
                continue
            local_object_text, local_object_path = object_urdf_for_env(inner, env_id)
            (output_dir / f"env_{env_id:04d}.html").write_text(build_pose_viewer_html(
                frames=env_frames,
                object_urdf_text=local_object_text,
                table_urdf_text=table_text,
                workpiece_urdf_text=workpiece_text,
                object_urdf_path=local_object_path,
                table_urdf_path=table_path,
                workpiece_urdf_path=workpiece_path,
            ))
        succeeded = arrays["allen_turn_full_success"] > 0.5
        acquired = arrays["allen_turn_ever_loaded_grasp"] > 0.5
        summary = {
            "checkpoint": str(ARGS.checkpoint.resolve()),
            "curriculum_stage": int(ARGS.curriculum_stage),
            "coulomb_friction_range_nm": list(
                cfg.allen_turn_friction_ranges_nm[int(ARGS.curriculum_stage)]
            ),
            "damping_range_nm_per_radps": list(
                cfg.allen_turn_damping_ranges_nm_per_radps[
                    int(ARGS.curriculum_stage)
                ]
            ),
            "num_envs": inner.num_envs,
            "full_turn_success_rate": float(succeeded.mean()),
            "loaded_grasp_acquisition_rate": float(acquired.mean()),
            "full_turn_success_given_loaded_grasp": float(
                succeeded[acquired].mean() if acquired.any() else 0.0
            ),
            "loaded_grasp_at_end_rate": float(
                (arrays["allen_turn_loaded_grasp_at_end"] > 0.5).mean()
            ),
            "maximum_grasp_quality_mean": float(
                arrays["allen_turn_max_grasp_quality"].mean()
            ),
            "maximum_grasp_quality_positive_rate": float(
                (arrays["allen_turn_max_grasp_quality"] > 0.0).mean()
            ),
            "palm_contact_ever_rate": float(
                (arrays["allen_turn_ever_palm_contact"] > 0.5).mean()
            ),
            "minimum_palm_handle_distance_m_percentiles": {
                str(percentile): float(np.percentile(
                    arrays["allen_turn_min_palm_handle_distance_m"], percentile
                ))
                for percentile in (0, 10, 25, 50, 75, 90, 100)
            },
            "maximum_finger_contact_count_mean": float(
                arrays["allen_turn_max_finger_contact_count"].mean()
            ),
            "subgoals_completed_mean": float(
                arrays["allen_turn_subgoals_completed"].mean()
            ),
            "apparent_regrasp_rate": float(regrasp.mean()),
            "apparent_regrasp_success_rate": float(
                succeeded[regrasp].mean() if regrasp.any() else 0.0
            ),
            "output_dir": str(output_dir.resolve()),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[pass] {json.dumps(summary, sort_keys=True)}", flush=True)
    finally:
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
