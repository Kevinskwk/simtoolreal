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
    parser.add_argument("--calibration-json", type=Path, required=True)
    parser.add_argument(
        "--policy-config", type=Path, default=ROOT / "pretrained_policy/config.yaml"
    )
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--curriculum-stage", type=int, default=6)
    parser.add_argument("--maximum-steps", type=int, default=1800)
    parser.add_argument("--viewer-env-ids", type=int, nargs="*", default=(0, 1, 2, 3))
    parser.add_argument("--viewer-stride", type=int, default=15)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--policy-coef-id", type=float, default=0.0)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    return parser.parse_args()


ARGS = parse_args()
APP = AppLauncher(ARGS).app


import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

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


def calibrated_torque() -> float:
    for path in (ARGS.checkpoint, ARGS.calibration_json, ARGS.policy_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    payload = json.loads(ARGS.calibration_json.read_text())
    value = payload.get("recommended_training_torque_nm")
    if (
        payload.get("schema_version") != 1
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0.0
    ):
        raise RuntimeError(f"invalid resistance calibration: {ARGS.calibration_json}")
    return float(value)


def main() -> None:
    torque = calibrated_torque()
    if not 1 <= int(ARGS.num_envs) <= 4096:
        raise ValueError("--num-envs must lie in [1, 4096]")
    if int(ARGS.viewer_stride) <= 0:
        raise ValueError("--viewer-stride must be positive")
    cfg = SimToolRealAllenKeyTurningEnvCfg()
    if not 0 <= int(ARGS.curriculum_stage) < len(cfg.allen_turn_resistance_fractions):
        raise ValueError("--curriculum-stage is outside the configured curriculum")
    cfg.scene.num_envs = int(ARGS.num_envs)
    cfg.allen_turn_calibrated_torque_nm = torque
    cfg.allen_turn_require_calibrated_load = True
    cfg.allen_turn_curriculum_min_episodes = 1_000_000_000
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
    frames = {env_id: [] for env_id in selected}
    finished = torch.zeros(inner.num_envs, dtype=torch.bool, device=inner.device)
    final_fields = (
        "allen_turn_acquired",
        "allen_turn_subgoals_completed",
        "allen_turn_full_success",
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
        )
        output_dir = ARGS.output_dir or (
            ROOT / "outputs" / "allen_key_turning_evaluation"
            / time.strftime("%Y%m%d_%H%M%S")
        )
        output_dir.mkdir(parents=True, exist_ok=False)
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
        acquired = arrays["allen_turn_acquired"] > 0.5
        succeeded = arrays["allen_turn_full_success"] > 0.5
        summary = {
            "checkpoint": str(ARGS.checkpoint.resolve()),
            "calibration_json": str(ARGS.calibration_json.resolve()),
            "curriculum_stage": int(ARGS.curriculum_stage),
            "resistance_torque_nm": torque * float(
                cfg.allen_turn_resistance_fractions[int(ARGS.curriculum_stage)]
            ),
            "num_envs": inner.num_envs,
            "acquisition_success_rate": float(acquired.mean()),
            "full_turn_success_rate": float(succeeded.mean()),
            "full_turn_success_given_acquisition": float(
                succeeded.sum() / max(acquired.sum(), 1)
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
