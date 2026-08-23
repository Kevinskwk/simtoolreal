#!/usr/bin/env python3
"""Deterministic smoke test for the Allen-key turning fixture and objective."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--test-torque-nm", type=float, default=0.20)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/allen_key_turning_validation/fixture_rollout.html",
    )
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    return parser.parse_args()


ARGS = parse_args()
APP = AppLauncher(ARGS).app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaacsimenvs  # noqa: E402,F401
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
from isaacsimenvs.tasks.simtoolreal.utils.allen_key_turning_utils import (  # noqa: E402
    yaw_from_quaternion,
)


TASK_ID = "Isaacsimenvs-SimToolReal-AllenKey-Turning-Direct-v0"


def require_finite(step: int, observation, reward) -> None:
    tensors = {
        "policy observation": observation["policy"],
        "critic observation": observation["critic"],
        "reward": reward,
    }
    for name, value in tensors.items():
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"{name} became non-finite at step {step}")


def main() -> None:
    if not 2 <= int(ARGS.num_envs) <= 64:
        raise ValueError("--num-envs must lie in [2, 64]")
    if int(ARGS.steps) < 20:
        raise ValueError("--steps must be at least 20")
    if float(ARGS.test_torque_nm) <= 0.0:
        raise ValueError("--test-torque-nm must be positive")
    cfg = SimToolRealAllenKeyTurningEnvCfg()
    cfg.scene.num_envs = int(ARGS.num_envs)
    cfg.allen_turn_require_calibrated_load = False
    cfg.allen_turn_calibrated_torque_nm = float(ARGS.test_torque_nm)
    cfg.allen_turn_resistance_fractions = (1.0,) * len(
        cfg.allen_turn_resistance_fractions
    )
    cfg.allen_turn_acquisition_timeout_steps = 1_000_000
    cfg.termination.episode_length = 1_000_000
    cfg.episode_length_s = 20_000.0
    cfg.allen_turn_curriculum_min_episodes = 1_000_000
    env = gym.make(TASK_ID, cfg=cfg)
    print("[smoke] environment constructed", flush=True)
    inner = env.unwrapped
    action = torch.zeros(inner.num_envs, inner.cfg.action_space, device=inner.device)
    frames = []
    try:
        print("[smoke] resetting environment", flush=True)
        observation, _ = env.reset()
        print("[smoke] reset complete", flush=True)
        if observation["policy"].shape != (inner.num_envs, inner.cfg.observation_space):
            raise RuntimeError(
                "policy observation shape mismatch: "
                f"{tuple(observation['policy'].shape)} vs {inner.cfg.observation_space}"
            )
        if observation["critic"].shape != (inner.num_envs, inner.cfg.state_space):
            raise RuntimeError(
                "critic observation shape mismatch: "
                f"{tuple(observation['critic'].shape)} vs {inner.cfg.state_space}"
            )
        env_ids = torch.arange(inner.num_envs, device=inner.device)
        pivot = inner.scene.env_origins + torch.tensor(
            [0.45, 0.45, 0.70], device=inner.device
        )
        quaternion = torch.zeros(inner.num_envs, 4, device=inner.device)
        quaternion[:, 0] = 1.0
        pivot_tool = torch.tensor(
            inner.cfg.allen_turn_screw_pivot_tool_m, device=inner.device
        ).expand(inner.num_envs, -1)
        position = pivot - torch.cat((
            pivot_tool[:, :2], pivot_tool[:, 2:3]
        ), dim=-1)
        inner.object.write_root_pose_to_sim(
            torch.cat((position, quaternion), dim=-1), env_ids=env_ids
        )
        inner.object.write_root_velocity_to_sim(
            torch.zeros(inner.num_envs, 6, device=inner.device), env_ids=env_ids
        )
        socket_position = pivot + torch.tensor(
            inner.cfg.allen_turn_socket_root_from_pivot_m, device=inner.device
        )
        inner.workpiece.write_root_pose_to_sim(
            torch.cat((socket_position, quaternion), dim=-1), env_ids=env_ids
        )
        inner._turn_pivot_w.copy_(pivot)
        inner._turn_initial_tool_pos.copy_(position)
        inner._turn_initial_tool_quat.copy_(quaternion)
        inner._turn_initial_yaw.zero_()
        inner._turn_previous_yaw.zero_()
        inner._turn_cumulative_angle.zero_()
        inner._write_turn_goal(env_ids, torch.zeros(inner.num_envs, device=inner.device))
        frames.append(capture_pose_viewer_frame(inner, 0))
        for step in range(10):
            observation, reward, terminated, truncated, _ = env.step(action)
            require_finite(step, observation, reward)
            if bool(terminated.any()) or bool(truncated.any()):
                raise RuntimeError("fixture task ended during acquisition smoke phase")
        initial_angle = inner._turn_cumulative_angle.clone()
        inner._turn_phase.fill_(1)
        inner._turn_acquired.fill_(True)
        directions = torch.ones(inner.num_envs, device=inner.device)
        directions[inner.num_envs // 2 :] = -1.0
        inner._turn_direction.copy_(directions)
        inner._turn_external_torque_override_nm = (
            directions * float(ARGS.test_torque_nm)
        )
        inner._turn_target_angle.copy_(directions * math.radians(30.0))
        inner.object.write_root_velocity_to_sim(
            torch.zeros(inner.num_envs, 6, device=inner.device), env_ids=env_ids
        )
        inner._turn_previous_yaw.copy_(yaw_from_quaternion(inner.object.data.root_quat_w))
        inner._turn_cumulative_angle.zero_()
        initial_angle.zero_()
        inner._write_turn_goal(env_ids, inner._turn_target_angle)
        for step in range(int(ARGS.steps)):
            observation, reward, terminated, truncated, _ = env.step(action)
            require_finite(step + 10, observation, reward)
            if bool(terminated.any()) or bool(truncated.any()):
                raise RuntimeError("fixture task ended during loaded smoke phase")
            if step in (0, int(ARGS.steps) // 2, int(ARGS.steps) - 1):
                frames.append(capture_pose_viewer_frame(inner, 0))
        directed_motion = directions * (inner._turn_cumulative_angle - initial_angle)
        if not bool((directed_motion < 0.0).all()):
            raise RuntimeError(
                "opposing torque sign is incorrect for one or both directions: "
                f"directed_motion={directed_motion.tolist()}"
            )
        if float(inner._turn_constraint_position_error.max()) > float(
            inner.cfg.allen_turn_constraint_position_tolerance_m
        ):
            raise RuntimeError(
                "screw pivot constraint is unstable: maximum error="
                f"{float(inner._turn_constraint_position_error.max()):.6f} m"
            )
        if float(torch.rad2deg(inner._turn_constraint_tilt_error).max()) > float(
            inner.cfg.allen_turn_constraint_tilt_tolerance_deg
        ):
            raise RuntimeError(
                "screw axis constraint is unstable: maximum tilt="
                f"{float(torch.rad2deg(inner._turn_constraint_tilt_error).max()):.3f} deg"
            )
        ARGS.output.parent.mkdir(parents=True, exist_ok=True)
        object_text, object_path = object_urdf_for_env(inner, 0)
        table_text, table_path = table_urdf_for_env(inner, 0)
        workpiece_text, workpiece_path = workpiece_urdf_for_env(inner)
        ARGS.output.write_text(build_pose_viewer_html(
            frames=frames,
            object_urdf_text=object_text,
            table_urdf_text=table_text,
            workpiece_urdf_text=workpiece_text,
            object_urdf_path=object_path,
            table_urdf_path=table_path,
            workpiece_urdf_path=workpiece_path,
        ))
        summary = {
            "num_envs": inner.num_envs,
            "policy_observation_dim": inner.cfg.observation_space,
            "critic_observation_dim": inner.cfg.state_space,
            "test_torque_nm": float(ARGS.test_torque_nm),
            "directed_motion_deg": torch.rad2deg(directed_motion).tolist(),
            "maximum_pivot_error_m": float(inner._turn_constraint_position_error.max()),
            "maximum_tilt_error_deg": float(
                torch.rad2deg(inner._turn_constraint_tilt_error).max()
            ),
            "viewer": str(ARGS.output.resolve()),
        }
        summary_path = ARGS.output.with_suffix(".json")
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[pass] {json.dumps(summary, sort_keys=True)}", flush=True)
    finally:
        if hasattr(inner, "_turn_external_torque_override_nm"):
            del inner._turn_external_torque_override_nm
        env.close()
        print("[smoke] environment closed", flush=True)


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
