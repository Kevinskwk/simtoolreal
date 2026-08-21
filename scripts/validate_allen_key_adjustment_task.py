#!/usr/bin/env python3
"""Validate Allen-key grasp, socket support, target reachability, and episode logic."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--settle-steps", type=int, default=60)
    parser.add_argument("--reset-yaw-range-deg", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    return parser.parse_args()


ARGS = parse_args()
APP = AppLauncher(ARGS).app


import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
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
    SimToolRealAllenKeyAdjustmentEnvCfg,
)
from isaacsimenvs.tasks.simtoolreal.utils.grasp_evaluator import (  # noqa: E402
    GraspEvaluatorThresholds,
    UrdfKinematics,
    pose_matrix,
    solve_arm_ik,
)
from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import compute_obs_dim  # noqa: E402


TASK_ID = "Isaacsimenvs-SimToolReal-AllenKey-Adjustment-Direct-v0"


def require_finite(step: int, observation: dict, reward: torch.Tensor) -> None:
    named = {"policy": observation["policy"], "critic": observation["critic"], "reward": reward}
    for name, value in named.items():
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"{name} contains NaN or Inf at validation step {step}")


def verify_sampled_target_reachability(inner) -> tuple[float, float]:
    robot_urdf = ROOT / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
    kinematics = UrdfKinematics(robot_urdf)
    base = np.eye(4)
    base[1, 3] = 0.8
    env_id = 0
    target_pos_local = (
        inner._adjustment_target_palm_pos_w[env_id] - inner.scene.env_origins[env_id]
    )
    target = pose_matrix(
        target_pos_local.detach().cpu().numpy(),
        inner._adjustment_target_palm_quat_w[env_id].detach().cpu().numpy(),
    )
    joints = inner.robot.data.joint_pos[env_id, inner._perm_lab_to_canon].detach().cpu().numpy()
    _, position_error, rotation_error, _, _ = solve_arm_ik(
        kinematics,
        target,
        joints[:7],
        joints[7:],
        base,
        GraspEvaluatorThresholds(
            ik_position_m=0.003, ik_orientation_deg=3.0, max_ik_iterations=300
        ),
    )
    if position_error > 0.003 or rotation_error > 3.0:
        raise RuntimeError(
            "sampled target palm transform is not arm-reachable: "
            f"position={position_error:.6f}m rotation={rotation_error:.3f}deg"
        )
    return float(position_error), float(rotation_error)


def main() -> None:
    if ARGS.num_envs <= 0 or ARGS.settle_steps <= 0:
        raise ValueError("num-envs and settle-steps must be positive")
    cfg = SimToolRealAllenKeyAdjustmentEnvCfg()
    cfg.scene.num_envs = int(ARGS.num_envs)
    if ARGS.reset_yaw_range_deg is not None:
        cfg.allen_reset_yaw_range_stages_deg = (
            float(ARGS.reset_yaw_range_deg),
        ) * len(cfg.adjustment_target_rotation_deg)
    cfg.adjustment_curriculum_min_eligible_count = 1_000_000
    env = gym.make(TASK_ID, cfg=cfg)
    frames: list[dict] = []
    try:
        observation, _ = env.reset()
        inner = env.unwrapped
        expected_policy = (ARGS.num_envs, compute_obs_dim(cfg.obs.obs_list))
        expected_critic = (ARGS.num_envs, compute_obs_dim(cfg.obs.state_list))
        if observation["policy"].shape != expected_policy:
            raise RuntimeError(f"unexpected policy shape {tuple(observation['policy'].shape)}")
        if observation["critic"].shape != expected_critic:
            raise RuntimeError(f"unexpected critic shape {tuple(observation['critic'].shape)}")
        ik_position, ik_rotation = verify_sampled_target_reachability(inner)

        inner._replay_target_lab_order = inner._cur_targets.clone()
        actions = inner._stable_previous_action.clone()
        for step in range(int(ARGS.settle_steps)):
            observation, reward, terminated, truncated, _ = env.step(actions)
            require_finite(step, observation, reward)
            if bool(terminated.any()) or bool(truncated.any()):
                raise RuntimeError("episode ended during deterministic grasp settling")

        if not bool(inner._allen_socket_valid.all()):
            raise RuntimeError(
                "socket engagement was not retained during settling: "
                f"ratio={float(inner._allen_socket_valid.float().mean().item()):.3f}"
            )
        if not bool(inner._allen_palm_contact.all()):
            current_pos, current_quat = inner._palm_tool_relative()
            relative_drift = torch.linalg.vector_norm(
                current_pos - inner._inhand_bank_relative_pos[0], dim=-1
            )
            raise RuntimeError(
                "whole-palm tool contact was not retained during settling: "
                f"ratio={float(inner._allen_palm_contact.float().mean().item()):.3f} "
                f"contact={inner._allen_palm_contact.tolist()} "
                f"force_n={inner._allen_palm_force_n.tolist()} "
                f"yaw_deg={torch.rad2deg(inner._allen_reset_yaw_rad).tolist()} "
                f"support={inner._stable_support_count.tolist()} "
                f"relative_drift_m={relative_drift.tolist()}"
            )

        # Make the current physically settled relationship the deterministic hold target.
        current_pos, current_quat = inner._palm_tool_relative()
        inner._adjustment_target_relative_pos.copy_(current_pos)
        inner._adjustment_target_relative_quat.copy_(current_quat)
        inner._adjustment_initial_tool_pos.copy_(inner.object.data.root_pos_w)
        inner._adjustment_initial_tool_quat.copy_(inner.object.data.root_quat_w)
        inner.episode_length_buf[:] = int(cfg.allen_adjustment_steps)
        for step in range(int(cfg.allen_success_hold_steps)):
            observation, reward, terminated, truncated, _ = env.step(actions)
            require_finite(ARGS.settle_steps + step, observation, reward)
            frames.append(capture_pose_viewer_frame(inner, 0))
            if bool(terminated.any()) or bool(truncated.any()):
                raise RuntimeError("episode ended before the post-adjustment hold completed")
        if int(inner._allen_hold_count.min().item()) < int(cfg.allen_success_hold_steps):
            raise RuntimeError(
                "deterministic valid target did not satisfy the post-adjustment hold gate"
            )

        # A separate reset audits that only the fixed 480-step timeout ends episodes.
        env.reset()
        inner._replay_target_lab_order = inner._cur_targets.clone()
        actions = inner._stable_previous_action.clone()
        for step in range(480):
            observation, reward, terminated, truncated, _ = env.step(actions)
            require_finite(step, observation, reward)
            if bool(terminated.any()):
                raise RuntimeError(f"Allen task terminated early at step {step + 1}")
            expected_timeout = step == 479
            if bool(truncated.any()) != expected_timeout:
                raise RuntimeError(
                    f"fixed-length timeout mismatch at step {step + 1}: "
                    f"truncated={int(truncated.sum().item())}"
                )

        output = ARGS.output or (
            ROOT / "outputs/allen_key_adjustment_validation/allen_key_hold_rollout.html"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        object_text, object_path = object_urdf_for_env(inner, 0)
        table_text, table_path = table_urdf_for_env(inner, 0)
        workpiece_text, workpiece_path = workpiece_urdf_for_env(inner)
        output.write_text(build_pose_viewer_html(
            frames=frames,
            object_urdf_text=object_text,
            table_urdf_text=table_text,
            workpiece_urdf_text=workpiece_text,
            object_urdf_path=object_path,
            table_urdf_path=table_path,
            workpiece_urdf_path=workpiece_path,
        ))
        summary = {
            "policy_observation_dim": expected_policy[1],
            "critic_observation_dim": expected_critic[1],
            "sampled_target_ik_position_error_m": ik_position,
            "sampled_target_ik_rotation_error_deg": ik_rotation,
            "hold_steps": int(cfg.allen_success_hold_steps),
            "episode_steps": 480,
            "viewer": str(output.resolve()),
        }
        output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
        print(f"[pass] {json.dumps(summary, sort_keys=True)}", flush=True)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        env.close()
        APP.close()


if __name__ == "__main__":
    main()
