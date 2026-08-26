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
    parser.add_argument("--initial-angular-speed-radps", type=float, default=2.0)
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
    quat_apply,
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
    if float(ARGS.initial_angular_speed_radps) <= 0.0:
        raise ValueError("--initial-angular-speed-radps must be positive")
    cfg = SimToolRealAllenKeyTurningEnvCfg()
    cfg.scene.num_envs = int(ARGS.num_envs)
    stage_count = len(cfg.allen_turn_friction_ranges_nm)
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
        if inner.cfg.observation_space != 140:
            raise RuntimeError(
                "Allen-key actor observation no longer matches the vanilla policy: "
                f"expected 140, got {inner.cfg.observation_space}"
            )
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
        finger_sensor_groups = getattr(inner, "_turn_finger_contact_sensors", None)
        if (
            finger_sensor_groups is None
            or len(finger_sensor_groups) != 5
            or sum(len(group) for group in finger_sensor_groups) != 22
        ):
            raise RuntimeError(
                "Allen-key task did not construct five finger groups with 22 link sensors"
            )
        palm_position, _ = inner._current_palm_center_pose_w()
        wrist_position = inner.robot.data.body_link_pos_w[:, inner._palm_body_id]
        palm_offset = torch.linalg.vector_norm(palm_position - wrist_position, dim=-1)
        if not torch.allclose(
            palm_offset,
            torch.full_like(palm_offset, 0.13821),
            atol=1.0e-5,
        ):
            raise RuntimeError(
                "physical palm center is not offset correctly from merged wrist frame"
            )
        expected_first_target = math.radians(
            float(inner.cfg.allen_turn_goal_increment_deg)
        )
        if not bool((inner._turn_phase == 1).all()):
            raise RuntimeError("Allen-key reset did not start directly in turning mode")
        if not torch.allclose(
            inner._turn_target_angle.abs(),
            torch.full_like(inner._turn_target_angle, expected_first_target),
            atol=1.0e-6,
        ):
            raise RuntimeError(
                "Allen-key reset did not publish the first nonzero tool-pose goal"
            )
        sampled_reach = inner._turn_initial_shoulder_to_handle_m.clone()
        reach_limit = float(
            inner.cfg.allen_turn_max_shoulder_to_handle_stages_m[
                inner._turn_curriculum_stage
            ]
        )
        if float(sampled_reach.max()) > reach_limit + 1.0e-6:
            raise RuntimeError(
                "Allen-key reset admitted an overextended handle pose: "
                f"maximum={float(sampled_reach.max()):.4f} m, "
                f"limit={reach_limit:.4f} m"
            )
        sampled_palm_distance = inner._turn_initial_palm_handle_distance.clone()
        palm_distance_limit = float(
            inner.cfg.allen_turn_max_initial_palm_handle_distance_stages_m[
                inner._turn_curriculum_stage
            ]
        )
        if float(sampled_palm_distance.max()) > palm_distance_limit + 1.0e-6:
            raise RuntimeError(
                "Allen-key reset admitted a handle outside the acquisition workspace: "
                f"maximum={float(sampled_palm_distance.max()):.4f} m, "
                f"limit={palm_distance_limit:.4f} m"
            )
        env_ids = torch.arange(inner.num_envs, device=inner.device)
        # Hold the randomized reset posture throughout the fixture-only test.
        # Zero policy actions drive hand joints toward their normalized
        # midpoint, which can otherwise make the robot strike the test key.
        inner._replay_target_lab_order = inner.robot.data.joint_pos.clone()
        pivot = inner._turn_pivot_w.clone()
        quaternion = inner._turn_initial_tool_quat.clone()
        pivot_tool = torch.tensor(
            inner.cfg.allen_turn_screw_pivot_tool_m, device=inner.device
        ).expand(inner.num_envs, -1)
        position = pivot - quat_apply(quaternion, pivot_tool)
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
        inner._turn_stiction_yaw.zero_()
        inner._turn_friction_stuck.fill_(True)
        inner._turn_fixture_torque_nm.zero_()
        inner._turn_cumulative_angle.zero_()
        inner._write_turn_goal(env_ids, torch.zeros(inner.num_envs, device=inner.device))
        frames.append(capture_pose_viewer_frame(inner, 0))
        # Joint creation happens after the aligned fixture reset. Exclude its
        # short solver-settling transient from the passive-dwell measurement.
        for step in range(5):
            observation, reward, terminated, truncated, _ = env.step(action)
            require_finite(step, observation, reward)
            if bool(terminated.any()) or bool(truncated.any()):
                raise RuntimeError("fixture task ended while settling the revolute joint")
        inner._turn_stiction_yaw.copy_(yaw_from_quaternion(inner.object.data.root_quat_w))
        inner.object.write_root_velocity_to_sim(
            torch.zeros(inner.num_envs, 6, device=inner.device), env_ids=env_ids
        )
        passive_initial_yaw = yaw_from_quaternion(inner.object.data.root_quat_w).clone()
        passive_max_fixture_torque = torch.zeros(
            inner.num_envs, device=inner.device
        )
        for step in range(10):
            observation, reward, terminated, truncated, _ = env.step(action)
            require_finite(step, observation, reward)
            passive_max_fixture_torque = torch.maximum(
                passive_max_fixture_torque,
                inner._turn_fixture_torque_nm.abs(),
            )
            if bool(terminated.any()) or bool(truncated.any()):
                raise RuntimeError("fixture task ended during passive smoke phase")
        if not bool((inner._reward_terms["lifting_rew"] == 0.0).all()):
            raise RuntimeError("engaged Allen-key task emitted an inherited lifting reward")
        if not bool((inner._reward_terms["lift_bonus_rew"] == 0.0).all()):
            raise RuntimeError("engaged Allen-key task emitted an inherited lift bonus")
        if not bool((inner._reward_terms["fingertip_delta_rew"] == 0.0).all()):
            raise RuntimeError("Allen-key task used fingertip distance to the socket root")
        required_reward_terms = {
            "handle_approach_rew",
            "first_loaded_grasp_bonus",
            "grasp_maintenance_rew",
            "keypoint_rew",
            "subgoal_bonus",
            "full_turn_bonus",
            "finger_effort_penalty",
            "action_rate_penalty",
        }
        missing = required_reward_terms.difference(inner._reward_terms)
        if missing:
            raise RuntimeError(f"Allen-key clean reward terms are missing: {sorted(missing)}")
        if not bool((inner._reward_terms["handle_approach_rew"] < 0.0).all()):
            raise RuntimeError(
                "ungrasped passive states did not receive a persistent approach cost"
            )
        if not bool((inner._reward_terms["grasp_maintenance_rew"] == 0.0).all()):
            raise RuntimeError(
                "grasp maintenance reward was active before confirmed acquisition"
            )
        removed_reward_terms = {
            "turn_progress_rew",
            "subgoal_progress_rew",
            "constraint_penalty",
            "final_hold_rew",
            "pregrasp_palm_distance_penalty",
        }
        unexpected = removed_reward_terms.intersection(inner._reward_terms)
        if unexpected:
            raise RuntimeError(f"redundant Allen-key reward terms remain: {sorted(unexpected)}")
        passive_yaw_delta = torch.atan2(
            torch.sin(yaw_from_quaternion(inner.object.data.root_quat_w) - passive_initial_yaw),
            torch.cos(yaw_from_quaternion(inner.object.data.root_quat_w) - passive_initial_yaw),
        ).abs()
        if float(torch.rad2deg(passive_yaw_delta).max()) > 0.5:
            raise RuntimeError(
                "friction fixture self-rotated from passive rest: maximum drift="
                f"{float(torch.rad2deg(passive_yaw_delta).max()):.4f} deg"
            )
        initial_angle = inner._turn_cumulative_angle.clone()
        inner._turn_phase.fill_(1)
        directions = torch.ones(inner.num_envs, device=inner.device)
        directions[inner.num_envs // 2 :] = -1.0
        inner._turn_direction.copy_(directions)
        kinetic_limit_nm = float(
            inner.cfg.allen_turn_friction_ranges_nm[0][1]
        )
        inner._turn_coulomb_friction_nm.fill_(kinetic_limit_nm)
        inner._turn_damping_nm_per_radps.fill_(float(
            inner.cfg.allen_turn_damping_ranges_nm_per_radps[0][1]
        ))
        static_limit_nm = kinetic_limit_nm * float(
            inner.cfg.allen_turn_static_to_kinetic_friction_ratio
        )
        inner._turn_target_angle.copy_(directions * math.radians(30.0))
        initial_angular_speed = float(ARGS.initial_angular_speed_radps)
        initial_velocity = torch.zeros(inner.num_envs, 6, device=inner.device)
        initial_velocity[:, 5] = directions * initial_angular_speed
        inner.object.write_root_velocity_to_sim(
            initial_velocity, env_ids=env_ids
        )
        inner._turn_friction_stuck.fill_(False)
        inner._turn_previous_yaw.copy_(yaw_from_quaternion(inner.object.data.root_quat_w))
        inner._turn_cumulative_angle.zero_()
        initial_angle.zero_()
        inner._write_turn_goal(env_ids, inner._turn_target_angle)
        first_fixture_torque = None
        for step in range(int(ARGS.steps)):
            observation, reward, terminated, truncated, _ = env.step(action)
            require_finite(step + 10, observation, reward)
            if first_fixture_torque is None:
                first_fixture_torque = inner._turn_fixture_torque_nm.clone()
            if bool(terminated.any()) or bool(truncated.any()):
                raise RuntimeError("fixture task ended during loaded smoke phase")
            if step in (0, int(ARGS.steps) // 2, int(ARGS.steps) - 1):
                frames.append(capture_pose_viewer_frame(inner, 0))
        directed_motion = directions * (inner._turn_cumulative_angle - initial_angle)
        if first_fixture_torque is None or not bool(
            (directions * first_fixture_torque < 0.0).all()
        ):
            raise RuntimeError(
                "friction torque did not oppose the initial angular velocity"
            )
        final_angular_speed = inner.object.data.root_ang_vel_w[:, 2].abs()
        if not bool((final_angular_speed < initial_angular_speed).all()):
            raise RuntimeError(
                "friction and damping failed to dissipate the initial angular motion: "
                f"final_speed={final_angular_speed.tolist()}"
            )
        if float(inner._turn_constraint_position_error.max()) > float(
            inner.cfg.allen_turn_constraint_position_tolerance_m
        ):
            current_pivot = inner.object.data.root_pos_w + quat_apply(
                inner.object.data.root_quat_w, pivot_tool
            )
            raise RuntimeError(
                "screw pivot constraint is unstable: maximum error="
                f"{float(inner._turn_constraint_position_error.max()):.6f} m; "
                f"object_position={inner.object.data.root_pos_w.tolist()}; "
                f"object_quaternion={inner.object.data.root_quat_w.tolist()}; "
                f"current_pivot={current_pivot.tolist()}; "
                f"target_pivot={inner._turn_pivot_w.tolist()}; "
                f"workpiece_position={inner.workpiece.data.root_pos_w.tolist()}"
            )
        if float(torch.rad2deg(inner._turn_constraint_tilt_error).max()) > float(
            inner.cfg.allen_turn_constraint_tilt_tolerance_deg
        ):
            raise RuntimeError(
                "screw axis constraint is unstable: maximum tilt="
                f"{float(torch.rad2deg(inner._turn_constraint_tilt_error).max()):.3f} deg"
            )
        stage_results = []
        maximum_torque_multiplier = float(
            inner.cfg.allen_turn_max_fixture_torque_multiplier
        )
        for stage, (friction_range, damping_range) in enumerate(zip(
            inner.cfg.allen_turn_friction_ranges_nm,
            inner.cfg.allen_turn_damping_ranges_nm_per_radps,
        )):
            friction = float(friction_range[1])
            damping = float(damping_range[1])
            torque_limit = maximum_torque_multiplier * friction
            inner.object.write_root_pose_to_sim(
                torch.cat((position, quaternion), dim=-1), env_ids=env_ids
            )
            inner.object.write_root_velocity_to_sim(
                torch.zeros(inner.num_envs, 6, device=inner.device), env_ids=env_ids
            )
            inner._turn_coulomb_friction_nm.fill_(friction)
            inner._turn_damping_nm_per_radps.fill_(damping)
            inner._turn_stiction_yaw.copy_(yaw_from_quaternion(inner.object.data.root_quat_w))
            inner._turn_friction_stuck.fill_(True)
            inner._turn_fixture_torque_peak_nm.zero_()
            inner._turn_fixture_torque_clipped_steps.zero_()
            for settle_step in range(3):
                observation, reward, terminated, truncated, _ = env.step(action)
                require_finite(1000 + stage * 100 + settle_step, observation, reward)
            initial_velocity[:, 5] = directions * initial_angular_speed
            inner.object.write_root_velocity_to_sim(initial_velocity, env_ids=env_ids)
            inner._turn_friction_stuck.fill_(False)
            inner._turn_stiction_yaw.copy_(yaw_from_quaternion(inner.object.data.root_quat_w))
            peak_pivot_error = torch.zeros(inner.num_envs, device=inner.device)
            peak_tilt_error = torch.zeros(inner.num_envs, device=inner.device)
            peak_torque = torch.zeros(inner.num_envs, device=inner.device)
            clip_steps = torch.zeros(inner.num_envs, dtype=torch.long, device=inner.device)
            first_torque = None
            for loaded_step in range(int(ARGS.steps)):
                observation, reward, terminated, truncated, _ = env.step(action)
                require_finite(
                    2000 + stage * int(ARGS.steps) + loaded_step,
                    observation,
                    reward,
                )
                if bool(terminated.any()) or bool(truncated.any()):
                    raise RuntimeError(
                        f"fixture task ended during curriculum stage {stage}"
                    )
                if first_torque is None:
                    first_torque = inner._turn_fixture_torque_nm.clone()
                peak_pivot_error = torch.maximum(
                    peak_pivot_error, inner._turn_constraint_position_error
                )
                peak_tilt_error = torch.maximum(
                    peak_tilt_error, inner._turn_constraint_tilt_error
                )
                peak_torque = torch.maximum(
                    peak_torque, inner._turn_fixture_torque_nm.abs()
                )
                clip_steps += inner._turn_fixture_torque_clipped.long()
            if first_torque is None or not bool((directions * first_torque < 0.0).all()):
                raise RuntimeError(
                    f"stage {stage} fixture torque did not oppose initial motion"
                )
            if float(peak_torque.max()) > torque_limit + 1.0e-5:
                raise RuntimeError(
                    f"stage {stage} exceeded fixture torque cap: "
                    f"peak={float(peak_torque.max()):.6f}, limit={torque_limit:.6f}"
                )
            if float(peak_pivot_error.max()) > float(
                inner.cfg.allen_turn_constraint_position_tolerance_m
            ):
                raise RuntimeError(
                    f"stage {stage} exceeded pivot tolerance: "
                    f"{float(peak_pivot_error.max()):.6f} m"
                )
            if float(torch.rad2deg(peak_tilt_error).max()) > float(
                inner.cfg.allen_turn_constraint_tilt_tolerance_deg
            ):
                raise RuntimeError(
                    f"stage {stage} exceeded tilt tolerance: "
                    f"{float(torch.rad2deg(peak_tilt_error).max()):.6f} deg"
                )
            stage_results.append({
                "stage": stage,
                "friction_nm": friction,
                "damping_nm_per_radps": damping,
                "torque_limit_nm": torque_limit,
                "peak_torque_nm": float(peak_torque.max()),
                "clip_step_ratio": float(
                    clip_steps.float().sum() / (inner.num_envs * int(ARGS.steps))
                ),
                "peak_pivot_error_m": float(peak_pivot_error.max()),
                "peak_tilt_error_deg": float(torch.rad2deg(peak_tilt_error).max()),
            })
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
            "sampled_shoulder_to_handle_mean_m": float(sampled_reach.mean()),
            "sampled_shoulder_to_handle_max_m": float(sampled_reach.max()),
            "sampled_shoulder_to_handle_limit_m": reach_limit,
            "sampled_palm_to_handle_mean_m": float(sampled_palm_distance.mean()),
            "sampled_palm_to_handle_max_m": float(sampled_palm_distance.max()),
            "sampled_palm_to_handle_limit_m": palm_distance_limit,
            "robot_reset_resample_count_max": int(
                inner._turn_initial_robot_resample_count.max()
            ),
            "kinetic_friction_limit_nm": kinetic_limit_nm,
            "static_friction_limit_nm": static_limit_nm,
            "initial_angular_speed_radps": initial_angular_speed,
            "final_angular_speed_radps": final_angular_speed.tolist(),
            "passive_yaw_drift_max_deg": float(
                torch.rad2deg(passive_yaw_delta).max()
            ),
            "passive_fixture_torque_max_nm": float(
                passive_max_fixture_torque.max()
            ),
            "directed_motion_deg": torch.rad2deg(directed_motion).tolist(),
            "maximum_pivot_error_m": float(inner._turn_constraint_position_error.max()),
            "maximum_tilt_error_deg": float(
                torch.rad2deg(inner._turn_constraint_tilt_error).max()
            ),
            "curriculum_stage_results": stage_results,
            "viewer": str(ARGS.output.resolve()),
        }
        summary_path = ARGS.output.with_suffix(".json")
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[pass] {json.dumps(summary, sort_keys=True)}", flush=True)
    finally:
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
