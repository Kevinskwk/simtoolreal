#!/usr/bin/env python3
"""Validate Allen-key fixture, contacts, target reachability, and episode logic."""

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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grasp-bank", type=Path, default=None)
    parser.add_argument("--reset-yaw-range-deg", type=float, default=None)
    parser.add_argument("--target-angles-deg", type=float, nargs="+", default=None)
    parser.add_argument("--all-target-pairs", action="store_true")
    parser.add_argument("--write-target-graph", action="store_true")
    parser.add_argument(
        "--workspace-conditioned", action="store_true",
        help="Validate the rollout-conditioned workspace curriculum task.",
    )
    parser.add_argument(
        "--palm-down", action="store_true",
        help="Validate the thick-handle palm-down adjustment task.",
    )
    parser.add_argument("--output", type=Path, default=None)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    return parser.parse_args()


ARGS = parse_args()
APP = AppLauncher(ARGS).app


import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_inv  # noqa: E402

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
    SimToolRealAllenKeyPalmDownAdjustmentEnvCfg,
    SimToolRealAllenKeyWorkspaceAdjustmentEnvCfg,
)
from isaacsimenvs.tasks.simtoolreal.utils.grasp_evaluator import (  # noqa: E402
    GraspEvaluatorThresholds,
    UrdfKinematics,
    pose_matrix,
    solve_arm_ik,
)
from isaacsimenvs.tasks.simtoolreal.utils.adjustment_utils import (  # noqa: E402
    rotate_palm_about_tool_axis_in_place,
)
from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import compute_obs_dim  # noqa: E402


TASK_ID = "Isaacsimenvs-SimToolReal-AllenKey-Adjustment-Direct-v0"
WORKSPACE_TASK_ID = (
    "Isaacsimenvs-SimToolReal-AllenKey-Workspace-Adjustment-Direct-v0"
)
PALM_DOWN_TASK_ID = (
    "Isaacsimenvs-SimToolReal-AllenKey-PalmDown-Adjustment-Direct-v0"
)


def require_finite(step: int, observation: dict, reward: torch.Tensor) -> None:
    named = {"policy": observation["policy"], "critic": observation["critic"], "reward": reward}
    for name, value in named.items():
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"{name} contains NaN or Inf at validation step {step}")


def solve_sampled_target_arms(inner) -> tuple[np.ndarray, float, float]:
    robot_urdf = ROOT / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
    kinematics = UrdfKinematics(robot_urdf)
    base = np.eye(4)
    base[1, 3] = 0.8
    arms = []
    position_errors = []
    rotation_errors = []
    for env_id in range(inner.num_envs):
        target_pos_local = (
            inner._adjustment_target_palm_pos_w[env_id]
            - inner.scene.env_origins[env_id]
        )
        target = pose_matrix(
            target_pos_local.detach().cpu().numpy(),
            inner._adjustment_target_palm_quat_w[env_id].detach().cpu().numpy(),
        )
        joints = inner.robot.data.joint_pos[
            env_id, inner._perm_lab_to_canon
        ].detach().cpu().numpy()
        arm, position_error, rotation_error, _, _ = solve_arm_ik(
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
                f"sampled target {env_id} is not arm-reachable: "
                f"position={position_error:.6f}m rotation={rotation_error:.3f}deg"
            )
        arms.append(arm)
        position_errors.append(float(position_error))
        rotation_errors.append(float(rotation_error))
    return (
        np.stack(arms), max(position_errors), max(rotation_errors)
    )


def main() -> None:
    if ARGS.num_envs <= 0 or ARGS.settle_steps <= 0:
        raise ValueError("num-envs and settle-steps must be positive")
    if ARGS.workspace_conditioned and ARGS.palm_down:
        raise ValueError("--workspace-conditioned and --palm-down are mutually exclusive")
    if ARGS.target_angles_deg is not None and len(ARGS.target_angles_deg) != ARGS.num_envs:
        raise ValueError("target-angles-deg must provide exactly one angle per environment")
    bank_payload = None
    pair_sources: list[int] = []
    pair_targets: list[int] = []
    num_envs = int(ARGS.num_envs)
    if ARGS.all_target_pairs:
        if ARGS.grasp_bank is None:
            raise ValueError("--all-target-pairs requires --grasp-bank")
        if ARGS.target_angles_deg is not None:
            raise ValueError("--all-target-pairs cannot be combined with target angles")
        bank_payload = json.loads(ARGS.grasp_bank.read_text())
        for source_id, entry in enumerate(bank_payload["entries"]):
            for target_id in entry["verification"].get("valid_target_ids", []):
                pair_sources.append(source_id)
                pair_targets.append(int(target_id))
        if not pair_sources:
            raise RuntimeError("Allen-key bank has no candidate target pairs to screen")
        num_envs = len(pair_sources)
    elif ARGS.write_target_graph:
        raise ValueError("--write-target-graph requires --all-target-pairs")

    if ARGS.palm_down:
        cfg = SimToolRealAllenKeyPalmDownAdjustmentEnvCfg()
        task_id = PALM_DOWN_TASK_ID
    elif ARGS.workspace_conditioned:
        cfg = SimToolRealAllenKeyWorkspaceAdjustmentEnvCfg()
        task_id = WORKSPACE_TASK_ID
    else:
        cfg = SimToolRealAllenKeyAdjustmentEnvCfg()
        task_id = TASK_ID
    cfg.seed = int(ARGS.seed)
    cfg.scene.num_envs = num_envs
    if ARGS.grasp_bank is not None:
        cfg.grasp_bank_path = str(ARGS.grasp_bank.resolve())
    if ARGS.reset_yaw_range_deg is not None:
        cfg.allen_reset_yaw_range_stages_deg = (
            float(ARGS.reset_yaw_range_deg),
        ) * len(cfg.adjustment_target_rotation_deg)
    if ARGS.all_target_pairs:
        cfg.allen_reset_yaw_range_stages_deg = (0.0,) * len(
            cfg.adjustment_target_rotation_deg
        )
    cfg.adjustment_curriculum_min_eligible_count = 1_000_000
    env = gym.make(task_id, cfg=cfg)
    frames: list[dict] = []
    try:
        observation, _ = env.reset()
        inner = env.unwrapped
        env_ids = torch.arange(inner.num_envs, device=inner.device)
        if ARGS.all_target_pairs:
            source_ids = torch.tensor(
                pair_sources, device=inner.device, dtype=torch.long
            )
            target_ids = torch.tensor(
                pair_targets, device=inner.device, dtype=torch.long
            )
            inner._restore_inhand_state(env_ids, source_ids)
            inner._allen_target_bank_index.copy_(target_ids)
            target_pos = inner._inhand_bank_relative_pos[target_ids]
            target_quat = inner._inhand_bank_relative_quat[target_ids]
            inner._adjustment_target_relative_pos.copy_(target_pos)
            inner._adjustment_target_relative_quat.copy_(target_quat)
            tool_to_palm_quat = quat_inv(target_quat)
            tool_to_palm_pos = quat_apply(tool_to_palm_quat, -target_pos)
            target_palm_pos, target_palm_quat = combine_frame_transforms(
                inner.object.data.root_pos_w,
                inner.object.data.root_quat_w,
                tool_to_palm_pos,
                tool_to_palm_quat,
            )
            inner._adjustment_target_palm_pos_w.copy_(target_palm_pos)
            inner._adjustment_target_palm_quat_w.copy_(target_palm_quat)
        elif ARGS.grasp_bank is not None:
            bank_size = len(json.loads(ARGS.grasp_bank.read_text())["entries"])
            if bank_size == inner.num_envs:
                inner._restore_inhand_state(env_ids, env_ids)
        expected_policy = (num_envs, compute_obs_dim(cfg.obs.obs_list))
        expected_critic = (num_envs, compute_obs_dim(cfg.obs.state_list))
        if observation["policy"].shape != expected_policy:
            raise RuntimeError(f"unexpected policy shape {tuple(observation['policy'].shape)}")
        if observation["critic"].shape != expected_critic:
            raise RuntimeError(f"unexpected critic shape {tuple(observation['critic'].shape)}")
        if ARGS.target_angles_deg is not None:
            angles = torch.deg2rad(torch.tensor(
                ARGS.target_angles_deg, device=inner.device, dtype=torch.float32
            ))
            target_pos, target_quat = rotate_palm_about_tool_axis_in_place(
                inner._stable_relative_pos,
                inner._stable_relative_quat,
                angles,
                torch.tensor(inner.cfg.allen_screw_axis_tool, device=inner.device),
            )
            inner._adjustment_target_relative_pos.copy_(target_pos)
            inner._adjustment_target_relative_quat.copy_(target_quat)
            tool_to_palm_quat = quat_inv(target_quat)
            tool_to_palm_pos = quat_apply(tool_to_palm_quat, -target_pos)
            target_palm_pos, target_palm_quat = combine_frame_transforms(
                inner.object.data.root_pos_w,
                inner.object.data.root_quat_w,
                tool_to_palm_pos,
                tool_to_palm_quat,
            )
            inner._adjustment_target_palm_pos_w.copy_(target_palm_pos)
            inner._adjustment_target_palm_quat_w.copy_(target_palm_quat)
            inner._adjustment_target_axial_translation.copy_(angles.abs())
        target_arms, ik_position, ik_rotation = solve_sampled_target_arms(inner)

        # IK alone is insufficient. Place every sampled wrist target with its
        # validated bank finger posture and test the physical grasps.
        target_ids = inner._allen_target_bank_index
        target_joints = inner._inhand_bank_joint_pos[target_ids][
            :, inner._perm_canon_to_lab
        ].clone()
        target_arm_tensor = torch.as_tensor(
            target_arms, device=inner.device, dtype=target_joints.dtype
        )
        target_joints[:, inner._arm_joint_ids] = target_arm_tensor
        inner.robot.write_joint_state_to_sim(
            target_joints, torch.zeros_like(target_joints), env_ids=env_ids
        )
        target_controls = inner._inhand_bank_joint_targets[target_ids][
            :, inner._perm_canon_to_lab
        ].clone()
        target_controls[:, inner._arm_joint_ids] = target_arm_tensor
        inner._replay_target_lab_order = target_controls
        target_actions = inner._inhand_bank_last_action[target_ids]
        for step in range(int(ARGS.settle_steps)):
            observation, reward, terminated, truncated, _ = env.step(target_actions)
            require_finite(step, observation, reward)
            if bool(terminated.any()) or bool(truncated.any()):
                raise RuntimeError("episode ended during sampled-target grasp validation")
        target_position_error = inner._adjustment_target_palm_position_error
        target_rotation_error = torch.rad2deg(
            inner._adjustment_target_palm_rotation_error
        )
        feasible = (
            (target_position_error <= float(cfg.allen_pose_sigma_stages_m[-1]))
            & (target_rotation_error <= 5.0)
            & inner._allen_final_grasp_valid
        )
        if ARGS.all_target_pairs:
            valid_by_source: list[list[tuple[int, float]]] = [
                [] for _ in bank_payload["entries"]
            ]
            for pair_id, is_feasible in enumerate(feasible.tolist()):
                if is_feasible:
                    valid_by_source[pair_sources[pair_id]].append((
                        pair_targets[pair_id], float(target_arms[pair_id, 0])
                    ))
            isolated = [
                source_id for source_id, targets in enumerate(valid_by_source)
                if (
                    bank_payload["entries"][source_id]["verification"].get(
                        "valid_target_ids", []
                    )
                    and not targets
                )
            ]
            if isolated:
                raise RuntimeError(
                    "physical target screening left starts without targets: "
                    f"{isolated}; passing={int(feasible.sum().item())}/{num_envs}; "
                    f"position_m={target_position_error.tolist()} "
                    f"rotation_deg={target_rotation_error.tolist()} "
                    f"grasp_valid={inner._allen_final_grasp_valid.tolist()} "
                    f"palm_contact={inner._allen_palm_contact.tolist()} "
                    f"fingertip_contacts={inner._allen_fingertip_contact_count.tolist()} "
                    f"support={inner._stable_support_count.tolist()} "
                    f"flexion_closure={inner._allen_flexion_closure.tolist()}"
                )
            if ARGS.write_target_graph:
                for source_id, targets in enumerate(valid_by_source):
                    verification = bank_payload["entries"][source_id]["verification"]
                    verification["valid_target_ids"] = [item[0] for item in targets]
                    verification["target_arm_joint_0_rad"] = [item[1] for item in targets]
                ARGS.grasp_bank.write_text(json.dumps(bank_payload, indent=2))
            print(
                f"[pass] physical target graph retained "
                f"{int(feasible.sum().item())}/{num_envs} directed pairs; "
                f"per_start={[len(targets) for targets in valid_by_source]} "
                f"written={bool(ARGS.write_target_graph)}",
                flush=True,
            )
            return
        if not bool(feasible.all()):
            raise RuntimeError(
                "sampled target is arm-reachable but not a feasible supported grasp: "
                f"valid={feasible.tolist()} "
                f"position={target_position_error.tolist()} "
                f"rotation={target_rotation_error.tolist()} "
                f"palm_contact={inner._allen_palm_contact.tolist()} "
                f"fingertip_contacts={inner._allen_fingertip_contact_count.tolist()} "
                f"support={inner._stable_support_count.tolist()} "
                f"flexion_closure={inner._allen_flexion_closure.tolist()} "
                f"start_bank={inner._inhand_reset_bank_index.tolist()} "
                f"target_bank={inner._allen_target_bank_index.tolist()}"
            )

        observation, _ = env.reset()
        if ARGS.grasp_bank is not None and bank_size == inner.num_envs:
            inner._restore_inhand_state(env_ids, env_ids)

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
        if not bool(inner._allen_final_grasp_valid.all()):
            current_pos, current_quat = inner._palm_tool_relative()
            source_ids = inner._inhand_reset_bank_index
            relative_drift = torch.linalg.vector_norm(
                current_pos - inner._inhand_bank_relative_pos[source_ids], dim=-1
            )
            raise RuntimeError(
                "multi-contact grasp support was not retained during settling: "
                f"valid={inner._allen_final_grasp_valid.tolist()} "
                f"contact={inner._allen_palm_contact.tolist()} "
                f"force_n={inner._allen_palm_force_n.tolist()} "
                f"yaw_deg={torch.rad2deg(inner._allen_reset_yaw_rad).tolist()} "
                f"start_bank={source_ids.tolist()} "
                f"support={inner._stable_support_count.tolist()} "
                f"relative_drift_m={relative_drift.tolist()}"
            )
        if len(inner._fingertip_tool_contact_sensors) != 5:
            raise RuntimeError("five fingertip-tool contact sensors were not created")

        fixture_position = inner._allen_fixture_tool_pos.clone()
        fixture_position_error = torch.linalg.vector_norm(
            inner.object.data.root_pos_w - fixture_position, dim=-1
        )
        if float(fixture_position_error.max().item()) > float(
            cfg.allen_tool_position_tolerance_m
        ):
            raise RuntimeError(
                "fixture did not preserve tool position during adjustment: "
                f"max_error={float(fixture_position_error.max().item()):.6f}m"
            )

        # Make the current physically settled relationship the deterministic hold target.
        current_pos, current_quat = inner._palm_tool_relative()
        inner._adjustment_target_relative_pos.copy_(current_pos)
        inner._adjustment_target_relative_quat.copy_(current_quat)
        inner._adjustment_initial_tool_pos.copy_(inner.object.data.root_pos_w)
        inner._adjustment_initial_tool_quat.copy_(inner.object.data.root_quat_w)
        inner.episode_length_buf[:] = int(cfg.allen_adjustment_steps)
        for step in range(int(cfg.allen_closure_steps)):
            observation, reward, terminated, truncated, _ = env.step(actions)
            require_finite(ARGS.settle_steps + step, observation, reward)
            frames.append(capture_pose_viewer_frame(inner, 0))
            if bool(terminated.any()) or bool(truncated.any()):
                raise RuntimeError("episode ended before the closure phase completed")
        if not bool((inner._allen_phase_obs[:, 2] > 0.5).all()):
            raise RuntimeError("task did not enter release validation after closure")
        # The release phase proves that the fixture is removed and the settled
        # bank grasp satisfies the same dynamic hold gate used by training.
        for step in range(int(cfg.allen_release_steps) - 1):
            observation, reward, terminated, truncated, _ = env.step(actions)
            require_finite(ARGS.settle_steps + int(cfg.allen_closure_steps) + step, observation, reward)
            frames.append(capture_pose_viewer_frame(inner, 0))
            if bool(terminated.any()) or bool(truncated.any()):
                raise RuntimeError("episode ended early during release validation")
        release_success = inner._allen_combined_valid & (
            inner._allen_hold_count >= int(cfg.allen_success_hold_steps)
        )
        release_success_count = int(release_success.sum().item())
        release_success_ratio = float(release_success.float().mean().item())
        print(
            "[diagnostic] open-loop challenged release: "
            f"{release_success_count}/{inner.num_envs} grasps passed "
            f"({release_success_ratio:.3f})",
            flush=True,
        )
        if release_success_count == 0:
            raise RuntimeError(
                "no settled bank grasp passed dynamic release validation: "
                f"valid={inner._allen_combined_valid.tolist()} "
                f"hold={inner._allen_hold_count.tolist()} "
                f"contacts={inner._allen_fingertip_contact_count.tolist()} "
                f"support={inner._stable_support_count.tolist()} "
                f"flexion_closure={inner._allen_flexion_closure.tolist()} "
                f"palm={inner._allen_palm_contact.tolist()} "
                f"tool_position_error={inner._adjustment_tool_position_error.tolist()} "
                f"relative_speed={inner._stable_relative_linear_speed.tolist()}"
                f" palm_keypoint_error={inner._allen_palm_keypoint_error.tolist()}"
                f" socket_valid={inner._allen_socket_valid.tolist()}"
                f" tool_valid={inner._allen_validity_obs[:, 5].tolist()}"
                f" motion_valid={inner._allen_validity_obs[:, 6].tolist()}"
                f" start_bank={inner._inhand_reset_bank_index.tolist()}"
            )
        observation, reward, terminated, truncated, _ = env.step(actions)
        require_finite(480, observation, reward)
        if not bool(truncated.all()):
            raise RuntimeError("dynamic release episode did not end at its fixed timeout")

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
            "closure_steps": int(cfg.allen_closure_steps),
            "release_steps": int(cfg.allen_release_steps),
            "open_loop_release_success_count": release_success_count,
            "open_loop_release_success_ratio": release_success_ratio,
            "fingertip_contact_count_mean": float(
                inner._allen_fingertip_contact_count.float().mean().item()
            ),
            "episode_steps": 480,
            "viewer": str(output.resolve()),
        }
        if ARGS.workspace_conditioned:
            summary.update({
                "source_workspace_tier_counts": [
                    int((inner._allen_source_workspace_tier == tier).sum().item())
                    for tier in range(3)
                ],
                "target_quality_improvement_min": float(
                    inner._allen_target_quality_improvement.min().item()
                ),
                "sampled_pair_translation_max_m": float(
                    inner._allen_sampled_pair_translation_m.max().item()
                ),
                "sampled_pair_rotation_max_deg": float(
                    inner._allen_sampled_pair_rotation_deg.max().item()
                ),
            })
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
