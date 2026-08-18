#!/usr/bin/env python3
"""Run static-wrench and task-conditioned simulation tests for a grasp pilot."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import traceback

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-results", type=Path, required=True)
    parser.add_argument("--max-force-n", type=float, default=10.0)
    parser.add_argument("--max-torque-nm", type=float, default=0.5)
    parser.add_argument("--ramp-seconds", type=float, default=2.0)
    parser.add_argument("--settle-control-steps", type=int, default=30)
    parser.add_argument("--slip-position-m", type=float, default=0.005)
    parser.add_argument("--slip-rotation-deg", type=float, default=2.0)
    parser.add_argument("--task-position-tolerance-m", type=float, default=0.03)
    parser.add_argument("--task-orientation-tolerance-deg", type=float, default=15.0)
    parser.add_argument("--task-contact-ratio-threshold", type=float, default=0.5)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    return parser.parse_args()


ARGS = parse_args()
APP = AppLauncher(ARGS).app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402

import isaacsimenvs  # noqa: E402,F401
from isaacsimenvs.tasks.simtoolreal.simtoolreal_tacmap_env_cfg import (  # noqa: E402
    SimToolRealInHandStableScrapeEnvCfg,
)
from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import (  # noqa: E402
    compute_intermediate_values,
)
from isaacsimenvs.tasks.simtoolreal.utils.grasp_stability import (  # noqa: E402
    static_stable_fraction,
    task_stability_gates,
)
from isaacsimenvs.tasks.simtoolreal.utils.scrape_pose_utils import (  # noqa: E402
    edge_contact_points_w,
    quat_apply_wxyz,
)


TASK_ID = "Isaacsimenvs-SimToolReal-Stable-Scrape-InHand-Direct-v0"
FORCE_DIRECTIONS = ("+Fx", "-Fx", "+Fy", "-Fy", "+Fz", "-Fz")
TORQUE_DIRECTIONS = ("+Tx", "-Tx", "+Ty", "-Ty", "+Tz", "-Tz")
ALL_DIRECTIONS = FORCE_DIRECTIONS + TORQUE_DIRECTIONS


@dataclass
class FailureState:
    failed: torch.Tensor
    failure_step: torch.Tensor
    reason: list[str]
    max_position_slip: torch.Tensor
    max_rotation_slip: torch.Tensor
    min_support: torch.Tensor


def validate_args(payload: dict) -> None:
    if payload.get("schema_version") != 1 or payload.get("kind") != "eraser_grasp_evaluator_pilot":
        raise ValueError("--pilot-results is not a supported eraser pilot manifest")
    if not payload.get("results"):
        raise ValueError("pilot manifest contains no results")
    for name in ("max_force_n", "max_torque_nm", "ramp_seconds", "slip_position_m", "slip_rotation_deg"):
        value = float(getattr(ARGS, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if ARGS.settle_control_steps <= 0:
        raise ValueError("--settle-control-steps must be positive")
    if ARGS.task_position_tolerance_m <= 0.0 or ARGS.task_orientation_tolerance_deg <= 0.0:
        raise ValueError("task pose tolerances must be positive")
    if not 0.0 <= ARGS.task_contact_ratio_threshold <= 1.0:
        raise ValueError("--task-contact-ratio-threshold must be in [0, 1]")


def make_cfg(num_envs: int, grasp_bank: str) -> SimToolRealInHandStableScrapeEnvCfg:
    cfg = SimToolRealInHandStableScrapeEnvCfg()
    cfg.seed = int(ARGS.seed if hasattr(ARGS, "seed") else 20260818)
    cfg.scene.num_envs = num_envs
    cfg.grasp_bank_path = grasp_bank
    cfg.grasp_bank_min_entries = 1
    cfg.episode_length_s = 600.0
    cfg.domain_randomization.use_obs_delay = False
    cfg.domain_randomization.use_action_delay = False
    cfg.domain_randomization.use_object_state_delay_noise = False
    cfg.domain_randomization.force_scale = 0.0
    cfg.domain_randomization.torque_scale = 0.0
    cfg.domain_randomization.force_prob_range = (1.0e-12, 1.0e-12)
    cfg.domain_randomization.torque_prob_range = (1.0e-12, 1.0e-12)
    cfg.inhand_table_angle_stages_deg = (0.0,)
    cfg.inhand_clearance_stages_m = ((0.05, 0.05),)
    cfg.enable_tool_table_contact_sensor = True
    cfg.enable_tool_table_contact_force_reward = False
    cfg.tool_table_contact_sensor_update_period = 0.0
    cfg.tool_table_contact_sensor_history_len = int(cfg.decimation)
    cfg.tool_table_contact_sensor_force_threshold = 0.0
    cfg.contact_force_use_control_interval_average = True
    return cfg


def direct_control_step(
    inner, targets_lab: torch.Tensor, forces_w: torch.Tensor | None = None,
    torques_w: torch.Tensor | None = None,
) -> None:
    zeros = torch.zeros(inner.num_envs, 1, 3, device=inner.device)
    forces = zeros if forces_w is None else forces_w[:, None, :]
    torques = zeros if torques_w is None else torques_w[:, None, :]
    if forces.shape != zeros.shape or torques.shape != zeros.shape:
        raise ValueError("external wrench tensors have an invalid shape")
    inner.object.set_external_force_and_torque(forces, torques, is_global=True)
    for _ in range(int(inner.cfg.decimation)):
        inner.robot.set_joint_position_target(targets_lab)
        inner.scene.write_data_to_sim()
        inner.sim.step(render=False)
        inner.scene.update(dt=inner.physics_dt)
    compute_intermediate_values(inner)


def relative_pose(inner) -> tuple[torch.Tensor, torch.Tensor]:
    return subtract_frame_transforms(
        inner.robot.data.body_link_pos_w[:, inner._palm_body_id],
        inner.robot.data.body_link_quat_w[:, inner._palm_body_id],
        inner.object.data.root_pos_w,
        inner.object.data.root_quat_w,
    )


def rotation_error_deg(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    alignment = torch.abs((current * reference).sum(-1)).clamp(0.0, 1.0)
    return torch.rad2deg(2.0 * torch.acos(alignment))


def support_count(inner) -> torch.Tensor:
    object_size = inner._object_scale_per_env * 0.04
    radius = 0.5 * torch.linalg.vector_norm(object_size, dim=-1)
    return (inner._curr_fingertip_distances < (radius + 0.012).unsqueeze(-1)).sum(-1)


def new_failure_state(count: int, device: torch.device) -> FailureState:
    return FailureState(
        failed=torch.zeros(count, dtype=torch.bool, device=device),
        failure_step=torch.full((count,), -1, dtype=torch.long, device=device),
        reason=[""] * count,
        max_position_slip=torch.zeros(count, device=device),
        max_rotation_slip=torch.zeros(count, device=device),
        min_support=torch.full((count,), 5, dtype=torch.long, device=device),
    )


def update_failures(
    inner, state: FailureState, reference_pos: torch.Tensor,
    reference_quat: torch.Tensor, step: int, active_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    position, quaternion = relative_pose(inner)
    position_slip = torch.linalg.vector_norm(position[:active_count] - reference_pos, dim=-1)
    rotation_slip = rotation_error_deg(quaternion[:active_count], reference_quat)
    support = support_count(inner)[:active_count]
    finite = (
        torch.isfinite(position_slip) & torch.isfinite(rotation_slip)
        & torch.isfinite(inner.object.data.root_pos_w[:active_count]).all(-1)
    )
    conditions = (
        ("nonfinite", ~finite),
        ("position_slip", position_slip > float(ARGS.slip_position_m)),
        ("rotation_slip", rotation_slip > float(ARGS.slip_rotation_deg)),
        ("support_loss", support < 2),
        ("drop", inner.object.data.root_pos_w[:active_count, 2] - inner.scene.env_origins[:active_count, 2] < 0.1),
    )
    state.max_position_slip = torch.maximum(state.max_position_slip, position_slip)
    state.max_rotation_slip = torch.maximum(state.max_rotation_slip, rotation_slip)
    state.min_support = torch.minimum(state.min_support, support)
    for name, condition in conditions:
        newly_failed = condition & ~state.failed
        for index in newly_failed.nonzero(as_tuple=False).squeeze(-1).tolist():
            state.reason[index] = name
        state.failure_step[newly_failed] = step
        state.failed |= newly_failed
    return position_slip, rotation_slip, support


def restore_bank_entries(inner, bank_ids: torch.Tensor) -> torch.Tensor:
    env_ids = torch.arange(bank_ids.numel(), device=inner.device)
    inner._restore_inhand_state(env_ids, bank_ids)
    targets = inner._inhand_bank_joint_targets[bank_ids][:, inner._perm_canon_to_lab]
    zeros = torch.zeros(inner.num_envs, 3, device=inner.device)
    all_targets = inner._cur_targets.clone()
    all_targets[: bank_ids.numel()] = targets
    direct_control_step(inner, all_targets, zeros, zeros)
    return all_targets


def run_static(inner, selected_entry_ids: list[int]) -> list[dict]:
    count = len(selected_entry_ids) * len(ALL_DIRECTIONS)
    bank_ids = torch.tensor(
        [entry_id for entry_id in selected_entry_ids for _ in ALL_DIRECTIONS],
        device=inner.device, dtype=torch.long,
    )
    targets = restore_bank_entries(inner, bank_ids)
    env_ids = torch.arange(count, device=inner.device)
    table_pose = torch.zeros(count, 7, device=inner.device)
    table_pose[:, :3] = inner.scene.env_origins[env_ids]
    table_pose[:, 2] -= 1.0
    table_pose[:, 3] = 1.0
    inner.table.write_root_pose_to_sim(table_pose, env_ids=env_ids)
    state = new_failure_state(count, inner.device)
    reference_pos = inner._inhand_bank_relative_pos[bank_ids].clone()
    reference_quat = inner._inhand_bank_relative_quat[bank_ids].clone()
    zeros = torch.zeros(inner.num_envs, 3, device=inner.device)
    for settle_step in range(int(ARGS.settle_control_steps)):
        direct_control_step(inner, targets, zeros, zeros)
        update_failures(
            inner, state, reference_pos, reference_quat,
            -int(ARGS.settle_control_steps) + settle_step, count,
        )
    ramp_steps = max(1, round(float(ARGS.ramp_seconds) / float(inner.step_dt)))
    local_force = torch.zeros(count, 3, device=inner.device)
    local_torque = torch.zeros_like(local_force)
    for grasp_index in range(len(selected_entry_ids)):
        for direction_index, name in enumerate(ALL_DIRECTIONS):
            index = grasp_index * len(ALL_DIRECTIONS) + direction_index
            sign = 1.0 if name[0] == "+" else -1.0
            axis = "xyz".index(name[-1].lower())
            if "F" in name:
                local_force[index, axis] = sign
            else:
                local_torque[index, axis] = sign
    zeros_full = torch.zeros(inner.num_envs, 3, device=inner.device)
    for step in range(ramp_steps):
        fraction = (step + 1) / ramp_steps
        object_quat = inner.object.data.root_quat_w[:count]
        force_w = quat_apply_wxyz(object_quat, local_force) * float(ARGS.max_force_n) * fraction
        torque_w = quat_apply_wxyz(object_quat, local_torque) * float(ARGS.max_torque_nm) * fraction
        forces, torques = zeros_full.clone(), zeros_full.clone()
        active = ~state.failed
        forces[:count] = force_w * active.unsqueeze(-1)
        torques[:count] = torque_w * active.unsqueeze(-1)
        direct_control_step(inner, targets, forces, torques)
        update_failures(inner, state, reference_pos, reference_quat, step, count)
    direct_control_step(inner, targets, zeros_full, zeros_full)
    results = []
    for grasp_index, entry_id in enumerate(selected_entry_ids):
        tests = []
        for direction_index, name in enumerate(ALL_DIRECTIONS):
            index = grasp_index * len(ALL_DIRECTIONS) + direction_index
            failure_step = int(state.failure_step[index].item())
            fraction = static_stable_fraction(
                failed=bool(state.failed[index].item()),
                failure_step=failure_step,
                ramp_steps=ramp_steps,
            )
            tests.append({
                "direction": name,
                "passed": not bool(state.failed[index].item()),
                "stable_fraction": fraction,
                "failure_force_n": float(ARGS.max_force_n) * fraction if "F" in name else None,
                "failure_torque_nm": float(ARGS.max_torque_nm) * fraction if "T" in name else None,
                "maximum_position_slip_m": float(state.max_position_slip[index].item()),
                "maximum_rotation_slip_deg": float(state.max_rotation_slip[index].item()),
                "minimum_support_count": int(state.min_support[index].item()),
                "failure_reason": state.reason[index],
            })
        results.append({
            "grasp_entry_id": entry_id,
            "mean_stable_fraction": sum(test["stable_fraction"] for test in tests) / len(tests),
            "passed_all_directions": all(test["passed"] for test in tests),
            "tests": tests,
        })
    print(f"[static] grasps={len(results)} directions={count}", flush=True)
    return results


def run_task(inner, pilot_results: list[dict]) -> list[dict]:
    count = len(pilot_results)
    bank_ids = torch.tensor(
        [int(result["grasp_entry_id"]) for result in pilot_results],
        device=inner.device, dtype=torch.long,
    )
    targets = restore_bank_entries(inner, bank_ids)
    env_ids = torch.arange(count, device=inner.device)
    table_pose = torch.zeros(count, 7, device=inner.device)
    desired_poses: list[torch.Tensor] = []
    arm_trajectories: list[torch.Tensor] = []
    for result in pilot_results:
        trajectory = result["trajectory"]
        table_pose[len(desired_poses), :3] = torch.tensor(
            trajectory["table_root_position"], device=inner.device
        ) + inner.scene.env_origins[len(desired_poses)]
        table_pose[len(desired_poses), 3:] = torch.tensor(
            trajectory["table_quaternion_wxyz"], device=inner.device
        )
        desired_poses.append(torch.tensor(result["tool_poses_wxyz"], device=inner.device))
        arm_trajectories.append(torch.tensor(
            result["computed"]["arm_trajectory"], device=inner.device
        ))
    inner.table.write_root_pose_to_sim(table_pose, env_ids=env_ids)
    inner._table_quat_wxyz_per_env[env_ids] = table_pose[:, 3:]
    inner._table_z_per_env[env_ids] = table_pose[:, 2] - inner.scene.env_origins[env_ids, 2]
    desired = torch.stack(desired_poses)
    arms = torch.stack(arm_trajectories)
    if desired.shape[:2] != arms.shape[:2] or desired.shape[0] != count:
        raise ValueError("pilot trajectories have inconsistent lengths")
    reference_pos = inner._inhand_bank_relative_pos[bank_ids].clone()
    reference_quat = inner._inhand_bank_relative_quat[bank_ids].clone()
    state = new_failure_state(count, inner.device)
    pose_position_sum = torch.zeros(count, device=inner.device)
    pose_rotation_sum = torch.zeros(count, device=inner.device)
    edge_error_sum = torch.zeros(count, device=inner.device)
    contact_steps = torch.zeros(count, dtype=torch.long, device=inner.device)
    contact_force_sum = torch.zeros(count, device=inner.device)
    contact_force_max = torch.zeros(count, device=inner.device)
    geometric_contact_steps = torch.zeros(count, dtype=torch.long, device=inner.device)
    sensed_contact_steps = torch.zeros(count, dtype=torch.long, device=inner.device)
    final_position_error = torch.zeros(count, device=inner.device)
    final_orientation_error = torch.zeros(count, device=inner.device)
    final_edge_error = torch.zeros(count, device=inner.device)
    arm_tracking_sum = torch.zeros(count, device=inner.device)
    first_contact_step = int(json.loads(ARGS.pilot_results.read_text())["config"]["transition_steps"])
    zeros = torch.zeros(inner.num_envs, 3, device=inner.device)
    for step in range(desired.shape[1]):
        canonical_targets = inner._inhand_bank_joint_targets[bank_ids].clone()
        canonical_targets[:, :7] = arms[:, step]
        targets[:count] = canonical_targets[:, inner._perm_canon_to_lab]
        direct_control_step(inner, targets, zeros, zeros)
        actual_arm = inner.robot.data.joint_pos[:count, inner._perm_lab_to_canon][:, :7]
        arm_tracking_sum += torch.linalg.vector_norm(actual_arm - arms[:, step], dim=-1)
        actual_pos = inner.object.data.root_pos_w[:count] - inner.scene.env_origins[:count]
        actual_quat = inner.object.data.root_quat_w[:count]
        desired_pos, desired_quat = desired[:, step, :3], desired[:, step, 3:]
        position_error = torch.linalg.vector_norm(actual_pos - desired_pos, dim=-1)
        orientation_error = rotation_error_deg(actual_quat, desired_quat)
        pose_position_sum += position_error
        pose_rotation_sum += orientation_error
        edge_points = edge_contact_points_w(
            inner.object.data.root_pos_w[:count], actual_quat,
            inner._scrape_x_tip_per_env[:count], inner._scrape_y_min_per_env[:count],
            inner._scrape_y_max_per_env[:count], inner._scrape_z_contact_per_env[:count],
        )
        table_top = table_pose[:, :3] + quat_apply_wxyz(
            table_pose[:, 3:], torch.tensor([0.0, 0.0, 0.15], device=inner.device).expand(count, -1)
        )
        normal = quat_apply_wxyz(
            table_pose[:, 3:], torch.tensor([0.0, 0.0, 1.0], device=inner.device).expand(count, -1)
        )
        edge_error = torch.abs(
            ((edge_points - table_top.unsqueeze(1)) * normal.unsqueeze(1)).sum(-1)
        ).mean(-1)
        final_position_error = position_error
        final_orientation_error = orientation_error
        final_edge_error = edge_error
        if step >= first_contact_step:
            edge_error_sum += edge_error
            contact_steps += 1
            force = inner._sensor_normal_force()[:count]
            contact_force_sum += force
            contact_force_max = torch.maximum(contact_force_max, force)
            geometric_contact_steps += (edge_error <= 0.01).long()
            sensed_contact_steps += (force >= 0.1).long()
        update_failures(inner, state, reference_pos, reference_quat, step, count)
    results = []
    for index, source in enumerate(pilot_results):
        divisor = desired.shape[1]
        contact_divisor = max(1, int(contact_steps[index].item()))
        geometric_contact_ratio = int(geometric_contact_steps[index].item()) / contact_divisor
        sensed_contact_ratio = int(sensed_contact_steps[index].item()) / contact_divisor
        grasp_survived = not bool(state.failed[index].item())
        computed_feasible = all(bool(value) for value in source["computed"]["gates"].values())
        gates = task_stability_gates(
            computed_feasible=computed_feasible,
            grasp_survived=grasp_survived,
            final_position_error_m=float(final_position_error[index].item()),
            final_orientation_error_deg=float(final_orientation_error[index].item()),
            geometric_contact_ratio=geometric_contact_ratio,
            sensed_contact_ratio=sensed_contact_ratio,
            position_tolerance_m=float(ARGS.task_position_tolerance_m),
            orientation_tolerance_deg=float(ARGS.task_orientation_tolerance_deg),
            contact_ratio_threshold=float(ARGS.task_contact_ratio_threshold),
        )
        results.append({
            "grasp_entry_id": int(source["grasp_entry_id"]),
            "grasp_fingerprint": source["grasp_fingerprint"],
            "trajectory_id": int(source["trajectory_id"]),
            "passed": gates.passed,
            "computed_feasible": gates.computed_feasible,
            "grasp_survived": gates.grasp_survived,
            "tracking_succeeded": gates.tracking_succeeded,
            "contact_succeeded": gates.contact_succeeded,
            "completion_fraction": (
                1.0 if int(state.failure_step[index].item()) < 0
                else int(state.failure_step[index].item()) / divisor
            ),
            "mean_tool_position_error_m": float(pose_position_sum[index].item()) / divisor,
            "mean_tool_orientation_error_deg": float(pose_rotation_sum[index].item()) / divisor,
            "final_tool_position_error_m": float(final_position_error[index].item()),
            "final_tool_orientation_error_deg": float(final_orientation_error[index].item()),
            "mean_edge_contact_error_m": float(edge_error_sum[index].item()) / contact_divisor,
            "final_edge_contact_error_m": float(final_edge_error[index].item()),
            "geometric_contact_ratio": geometric_contact_ratio,
            "sensed_contact_ratio": sensed_contact_ratio,
            "mean_contact_force_n": float(contact_force_sum[index].item()) / contact_divisor,
            "maximum_contact_force_n": float(contact_force_max[index].item()),
            "maximum_position_slip_m": float(state.max_position_slip[index].item()),
            "maximum_rotation_slip_deg": float(state.max_rotation_slip[index].item()),
            "minimum_support_count": int(state.min_support[index].item()),
            "mean_arm_tracking_error_rad": float(arm_tracking_sum[index].item()) / divisor,
            "failure_reason": state.reason[index],
            "task_failure_reasons": gates.failure_reasons,
        })
    print(f"[task] trajectories={len(results)}", flush=True)
    return results


def main() -> None:
    if not ARGS.pilot_results.is_file():
        raise FileNotFoundError(f"pilot results do not exist: {ARGS.pilot_results}")
    pilot = json.loads(ARGS.pilot_results.read_text())
    validate_args(pilot)
    selected = [int(value) for value in pilot["selected_entry_ids"]]
    required_envs = max(len(selected) * len(ALL_DIRECTIONS), len(pilot["results"]))
    cfg = make_cfg(required_envs, pilot["config"]["grasp_bank"])
    env = gym.make(TASK_ID, cfg=cfg)
    inner = env.unwrapped
    try:
        if getattr(inner, "_tool_table_contact_sensor_failed", False):
            raise RuntimeError(
                f"tool-table ContactSensor failed: {inner._tool_table_contact_sensor_error}"
            )
        if getattr(inner, "_tool_table_contact_sensor", None) is None:
            raise RuntimeError("tool-table ContactSensor is unavailable")
        static_results = run_static(inner, selected)
        task_results = run_task(inner, pilot["results"])
        payload = {
            "schema_version": 1,
            "pilot_results": str(ARGS.pilot_results.resolve()),
            "config": {
                "max_force_n": ARGS.max_force_n,
                "max_torque_nm": ARGS.max_torque_nm,
                "ramp_seconds": ARGS.ramp_seconds,
                "settle_control_steps": ARGS.settle_control_steps,
                "slip_position_m": ARGS.slip_position_m,
                "slip_rotation_deg": ARGS.slip_rotation_deg,
                "task_position_tolerance_m": ARGS.task_position_tolerance_m,
                "task_orientation_tolerance_deg": ARGS.task_orientation_tolerance_deg,
                "task_contact_ratio_threshold": ARGS.task_contact_ratio_threshold,
            },
            "provenance": {
                "code_commit": subprocess.run(
                    ("git", "rev-parse", "HEAD"), cwd=REPO_ROOT,
                    check=True, capture_output=True, text=True,
                ).stdout.strip(),
            },
            "static_stability": static_results,
            "task_stability": task_results,
        }
        output = ARGS.pilot_results.parent / "stability_results.json"
        output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        print(
            f"[summary] static_all_pass={sum(item['passed_all_directions'] for item in static_results)}/{len(static_results)} "
            f"task_pass={sum(item['passed'] for item in task_results)}/{len(task_results)}",
            flush=True,
        )
        print(f"[output] {output.resolve()}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        APP.close()
