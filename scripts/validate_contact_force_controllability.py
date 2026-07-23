#!/usr/bin/env python3
"""Validate tool-table force measurement and controllability without RL training.

The phases intentionally run in this order:
  1. known external loads on a bare tool,
  2. Cartesian DLS and PI force control after policy grasp acquisition,
  3. deterministic policy response to normal offsets in the tool-pose goal.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "outputs/2026-07-20/17-51-49/0_simtoolreal_sapg/last/model.pth"
)
DEFAULT_CONFIG = REPO_ROOT / "pretrained_policy/config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=("all", "bare-load", "bare-dynamics", "held-dls", "policy-goal"),
        default=("all",),
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--policy-config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default=str(REPO_ROOT / "outputs/contact_force_controllability"))
    parser.add_argument("--tool-type", default="spatula")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy-coef-id", type=float, default=0.0)
    parser.add_argument(
        "--held-tool-mode",
        choices=("fixed-joint", "policy-grasp"),
        default="fixed-joint",
    )
    parser.add_argument("--acquisition-steps", type=int, default=3600)
    parser.add_argument("--acquisition-stable-steps", type=int, default=5)
    parser.add_argument("--minimum-contact-force", type=float, default=0.5)
    parser.add_argument("--maximum-force", type=float, default=20.0)
    parser.add_argument("--dls-damping", type=float, default=0.05)
    parser.add_argument("--pi-kp", type=float, default=8.0e-4)
    parser.add_argument("--pi-ki", type=float, default=2.0e-4)
    parser.add_argument("--force-velocity-limit", type=float, default=0.005)
    parser.add_argument(
        "--physics-dt",
        type=float,
        default=None,
        help="Physics timestep; decimation is adjusted to retain 60 Hz control.",
    )
    parser.add_argument("--solver-position-iterations", type=int, default=None)
    parser.add_argument("--solver-velocity-iterations", type=int, default=None)
    parser.add_argument(
        "--external-forces-every-iteration",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--fixed-grasp-export",
        default="",
        help="Write the acquired robot pose and palm-to-tool fixed-joint frame to JSON.",
    )
    parser.add_argument("--require-pass", action=argparse.BooleanOptionalAction, default=True)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    return parser.parse_args()


ARGS = parse_args()
APP = AppLauncher(ARGS).app


import gymnasium as gym  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import omni.usd  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402
from pxr import Gf, Sdf, UsdPhysics  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

import isaacsimenvs  # noqa: E402,F401
from deployment.rl_player import RlPlayer  # noqa: E402
from isaacsimenvs.tasks.simtoolreal.simtoolreal_tacmap_env_cfg import (  # noqa: E402
    SimToolRealTacMapScrapePoseEnvCfg,
)
from isaacsimenvs.tasks.simtoolreal.utils.contact_force_controllability import (  # noqa: E402
    coefficient_of_determination,
    damped_least_squares,
    expected_normal_reaction,
    interval_normal_force,
    pi_force_step,
    quaternion_error_vector,
)
from isaacsimenvs.tasks.simtoolreal.utils.scrape_pose_utils import (  # noqa: E402
    table_top_state,
)


OFFSETS_M = np.asarray([0.003, 0.001, 0.0, -0.00025, -0.0005, -0.001, -0.0015, -0.0025])
LOADS_N = np.asarray([0.0, 2.0, 4.0, 6.0, 10.0])
STEP_LOADS_N = np.asarray([0.0, 2.0, 6.0, 10.0, 6.0, 2.0, 0.0])
RAMP_RATES_NPS = (1.0, 5.0, 20.0)
FORCE_TARGETS_N = np.asarray([2.0, 4.0, 6.0])


class PhaseFailure(RuntimeError):
    """Raised when a phase cannot produce trustworthy measurements."""


FIXED_GRASP_JOINT_PATH = "/World/envs/env_0/ControllabilityGraspJoint"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_text(*args: str) -> str:
    result = subprocess.run(
        ("git", *args), cwd=REPO_ROOT, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        raise PhaseFailure(f"refusing to write empty phase data: {path.name}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table_state(inner) -> tuple[torch.Tensor, torch.Tensor]:
    quat = getattr(inner, "_table_quat_wxyz_per_env", inner.table.data.root_quat_w)
    return table_top_state(inner.table.data.root_pos_w, quat)


def measured_interval_force(inner) -> torch.Tensor:
    sensor = getattr(inner, "_tool_table_contact_sensor", None)
    if sensor is None:
        raise PhaseFailure("tool-table ContactSensor is unavailable")
    history = getattr(sensor.data, "force_matrix_w_history", None)
    if history is None:
        raise PhaseFailure("filtered contact-force history is unavailable")
    required = int(inner.cfg.decimation)
    if history.shape[1] < required:
        raise PhaseFailure(
            f"contact history has {history.shape[1]} samples; {required} required"
        )
    _, normal = table_state(inner)
    force = interval_normal_force(history[:, :required], normal)
    if not torch.isfinite(force).all():
        raise PhaseFailure("contact-force measurement contains NaN or Inf")
    return force


def measured_current_force(inner, normal: torch.Tensor) -> torch.Tensor:
    sensor = getattr(inner, "_tool_table_contact_sensor", None)
    if sensor is None or sensor.data is None:
        raise PhaseFailure("tool-table ContactSensor current data is unavailable")
    matrix = getattr(sensor.data, "force_matrix_w", None)
    if matrix is None or matrix.ndim < 3 or matrix.shape[-1] != 3:
        raise PhaseFailure(
            "tool-table ContactSensor has no valid pair-filtered force_matrix_w"
        )
    force_w = matrix.sum(dim=tuple(range(1, matrix.ndim - 1)))
    force = (force_w * normal).sum(dim=-1)
    if force.shape != (inner.num_envs,) or not torch.isfinite(force).all():
        raise PhaseFailure(
            f"current normal contact force is invalid: shape={tuple(force.shape)}"
        )
    return force


def make_cfg() -> SimToolRealTacMapScrapePoseEnvCfg:
    cfg = SimToolRealTacMapScrapePoseEnvCfg()
    cfg.seed = ARGS.seed
    cfg.episode_length_s = 180.0
    cfg.scene.num_envs = 1
    cfg.assets.handle_head_types = (ARGS.tool_type,)
    cfg.assets.num_assets_per_type = 1
    cfg.assets.shuffle_assets = False
    cfg.table_pitch_roll_range_deg = 0.0
    cfg.reset.table_reset_pitch_roll_range_deg = 0.0
    cfg.reset.table_reset_z_range = 0.0
    cfg.reset.reset_position_noise_x = 0.0
    cfg.reset.reset_position_noise_y = 0.0
    cfg.reset.reset_position_noise_z = 0.0
    cfg.reset.reset_dof_pos_random_interval_arm = 0.0
    cfg.reset.reset_dof_pos_random_interval_fingers = 0.0
    cfg.reset.reset_dof_vel_random_interval = 0.0
    cfg.domain_randomization.use_obs_delay = False
    cfg.domain_randomization.use_action_delay = False
    cfg.domain_randomization.use_object_state_delay_noise = False
    cfg.domain_randomization.joint_velocity_obs_noise_std = 0.0
    cfg.domain_randomization.force_scale = 0.0
    cfg.domain_randomization.torque_scale = 0.0
    cfg.domain_randomization.force_prob_range = (1.0e-12, 1.0e-12)
    cfg.domain_randomization.torque_prob_range = (1.0e-12, 1.0e-12)
    cfg.enable_tool_table_contact_force_reward = True
    cfg.target_contact_normal_force = 4.0
    cfg.tool_table_contact_sensor_update_period = 0.0
    cfg.tool_table_contact_sensor_history_len = int(cfg.decimation)
    cfg.tool_table_contact_sensor_force_threshold = 0.0
    cfg.contact_force_filter_alpha = 0.2
    cfg.termination.max_consecutive_successes = 0
    if ARGS.physics_dt is not None:
        physics_dt = float(ARGS.physics_dt)
        if physics_dt <= 0.0:
            raise ValueError("--physics-dt must be positive")
        decimation = round((1.0 / 60.0) / physics_dt)
        if decimation <= 0 or not np.isclose(
            decimation * physics_dt, 1.0 / 60.0, rtol=0.0, atol=1.0e-9
        ):
            raise ValueError("--physics-dt must divide the 1/60 s control period")
        cfg.sim.dt = physics_dt
        cfg.decimation = decimation
        cfg.sim.render_interval = decimation
        cfg.tool_table_contact_sensor_history_len = decimation
    if ARGS.solver_position_iterations is not None:
        iterations = int(ARGS.solver_position_iterations)
        if iterations <= 0:
            raise ValueError("--solver-position-iterations must be positive")
        cfg.sim.physx.min_position_iteration_count = iterations
        cfg.sim.physx.max_position_iteration_count = iterations
    if ARGS.solver_velocity_iterations is not None:
        iterations = int(ARGS.solver_velocity_iterations)
        if iterations < 0:
            raise ValueError("--solver-velocity-iterations must be non-negative")
        cfg.sim.physx.min_velocity_iteration_count = iterations
        cfg.sim.physx.max_velocity_iteration_count = iterations
    if ARGS.external_forces_every_iteration is not None:
        cfg.sim.physx.enable_external_forces_every_iteration = bool(
            ARGS.external_forces_every_iteration
        )
    return cfg


def bare_tool_state(inner) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    top, normal = table_state(inner)
    tool_quat = getattr(
        inner, "_table_quat_wxyz_per_env", inner.table.data.root_quat_w
    ).clone()
    z_min = inner._scrape_z_contact_per_env
    tool_pos = top + normal * (-z_min[:, None] + 0.002)
    pose = torch.cat((tool_pos, tool_quat), dim=-1)
    mass = inner.object.data.default_mass.reshape(1, -1).sum(dim=1).to(
        device=inner.device, dtype=torch.float32
    )
    return pose, normal.clone(), mass, torch.zeros(1, 6, device=inner.device)


def reset_bare_tool(
    inner,
    pose: torch.Tensor,
    zero_velocity: torch.Tensor,
    normal: torch.Tensor,
    *,
    settle_steps: int = 240,
) -> float:
    inner.object.write_root_pose_to_sim(pose)
    inner.object.write_root_velocity_to_sim(zero_velocity)
    for _ in range(settle_steps):
        direct_physics_step(inner, torch.zeros_like(normal))
    return float((inner.object.data.root_lin_vel_w[0] * normal[0]).sum())


def direct_physics_step(inner, applied_force_w: torch.Tensor) -> float:
    zeros = torch.zeros_like(applied_force_w)
    inner.object.set_external_force_and_torque(
        applied_force_w[:, None, :], zeros[:, None, :], is_global=True
    )
    inner.robot.set_joint_position_target(inner.robot.data.joint_pos)
    inner.scene.write_data_to_sim()
    inner.sim.step(render=False)
    inner.scene.update(dt=inner.physics_dt)
    return float(measured_interval_force(inner)[0].item())


def run_bare_load(inner, output_dir: Path) -> dict:
    print("[phase 1/3] bare-load calibration")
    inner._replay_target_lab_order = None
    inner.reset()
    pose, normal, mass, zero_vel = bare_tool_state(inner)
    gravity = torch.tensor(inner.cfg.sim.gravity, device=inner.device, dtype=mass.dtype)
    rows: list[dict[str, float | int | str]] = []
    summaries = []

    for load in LOADS_N:
        inner.object.write_root_pose_to_sim(pose)
        inner.object.write_root_velocity_to_sim(zero_vel)
        applied = -float(load) * normal
        for _ in range(120):
            direct_physics_step(inner, applied)
        samples = []
        for sample in range(240):
            force = direct_physics_step(inner, applied)
            samples.append(force)
            rows.append(
                {
                    "phase": "bare-load",
                    "load_n": float(load),
                    "sample": sample,
                    "time_s": sample * float(inner.physics_dt),
                    "measured_force_n": force,
                    "object_normal_velocity_mps": float(
                        (inner.object.data.root_lin_vel_w[0] * normal[0]).sum().item()
                    ),
                }
            )
        expected = expected_normal_reaction(
            mass, gravity, normal, torch.tensor([load], device=inner.device)
        )[0].item()
        measured = float(np.mean(samples))
        summaries.append(
            {
                "load_n": float(load),
                "expected_n": float(expected),
                "measured_n": measured,
                "std_n": float(np.std(samples)),
            }
        )

    inner.object.set_external_force_and_torque(
        torch.zeros(1, 1, 3, device=inner.device),
        torch.zeros(1, 1, 3, device=inner.device),
        is_global=True,
    )
    expected = np.asarray([item["expected_n"] for item in summaries])
    measured = np.asarray([item["measured_n"] for item in summaries])
    slope, intercept = np.polyfit(expected, measured, 1)
    r2 = coefficient_of_determination(torch.tensor(expected), torch.tensor(measured))
    tolerances = np.maximum(0.5, 0.1 * expected)
    errors = np.abs(measured - expected)
    passed = bool(0.9 <= slope <= 1.1 and r2 >= 0.98 and np.all(errors <= tolerances))

    write_csv(output_dir / "bare_load.csv", rows)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(expected, expected, "k--", label="ideal")
    ax.errorbar(
        expected,
        measured,
        yerr=[item["std_n"] for item in summaries],
        marker="o",
        label="PhysX",
    )
    ax.set(xlabel="Expected reaction (N)", ylabel="Measured normal force (N)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "bare_load.png", dpi=160)
    plt.close(fig)
    return {
        "passed": passed,
        "slope": float(slope),
        "intercept_n": float(intercept),
        "r2": r2,
        "points": summaries,
    }


def run_bare_dynamics(inner, output_dir: Path) -> dict:
    """Validate force transitions on a bare tool under known external loads."""
    print("[gate] bare-tool dynamic force transitions")
    inner._replay_target_lab_order = None
    inner.reset()
    pose, normal, mass, zero_velocity = bare_tool_state(inner)
    gravity = torch.tensor(inner.cfg.sim.gravity, device=inner.device)
    gravity_normal_n = float((mass[:, None] * gravity * normal).sum())
    dt = float(inner.physics_dt)
    rows: list[dict[str, float | int | str]] = []

    def sample(
        *,
        protocol: str,
        repetition: int,
        segment: int,
        sample_index: int,
        load_n: float,
        previous_velocity: float,
        ramp_rate_nps: float = 0.0,
        direction: str = "hold",
        initial_speed_mps: float = 0.0,
    ) -> float:
        applied = -float(load_n) * normal
        inner.object.set_external_force_and_torque(
            applied[:, None, :],
            torch.zeros_like(applied)[:, None, :],
            is_global=True,
        )
        inner.robot.set_joint_position_target(inner.robot.data.joint_pos)
        inner.scene.write_data_to_sim()
        inner.sim.step(render=False)
        inner.scene.update(dt=inner.physics_dt)
        measured = float(measured_current_force(inner, normal)[0])
        velocity = float(
            (inner.object.data.root_lin_vel_w[0] * normal[0]).sum()
        )
        acceleration = (velocity - previous_velocity) / dt
        equilibrium = float(load_n) - gravity_normal_n
        dynamic_expected = equilibrium + float(mass[0]) * acceleration
        rows.append(
            {
                "protocol": protocol,
                "repetition": repetition,
                "segment": segment,
                "sample": sample_index,
                "time_s": len(rows) * dt,
                "direction": direction,
                "ramp_rate_nps": ramp_rate_nps,
                "initial_speed_mps": initial_speed_mps,
                "load_n": load_n,
                "equilibrium_reaction_n": equilibrium,
                "dynamic_expected_reaction_n": dynamic_expected,
                "measured_force_n": measured,
                "force_balance_residual_n": measured - dynamic_expected,
                "object_normal_velocity_mps": velocity,
                "object_normal_acceleration_mps2": acceleration,
            }
        )
        return velocity

    step_hold_steps = 120
    step_repetitions = 3
    previous_velocity = reset_bare_tool(
        inner, pose, zero_velocity, normal, settle_steps=240
    )
    for repetition in range(step_repetitions):
        for segment, load in enumerate(STEP_LOADS_N):
            for index in range(step_hold_steps):
                previous_velocity = sample(
                    protocol="step",
                    repetition=repetition,
                    segment=segment,
                    sample_index=index,
                    load_n=float(load),
                    previous_velocity=previous_velocity,
                )

    for rate_index, rate in enumerate(RAMP_RATES_NPS):
        previous_velocity = reset_bare_tool(
            inner, pose, zero_velocity, normal, settle_steps=240
        )
        ramp_step = float(rate) * dt
        up = np.arange(0.0, 6.0, ramp_step, dtype=np.float64)
        loads = np.concatenate((up, np.asarray([6.0]), up[::-1]))
        for index, load in enumerate(loads):
            previous_velocity = sample(
                protocol="ramp",
                repetition=rate_index,
                segment=0,
                sample_index=index,
                load_n=float(load),
                previous_velocity=previous_velocity,
                ramp_rate_nps=float(rate),
                direction="loading" if index <= len(up) else "unloading",
            )

    acquisition_summaries = []
    control_interval_steps = round((1.0 / 60.0) / dt)
    for speed_index, initial_speed in enumerate((0.0, 0.005, 0.02)):
        repetitions = []
        for repetition in range(3):
            velocity = zero_velocity.clone()
            velocity[:, :3] = -float(initial_speed) * normal
            inner.object.write_root_pose_to_sim(pose)
            inner.object.write_root_velocity_to_sim(velocity)
            inner.object.set_external_force_and_torque(
                torch.zeros(1, 1, 3, device=inner.device),
                torch.zeros(1, 1, 3, device=inner.device),
                is_global=True,
            )
            previous_velocity = -float(initial_speed)
            row_start = len(rows)
            for index in range(180):
                previous_velocity = sample(
                    protocol="acquisition",
                    repetition=repetition,
                    segment=speed_index,
                    sample_index=index,
                    load_n=0.0,
                    previous_velocity=previous_velocity,
                    direction="impact",
                    initial_speed_mps=float(initial_speed),
                )
            run_rows = rows[row_start:]
            measured = np.asarray(
                [float(row["measured_force_n"]) for row in run_rows]
            )
            interval_force = np.convolve(
                measured,
                np.ones(control_interval_steps) / control_interval_steps,
                mode="valid",
            )
            impulse = float(
                np.sum(measured + gravity_normal_n) * dt
                - float(mass[0]) * (previous_velocity + float(initial_speed))
            )
            repetitions.append(
                {
                    "peak_force_n": float(measured.max()),
                    "control_interval_peak_force_n": float(interval_force.max()),
                    "contact_impulse_residual_ns": impulse,
                    "force_balance_p95_n": float(
                        np.quantile(
                            np.abs(
                                [
                                    float(row["force_balance_residual_n"])
                                    for row in run_rows
                                ]
                            ),
                            0.95,
                        )
                    ),
                }
            )
        peaks = np.asarray([item["peak_force_n"] for item in repetitions])
        interval_peaks = np.asarray(
            [item["control_interval_peak_force_n"] for item in repetitions]
        )
        acquisition_summaries.append(
            {
                "initial_speed_mps": initial_speed,
                "peak_force_mean_n": float(peaks.mean()),
                "peak_force_range_n": float(peaks.max() - peaks.min()),
                "control_interval_peak_force_mean_n": float(
                    interval_peaks.mean()
                ),
                "maximum_abs_impulse_residual_ns": float(
                    max(
                        abs(item["contact_impulse_residual_ns"])
                        for item in repetitions
                    )
                ),
                "force_balance_p95_max_n": float(
                    max(item["force_balance_p95_n"] for item in repetitions)
                ),
                "repetitions": repetitions,
            }
        )

    inner.object.set_external_force_and_torque(
        torch.zeros(1, 1, 3, device=inner.device),
        torch.zeros(1, 1, 3, device=inner.device),
        is_global=True,
    )
    write_csv(output_dir / "bare_dynamics.csv", rows)

    step_rows = [row for row in rows if row["protocol"] == "step"]
    settled_rows = [
        row for row in step_rows if int(row["sample"]) >= 3 * step_hold_steps // 4
    ]
    settled_error = np.asarray(
        [
            float(row["measured_force_n"]) - float(row["equilibrium_reaction_n"])
            for row in settled_rows
        ]
    )
    residual = np.asarray(
        [float(row["force_balance_residual_n"]) for row in step_rows]
    )
    cycle_length = len(STEP_LOADS_N) * step_hold_steps
    step_force = np.asarray([float(row["measured_force_n"]) for row in step_rows])
    repeated = step_force.reshape(step_repetitions, cycle_length)
    repeatability_std = repeated.std(axis=0)

    ramp_summaries = []
    for rate in RAMP_RATES_NPS:
        rate_rows = [
            row
            for row in rows
            if row["protocol"] == "ramp"
            and np.isclose(float(row["ramp_rate_nps"]), rate)
        ]
        rate_residual = np.abs(
            np.asarray(
                [float(row["force_balance_residual_n"]) for row in rate_rows]
            )
        )
        loading = [row for row in rate_rows if row["direction"] == "loading"]
        unloading = [row for row in rate_rows if row["direction"] == "unloading"]
        load_grid = np.linspace(0.5, 5.5, 11)
        loading_force = np.interp(
            load_grid,
            [float(row["load_n"]) for row in loading],
            [float(row["measured_force_n"]) for row in loading],
        )
        unloading_force = np.interp(
            load_grid,
            [float(row["load_n"]) for row in reversed(unloading)],
            [float(row["measured_force_n"]) for row in reversed(unloading)],
        )
        ramp_summaries.append(
            {
                "rate_nps": rate,
                "force_balance_mae_n": float(rate_residual.mean()),
                "force_balance_p95_n": float(np.quantile(rate_residual, 0.95)),
                "hysteresis_mae_n": float(
                    np.mean(np.abs(loading_force - unloading_force))
                ),
            }
        )

    maximum_measured = float(
        max(float(row["measured_force_n"]) for row in rows)
    )
    result = {
        "passed": bool(
            np.mean(np.abs(settled_error)) <= 0.1
            and np.quantile(np.abs(residual), 0.95) <= 1.0
            and np.quantile(repeatability_std, 0.95) <= 0.1
            and ramp_summaries[0]["hysteresis_mae_n"] <= 0.25
            and max(
                item["peak_force_range_n"] for item in acquisition_summaries
            )
            <= 1.0
            and max(
                item["maximum_abs_impulse_residual_ns"]
                for item in acquisition_summaries
            )
            <= 0.01
            and maximum_measured <= 20.0
        ),
        "tool_mass_kg": float(mass[0]),
        "gravity_normal_force_n": -gravity_normal_n,
        "step": {
            "settled_mae_n": float(np.mean(np.abs(settled_error))),
            "force_balance_mae_n": float(np.mean(np.abs(residual))),
            "force_balance_p95_n": float(
                np.quantile(np.abs(residual), 0.95)
            ),
            "repeatability_std_p95_n": float(
                np.quantile(repeatability_std, 0.95)
            ),
            "maximum_measured_force_n": maximum_measured,
        },
        "ramps": ramp_summaries,
        "acquisition": acquisition_summaries,
        "criteria": {
            "maximum_settled_mae_n": 0.1,
            "maximum_force_balance_p95_n": 1.0,
            "maximum_repeatability_std_p95_n": 0.1,
            "maximum_slow_ramp_hysteresis_mae_n": 0.25,
            "maximum_acquisition_peak_range_n": 1.0,
            "maximum_acquisition_impulse_residual_ns": 0.01,
            "maximum_measured_force_n": 20.0,
        },
    }

    time = np.asarray([float(row["time_s"]) for row in rows])
    measured = np.asarray([float(row["measured_force_n"]) for row in rows])
    equilibrium = np.asarray(
        [float(row["equilibrium_reaction_n"]) for row in rows]
    )
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(time, equilibrium, label="quasi-static expected", linewidth=1.0)
    axes[0].plot(time, measured, label="measured", linewidth=0.8)
    axes[0].set_ylabel("Normal force (N)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(
        time,
        [float(row["force_balance_residual_n"]) for row in rows],
        linewidth=0.8,
    )
    axes[1].set(xlabel="Time (s)", ylabel="Force-balance residual (N)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "bare_dynamics.png", dpi=160)
    plt.close(fig)
    return result


def make_player(inner, checkpoint: Path, config: Path) -> RlPlayer:
    if not checkpoint.is_file():
        raise PhaseFailure(f"checkpoint does not exist: {checkpoint}")
    if not config.is_file():
        raise PhaseFailure(f"policy config does not exist: {config}")
    return RlPlayer(
        num_observations=inner.cfg.observation_space,
        num_actions=inner.cfg.action_space,
        config_path=str(config),
        checkpoint_path=str(checkpoint),
        device=str(inner.device),
        num_envs=1,
        coefficient_id=ARGS.policy_coef_id,
    )


def acquire_grasp(env, inner, player: RlPlayer) -> dict[str, torch.Tensor]:
    inner._replay_target_lab_order = None
    inner.cfg.termination.eval_success_tolerance = None
    player.reset()
    obs, _ = env.reset()
    obs, _, _, _, _ = env.step(torch.zeros(1, inner.cfg.action_space, device=inner.device))
    relative_positions: deque[torch.Tensor] = deque(maxlen=ARGS.acquisition_stable_steps)
    best = {"lifted": False, "fingertips": 0, "force_n": 0.0, "edge_error_m": float("inf")}

    for step in range(ARGS.acquisition_steps):
        action = player.get_normalized_action(obs["policy"], deterministic_actions=True)
        obs, _, terminated, truncated, _ = env.step(action.to(inner.device))
        if bool(terminated[0] or truncated[0]):
            player.reset()
            relative_positions.clear()
            continue
        force = float(measured_interval_force(inner)[0].item())
        palm = inner.robot.data.body_pos_w[0, inner._palm_body_id]
        obj = inner.object.data.root_pos_w[0]
        relative_positions.append((obj - palm).detach().clone())
        fingertip_count = int((inner._curr_fingertip_distances[0] < 0.12).sum().item())
        edge_error = float(inner._scrape_edge_contact_error[0].item())
        best["lifted"] = bool(best["lifted"] or bool(inner._lifted_object[0]))
        best["fingertips"] = max(int(best["fingertips"]), fingertip_count)
        best["force_n"] = max(float(best["force_n"]), force)
        best["edge_error_m"] = min(float(best["edge_error_m"]), edge_error)
        if (step + 1) % 300 == 0:
            print(
                f"[acquire] step={step + 1} lifted={bool(inner._lifted_object[0])} "
                f"fingertips={fingertip_count} edge={edge_error:.4f} m "
                f"force={force:.3f} N"
            )
        if ARGS.held_tool_mode == "fixed-joint":
            fixed_candidate = (
                fingertip_count >= 2
                and edge_error <= 0.01
                and force >= ARGS.minimum_contact_force
            )
            if fixed_candidate:
                print(
                    f"[acquire] edge-contact attachment pose at policy step {step + 1}, "
                    f"edge={edge_error:.4f} m, force={force:.3f} N"
                )
                return obs
            continue

        candidate = fingertip_count >= 2
        if candidate and len(relative_positions) == relative_positions.maxlen:
            stack = torch.stack(tuple(relative_positions))
            stable = (stack - stack.mean(dim=0)).norm(dim=-1).max() <= 0.005
            if bool(stable):
                print(f"[acquire] stable grasp candidate at policy step {step + 1}")
                return obs
        elif not candidate:
            relative_positions.clear()
    raise PhaseFailure(
        f"policy failed to acquire a stable grasp candidate in {ARGS.acquisition_steps} steps; "
        f"best lifted={best['lifted']}, fingertips={best['fingertips']}, "
        f"edge={best['edge_error_m']:.4f} m, force={best['force_n']:.3f} N"
    )


def attach_tool_to_palm(inner) -> dict:
    """Create an explicit rigid grasp while preserving the current tool transform."""
    stage = omni.usd.get_context().get_stage()
    palm_path = "/World/envs/env_0/Robot/iiwa14_link_7"
    tool_path = "/World/envs/env_0/Object/object_root"
    for path in (palm_path, tool_path):
        if not stage.GetPrimAtPath(path).IsValid():
            raise PhaseFailure(f"fixed-grasp rigid body prim does not exist: {path}")

    palm_pos = inner.robot.data.body_link_pos_w[:, inner._palm_body_id]
    palm_quat = inner.robot.data.body_link_quat_w[:, inner._palm_body_id]
    rel_pos, rel_quat = subtract_frame_transforms(
        palm_pos,
        palm_quat,
        inner.object.data.root_link_pos_w,
        inner.object.data.root_link_quat_w,
    )
    p = rel_pos[0].detach().cpu().tolist()
    q = rel_quat[0].detach().cpu().tolist()
    joint = UsdPhysics.FixedJoint.Define(stage, FIXED_GRASP_JOINT_PATH)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(palm_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(tool_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*p))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(q[0], Gf.Vec3f(*q[1:])))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0)))
    print(f"[acquire] synthetic fixed grasp created at {FIXED_GRASP_JOINT_PATH}")
    spec = {
        "version": 1,
        "tool_type": ARGS.tool_type,
        "joint_positions": {
            name: float(inner.robot.data.joint_pos[0, index].item())
            for index, name in enumerate(inner.robot.data.joint_names)
        },
        "palm_to_tool_pos": [float(value) for value in p],
        "palm_to_tool_quat_wxyz": [float(value) for value in q],
        "source_checkpoint": str(Path(ARGS.checkpoint).resolve()),
    }
    if ARGS.fixed_grasp_export:
        export_path = Path(ARGS.fixed_grasp_export).resolve()
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with export_path.open("w") as stream:
            json.dump(spec, stream, indent=2)
            stream.write("\n")
        print(f"[acquire] fixed-grasp specification written to {export_path}")
    return spec


def palm_jacobian(inner) -> torch.Tensor:
    jacobians = inner.robot.root_physx_view.get_jacobians()
    body_index = int(inner._palm_body_id) - (1 if inner.robot.is_fixed_base else 0)
    if body_index < 0 or body_index >= jacobians.shape[1]:
        raise PhaseFailure(
            f"palm Jacobian index {body_index} invalid for shape {tuple(jacobians.shape)}"
        )
    jacobian = jacobians[:, body_index, :, inner._arm_joint_ids]
    if jacobian.shape != (1, 6, 7) or not torch.isfinite(jacobian).all():
        raise PhaseFailure(f"expected finite palm Jacobian (1, 6, 7), got {jacobian.shape}")
    return jacobian


def dls_step(
    env,
    inner,
    joint_target: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
) -> tuple[dict, torch.Tensor]:
    palm_pos = inner.robot.data.body_link_pos_w[:, inner._palm_body_id]
    palm_quat = inner.robot.data.body_link_quat_w[:, inner._palm_body_id]
    linear = 5.0 * (target_pos - palm_pos)
    angular = 3.0 * quaternion_error_vector(palm_quat, target_quat)
    twist = torch.cat((linear, angular), dim=-1)
    qdot = damped_least_squares(palm_jacobian(inner), twist, ARGS.dls_damping)
    qdot = torch.clamp(qdot, -0.5, 0.5)
    joint_target[:, inner._arm_joint_ids] += qdot * float(inner.step_dt)
    joint_target[:, inner._arm_joint_ids] = torch.clamp(
        joint_target[:, inner._arm_joint_ids], inner._arm_lower, inner._arm_upper
    )
    inner._replay_target_lab_order = joint_target
    obs, _, terminated, truncated, _ = env.step(
        torch.zeros(1, inner.cfg.action_space, device=inner.device)
    )
    if bool(terminated[0] or truncated[0]):
        raise PhaseFailure("environment reset during held-tool control")
    return obs, joint_target


def assert_grasp_preserved(inner, baseline_relative_pos: torch.Tensor) -> None:
    palm = inner.robot.data.body_pos_w[:, inner._palm_body_id]
    relative = inner.object.data.root_pos_w - palm
    drift = float((relative - baseline_relative_pos).norm(dim=-1).max().item())
    force = float(measured_interval_force(inner)[0].item())
    if drift > 0.03:
        raise PhaseFailure(f"tool-palm translation drifted {drift:.4f} m; grasp lost")
    if force > ARGS.maximum_force:
        raise PhaseFailure(f"measured force {force:.3f} N exceeds safety limit")


def monotonic_summary(offsets: np.ndarray, forces: np.ndarray) -> dict:
    result = spearmanr(-offsets, forces)
    rho = float(result.statistic)
    span = float(forces.max() - forces.min())
    return {
        "spearman_rho": rho,
        "force_span_n": span,
        "passed": bool(np.isfinite(rho) and rho >= 0.8 and span >= 4.0),
    }


def run_held_dls(env, inner, player: RlPlayer, output_dir: Path) -> dict:
    print("[phase 2/3] held-tool DLS displacement and PI force control")
    acquire_grasp(env, inner, player)
    if ARGS.held_tool_mode == "fixed-joint":
        attach_tool_to_palm(inner)
        inner._replay_target_lab_order = inner.robot.data.joint_pos.clone()
        for _ in range(30):
            _, _, terminated, truncated, _ = env.step(
                torch.zeros(1, inner.cfg.action_space, device=inner.device)
            )
            if bool(terminated[0] or truncated[0]):
                raise PhaseFailure("environment reset while settling fixed grasp")
        print("[acquire] synthetic fixed grasp settled for 30 control steps")
    inner.cfg.termination.eval_success_tolerance = 0.0
    inner._current_success_tolerance = 0.0
    _, normal = table_state(inner)
    contact_palm_pos = inner.robot.data.body_link_pos_w[:, inner._palm_body_id].clone()
    contact_palm_quat = inner.robot.data.body_link_quat_w[:, inner._palm_body_id].clone()
    contact_object_pos = inner.object.data.root_pos_w.clone()
    contact_relative = contact_object_pos - contact_palm_pos
    joint_target = inner.robot.data.joint_pos.clone()
    joint_target[:, inner._hand_joint_ids] = inner.robot.data.joint_pos[:, inner._hand_joint_ids]

    for _ in range(180):
        _, joint_target = dls_step(
            env,
            inner,
            joint_target,
            contact_palm_pos + 0.04 * normal,
            contact_palm_quat,
        )
    lift_distance = float(
        ((inner.object.data.root_pos_w - contact_object_pos) * normal).sum(dim=-1)[0].item()
    )
    lift_drift = float(
        (
            inner.object.data.root_pos_w
            - inner.robot.data.body_pos_w[:, inner._palm_body_id]
            - contact_relative
        ).norm(dim=-1)[0].item()
    )
    if lift_distance < 0.025 or lift_drift > 0.015:
        raise PhaseFailure(
            "stable grasp candidate failed grasp lift challenge: "
            f"tool lift={lift_distance:.4f} m, tool-palm drift={lift_drift:.4f} m"
        )
    print(
        f"[acquire] grasp verified by lift: tool={lift_distance:.4f} m, "
        f"tool-palm drift={lift_drift:.4f} m"
    )
    lifted_palm_pos = inner.robot.data.body_link_pos_w[:, inner._palm_body_id].clone()
    contact_steps = 0
    approach_depth = 0.0
    held_contact_depth: float | None = None
    for _ in range(1200):
        if held_contact_depth is None:
            approach_depth = min(0.06, approach_depth + 5.0e-5)
        else:
            approach_depth = held_contact_depth
        _, joint_target = dls_step(
            env,
            inner,
            joint_target,
            lifted_palm_pos - approach_depth * normal,
            contact_palm_quat,
        )
        assert_grasp_preserved(inner, contact_relative)
        force = float(measured_interval_force(inner)[0].item())
        edge_error = float(inner._scrape_edge_contact_error[0].item())
        contact_now = force >= ARGS.minimum_contact_force and edge_error <= 0.01
        if contact_now:
            if contact_steps == 0:
                held_contact_depth = approach_depth
            contact_steps += 1
        else:
            contact_steps = 0
            held_contact_depth = None
        if contact_steps >= 10:
            print(
                f"[acquire] controlled table contact at depth={approach_depth:.4f} m, "
                f"edge={edge_error:.4f} m, force={force:.3f} N"
            )
            break
    else:
        raise PhaseFailure("verified grasp failed to establish controlled edge contact")

    palm_pos0 = inner.robot.data.body_pos_w[:, inner._palm_body_id].clone()
    palm_quat0 = inner.robot.data.body_quat_w[:, inner._palm_body_id].clone()
    baseline_relative = inner.object.data.root_pos_w.clone() - palm_pos0
    rows: list[dict[str, float | int | str]] = []
    offset_summaries = []

    for stage, offset in enumerate(OFFSETS_M):
        stage_forces = []
        target_pos = palm_pos0 + float(offset) * normal
        for step in range(180):
            _, joint_target = dls_step(env, inner, joint_target, target_pos, palm_quat0)
            assert_grasp_preserved(inner, baseline_relative)
            force = float(measured_interval_force(inner)[0].item())
            stage_forces.append(force)
            rows.append(
                {
                    "phase": "held-dls-offset",
                    "stage": stage,
                    "step": step,
                    "offset_m": float(offset),
                    "target_force_n": "",
                    "measured_force_n": force,
                }
            )
        offset_summaries.append(float(np.mean(stage_forces[-60:])))

    monotonic = monotonic_summary(OFFSETS_M, np.asarray(offset_summaries))
    target_summaries = []
    offset = torch.zeros(1, device=inner.device)
    integral = torch.zeros_like(offset)
    for stage, target_force in enumerate(FORCE_TARGETS_N):
        stage_forces = []
        for step in range(600):
            force_tensor = measured_interval_force(inner)
            control = pi_force_step(
                torch.full_like(force_tensor, float(target_force)) - force_tensor,
                integral,
                dt=float(inner.step_dt),
                kp=ARGS.pi_kp,
                ki=ARGS.pi_ki,
                velocity_limit=ARGS.force_velocity_limit,
                integral_limit=20.0,
            )
            integral = control.integral
            offset = torch.clamp(
                offset - control.velocity * float(inner.step_dt), -0.008, 0.004
            )
            target_pos = palm_pos0 + offset[:, None] * normal
            _, joint_target = dls_step(env, inner, joint_target, target_pos, palm_quat0)
            assert_grasp_preserved(inner, baseline_relative)
            force = float(measured_interval_force(inner)[0].item())
            stage_forces.append(force)
            rows.append(
                {
                    "phase": "held-dls-pi",
                    "stage": stage,
                    "step": step,
                    "offset_m": float(offset.item()),
                    "target_force_n": float(target_force),
                    "measured_force_n": force,
                }
            )
        steady = np.asarray(stage_forces[-120:])
        target_summaries.append(
            {
                "target_n": float(target_force),
                "mean_n": float(steady.mean()),
                "mae_n": float(np.abs(steady - target_force).mean()),
                "rmse_n": float(np.sqrt(np.mean((steady - target_force) ** 2))),
            }
        )
    pi_passed = all(item["mae_n"] <= 1.0 for item in target_summaries)
    passed = bool(monotonic["passed"] and pi_passed)
    write_csv(output_dir / "held_dls.csv", rows)
    plot_force_series(output_dir / "held_dls.png", rows, "Held-tool DLS/PI")
    return {
        "passed": passed,
        "held_tool_mode": ARGS.held_tool_mode,
        "offsets_m": OFFSETS_M.tolist(),
        "steady_offset_forces_n": offset_summaries,
        "monotonic": monotonic,
        "pi_passed": pi_passed,
        "pi_targets": target_summaries,
    }


def run_policy_goal(env, inner, player: RlPlayer, output_dir: Path) -> dict:
    print("[phase 3/3] policy response to tool-goal normal offsets")
    if ARGS.held_tool_mode == "fixed-joint":
        stage = omni.usd.get_context().get_stage()
        if not stage.GetPrimAtPath(FIXED_GRASP_JOINT_PATH).IsValid():
            raise PhaseFailure("fixed-joint policy phase requires held-dls to run first")
        player.reset()
        obs = inner._get_observations()
    else:
        obs = acquire_grasp(env, inner, player)
    inner.cfg.termination.eval_success_tolerance = 0.0
    inner._current_success_tolerance = 0.0
    baseline_goal_pos = inner.goal_viz.data.root_pos_w.clone()
    baseline_goal_quat = inner.goal_viz.data.root_quat_w.clone()
    baseline_relative = (
        inner.object.data.root_pos_w.clone()
        - inner.robot.data.body_pos_w[:, inner._palm_body_id].clone()
    )
    _, normal = table_state(inner)
    inner._scrape_target_contact_normal_force.fill_(4.0)
    rows: list[dict[str, float | int | str]] = []
    offset_summaries = []

    for stage, offset in enumerate(OFFSETS_M):
        goal_pos = baseline_goal_pos + float(offset) * normal
        goal_pose = torch.cat((goal_pos, baseline_goal_quat), dim=-1)
        stage_forces = []
        for step in range(180):
            inner.goal_viz.write_root_pose_to_sim(goal_pose)
            inner._scrape_target_contact_normal_force.fill_(4.0)
            obs = inner._get_observations()
            action = player.get_normalized_action(obs["policy"], deterministic_actions=True)
            obs, _, terminated, truncated, _ = env.step(action.to(inner.device))
            if bool(terminated[0] or truncated[0]):
                raise PhaseFailure("environment reset during policy-goal sweep")
            assert_grasp_preserved(inner, baseline_relative)
            force = float(measured_interval_force(inner)[0].item())
            stage_forces.append(force)
            rows.append(
                {
                    "phase": "policy-goal",
                    "stage": stage,
                    "step": step,
                    "goal_offset_m": float(offset),
                    "target_force_n": 4.0,
                    "measured_force_n": force,
                    "pose_error_m": float(inner._keypoints_max_dist[0].item()),
                }
            )
        offset_summaries.append(float(np.mean(stage_forces[-60:])))

    monotonic = monotonic_summary(OFFSETS_M, np.asarray(offset_summaries))
    write_csv(output_dir / "policy_goal.csv", rows)
    plot_force_series(output_dir / "policy_goal.png", rows, "Policy goal-offset response")
    return {
        "passed": bool(monotonic["passed"]),
        "offsets_m": OFFSETS_M.tolist(),
        "steady_forces_n": offset_summaries,
        "monotonic": monotonic,
    }


def plot_force_series(path: Path, rows: list[dict], title: str) -> None:
    measured = np.asarray([float(row["measured_force_n"]) for row in rows])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(np.arange(measured.size) / 60.0, measured, label="measured")
    targets = [row.get("target_force_n", "") for row in rows]
    if any(value != "" for value in targets):
        target_values = np.asarray(
            [np.nan if value == "" else float(value) for value in targets]
        )
        ax.plot(np.arange(measured.size) / 60.0, target_values, "--", label="target")
    ax.axhline(ARGS.maximum_force, color="r", linestyle=":", label="force limit")
    ax.set(title=title, xlabel="Control time (s)", ylabel="Normal force (N)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_summary(output_dir: Path, metadata: dict, summary: dict) -> None:
    payload = {"metadata": metadata, "phases": summary}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# Contact-force controllability", "", f"Commit: `{metadata['git_commit']}`", ""]
    for name, result in summary.items():
        status = "PASS" if result.get("passed") else "FAIL"
        lines.append(f"- **{name}: {status}**")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    torch.manual_seed(ARGS.seed)
    np.random.seed(ARGS.seed)
    checkpoint = Path(ARGS.checkpoint).resolve()
    policy_config = Path(ARGS.policy_config).resolve()
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(ARGS.output_root).resolve() / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    requested = list(ARGS.phases)
    phases = (
        ["bare-load", "bare-dynamics", "held-dls", "policy-goal"]
        if "all" in requested
        else requested
    )
    cfg = make_cfg()
    metadata = {
        "created_at": datetime.now().astimezone().isoformat(),
        "git_commit": git_text("rev-parse", "HEAD"),
        "git_dirty": bool(git_text("status", "--porcelain")),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "policy_config": str(policy_config),
        "tool_type": ARGS.tool_type,
        "seed": ARGS.seed,
        "policy_coefficient_id": ARGS.policy_coef_id,
        "held_tool_mode": ARGS.held_tool_mode,
        "phases": phases,
        "physics_hz": round(1.0 / float(cfg.sim.dt)),
        "control_hz": round(1.0 / (float(cfg.sim.dt) * int(cfg.decimation))),
        "solver_position_iterations": int(
            cfg.sim.physx.max_position_iteration_count
        ),
        "solver_velocity_iterations": int(
            cfg.sim.physx.max_velocity_iteration_count
        ),
        "external_forces_every_iteration": bool(
            cfg.sim.physx.enable_external_forces_every_iteration
        ),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    env = gym.make("Isaacsimenvs-SimToolReal-TacMap-Scrape-Direct-v0", cfg=cfg)
    inner = env.unwrapped
    player = None
    summary: dict[str, dict] = {}
    active_phase: str | None = None
    try:
        if "bare-load" in phases:
            active_phase = "bare-load"
            summary["bare-load"] = run_bare_load(inner, output_dir)
        if "bare-dynamics" in phases:
            active_phase = "bare-dynamics"
            summary["bare-dynamics"] = run_bare_dynamics(inner, output_dir)
        if any(phase in phases for phase in ("held-dls", "policy-goal")):
            player = make_player(inner, checkpoint, policy_config)
        if "held-dls" in phases:
            active_phase = "held-dls"
            summary["held-dls"] = run_held_dls(env, inner, player, output_dir)
        if "policy-goal" in phases:
            active_phase = "policy-goal"
            inner._replay_target_lab_order = None
            summary["policy-goal"] = run_policy_goal(env, inner, player, output_dir)
    except Exception as exc:
        failure_key = active_phase if isinstance(exc, PhaseFailure) else "infrastructure-error"
        summary[failure_key or "infrastructure-error"] = {
            "passed": False,
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_summary(output_dir, metadata, summary)
        traceback.print_exc()
        print(f"[output] {output_dir}", file=sys.stderr)
        return 1
    finally:
        env.close()

    write_summary(output_dir, metadata, summary)
    print("\nPhase results")
    for name, result in summary.items():
        print(f"  {name:12s} {'PASS' if result['passed'] else 'FAIL'}")
    print(f"[output] {output_dir}")
    all_passed = all(item.get("passed", False) for item in summary.values())
    return 0 if all_passed or not ARGS.require_pass else 2


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        APP._app.post_quit(exit_code)
        APP.close()
    raise SystemExit(exit_code)
