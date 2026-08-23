#!/usr/bin/env python3
"""Evaluate pretrained SimToolReal acquisition and zero-load Allen-key turning."""

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
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "pretrained_policy/model.pth")
    parser.add_argument("--policy-config", type=Path, default=ROOT / "pretrained_policy/config.yaml")
    parser.add_argument(
        "--object-urdf", type=Path,
        default=ROOT / "assets/urdf/objects/allen_key_canonical.urdf",
    )
    parser.add_argument(
        "--object-scale", type=float, nargs=3, default=(1.0, 1.0, 1.0),
        metavar=("X", "Y", "Z"),
        help="Policy-normalized object scale observation; this does not resize the URDF.",
    )
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--acquisition-steps", type=int, default=240)
    parser.add_argument("--steps-per-goal", type=int, default=240)
    parser.add_argument("--turn-angles-deg", type=float, nargs="+", default=(30.0, 60.0, 90.0))
    parser.add_argument("--socket-z-range-m", type=float, nargs=2, default=(0.37, 0.77))
    parser.add_argument("--socket-xy-range-m", type=float, nargs=2, default=(0.20, 0.20))
    parser.add_argument("--yaw-range-deg", type=float, nargs=2, default=(-180.0, 180.0))
    parser.add_argument("--pose-position-tolerance-m", type=float, default=0.02)
    parser.add_argument("--pose-rotation-tolerance-deg", type=float, default=10.0)
    parser.add_argument("--socket-lateral-tolerance-m", type=float, default=0.01)
    parser.add_argument("--socket-tilt-tolerance-deg", type=float, default=8.0)
    parser.add_argument("--minimum-contact-fingers", type=int, default=2)
    parser.add_argument("--contact-force-threshold-n", type=float, default=0.05)
    parser.add_argument("--stable-grasp-steps", type=int, default=15)
    parser.add_argument("--goal-success-steps", type=int, default=10)
    parser.add_argument("--grasp-step-translation-m", type=float, default=0.003)
    parser.add_argument("--grasp-step-rotation-deg", type=float, default=3.0)
    parser.add_argument("--initial-settle-steps", type=int, default=12)
    parser.add_argument("--initialization-max-trials", type=int, default=20)
    parser.add_argument("--initialization-max-joint-speed-rad-s", type=float, default=0.05)
    parser.add_argument("--initialization-max-socket-error-m", type=float, default=0.03)
    parser.add_argument("--early-motion-window-steps", type=int, default=30)
    parser.add_argument("--early-flyout-distance-m", type=float, default=0.08)
    parser.add_argument(
        "--max-grasp-candidates", type=int, default=0,
        help="Maximum saved grasp snapshots; 0 saves every eligible stage snapshot.",
    )
    parser.add_argument("--video-env-ids", type=int, nargs="*", default=(0, 1, 2))
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--policy-coef-id", type=float, default=0.0)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    args = parser.parse_args()
    args.enable_cameras = bool(args.video)
    return args


ARGS = parse_args()
APP = AppLauncher(ARGS).app


import imageio.v2 as imageio  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab import sim as sim_utils  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from isaaclab.utils.math import (  # noqa: E402
    quat_apply,
    quat_from_angle_axis,
    quat_mul,
    subtract_frame_transforms,
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
from isaacsimenvs.tasks.simtoolreal.simtoolreal_scrape_pose_env import (  # noqa: E402
    SimToolRealTacMapScrapePoseEnv,
)
from isaacsimenvs.tasks.simtoolreal.simtoolreal_tacmap_env_cfg import (  # noqa: E402
    SimToolRealTacMapScrapePoseEnvCfg,
)
from isaacsimenvs.tasks.simtoolreal.utils.inhand_grasp_bank import sha256_file  # noqa: E402
from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import compute_intermediate_values  # noqa: E402
from isaacsimenvs.tasks.simtoolreal.utils.scene_utils import JOINT_NAMES_CANONICAL  # noqa: E402


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


BASE_OBS = (
    "joint_pos", "joint_vel", "prev_action_targets", "palm_pos", "palm_rot",
    "object_rot", "fingertip_pos_rel_palm", "keypoints_rel_palm",
    "keypoints_rel_goal", "object_scales",
)
ALLEN_URDF = ROOT / "assets/urdf/objects/allen_key_canonical.urdf"
SOCKET_URDF = ROOT / "assets/urdf/workpieces/allen_key_hex_socket.urdf"
HIDDEN_TABLE_URDF = ROOT / "assets/urdf/table_allen_disabled.urdf"
SCREW_AXIS_TOOL = (0.0, 0.0, -1.0)
SCREW_PIVOT_TOOL_M = (0.192, 0.0, -0.03)
LONG_HANDLE_CENTER_TOOL_M = (0.067, 0.0, 0.0)
SOCKET_ROOT_FROM_PIVOT_M = (0.0, 0.0, -0.065)


class EvaluationFailure(RuntimeError):
    pass


def quaternion_error_rad(actual: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    alignment = torch.abs((actual * target).sum(dim=-1)).clamp(0.0, 1.0)
    return 2.0 * torch.acos(alignment)


def goal_pose_about_screw_axis(
    initial_pos: torch.Tensor,
    initial_quat: torch.Tensor,
    pivot_tool: torch.Tensor,
    axis_tool: torch.Tensor,
    angle_rad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    axis_world = quat_apply(initial_quat, axis_tool)
    delta = quat_from_angle_axis(angle_rad, axis_world)
    goal_quat = quat_mul(delta, initial_quat)
    pivot_world = initial_pos + quat_apply(initial_quat, pivot_tool)
    goal_pos = pivot_world - quat_apply(goal_quat, pivot_tool)
    return goal_pos, goal_quat, pivot_world


def yaw_quaternion_from_tool(tool_quat: torch.Tensor) -> torch.Tensor:
    local_x = torch.zeros(tool_quat.shape[0], 3, device=tool_quat.device)
    local_x[:, 0] = 1.0
    x_world = quat_apply(tool_quat, local_x)
    yaw = torch.atan2(x_world[:, 1], x_world[:, 0])
    z_axis = torch.zeros_like(local_x)
    z_axis[:, 2] = 1.0
    return quat_from_angle_axis(yaw, z_axis)


class AllenKeyTurningCapabilityEnv(SimToolRealTacMapScrapePoseEnv):
    """Original policy interface with a fixed-axis, freely rotating socket."""

    def __init__(self, cfg, render_mode=None, **kwargs):
        self._turn_fixture_ready = False
        super().__init__(cfg, render_mode, **kwargs)
        self._turn_socket_root_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._turn_hidden_table_pos = self.scene.env_origins.clone()
        self._turn_hidden_table_pos[:, 2] -= 1.0
        self._turn_fixture_ready = True

    def _get_dones(self):
        compute_intermediate_values(self)
        zeros = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._termination_reasons = {"capability_gate": zeros}
        return zeros, zeros

    def _apply_action(self) -> None:
        super()._apply_action()
        if not self._turn_fixture_ready:
            return
        env_ids = torch.arange(self.num_envs, device=self.device)
        socket_quat = yaw_quaternion_from_tool(self.object.data.root_quat_w)
        self.workpiece.write_root_pose_to_sim(
            torch.cat((self._turn_socket_root_pos, socket_quat), dim=-1), env_ids=env_ids
        )
        self.workpiece.write_root_velocity_to_sim(
            torch.zeros(self.num_envs, 6, device=self.device), env_ids=env_ids
        )
        table_quat = torch.zeros(self.num_envs, 4, device=self.device)
        table_quat[:, 0] = 1.0
        self.table.write_root_pose_to_sim(
            torch.cat((self._turn_hidden_table_pos, table_quat), dim=-1), env_ids=env_ids
        )
        self.table.write_root_velocity_to_sim(
            torch.zeros(self.num_envs, 6, device=self.device), env_ids=env_ids
        )


class VideoRecorder:
    def __init__(self, inner, output_dir: Path, env_ids: list[int]):
        self.inner = inner
        self.env_ids = env_ids
        self.capture_every = max(1, round(60 / int(ARGS.video_fps)))
        self.step = 0
        self.cameras: dict[int, Camera] = {}
        self.writers = {}
        if not ARGS.video:
            return
        for env_id in env_ids:
            cfg = CameraCfg(
                prim_path=f"/World/AllenTurningCamera_{env_id}",
                update_period=0,
                height=int(ARGS.camera_height),
                width=int(ARGS.camera_width),
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=24.0,
                    clipping_range=(0.1, 10.0),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=(0.0, 0.0, 10.0), rot=(1.0, 0.0, 0.0, 0.0),
                    convention="opengl",
                ),
            )
            self.cameras[env_id] = Camera(cfg=cfg)
            self.writers[env_id] = imageio.get_writer(
                output_dir / f"env_{env_id:04d}.mp4",
                fps=int(ARGS.video_fps), codec="libx264", quality=8,
                macro_block_size=None,
            )
        inner.sim.reset()

    def set_views(self, pivots: torch.Tensor) -> None:
        if not self.cameras:
            return
        for env_id, camera in self.cameras.items():
            pivot = pivots[env_id]
            eye = pivot + torch.tensor((0.52, -0.78, 0.38), device=self.inner.device)
            target = pivot + torch.tensor((0.0, 0.0, 0.05), device=self.inner.device)
            camera.set_world_poses_from_view(eye.unsqueeze(0), target.unsqueeze(0))

    def capture(self) -> None:
        if not self.cameras:
            return
        self.step += 1
        if self.step % self.capture_every:
            return
        self.inner.sim.render()
        for env_id, camera in self.cameras.items():
            camera.update(self.capture_every * float(self.inner.step_dt))
            rgb = camera.data.output.get("rgb")
            if rgb is None or rgb.shape[0] != 1:
                raise EvaluationFailure(f"camera for env {env_id} returned no RGB frame")
            frame = rgb[0, :, :, :3].detach().cpu().numpy()
            if not np.isfinite(frame).all() or float(frame.mean()) <= 0.0:
                raise EvaluationFailure(f"camera for env {env_id} returned a blank frame")
            self.writers[env_id].append_data(frame.astype(np.uint8))

    def close(self) -> None:
        for writer in self.writers.values():
            writer.close()
        self.writers.clear()


def validate_args() -> None:
    for path in (
        ARGS.checkpoint, ARGS.policy_config, ARGS.object_urdf,
        SOCKET_URDF, HIDDEN_TABLE_URDF,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    for name in (
        "num_envs", "acquisition_steps", "steps_per_goal", "stable_grasp_steps",
        "goal_success_steps", "initial_settle_steps", "early_motion_window_steps",
        "initialization_max_trials",
    ):
        if int(getattr(ARGS, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not ARGS.turn_angles_deg or any(not math.isfinite(v) for v in ARGS.turn_angles_deg):
        raise ValueError("turn angles must be finite and nonempty")
    z0, z1 = (float(v) for v in ARGS.socket_z_range_m)
    if not 0.1 < z0 < z1 < 1.2:
        raise ValueError("socket z range is invalid")
    if any(v < 0.0 or not math.isfinite(v) for v in ARGS.socket_xy_range_m):
        raise ValueError("socket xy ranges must be finite and non-negative")
    yaw0, yaw1 = (float(v) for v in ARGS.yaw_range_deg)
    if not -360.0 <= yaw0 < yaw1 <= 360.0:
        raise ValueError("yaw range is invalid")
    if not 1 <= int(ARGS.minimum_contact_fingers) <= 5:
        raise ValueError("minimum contact fingers must be in [1, 5]")
    if float(ARGS.early_flyout_distance_m) <= 0.0:
        raise ValueError("--early-flyout-distance-m must be positive")
    if float(ARGS.initialization_max_joint_speed_rad_s) <= 0.0:
        raise ValueError("--initialization-max-joint-speed-rad-s must be positive")
    if float(ARGS.initialization_max_socket_error_m) <= 0.0:
        raise ValueError("--initialization-max-socket-error-m must be positive")
    if int(ARGS.max_grasp_candidates) < 0:
        raise ValueError("--max-grasp-candidates must be non-negative")
    invalid_video_ids = [i for i in ARGS.video_env_ids if not 0 <= i < ARGS.num_envs]
    if invalid_video_ids:
        raise ValueError(f"video env ids are out of range: {invalid_video_ids}")


def make_cfg() -> SimToolRealTacMapScrapePoseEnvCfg:
    cfg = SimToolRealTacMapScrapePoseEnvCfg()
    cfg.seed = int(ARGS.seed)
    cfg.scene.num_envs = int(ARGS.num_envs)
    cfg.episode_length_s = 3600.0
    cfg.assets.handle_head_types = ("screwdriver",)
    cfg.assets.object_urdf = str(ARGS.object_urdf.resolve())
    cfg.assets.object_scale = tuple(float(value) for value in ARGS.object_scale)
    cfg.assets.workpiece_urdf = str(SOCKET_URDF)
    cfg.assets.table_urdf = str(HIDDEN_TABLE_URDF)
    cfg.use_tacmap = False
    cfg.enable_vbts = False
    cfg.enable_tactile = False
    cfg.include_tacmap_in_policy = False
    cfg.enable_fingertip_tool_contact_sensors = True
    cfg.fingertip_tool_contact_filter_paths = (
        "/World/envs/env_.*/Object/object_root",
    )
    cfg.vbts_target_rigid_expr = "/World/envs/env_.*/Object/object_root"
    cfg.obs.obs_list = BASE_OBS
    cfg.obs.state_list = BASE_OBS
    cfg.enable_tool_table_contact_force_reward = False
    cfg.enable_tool_table_contact_sensor = False
    cfg.termination.success_steps = 1_000_000
    cfg.termination.max_consecutive_successes = 0
    cfg.domain_randomization.force_scale = 0.0
    cfg.domain_randomization.torque_scale = 0.0
    cfg.domain_randomization.use_obs_delay = False
    cfg.domain_randomization.use_action_delay = False
    cfg.domain_randomization.use_object_state_delay_noise = False
    cfg.domain_randomization.joint_velocity_obs_noise_std = 0.0
    cfg.reset.reset_dof_pos_random_interval_arm = 0.0
    cfg.reset.reset_dof_pos_random_interval_fingers = 0.0
    cfg.reset.reset_dof_vel_random_interval = 0.0
    return cfg


def initialize_robot_at_pretrained_default(
    inner, env_ids: torch.Tensor | None = None
) -> torch.Tensor:
    """Install the center of the pretrained reset distribution without transients."""
    if env_ids is None:
        env_ids = torch.arange(inner.num_envs, device=inner.device)
    joint_pos = inner.robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    inner.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    inner._cur_targets[env_ids] = joint_pos
    inner._prev_targets[env_ids] = joint_pos
    inner.robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    return joint_pos


def write_pose_outcomes(
    output_dir: Path,
    xy: torch.Tensor,
    z: torch.Tensor,
    yaw: torch.Tensor,
    acquired: torch.Tensor,
    acquisition_step: torch.Tensor,
    stage_success: torch.Tensor,
    retention: torch.Tensor,
    post_settle_socket_error: torch.Tensor,
    initialization_trials: torch.Tensor,
    early_max_socket_error: torch.Tensor,
    early_max_tool_speed: torch.Tensor,
    early_max_palm_speed: torch.Tensor,
) -> None:
    columns = [
        "env_id", "socket_x_m", "socket_y_m", "socket_z_m", "yaw_deg",
        "handle_center_x_m", "handle_center_y_m", "handle_center_z_m",
        "acquired", "acquisition_step", "all_turns_success",
        "contact_retention", "post_settle_socket_error_m",
        "initialization_trials",
        "early_max_socket_error_m", "early_max_tool_speed_mps",
        "early_max_palm_speed_mps",
    ] + [f"turn_{angle:g}_success" for angle in ARGS.turn_angles_deg]
    arrays = [
        xy.detach().cpu().numpy(), z.detach().cpu().numpy(),
        torch.rad2deg(yaw).detach().cpu().numpy(), acquired.detach().cpu().numpy(),
        acquisition_step.detach().cpu().numpy(), stage_success.detach().cpu().numpy(),
        retention.detach().cpu().numpy(), post_settle_socket_error.detach().cpu().numpy(),
        initialization_trials.detach().cpu().numpy(),
        early_max_socket_error.detach().cpu().numpy(),
        early_max_tool_speed.detach().cpu().numpy(),
        early_max_palm_speed.detach().cpu().numpy(),
    ]
    xy_np, z_np, yaw_np, acquired_np, step_np, success_np, retention_np, settle_np, trials_np, socket_np, tool_np, palm_np = arrays
    yaw_rad = np.deg2rad(yaw_np)
    handle_from_socket = np.asarray(LONG_HANDLE_CENTER_TOOL_M) - np.asarray(
        SCREW_PIVOT_TOOL_M
    )
    handle_x = xy_np[:, 0] + handle_from_socket[0] * np.cos(yaw_rad)
    handle_y = xy_np[:, 1] + handle_from_socket[0] * np.sin(yaw_rad)
    handle_z = z_np + handle_from_socket[2]
    with (output_dir / "pose_outcomes.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for env_id in range(len(z_np)):
            writer.writerow([
                env_id, *xy_np[env_id].tolist(), float(z_np[env_id]),
                float(yaw_np[env_id]), float(handle_x[env_id]),
                float(handle_y[env_id]), float(handle_z[env_id]),
                int(acquired_np[env_id]), int(step_np[env_id]),
                int(success_np[env_id].all()), float(retention_np[env_id]),
                float(settle_np[env_id]), int(trials_np[env_id]), float(socket_np[env_id]),
                float(tool_np[env_id]), float(palm_np[env_id]),
                *success_np[env_id].astype(np.int64).tolist(),
            ])

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    masks = [
        (~acquired_np.astype(bool), "not acquired", "0.75"),
        (acquired_np.astype(bool), "acquired", "#2878b5"),
        (success_np.all(axis=1), "all turns", "#d89000"),
    ]
    ax = axes[0, 0]
    for mask, label, color in masks:
        ax.scatter(xy_np[mask, 0], xy_np[mask, 1], s=10, alpha=0.65, c=color, label=label)
    ax.set(xlabel="socket x (m)", ylabel="socket y (m)", title="XY outcomes")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, fontsize=9)
    values = [xy_np[:, 0], xy_np[:, 1], z_np, yaw_np]
    labels = ["socket x (m)", "socket y (m)", "socket z (m)", "yaw (deg)"]
    for ax, value, label in zip(axes.flat[1:5], values, labels):
        bins = np.linspace(float(value.min()), float(value.max()), 13)
        ax.hist(value, bins=bins, color="0.82", label="all poses")
        ax.hist(value[acquired_np.astype(bool)], bins=bins, color="#2878b5", alpha=0.8, label="acquired")
        ax.hist(value[success_np.all(axis=1)], bins=bins, color="#d89000", alpha=0.9, label="all turns")
        ax.set(xlabel=label, ylabel="count")
    ax = axes[1, 2]
    ax.axis("off")
    acquired_count = int(acquired_np.sum())
    all_turns_count = int(success_np.all(axis=1).sum())
    ax.text(0.02, 0.95, "Pretrained Allen-key gate", va="top", fontsize=14, weight="bold")
    ax.text(
        0.02, 0.76,
        f"Acquired: {acquired_count}/{len(acquired_np)} ({acquired_np.mean():.1%})\n"
        f"All turns: {all_turns_count}/{len(acquired_np)} ({success_np.all(axis=1).mean():.1%})\n"
        f"All turns | acquired: {all_turns_count / max(acquired_count, 1):.1%}",
        va="top", fontsize=12, linespacing=1.5,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"success_pose_distribution.{suffix}", dpi=180)
    plt.close(fig)


def write_goal(inner, env_ids: torch.Tensor, pos: torch.Tensor, quat: torch.Tensor) -> None:
    inner.goal_viz.write_root_pose_to_sim(
        torch.cat((pos, quat), dim=-1), env_ids=env_ids
    )
    inner.goal_viz.write_root_velocity_to_sim(
        torch.zeros(env_ids.numel(), 6, device=inner.device), env_ids=env_ids
    )


def read_fingertip_forces(inner) -> torch.Tensor:
    sensors = getattr(inner, "_fingertip_tool_contact_sensors", None)
    if sensors is None or len(sensors) != 5:
        raise EvaluationFailure("five fingertip-tool contact sensors are required")
    forces = []
    for sensor_id, sensor in enumerate(sensors):
        matrix = getattr(getattr(sensor, "data", None), "force_matrix_w", None)
        if matrix is None or matrix.shape[0] != inner.num_envs or matrix.shape[-1] != 3:
            shape = None if matrix is None else tuple(matrix.shape)
            raise EvaluationFailure(
                f"fingertip contact sensor {sensor_id} has invalid force matrix {shape}"
            )
        force = torch.linalg.vector_norm(
            matrix.reshape(inner.num_envs, -1, 3), dim=-1
        ).sum(-1)
        if not bool(torch.isfinite(force).all()):
            raise EvaluationFailure(f"fingertip contact sensor {sensor_id} is non-finite")
        forces.append(force)
    return torch.stack(forces, dim=-1)


def snapshot_candidate(
    inner, env_id: int, action: torch.Tensor, stage: int, step: int,
    forces: torch.Tensor, position_error: torch.Tensor,
    rotation_error: torch.Tensor, socket_error: torch.Tensor,
) -> dict:
    palm_pos = inner.robot.data.body_link_pos_w[:, inner._palm_body_id]
    palm_quat = inner.robot.data.body_link_quat_w[:, inner._palm_body_id]
    rel_pos, rel_quat = subtract_frame_transforms(
        palm_pos, palm_quat, inner.object.data.root_pos_w, inner.object.data.root_quat_w
    )
    perm = inner._perm_lab_to_canon
    origin = inner.scene.env_origins[env_id]
    return {
        "source_env_id": int(env_id),
        "policy_step": int(step),
        "turn_stage": int(stage),
        "target_turn_angle_deg": (
            0.0 if stage < 0 else float(ARGS.turn_angles_deg[stage])
        ),
        "joint_pos_canonical": inner.robot.data.joint_pos[env_id, perm].tolist(),
        "joint_vel_canonical": inner.robot.data.joint_vel[env_id, perm].tolist(),
        "joint_targets_canonical": inner._cur_targets[env_id, perm].tolist(),
        "last_action_canonical": action[env_id].tolist(),
        "object_pos_local": (inner.object.data.root_pos_w[env_id] - origin).tolist(),
        "object_quat_wxyz": inner.object.data.root_quat_w[env_id].tolist(),
        "palm_to_tool_pos": rel_pos[env_id].tolist(),
        "palm_to_tool_quat_wxyz": rel_quat[env_id].tolist(),
        "fingertip_force_n": forces[env_id].tolist(),
        "pose_position_error_m": float(position_error[env_id]),
        "pose_rotation_error_deg": float(torch.rad2deg(rotation_error[env_id])),
        "socket_pivot_error_m": float(socket_error[env_id]),
    }


def write_viewers(inner, frames: dict[int, list[dict]], output_dir: Path) -> None:
    object_text, object_path = object_urdf_for_env(inner, 0)
    table_text, table_path = table_urdf_for_env(inner, 0)
    workpiece_text, workpiece_path = workpiece_urdf_for_env(inner)
    for env_id, env_frames in frames.items():
        if not env_frames:
            continue
        html = build_pose_viewer_html(
            frames=env_frames,
            object_urdf_text=object_text,
            table_urdf_text=table_text,
            workpiece_urdf_text=workpiece_text,
            object_urdf_path=object_path,
            table_urdf_path=table_path,
            workpiece_urdf_path=workpiece_path,
        )
        (output_dir / f"env_{env_id:04d}.html").write_text(html, encoding="utf-8")


def main() -> None:
    validate_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = ARGS.output_dir or ROOT / "outputs/allen_key_pretrained_turning" / stamp
    output_dir.mkdir(parents=True, exist_ok=False)
    cfg = make_cfg()
    env = AllenKeyTurningCapabilityEnv(cfg)
    recorder = None
    try:
        inner = env.unwrapped
        player = RlPlayer(
            num_observations=inner.cfg.observation_space,
            num_actions=inner.cfg.action_space,
            config_path=str(ARGS.policy_config),
            checkpoint_path=str(ARGS.checkpoint),
            device=str(inner.device),
            num_envs=inner.num_envs,
            coefficient_id=float(ARGS.policy_coef_id),
        )
        env.reset()
        video_ids = sorted(set(int(i) for i in ARGS.video_env_ids))
        recorder = VideoRecorder(inner, output_dir, video_ids)

        n = inner.num_envs
        env_ids = torch.arange(n, device=inner.device)
        generator = torch.Generator(device=inner.device).manual_seed(int(ARGS.seed))
        xy = torch.zeros(n, 2, device=inner.device)
        z = torch.zeros(n, device=inner.device)
        yaw = torch.zeros(n, device=inner.device)
        z_axis = torch.zeros(n, 3, device=inner.device)
        z_axis[:, 2] = 1.0
        initial_quat = torch.zeros(n, 4, device=inner.device)
        initial_quat[:, 0] = 1.0
        pivot = torch.zeros(n, 3, device=inner.device)
        pivot_tool = torch.tensor(SCREW_PIVOT_TOOL_M, device=inner.device).expand(n, -1)
        axis_tool = torch.tensor(SCREW_AXIS_TOOL, device=inner.device).expand(n, -1)
        initial_pos = torch.zeros(n, 3, device=inner.device)
        settle_max_socket_error = torch.zeros(n, device=inner.device)
        initialization_trials = torch.zeros(n, dtype=torch.long, device=inner.device)
        pending = env_ids
        observation = None
        zero_action = torch.zeros(n, inner.cfg.action_space, device=inner.device)
        default_joint_pos = inner.robot.data.default_joint_pos.clone()

        for trial in range(int(ARGS.initialization_max_trials)):
            if pending.numel() == 0:
                break
            count = pending.numel()
            sampled_xy = torch.empty(
                count, 2, device=inner.device
            ).uniform_(-1.0, 1.0, generator=generator)
            sampled_xy[:, 0] *= float(ARGS.socket_xy_range_m[0])
            sampled_xy[:, 1] *= float(ARGS.socket_xy_range_m[1])
            sampled_z = torch.empty(count, device=inner.device).uniform_(
                float(ARGS.socket_z_range_m[0]), float(ARGS.socket_z_range_m[1]),
                generator=generator,
            )
            sampled_yaw = torch.empty(count, device=inner.device).uniform_(
                math.radians(float(ARGS.yaw_range_deg[0])),
                math.radians(float(ARGS.yaw_range_deg[1])), generator=generator,
            )
            xy[pending] = sampled_xy
            z[pending] = sampled_z
            yaw[pending] = sampled_yaw
            initial_quat[pending] = quat_from_angle_axis(
                sampled_yaw, z_axis[pending]
            )
            pivot[pending] = (
                inner.scene.env_origins[pending]
                + torch.cat((sampled_xy, sampled_z.unsqueeze(-1)), dim=-1)
            )
            initial_pos[pending] = (
                pivot[pending] - quat_apply(initial_quat[pending], pivot_tool[pending])
            )
            inner._turn_socket_root_pos[pending] = (
                pivot[pending]
                + torch.tensor(SOCKET_ROOT_FROM_PIVOT_M, device=inner.device)
            )
            inner.object.write_root_pose_to_sim(
                torch.cat((initial_pos[pending], initial_quat[pending]), dim=-1),
                env_ids=pending,
            )
            inner.object.write_root_velocity_to_sim(
                torch.zeros(count, 6, device=inner.device), env_ids=pending
            )
            inner._object_init_z[pending] = (
                initial_pos[pending, 2] - inner.scene.env_origins[pending, 2]
            )
            initialize_robot_at_pretrained_default(inner, pending)
            write_goal(inner, pending, initial_pos[pending], initial_quat[pending])

            trial_max_socket_error = torch.zeros(n, device=inner.device)
            inner._replay_target_lab_order = default_joint_pos
            for _ in range(int(ARGS.initial_settle_steps)):
                observation, _, terminated, truncated, _ = env.step(zero_action)
                if bool(terminated.any()) or bool(truncated.any()):
                    raise EvaluationFailure(
                        "capability environment reset during initial settle"
                    )
                settled_pivot = inner.object.data.root_pos_w + quat_apply(
                    inner.object.data.root_quat_w, pivot_tool
                )
                trial_max_socket_error = torch.maximum(
                    trial_max_socket_error,
                    torch.linalg.vector_norm(settled_pivot - pivot, dim=-1),
                )
            joint_speed = inner.robot.data.joint_vel.abs().amax(dim=-1)
            valid = (
                torch.isfinite(joint_speed)
                & torch.isfinite(trial_max_socket_error)
                & (joint_speed <= float(ARGS.initialization_max_joint_speed_rad_s))
                & (trial_max_socket_error <= float(ARGS.initialization_max_socket_error_m))
            )
            accepted = env_ids[valid]
            settle_max_socket_error[valid] = trial_max_socket_error[valid]
            initialization_trials[pending] = trial + 1
            pending = env_ids[~valid]
            print(
                f"[initialization] trial={trial + 1} accepted={accepted.numel()} "
                f"remaining={pending.numel()}",
                flush=True,
            )
        del inner._replay_target_lab_order
        if pending.numel():
            raise EvaluationFailure(
                f"failed to sample collision-free initialization for {pending.numel()}/{n} "
                f"environments after {ARGS.initialization_max_trials} trials: "
                f"env_ids={pending[:32].tolist()}"
            )
        if observation is None:
            raise EvaluationFailure("initial settle produced no observation")
        initial_tool_pos = initial_pos.clone()
        initial_tool_quat = initial_quat.clone()

        angles = torch.deg2rad(torch.tensor(ARGS.turn_angles_deg, device=inner.device))
        goal_positions = []
        goal_quaternions = []
        for angle in angles:
            pos, quat, _ = goal_pose_about_screw_axis(
                initial_tool_pos, initial_tool_quat, pivot_tool, axis_tool,
                angle.expand(n),
            )
            goal_positions.append(pos)
            goal_quaternions.append(quat)
        goal_positions_t = torch.stack(goal_positions, dim=1)
        goal_quaternions_t = torch.stack(goal_quaternions, dim=1)
        # Match the original manipulation task: first acquire the tool at its
        # current pose, then command rotations only after grasp verification.
        write_goal(inner, env_ids, initial_tool_pos, initial_tool_quat)
        recorder.set_views(pivot)
        initial_joint_velocity = inner.robot.data.joint_vel
        if not bool(torch.isfinite(initial_joint_velocity).all()):
            raise EvaluationFailure("robot velocity is non-finite after initialization")
        max_initial_joint_speed = float(initial_joint_velocity.abs().max())
        print(
            "[initialization] "
            f"max_joint_speed={max_initial_joint_speed:.4f} rad/s "
            f"max_socket_error={float(settle_max_socket_error.max()):.4f} m",
            flush=True,
        )
        if max_initial_joint_speed > 0.05:
            raise EvaluationFailure(
                "robot did not settle at the pretrained default pose: "
                f"maximum joint speed={max_initial_joint_speed:.4f} rad/s"
            )
        player.reset()
        post_settle_pivot = inner.object.data.root_pos_w + quat_apply(
            inner.object.data.root_quat_w, pivot_tool
        )
        post_settle_socket_error = torch.linalg.vector_norm(
            post_settle_pivot - pivot, dim=-1
        )

        # -1 is acquisition, [0, num_turns) are turn goals, and num_turns is done.
        stage = torch.full((n,), -1, dtype=torch.long, device=inner.device)
        stage_step = torch.zeros_like(stage)
        goal_hold = torch.zeros_like(stage)
        grasp_hold = torch.zeros_like(stage)
        acquired = torch.zeros(n, dtype=torch.bool, device=inner.device)
        acquisition_step = torch.full((n,), -1, dtype=torch.long, device=inner.device)
        stage_success = torch.zeros(n, len(angles), dtype=torch.bool, device=inner.device)
        candidate_saved = torch.zeros(
            n, len(angles) + 1, dtype=torch.bool, device=inner.device
        )
        contact_after_acquire = torch.zeros(n, device=inner.device)
        steps_after_acquire = torch.zeros(n, device=inner.device)
        previous_rel_pos = None
        previous_rel_quat = None
        previous_palm_pos = inner.robot.data.body_link_pos_w[:, inner._palm_body_id].clone()
        early_max_socket_error = settle_max_socket_error.clone()
        early_max_tool_speed = torch.zeros(n, device=inner.device)
        early_max_palm_speed = torch.zeros(n, device=inner.device)
        candidates: list[dict] = []
        candidate_generator = torch.Generator(device=inner.device).manual_seed(
            int(ARGS.seed) + 10_000
        )
        candidate_slots = len(angles) + 1
        if int(ARGS.max_grasp_candidates) > 0:
            base_budget, remainder = divmod(
                int(ARGS.max_grasp_candidates), candidate_slots
            )
            candidate_budget = [
                base_budget + int(slot < remainder)
                for slot in range(candidate_slots)
            ]
        else:
            candidate_budget = [n] * candidate_slots
        candidate_count_by_slot = [0] * candidate_slots
        viewer_frames: dict[int, list[dict]] = {env_id: [] for env_id in video_ids}
        total_steps = int(ARGS.acquisition_steps) + int(ARGS.steps_per_goal) * len(angles)

        for policy_step in range(total_steps):
            action = player.get_normalized_action(
                observation["policy"], deterministic_actions=True
            ).to(inner.device)
            observation, _, terminated, truncated, _ = env.step(action)
            if bool(terminated.any()) or bool(truncated.any()):
                raise EvaluationFailure("capability environment reset unexpectedly")
            forces = read_fingertip_forces(inner)
            contacts = forces >= float(ARGS.contact_force_threshold_n)
            contact_count = contacts.sum(dim=-1)

            palm_pos = inner.robot.data.body_link_pos_w[:, inner._palm_body_id]
            palm_quat = inner.robot.data.body_link_quat_w[:, inner._palm_body_id]
            rel_pos, rel_quat = subtract_frame_transforms(
                palm_pos, palm_quat,
                inner.object.data.root_pos_w, inner.object.data.root_quat_w,
            )
            if previous_rel_pos is None:
                step_translation = torch.zeros(n, device=inner.device)
                step_rotation = torch.zeros(n, device=inner.device)
            else:
                step_translation = torch.linalg.vector_norm(rel_pos - previous_rel_pos, dim=-1)
                step_rotation = quaternion_error_rad(rel_quat, previous_rel_quat)
            previous_rel_pos, previous_rel_quat = rel_pos.clone(), rel_quat.clone()
            stable_contact = (
                (contact_count >= int(ARGS.minimum_contact_fingers))
                & (step_translation <= float(ARGS.grasp_step_translation_m))
                & (step_rotation <= math.radians(float(ARGS.grasp_step_rotation_deg)))
            )
            grasp_hold = torch.where(stable_contact, grasp_hold + 1, torch.zeros_like(grasp_hold))
            acquired_now = (
                (stage == -1) & (grasp_hold >= int(ARGS.stable_grasp_steps))
            )
            acquired |= acquired_now
            steps_after_acquire += acquired.float()
            contact_after_acquire += (acquired & stable_contact).float()

            active_turn = (stage >= 0) & (stage < len(angles))
            active_stage = stage.clamp(min=0, max=len(angles) - 1)
            batch = torch.arange(n, device=inner.device)
            goal_pos = torch.where(
                active_turn.unsqueeze(-1), goal_positions_t[batch, active_stage],
                initial_tool_pos,
            )
            goal_quat = torch.where(
                active_turn.unsqueeze(-1), goal_quaternions_t[batch, active_stage],
                initial_tool_quat,
            )
            position_error = torch.linalg.vector_norm(
                inner.object.data.root_pos_w - goal_pos, dim=-1
            )
            rotation_error = quaternion_error_rad(inner.object.data.root_quat_w, goal_quat)
            current_pivot = inner.object.data.root_pos_w + quat_apply(
                inner.object.data.root_quat_w, pivot_tool
            )
            socket_error = torch.linalg.vector_norm(current_pivot - pivot, dim=-1)
            if policy_step < int(ARGS.early_motion_window_steps):
                early_max_socket_error = torch.maximum(early_max_socket_error, socket_error)
                early_max_tool_speed = torch.maximum(
                    early_max_tool_speed,
                    torch.linalg.vector_norm(inner.object.data.root_lin_vel_w, dim=-1),
                )
                palm_speed = torch.linalg.vector_norm(
                    palm_pos - previous_palm_pos, dim=-1
                ) / float(inner.step_dt)
                early_max_palm_speed = torch.maximum(early_max_palm_speed, palm_speed)
            previous_palm_pos = palm_pos.clone()
            current_axis = quat_apply(inner.object.data.root_quat_w, axis_tool)
            socket_tilt = torch.acos((-current_axis[:, 2]).clamp(-1.0, 1.0))
            at_goal = (
                active_turn & acquired & stable_contact
                & (position_error <= float(ARGS.pose_position_tolerance_m))
                & (rotation_error <= math.radians(float(ARGS.pose_rotation_tolerance_deg)))
                & (socket_error <= float(ARGS.socket_lateral_tolerance_m))
                & (socket_tilt <= math.radians(float(ARGS.socket_tilt_tolerance_deg)))
            )
            goal_hold = torch.where(at_goal, goal_hold + 1, torch.zeros_like(goal_hold))
            reached = goal_hold >= int(ARGS.goal_success_steps)

            candidate_slot = (stage + 1).clamp(min=0, max=len(angles))
            save_mask = (
                acquired & stable_contact & (stage < len(angles))
                & ~candidate_saved[batch, candidate_slot]
            )
            save_ids = save_mask.nonzero(as_tuple=False).squeeze(-1)
            for slot in range(candidate_slots):
                slot_ids = save_ids[candidate_slot[save_ids] == slot]
                if slot_ids.numel() == 0:
                    continue
                remaining = candidate_budget[slot] - candidate_count_by_slot[slot]
                if remaining > 0 and slot_ids.numel() > remaining:
                    order = torch.randperm(
                        slot_ids.numel(), device=inner.device,
                        generator=candidate_generator,
                    )
                    selected_ids = slot_ids[order[:remaining]]
                elif remaining > 0:
                    selected_ids = slot_ids
                else:
                    selected_ids = slot_ids[:0]
                for env_id in selected_ids.tolist():
                    candidates.append(snapshot_candidate(
                        inner, env_id, action, int(stage[env_id]), policy_step,
                        forces, position_error, rotation_error, socket_error,
                    ))
                candidate_count_by_slot[slot] += selected_ids.numel()
                candidate_saved[slot_ids, slot] = True

            reached_turn = reached & active_turn
            stage_success[batch[reached_turn], active_stage[reached_turn]] = True
            stage_step += 1
            acquired_advance = (stage == -1) & acquired_now
            acquisition_step[acquired_advance] = policy_step
            acquisition_failed = (
                (stage == -1) & (stage_step >= int(ARGS.acquisition_steps))
                & ~acquired_now
            )
            turn_timed_out = (
                active_turn & (stage_step >= int(ARGS.steps_per_goal))
            )
            turn_advance = reached_turn | turn_timed_out
            stage[acquired_advance] = 0
            stage[acquisition_failed] = len(angles)
            stage[turn_advance] += 1
            changed_stage = acquired_advance | acquisition_failed | turn_advance
            stage_step[changed_stage] = 0
            goal_hold[changed_stage] = 0
            next_ids = torch.nonzero(
                changed_stage & (stage >= 0) & (stage < len(angles)),
                as_tuple=False,
            ).squeeze(-1)
            if next_ids.numel():
                next_stage = stage[next_ids]
                write_goal(
                    inner, next_ids,
                    goal_positions_t[next_ids, next_stage],
                    goal_quaternions_t[next_ids, next_stage],
                )

            recorder.capture()
            if policy_step % 6 == 0:
                for env_id in video_ids:
                    viewer_frames[env_id].append(capture_pose_viewer_frame(inner, env_id))
            if (policy_step + 1) % 60 == 0:
                rates = stage_success.float().mean(dim=0).tolist()
                print(
                    f"[rollout] step={policy_step + 1}/{total_steps} "
                    f"acquired={float(acquired.float().mean()):.3f} "
                    f"turn_success={[round(v, 3) for v in rates]}", flush=True,
                )

        recorder.close()
        recorder = None
        write_viewers(inner, viewer_frames, output_dir)
        retention = contact_after_acquire / steps_after_acquire.clamp_min(1.0)
        acquisition_success_env_ids = acquired.nonzero(as_tuple=False).squeeze(-1).tolist()
        turn_success_env_ids = [
            stage_success[:, turn_id].nonzero(as_tuple=False).squeeze(-1).tolist()
            for turn_id in range(len(angles))
        ]
        all_turns_success_env_ids = (
            stage_success.all(dim=1).nonzero(as_tuple=False).squeeze(-1).tolist()
        )
        acquired_count = int(acquired.sum())
        turn_success_count = stage_success.sum(dim=0)
        all_turns_success_count = int(stage_success.all(dim=1).sum())
        early_flyout = early_max_socket_error > float(ARGS.early_flyout_distance_m)
        write_pose_outcomes(
            output_dir, xy, z, yaw, acquired, acquisition_step, stage_success,
            retention, post_settle_socket_error, initialization_trials,
            early_max_socket_error,
            early_max_tool_speed, early_max_palm_speed,
        )
        summary = {
            "checkpoint": str(ARGS.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(ARGS.checkpoint),
            "num_envs": n,
            "seed": int(ARGS.seed),
            "acquisition_steps": int(ARGS.acquisition_steps),
            "steps_per_goal": int(ARGS.steps_per_goal),
            "initial_settle_steps": int(ARGS.initial_settle_steps),
            "initialization_max_trials_used": int(initialization_trials.max()),
            "initialization_resampled_pose_count": int(
                (initialization_trials > 1).sum()
            ),
            "max_robot_joint_speed_after_settle_rad_s": max_initial_joint_speed,
            "early_motion_window_steps": int(ARGS.early_motion_window_steps),
            "early_flyout_distance_m": float(ARGS.early_flyout_distance_m),
            "early_flyout_count": int(early_flyout.sum()),
            "early_flyout_rate": float(early_flyout.float().mean()),
            "post_settle_socket_error_mean_m": float(post_settle_socket_error.mean()),
            "early_max_tool_speed_mean_mps": float(early_max_tool_speed.mean()),
            "early_max_palm_speed_mean_mps": float(early_max_palm_speed.mean()),
            "socket_z_range_m": list(map(float, ARGS.socket_z_range_m)),
            "socket_xy_range_m": list(map(float, ARGS.socket_xy_range_m)),
            "yaw_range_deg": list(map(float, ARGS.yaw_range_deg)),
            "turn_angles_deg": list(map(float, ARGS.turn_angles_deg)),
            "acquisition_success_count": acquired_count,
            "acquisition_success_rate": float(acquired.float().mean()),
            "acquisition_success_env_ids": acquisition_success_env_ids,
            "turn_success_count": turn_success_count.tolist(),
            "turn_success_rate": stage_success.float().mean(dim=0).tolist(),
            "turn_success_rate_given_acquisition": (
                (turn_success_count.float() / acquired_count).tolist()
                if acquired_count else [0.0] * len(angles)
            ),
            "turn_success_env_ids": turn_success_env_ids,
            "all_turns_success_count": all_turns_success_count,
            "all_turns_success_rate": float(stage_success.all(dim=1).float().mean()),
            "all_turns_success_rate_given_acquisition": (
                all_turns_success_count / acquired_count if acquired_count else 0.0
            ),
            "all_turns_success_env_ids": all_turns_success_env_ids,
            "contact_retention_after_acquisition_mean": float(retention[acquired].mean())
            if bool(acquired.any()) else 0.0,
            "saved_grasp_candidates": len(candidates),
            "saved_grasp_candidates_by_stage": candidate_count_by_slot,
            "video_env_ids": video_ids,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        candidate_payload = {
            "format": "allen_key_live_policy_grasp_candidates_v1",
            "source_checkpoint": str(ARGS.checkpoint.resolve()),
            "source_checkpoint_sha256": sha256_file(ARGS.checkpoint),
            "asset": str(ARGS.object_urdf.resolve()),
            "asset_sha256": sha256_file(ARGS.object_urdf),
            "joint_names_canonical": list(JOINT_NAMES_CANONICAL),
            "entries": candidates,
        }
        (output_dir / "grasp_candidates.json").write_text(
            json.dumps(candidate_payload, indent=2) + "\n"
        )
        print(f"[pass] {json.dumps(summary, sort_keys=True)}", flush=True)
        print(f"[output] {output_dir.resolve()}", flush=True)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if recorder is not None:
            recorder.close()
        env.close()
        APP.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        APP.close()
        raise
