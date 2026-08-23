"""Pure helpers and procedural assets for Allen-key turning."""

from __future__ import annotations

import math
from pathlib import Path

import torch


def quat_mul(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Multiply wxyz quaternions."""
    aw, ax, ay, az = first.unbind(-1)
    bw, bx, by, bz = second.unbind(-1)
    return torch.stack((
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ), dim=-1)


def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Construct normalized wxyz quaternions."""
    normalized = axis / torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1e-12)
    half = 0.5 * angle
    return torch.cat((torch.cos(half).unsqueeze(-1), normalized * torch.sin(half).unsqueeze(-1)), -1)


def quat_apply(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by normalized wxyz quaternions."""
    q_vector = quaternion[..., 1:]
    twice_cross = 2.0 * torch.linalg.cross(q_vector, vector, dim=-1)
    return vector + quaternion[..., :1] * twice_cross + torch.linalg.cross(
        q_vector, twice_cross, dim=-1
    )


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap radians to [-pi, pi)."""
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def yaw_from_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    """Return world yaw from a wxyz quaternion using its local X axis."""
    local_x = torch.zeros(quaternion.shape[0], 3, device=quaternion.device)
    local_x[:, 0] = 1.0
    world_x = quat_apply(quaternion, local_x)
    return torch.atan2(world_x[:, 1], world_x[:, 0])


def update_unwrapped_angle(
    previous_yaw: torch.Tensor,
    current_yaw: torch.Tensor,
    cumulative_angle: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Accumulate the shortest signed yaw increment without a +/-pi jump."""
    delta = wrap_to_pi(current_yaw - previous_yaw)
    return cumulative_angle + delta, delta


def turn_goal_pose(
    initial_position: torch.Tensor,
    initial_quaternion: torch.Tensor,
    pivot_tool: torch.Tensor,
    axis_tool: torch.Tensor,
    angle: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate a rigid tool about a tool-frame screw axis and pivot."""
    axis_world = quat_apply(initial_quaternion, axis_tool)
    delta = quat_from_angle_axis(angle, axis_world)
    goal_quaternion = quat_mul(delta, initial_quaternion)
    pivot_world = initial_position + quat_apply(initial_quaternion, pivot_tool)
    goal_position = pivot_world - quat_apply(goal_quaternion, pivot_tool)
    return goal_position, goal_quaternion


def finger_effort_soft_penalty(
    applied_torque: torch.Tensor,
    effort_limit: torch.Tensor,
    threshold_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Penalize only actuator effort above a normalized soft threshold."""
    if applied_torque.shape != effort_limit.shape or applied_torque.ndim != 2:
        raise ValueError("finger torque and effort limit tensors must be matching matrices")
    if not 0.0 < threshold_fraction < 1.0:
        raise ValueError("finger effort threshold fraction must lie in (0, 1)")
    if not bool(torch.isfinite(applied_torque).all()):
        raise ValueError("finger applied torque contains NaN or Inf")
    if not bool(torch.isfinite(effort_limit).all()) or bool((effort_limit <= 0.0).any()):
        raise ValueError("finger effort limits must be finite and positive")
    ratio = applied_torque.abs() / effort_limit
    excess = torch.relu(ratio - threshold_fraction)
    return excess.square().mean(-1), ratio.amax(-1), (ratio >= 0.999).float().mean(-1)


def turning_curriculum_ready(
    acquired_count: int,
    completed_count: int,
    full_turn_success_count: int,
    *,
    minimum_episodes: int,
    acquisition_threshold: float,
    conditional_turn_threshold: float,
) -> tuple[bool, float, float]:
    """Gate difficulty on acquisition and full-turn success given acquisition."""
    if min(acquired_count, completed_count, full_turn_success_count) < 0:
        raise ValueError("curriculum counts must be non-negative")
    if acquired_count > completed_count or full_turn_success_count > acquired_count:
        raise ValueError("curriculum counts are inconsistent")
    if minimum_episodes <= 0:
        raise ValueError("minimum curriculum episode count must be positive")
    if not 0.0 <= acquisition_threshold <= 1.0:
        raise ValueError("acquisition threshold must lie in [0, 1]")
    if not 0.0 <= conditional_turn_threshold <= 1.0:
        raise ValueError("conditional turn threshold must lie in [0, 1]")
    acquisition_rate = acquired_count / completed_count if completed_count else 0.0
    conditional_rate = (
        full_turn_success_count / acquired_count if acquired_count else 0.0
    )
    ready = (
        completed_count >= minimum_episodes
        and acquisition_rate >= acquisition_threshold
        and conditional_rate >= conditional_turn_threshold
    )
    return ready, acquisition_rate, conditional_rate


def allen_key_urdf_text(
    *,
    long_handle_length_m: float,
    handle_across_flats_m: float,
    short_leg_length_m: float,
    elbow_x_m: float,
    mesh_path: Path,
) -> str:
    """Build an L-shaped hexagonal key while keeping the screw pivot fixed."""
    values = (
        long_handle_length_m,
        handle_across_flats_m,
        short_leg_length_m,
        elbow_x_m,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("Allen-key dimensions must be finite and positive")
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Allen-key prism mesh does not exist: {mesh_path}")
    overlap = 0.25 * handle_across_flats_m
    long_center_x = elbow_x_m - 0.5 * long_handle_length_m + overlap
    short_center_z = -0.5 * short_leg_length_m
    # A unit regular hexagonal prism has unit circumdiameter. Across-flats is
    # therefore sqrt(3)/2 times its Y/Z scale.
    prism_diameter = handle_across_flats_m * 2.0 / math.sqrt(3.0)
    short_prism_diameter = min(prism_diameter, 0.012 * 2.0 / math.sqrt(3.0))
    mass = 0.18 * long_handle_length_m / 0.264
    name_length_mm = round(long_handle_length_m * 1000.0)
    return f'''<?xml version="1.0"?>
<robot name="allen_key_turning_{name_length_mm}mm">
  <link name="object_root">
    <visual><origin xyz="{long_center_x:.7f} 0 0" rpy="0 0 0"/><geometry><mesh filename="{mesh_path}" scale="{long_handle_length_m:.7f} {prism_diameter:.7f} {prism_diameter:.7f}"/></geometry><material name="grip"><color rgba="0.16 0.20 0.23 1"/></material></visual>
    <collision><origin xyz="{long_center_x:.7f} 0 0" rpy="0 0 0"/><geometry><mesh filename="{mesh_path}" scale="{long_handle_length_m:.7f} {prism_diameter:.7f} {prism_diameter:.7f}"/></geometry></collision>
    <visual><origin xyz="{elbow_x_m:.7f} 0 {short_center_z:.7f}" rpy="0 1.57079632679 0"/><geometry><mesh filename="{mesh_path}" scale="{short_leg_length_m:.7f} {short_prism_diameter:.7f} {short_prism_diameter:.7f}"/></geometry><material name="steel"><color rgba="0.32 0.36 0.39 1"/></material></visual>
    <collision><origin xyz="{elbow_x_m:.7f} 0 {short_center_z:.7f}" rpy="0 1.57079632679 0"/><geometry><mesh filename="{mesh_path}" scale="{short_leg_length_m:.7f} {short_prism_diameter:.7f} {short_prism_diameter:.7f}"/></geometry></collision>
    <!-- Put the simulated COM on the screw axis. Together with the task's
         yaw-only projection this models an ideal revolute fixture without a
         synthetic grasp joint or solver-dependent socket impulses. -->
    <inertial><origin xyz="{elbow_x_m:.7f} 0 {short_center_z:.7f}" rpy="0 0 0"/><mass value="{mass:.7f}"/><inertia ixx="0.00008" iyy="0.0012" izz="0.0012" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
</robot>
'''


def generate_allen_key_urdf_pool(
    output_dir: Path,
    lengths_m: tuple[float, ...],
    *,
    handle_across_flats_m: float,
    short_leg_length_m: float,
    elbow_x_m: float,
    mesh_path: Path,
) -> tuple[list[Path], list[tuple[float, float, float]]]:
    """Generate deterministic physical length variants for scene cloning."""
    if not lengths_m:
        raise ValueError("Allen-key length pool must not be empty")
    if len(set(lengths_m)) != len(lengths_m):
        raise ValueError("Allen-key length pool contains duplicates")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    scales = []
    for index, length in enumerate(lengths_m):
        path = output_dir / f"allen_key_{index:02d}_{length:.3f}m.urdf"
        path.write_text(allen_key_urdf_text(
            long_handle_length_m=float(length),
            handle_across_flats_m=float(handle_across_flats_m),
            short_leg_length_m=float(short_leg_length_m),
            elbow_x_m=float(elbow_x_m),
            mesh_path=mesh_path.resolve(),
        ))
        paths.append(path)
        scales.append((
            float(length) / 0.1,
            float(handle_across_flats_m) / 0.1,
            float(short_leg_length_m) / 0.1,
        ))
    return paths, scales


__all__ = [
    "allen_key_urdf_text",
    "finger_effort_soft_penalty",
    "generate_allen_key_urdf_pool",
    "turn_goal_pose",
    "turning_curriculum_ready",
    "update_unwrapped_angle",
    "wrap_to_pi",
    "yaw_from_quaternion",
]
