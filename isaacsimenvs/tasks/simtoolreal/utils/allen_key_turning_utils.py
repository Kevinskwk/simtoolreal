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


def stick_slip_torsional_friction(
    angular_displacement: torch.Tensor,
    angular_velocity: torch.Tensor,
    was_stuck: torch.Tensor,
    *,
    kinetic_limit_nm: float | torch.Tensor,
    static_to_kinetic_ratio: float,
    stiction_stiffness_nm_per_rad: float,
    damping_nm_per_radps: float | torch.Tensor,
    maximum_abs_torque_nm: float | torch.Tensor,
    kinetic_transition_speed_radps: float,
    restick_speed_radps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return motion-reactive torsional friction and stick/slip state.

    While stuck, a clamped spring-damper reaction holds the current screw
    angle up to the static-friction limit. Once that demand exceeds the limit,
    Coulomb-like kinetic friction opposes motion. A slipping fixture re-sticks
    only near zero speed, at which point the caller should move its angular
    anchor to the current angle.
    """
    if angular_displacement.shape != angular_velocity.shape:
        raise ValueError("friction displacement and velocity shapes must match")
    if was_stuck.shape != angular_velocity.shape or was_stuck.dtype != torch.bool:
        raise ValueError("friction stick state must be a matching boolean tensor")
    if not bool(torch.isfinite(angular_displacement).all()) or not bool(
        torch.isfinite(angular_velocity).all()
    ):
        raise ValueError("friction state contains NaN or Inf")
    scalar_parameters = {
        "static_to_kinetic_ratio": static_to_kinetic_ratio,
        "stiction_stiffness_nm_per_rad": stiction_stiffness_nm_per_rad,
        "kinetic_transition_speed_radps": kinetic_transition_speed_radps,
        "restick_speed_radps": restick_speed_radps,
    }
    if any(not math.isfinite(float(value)) for value in scalar_parameters.values()):
        raise ValueError("friction parameters must be finite")

    def coefficient_tensor(value: float | torch.Tensor, name: str) -> torch.Tensor:
        coefficient = torch.as_tensor(
            value, dtype=angular_velocity.dtype, device=angular_velocity.device
        )
        if coefficient.ndim == 0:
            coefficient = coefficient.expand_as(angular_velocity)
        elif coefficient.shape != angular_velocity.shape:
            raise ValueError(
                f"{name} must be scalar or match friction state shape "
                f"{tuple(angular_velocity.shape)}, got {tuple(coefficient.shape)}"
            )
        if not bool(torch.isfinite(coefficient).all()):
            raise ValueError(f"{name} must be finite")
        return coefficient

    kinetic_limit = coefficient_tensor(kinetic_limit_nm, "kinetic_limit_nm")
    damping = coefficient_tensor(damping_nm_per_radps, "damping_nm_per_radps")
    maximum_abs_torque = coefficient_tensor(
        maximum_abs_torque_nm, "maximum_abs_torque_nm"
    )
    if bool((kinetic_limit < 0.0).any()):
        raise ValueError("kinetic friction limit must be non-negative")
    if static_to_kinetic_ratio < 1.0:
        raise ValueError("static friction must be at least kinetic friction")
    if stiction_stiffness_nm_per_rad <= 0.0 or bool((damping < 0.0).any()):
        raise ValueError("friction stiffness must be positive and damping non-negative")
    if bool((maximum_abs_torque <= 0.0).any()):
        raise ValueError("maximum absolute friction torque must be positive")
    if bool((maximum_abs_torque < static_to_kinetic_ratio * kinetic_limit).any()):
        raise ValueError("maximum friction torque must cover the static-friction limit")
    if kinetic_transition_speed_radps <= 0.0 or restick_speed_radps <= 0.0:
        raise ValueError("friction transition speeds must be positive")

    restuck = (~was_stuck) & (angular_velocity.abs() <= restick_speed_radps)
    effective_displacement = torch.where(
        restuck, torch.zeros_like(angular_displacement), angular_displacement
    )
    static_demand = (
        -stiction_stiffness_nm_per_rad * effective_displacement
        - damping * angular_velocity
    )
    static_limit_nm = static_to_kinetic_ratio * kinetic_limit
    remains_stuck = was_stuck & (static_demand.abs() <= static_limit_nm)
    stuck = restuck | remains_stuck
    static_torque = torch.maximum(
        torch.minimum(static_demand, static_limit_nm), -static_limit_nm
    )

    moving_direction = torch.tanh(
        angular_velocity / kinetic_transition_speed_radps
    )
    displacement_direction = torch.sign(effective_displacement)
    kinetic_direction = torch.where(
        angular_velocity.abs() > restick_speed_radps,
        moving_direction,
        displacement_direction,
    )
    kinetic_torque = (
        -kinetic_limit * kinetic_direction
        - damping * angular_velocity
    )
    unclamped_torque = torch.where(stuck, static_torque, kinetic_torque)
    torque = torch.maximum(
        torch.minimum(unclamped_torque, maximum_abs_torque), -maximum_abs_torque
    )
    clipped = unclamped_torque.abs() > maximum_abs_torque
    if not bool(torch.isfinite(torque).all()):
        raise RuntimeError("computed torsional friction is not finite")
    return torque, stuck, restuck, clipped


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


def loaded_grasp_quality(
    palm_support_quality: torch.Tensor,
    palm_supported: torch.Tensor,
    finger_contact_count: torch.Tensor,
    relative_linear_speed_mps: torch.Tensor,
    relative_angular_speed_radps: torch.Tensor,
    *,
    minimum_contact_fingers: int,
    maximum_relative_linear_speed_mps: float,
    maximum_relative_angular_speed_radps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score and classify a palm-supported, low-slip loaded grasp."""
    shape = palm_support_quality.shape
    tensors = (
        finger_contact_count,
        relative_linear_speed_mps,
        relative_angular_speed_radps,
    )
    if palm_supported.shape != shape or palm_supported.dtype != torch.bool or any(
        value.shape != shape for value in tensors
    ):
        raise ValueError("loaded-grasp inputs must have matching shapes")
    if minimum_contact_fingers <= 0:
        raise ValueError("loaded grasp requires at least one finger")
    if (
        maximum_relative_linear_speed_mps <= 0.0
        or maximum_relative_angular_speed_radps <= 0.0
    ):
        raise ValueError("loaded-grasp slip limits must be positive")
    if any(
        not bool(torch.isfinite(value).all())
        for value in (palm_support_quality, *tensors)
    ):
        raise ValueError("loaded-grasp inputs contain NaN or Inf")
    if bool(((palm_support_quality < 0.0) | (palm_support_quality > 1.0)).any()):
        raise ValueError("palm support quality must lie in [0, 1]")

    finger_quality = (
        finger_contact_count.float() / float(minimum_contact_fingers)
    ).clamp(0.0, 1.0)
    linear_quality = (
        1.0 - relative_linear_speed_mps / maximum_relative_linear_speed_mps
    ).clamp(0.0, 1.0)
    angular_quality = (
        1.0 - relative_angular_speed_radps / maximum_relative_angular_speed_radps
    ).clamp(0.0, 1.0)
    quality = (
        palm_support_quality
        * finger_quality
        * torch.sqrt(linear_quality * angular_quality)
    )
    valid = (
        palm_supported
        & (finger_contact_count >= minimum_contact_fingers)
        & (relative_linear_speed_mps <= maximum_relative_linear_speed_mps)
        & (relative_angular_speed_radps <= maximum_relative_angular_speed_radps)
    )
    return quality, valid


def deep_grasp_quality(
    finger_contact: torch.Tensor,
    proximal_finger_contact: torch.Tensor,
    palm_contact: torch.Tensor,
    finger_force_w: torch.Tensor,
    relative_linear_speed_mps: torch.Tensor,
    relative_angular_speed_radps: torch.Tensor,
    *,
    minimum_contact_fingers: int,
    maximum_opposition_cosine: float,
    maximum_relative_linear_speed_mps: float,
    maximum_relative_angular_speed_radps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score a deep, opposed whole-hand grasp rather than unilateral pushing.

    Finger index zero is the thumb. A valid grasp requires the thumb, at least
    one opposing finger, physical palm or proximal-link support, opposed thumb
    and finger normal forces, and low palm-tool slip.
    """
    if finger_contact.ndim != 2 or finger_contact.shape[1] != 5:
        raise ValueError("deep-grasp finger contact must have shape (N, 5)")
    if (
        proximal_finger_contact.shape != finger_contact.shape
        or proximal_finger_contact.dtype != torch.bool
        or finger_contact.dtype != torch.bool
    ):
        raise ValueError("deep-grasp contact masks must be matching boolean tensors")
    if finger_force_w.shape != (*finger_contact.shape, 3):
        raise ValueError("deep-grasp finger force must have shape (N, 5, 3)")
    shape = finger_contact.shape[:1]
    if (
        palm_contact.shape != shape
        or palm_contact.dtype != torch.bool
        or relative_linear_speed_mps.shape != shape
        or relative_angular_speed_radps.shape != shape
    ):
        raise ValueError("deep-grasp per-environment inputs have invalid shapes")
    if not 3 <= minimum_contact_fingers <= 5:
        raise ValueError("deep grasp requires between three and five fingers")
    if not -1.0 <= maximum_opposition_cosine < 1.0:
        raise ValueError("deep-grasp opposition cosine must lie in [-1, 1)")
    if maximum_relative_linear_speed_mps <= 0.0 or maximum_relative_angular_speed_radps <= 0.0:
        raise ValueError("deep-grasp slip limits must be positive")
    if not bool(torch.isfinite(finger_force_w).all()) or not bool(
        torch.isfinite(relative_linear_speed_mps).all()
    ) or not bool(torch.isfinite(relative_angular_speed_radps).all()):
        raise ValueError("deep-grasp inputs contain NaN or Inf")

    contact_count = finger_contact.sum(-1)
    thumb_contact = finger_contact[:, 0]
    opposing_finger_contact = finger_contact[:, 1:].any(-1)
    inner_hand_support = palm_contact | proximal_finger_contact.any(-1)

    thumb_force = finger_force_w[:, 0]
    opposing_force = finger_force_w[:, 1:].sum(1)
    thumb_norm = torch.linalg.vector_norm(thumb_force, dim=-1)
    opposing_norm = torch.linalg.vector_norm(opposing_force, dim=-1)
    force_pair_valid = thumb_contact & opposing_finger_contact & (
        thumb_norm > 1.0e-6
    ) & (opposing_norm > 1.0e-6)
    cosine = (thumb_force * opposing_force).sum(-1) / (
        thumb_norm * opposing_norm
    ).clamp_min(1.0e-12)
    cosine = torch.where(
        force_pair_valid, cosine.clamp(-1.0, 1.0), torch.ones_like(cosine)
    )
    opposition_quality = ((1.0 - cosine) * 0.5).clamp(0.0, 1.0)

    contact_quality = (
        contact_count.float() / float(minimum_contact_fingers)
    ).clamp(0.0, 1.0)
    linear_quality = (
        1.0 - relative_linear_speed_mps / maximum_relative_linear_speed_mps
    ).clamp(0.0, 1.0)
    angular_quality = (
        1.0 - relative_angular_speed_radps / maximum_relative_angular_speed_radps
    ).clamp(0.0, 1.0)
    topology_quality = (
        thumb_contact & opposing_finger_contact & inner_hand_support
    ).float()
    quality = (
        contact_quality
        * topology_quality
        * opposition_quality
        * torch.sqrt(linear_quality * angular_quality)
    )
    valid = (
        (contact_count >= minimum_contact_fingers)
        & thumb_contact
        & opposing_finger_contact
        & inner_hand_support
        & force_pair_valid
        & (cosine <= maximum_opposition_cosine)
        & (relative_linear_speed_mps <= maximum_relative_linear_speed_mps)
        & (relative_angular_speed_radps <= maximum_relative_angular_speed_radps)
    )
    return quality, valid, cosine


def gate_positive_progress(
    progress: torch.Tensor,
    grasp_quality: torch.Tensor,
) -> torch.Tensor:
    """Gate positive task progress while preserving penalties for regression."""
    if progress.shape != grasp_quality.shape:
        raise ValueError("progress and grasp quality must have matching shapes")
    if not bool(torch.isfinite(progress).all()) or not bool(
        torch.isfinite(grasp_quality).all()
    ):
        raise ValueError("progress gating inputs contain NaN or Inf")
    if bool(((grasp_quality < 0.0) | (grasp_quality > 1.0)).any()):
        raise ValueError("grasp quality must lie in [0, 1]")
    return torch.minimum(progress, torch.zeros_like(progress)) + (
        torch.relu(progress) * grasp_quality
    )


def update_consecutive_grasp_hold(
    valid_grasp: torch.Tensor,
    previous_count: torch.Tensor,
    already_confirmed: torch.Tensor,
    *,
    required_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update a consecutive grasp hold and report first confirmation."""
    if (
        valid_grasp.dtype != torch.bool
        or already_confirmed.dtype != torch.bool
        or previous_count.dtype != torch.long
        or valid_grasp.shape != previous_count.shape
        or valid_grasp.shape != already_confirmed.shape
    ):
        raise ValueError("grasp-hold inputs must be matching bool/long tensors")
    if required_steps <= 0:
        raise ValueError("required grasp-hold steps must be positive")
    count = torch.where(
        valid_grasp,
        previous_count + 1,
        torch.zeros_like(previous_count),
    )
    just_confirmed = (
        ~already_confirmed
        & valid_grasp
        & (count >= required_steps)
    )
    return count, just_confirmed


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
    # unit_hex_prism_x.obj has unit circumradius in its Y/Z cross-section, so
    # its unscaled across-flats distance is sqrt(3).
    prism_scale = handle_across_flats_m / math.sqrt(3.0)
    short_prism_scale = min(prism_scale, 0.012 / math.sqrt(3.0))
    mass = 0.18 * long_handle_length_m / 0.264
    name_length_mm = round(long_handle_length_m * 1000.0)
    return f'''<?xml version="1.0"?>
<robot name="allen_key_turning_{name_length_mm}mm">
  <link name="object_root">
    <visual><origin xyz="{long_center_x:.7f} 0 0" rpy="0 0 0"/><geometry><mesh filename="{mesh_path}" scale="{long_handle_length_m:.7f} {prism_scale:.7f} {prism_scale:.7f}"/></geometry><material name="grip"><color rgba="0.16 0.20 0.23 1"/></material></visual>
    <collision><origin xyz="{long_center_x:.7f} 0 0" rpy="0 0 0"/><geometry><mesh filename="{mesh_path}" scale="{long_handle_length_m:.7f} {prism_scale:.7f} {prism_scale:.7f}"/></geometry></collision>
    <visual><origin xyz="{elbow_x_m:.7f} 0 {short_center_z:.7f}" rpy="0 1.57079632679 0"/><geometry><mesh filename="{mesh_path}" scale="{short_leg_length_m:.7f} {short_prism_scale:.7f} {short_prism_scale:.7f}"/></geometry><material name="steel"><color rgba="0.32 0.36 0.39 1"/></material></visual>
    <collision><origin xyz="{elbow_x_m:.7f} 0 {short_center_z:.7f}" rpy="0 1.57079632679 0"/><geometry><mesh filename="{mesh_path}" scale="{short_leg_length_m:.7f} {short_prism_scale:.7f} {short_prism_scale:.7f}"/></geometry></collision>
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
    "gate_positive_progress",
    "generate_allen_key_urdf_pool",
    "loaded_grasp_quality",
    "stick_slip_torsional_friction",
    "turn_goal_pose",
    "turning_curriculum_ready",
    "update_consecutive_grasp_hold",
    "update_unwrapped_angle",
    "wrap_to_pi",
    "yaw_from_quaternion",
]
