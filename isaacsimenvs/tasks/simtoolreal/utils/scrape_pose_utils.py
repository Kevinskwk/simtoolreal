"""Goal-pose sampling helpers for table-edge scraping poses."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import torch
import trimesh


TABLE_HALF_HEIGHT: float = 0.15


def quat_apply_wxyz(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by wxyz quaternions."""
    q_vec = quat[:, 1:4]
    q_w = quat[:, 0:1]
    t = 2.0 * torch.cross(q_vec, vec, dim=-1)
    return vec + q_w * t + torch.cross(q_vec, t, dim=-1)


def table_top_state(table_pos_w: torch.Tensor, table_quat_wxyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a table-top point and unit normal in world frame."""
    local_normal = torch.zeros_like(table_pos_w)
    local_normal[:, 2] = 1.0
    normal_w = quat_apply_wxyz(table_quat_wxyz, local_normal)
    normal_w = torch.nn.functional.normalize(normal_w, dim=-1)
    return table_pos_w + normal_w * TABLE_HALF_HEIGHT, normal_w


def tangent_basis_from_yaw(yaw: torch.Tensor, normal_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build two orthonormal table tangents; first one follows world-yaw as closely as possible."""
    candidate = torch.stack((torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)), dim=-1)
    edge = candidate - (candidate * normal_w).sum(dim=-1, keepdim=True) * normal_w
    edge_norm = edge.norm(dim=-1, keepdim=True)
    fallback = torch.tensor([1.0, 0.0, 0.0], device=normal_w.device, dtype=normal_w.dtype).expand_as(normal_w)
    fallback = fallback - (fallback * normal_w).sum(dim=-1, keepdim=True) * normal_w
    fallback_norm = fallback.norm(dim=-1, keepdim=True)
    fallback_alt = torch.tensor([0.0, 1.0, 0.0], device=normal_w.device, dtype=normal_w.dtype).expand_as(normal_w)
    fallback_alt = fallback_alt - (fallback_alt * normal_w).sum(dim=-1, keepdim=True) * normal_w
    fallback = torch.where(fallback_norm > 1.0e-6, fallback / fallback_norm.clamp_min(1.0e-6), fallback_alt)
    fallback = torch.nn.functional.normalize(fallback, dim=-1)
    edge = torch.where(edge_norm > 1.0e-6, edge / edge_norm.clamp_min(1.0e-6), fallback)
    forward = torch.cross(edge, normal_w, dim=-1)
    forward = torch.nn.functional.normalize(forward, dim=-1)
    return edge, forward


def quat_from_matrix_wxyz(rot: torch.Tensor) -> torch.Tensor:
    """Convert batched rotation matrices to wxyz quaternions."""
    m00, m01, m02 = rot[:, 0, 0], rot[:, 0, 1], rot[:, 0, 2]
    m10, m11, m12 = rot[:, 1, 0], rot[:, 1, 1], rot[:, 1, 2]
    m20, m21, m22 = rot[:, 2, 0], rot[:, 2, 1], rot[:, 2, 2]
    qw = 0.5 * torch.sqrt(torch.clamp(1.0 + m00 + m11 + m22, min=0.0))
    qx = 0.5 * torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=0.0))
    qy = 0.5 * torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=0.0))
    qz = 0.5 * torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=0.0))
    qx = torch.copysign(qx, m21 - m12)
    qy = torch.copysign(qy, m02 - m20)
    qz = torch.copysign(qz, m10 - m01)
    quat = torch.stack((qw, qx, qy, qz), dim=-1)
    return torch.nn.functional.normalize(quat, dim=-1)


def _rpy_matrix(rpy: tuple[float, float, float]) -> list[list[float]]:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _matvec(mat: list[list[float]], vec: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(sum(mat[i][j] * vec[j] for j in range(3)) for i in range(3))


def load_urdf_collision_bounds(path: str | Path) -> tuple[float, float, float, float, float, float]:
    """Return local collision AABB bounds from primitive or mesh URDF geometry."""
    path = Path(path)
    root = ET.parse(path).getroot()
    mins = [float("inf"), float("inf"), float("inf")]
    maxs = [float("-inf"), float("-inf"), float("-inf")]
    for collision in root.findall(".//collision"):
        origin = collision.find("origin")
        xyz = (0.0, 0.0, 0.0)
        rpy = (0.0, 0.0, 0.0)
        if origin is not None:
            if origin.get("xyz"):
                xyz = tuple(float(v) for v in origin.get("xyz", "0 0 0").split())
            if origin.get("rpy"):
                rpy = tuple(float(v) for v in origin.get("rpy", "0 0 0").split())
        rot = _rpy_matrix(rpy)
        geom = collision.find("geometry")
        if geom is None:
            continue
        box = geom.find("box")
        cyl = geom.find("cylinder")
        mesh = geom.find("mesh")
        if box is not None:
            sx, sy, sz = (float(v) for v in box.get("size", "0 0 0").split())
            local_corners = [
                (x, y, z)
                for x in (-0.5 * sx, 0.5 * sx)
                for y in (-0.5 * sy, 0.5 * sy)
                for z in (-0.5 * sz, 0.5 * sz)
            ]
            for corner in local_corners:
                rotated = _matvec(rot, corner)
                point = [xyz[i] + rotated[i] for i in range(3)]
                for i in range(3):
                    mins[i] = min(mins[i], point[i])
                    maxs[i] = max(maxs[i], point[i])
        elif cyl is not None:
            length = float(cyl.get("length", "0"))
            radius = float(cyl.get("radius", "0"))
            axis = [rot[0][2], rot[1][2], rot[2][2]]
            extents = [
                abs(axis[i]) * 0.5 * length + radius * math.sqrt(max(0.0, 1.0 - axis[i] * axis[i]))
                for i in range(3)
            ]
            for i in range(3):
                mins[i] = min(mins[i], xyz[i] - extents[i])
                maxs[i] = max(maxs[i], xyz[i] + extents[i])
        elif mesh is not None:
            filename = mesh.get("filename")
            if not filename:
                raise ValueError(f"collision mesh in {path} has no filename")
            if filename.startswith("package://"):
                raise ValueError(
                    f"package:// collision mesh paths are unsupported in {path}: {filename}"
                )
            mesh_path = (path.parent / filename).resolve()
            if not mesh_path.is_file():
                raise FileNotFoundError(f"collision mesh does not exist: {mesh_path}")
            loaded = trimesh.load(str(mesh_path), force="scene", process=False)
            if not isinstance(loaded, trimesh.Scene) or not loaded.geometry:
                raise ValueError(f"collision mesh has no geometry: {mesh_path}")
            mesh_bounds = loaded.bounds
            scale = tuple(float(v) for v in mesh.get("scale", "1 1 1").split())
            if len(scale) != 3 or any(not math.isfinite(v) or v <= 0.0 for v in scale):
                raise ValueError(f"collision mesh has invalid scale in {path}: {scale}")
            local_corners = [
                (x * scale[0], y * scale[1], z * scale[2])
                for x in (float(mesh_bounds[0][0]), float(mesh_bounds[1][0]))
                for y in (float(mesh_bounds[0][1]), float(mesh_bounds[1][1]))
                for z in (float(mesh_bounds[0][2]), float(mesh_bounds[1][2]))
            ]
            for corner in local_corners:
                rotated = _matvec(rot, corner)
                point = [xyz[i] + rotated[i] for i in range(3)]
                for i in range(3):
                    mins[i] = min(mins[i], point[i])
                    maxs[i] = max(maxs[i], point[i])
    if not all(math.isfinite(v) for v in mins + maxs):
        raise ValueError(f"No supported collision geometry found in {path}")
    return (mins[0], mins[1], mins[2], maxs[0], maxs[1], maxs[2])


def sample_edge_contact_goal_pose(
    *,
    table_pos_w: torch.Tensor,
    table_quat_wxyz: torch.Tensor,
    x_tip: torch.Tensor,
    y_center: torch.Tensor,
    z_contact: torch.Tensor,
    xy_half_range: tuple[float, float],
    edge_yaw_range_rad: float,
    tilt_range_rad: tuple[float, float],
    device: torch.device,
    edge_yaw: torch.Tensor | None = None,
    edge_tilt: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample tool poses whose lower leading tip edge is anchored on the tabletop.

    The sampled contact edge is the local line at x=tip, z=-half_thickness.
    The target can translate in table x/y and rotate around that edge by a
    positive tilt angle. Positive tilt lifts the handle side above the table,
    so the conservative tool AABB stays non-penetrating.
    """
    n_envs = table_pos_w.shape[0]
    table_top_w, normal_w = table_top_state(table_pos_w, table_quat_wxyz)

    if edge_yaw is None:
        edge_yaw = torch.empty(n_envs, device=device).uniform_(
            -float(edge_yaw_range_rad), float(edge_yaw_range_rad)
        )
    edge_dir_w, forward_flat_w = tangent_basis_from_yaw(edge_yaw, normal_w)

    if edge_tilt is None:
        tilt_lo, tilt_hi = tilt_range_rad
        tilt = torch.empty(n_envs, device=device).uniform_(float(tilt_lo), float(tilt_hi))
    else:
        if edge_tilt.shape != (n_envs,) or not bool(torch.isfinite(edge_tilt).all()):
            raise ValueError(f"edge_tilt must be a finite ({n_envs},) tensor")
        tilt = edge_tilt
    cos_t = torch.cos(tilt).unsqueeze(-1)
    sin_t = torch.sin(tilt).unsqueeze(-1)
    local_x_w = cos_t * forward_flat_w - sin_t * normal_w
    local_z_w = sin_t * forward_flat_w + cos_t * normal_w
    local_y_w = edge_dir_w
    rot = torch.stack((local_x_w, local_y_w, local_z_w), dim=-1)
    goal_quat_wxyz = quat_from_matrix_wxyz(rot)

    dx_range, dy_range = xy_half_range
    dx = torch.empty(n_envs, device=device).uniform_(-float(dx_range), float(dx_range))
    dy = torch.empty(n_envs, device=device).uniform_(-float(dy_range), float(dy_range))
    table_x_w = forward_flat_w
    table_y_w = edge_dir_w
    edge_center_w = table_top_w + table_x_w * dx.unsqueeze(-1) + table_y_w * dy.unsqueeze(-1)

    contact_edge_local = torch.stack((x_tip, y_center, z_contact), dim=-1)
    goal_pos_w = edge_center_w - quat_apply_wxyz(goal_quat_wxyz, contact_edge_local)
    return goal_pos_w, goal_quat_wxyz, edge_center_w, edge_yaw


def edge_tilt_from_pose(
    object_quat_wxyz: torch.Tensor,
    table_normal_w: torch.Tensor,
    edge_yaw: torch.Tensor,
) -> torch.Tensor:
    """Recover signed edge tilt from an edge-contact orientation."""
    if object_quat_wxyz.shape != table_normal_w.shape[:1] + (4,):
        raise ValueError("object quaternion must have shape (N, 4)")
    if table_normal_w.shape[-1] != 3 or edge_yaw.shape != table_normal_w.shape[:-1]:
        raise ValueError("table normal or edge yaw shape is invalid")
    _, forward_w = tangent_basis_from_yaw(edge_yaw, table_normal_w)
    local_z = torch.zeros_like(table_normal_w)
    local_z[:, 2] = 1.0
    tool_z_w = quat_apply_wxyz(object_quat_wxyz, local_z)
    sin_tilt = (tool_z_w * forward_w).sum(dim=-1)
    cos_tilt = (tool_z_w * table_normal_w).sum(dim=-1)
    return torch.atan2(sin_tilt, cos_tilt)


def edge_contact_points_w(
    object_pos_w: torch.Tensor,
    object_quat_wxyz: torch.Tensor,
    x_tip: torch.Tensor,
    y_min: torch.Tensor,
    y_max: torch.Tensor,
    z_contact: torch.Tensor,
) -> torch.Tensor:
    """Return world endpoints and midpoint of the persistent local contact edge."""
    p0 = torch.stack((x_tip, y_min, z_contact), dim=-1)
    p1 = torch.stack((x_tip, y_max, z_contact), dim=-1)
    pm = torch.stack((x_tip, 0.5 * (y_min + y_max), z_contact), dim=-1)
    local_points = torch.stack((p0, pm, p1), dim=1)
    n_envs, n_pts, _ = local_points.shape
    points_w = object_pos_w.unsqueeze(1) + quat_apply_wxyz(
        object_quat_wxyz.unsqueeze(1).expand(-1, n_pts, -1).reshape(-1, 4),
        local_points.reshape(-1, 3),
    ).reshape(n_envs, n_pts, 3)
    return points_w


def contact_force_reward(
    normal_force: torch.Tensor,
    target_force: float | torch.Tensor,
    force_sigma: float,
    max_force: float,
    *,
    huber_delta_n: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reward normal contact force near scalar or per-env targets."""
    target = torch.as_tensor(
        target_force, device=normal_force.device, dtype=normal_force.dtype
    )
    force_error = torch.abs(normal_force - target)
    robust_error = force_error
    if huber_delta_n is not None:
        delta = float(huber_delta_n)
        if delta <= 0.0:
            raise ValueError(f"huber_delta_n must be positive, got {delta}.")
        robust_error = torch.where(
            force_error <= delta,
            0.5 * force_error.square() / delta,
            force_error - 0.5 * delta,
        )
    reward = torch.exp(-robust_error / max(float(force_sigma), 1.0e-6))
    over_force = torch.clamp(normal_force - float(max_force), min=0.0)
    return reward, over_force


def conditional_success_rate(
    success: torch.Tensor,
    eligible: torch.Tensor,
    *,
    min_eligible_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return success rate over eligible samples and whether support is sufficient."""
    if success.shape != eligible.shape:
        raise ValueError(
            "success and eligible must have matching shapes, got "
            f"{tuple(success.shape)} and {tuple(eligible.shape)}."
        )
    if success.dtype != torch.bool or eligible.dtype != torch.bool:
        raise ValueError("success and eligible must be boolean tensors.")
    if min_eligible_count <= 0:
        raise ValueError("min_eligible_count must be positive.")
    if bool((success & ~eligible).any()):
        raise ValueError("success contains samples that are not eligible.")

    eligible_count = eligible.sum()
    success_count = success.sum()
    success_rate = success_count.to(torch.float32) / eligible_count.clamp_min(1)
    has_minimum_support = eligible_count >= int(min_eligible_count)
    return success_rate, eligible_count, has_minimum_support


def contact_force_onset_gate(
    normal_force: torch.Tensor,
    previous_contact_age: torch.Tensor,
    *,
    contact_threshold_n: float,
    grace_steps: int,
    ramp_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Track consecutive contact and return age, reward ramp, and contact mask."""
    if normal_force.shape != previous_contact_age.shape:
        raise ValueError(
            "normal_force and previous_contact_age must have matching shapes, got "
            f"{tuple(normal_force.shape)} and {tuple(previous_contact_age.shape)}."
        )
    if previous_contact_age.dtype not in (torch.int32, torch.int64):
        raise ValueError("previous_contact_age must use an integer dtype.")
    if contact_threshold_n <= 0.0:
        raise ValueError("contact_threshold_n must be positive.")
    if grace_steps < 0:
        raise ValueError("grace_steps must be non-negative.")
    if ramp_steps <= 0:
        raise ValueError("ramp_steps must be positive.")
    if not torch.isfinite(normal_force).all():
        raise ValueError("normal_force contains NaN or Inf.")

    in_contact = normal_force >= float(contact_threshold_n)
    contact_age = torch.where(
        in_contact,
        previous_contact_age + 1,
        torch.zeros_like(previous_contact_age),
    )
    reward_ramp = torch.clamp(
        (contact_age.to(normal_force.dtype) - float(grace_steps))
        / float(ramp_steps),
        min=0.0,
        max=1.0,
    )
    return contact_age, reward_ramp, in_contact


def edge_contact_reward(
    edge_points_w: torch.Tensor,
    table_pos_w: torch.Tensor,
    table_quat_wxyz: torch.Tensor,
    distance_sigma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Geometry-only reward for keeping the selected edge on the tabletop."""
    table_top_w, normal_w = table_top_state(table_pos_w, table_quat_wxyz)
    signed_dist = ((edge_points_w - table_top_w.unsqueeze(1)) * normal_w.unsqueeze(1)).sum(dim=-1)
    edge_error = signed_dist.abs().amax(dim=-1)
    reward = torch.exp(-edge_error / max(float(distance_sigma), 1.0e-6))
    return reward, edge_error


__all__ = [
    "TABLE_HALF_HEIGHT",
    "quat_apply_wxyz",
    "table_top_state",
    "tangent_basis_from_yaw",
    "quat_from_matrix_wxyz",
    "load_urdf_collision_bounds",
    "sample_edge_contact_goal_pose",
    "edge_contact_points_w",
    "edge_tilt_from_pose",
    "contact_force_reward",
    "conditional_success_rate",
    "contact_force_onset_gate",
    "edge_contact_reward",
]
