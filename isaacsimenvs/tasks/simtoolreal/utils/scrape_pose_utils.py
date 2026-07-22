"""Goal-pose sampling helpers for table-edge scraping poses."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import torch


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
    """Return local collision AABB bounds from generated box/cylinder URDFs."""
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

    tilt_lo, tilt_hi = tilt_range_rad
    tilt = torch.empty(n_envs, device=device).uniform_(float(tilt_lo), float(tilt_hi))
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reward normal contact force near scalar or per-env targets."""
    target = torch.as_tensor(
        target_force, device=normal_force.device, dtype=normal_force.dtype
    )
    force_error = torch.abs(normal_force - target)
    reward = torch.exp(-force_error / max(float(force_sigma), 1.0e-6))
    over_force = torch.clamp(normal_force - float(max_force), min=0.0)
    return reward, over_force


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
    "contact_force_reward",
    "edge_contact_reward",
]
