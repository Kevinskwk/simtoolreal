"""Numerical helpers for tool-table contact-force controllability checks."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def interval_normal_force(
    force_history_w: torch.Tensor, table_normal_w: torch.Tensor
) -> torch.Tensor:
    """Average filtered force vectors over sensor history and project on table normal."""
    if force_history_w.ndim < 4 or force_history_w.shape[-1] != 3:
        raise ValueError(
            "force_history_w must have shape (env, history, ..., 3), got "
            f"{tuple(force_history_w.shape)}"
        )
    if table_normal_w.shape != (force_history_w.shape[0], 3):
        raise ValueError(
            f"table_normal_w must have shape ({force_history_w.shape[0]}, 3), "
            f"got {tuple(table_normal_w.shape)}"
        )
    if not torch.isfinite(force_history_w).all() or not torch.isfinite(table_normal_w).all():
        raise ValueError("contact force history and table normal must be finite")

    force_per_sample = force_history_w.sum(
        dim=tuple(range(2, force_history_w.ndim - 1))
    )
    return (force_per_sample.mean(dim=1) * table_normal_w).sum(dim=-1)


def expected_normal_reaction(
    mass: torch.Tensor,
    gravity_w: torch.Tensor,
    table_normal_w: torch.Tensor,
    applied_normal_load: torch.Tensor,
) -> torch.Tensor:
    """Return the static table reaction for a load applied into the table."""
    if mass.ndim != 1 or applied_normal_load.shape != mass.shape:
        raise ValueError("mass and applied_normal_load must be one-dimensional and aligned")
    if gravity_w.shape != (3,) or table_normal_w.shape != (mass.shape[0], 3):
        raise ValueError("gravity_w/table_normal_w have incompatible shapes")
    gravity_force_w = mass[:, None] * gravity_w[None, :]
    return applied_normal_load - (gravity_force_w * table_normal_w).sum(dim=-1)


def damped_least_squares(
    jacobian: torch.Tensor, twist: torch.Tensor, damping: float
) -> torch.Tensor:
    """Solve qdot = J^T (J J^T + lambda^2 I)^-1 twist in batch."""
    if jacobian.ndim != 3 or twist.shape != jacobian.shape[:2]:
        raise ValueError(
            f"expected J=(N, task, dof), twist=(N, task); got {jacobian.shape}, {twist.shape}"
        )
    if damping <= 0.0:
        raise ValueError("damping must be positive")
    jj_t = jacobian @ jacobian.transpose(-1, -2)
    eye = torch.eye(jacobian.shape[1], device=jacobian.device, dtype=jacobian.dtype)
    solved = torch.linalg.solve(jj_t + damping * damping * eye, twist.unsqueeze(-1))
    return (jacobian.transpose(-1, -2) @ solved).squeeze(-1)


@dataclass(frozen=True)
class PiStep:
    velocity: torch.Tensor
    integral: torch.Tensor


def pi_force_step(
    error: torch.Tensor,
    integral: torch.Tensor,
    *,
    dt: float,
    kp: float,
    ki: float,
    velocity_limit: float,
    integral_limit: float,
) -> PiStep:
    """PI force controller with symmetric clamps and anti-windup."""
    if dt <= 0.0 or velocity_limit <= 0.0 or integral_limit <= 0.0:
        raise ValueError("dt and PI limits must be positive")
    candidate = torch.clamp(integral + error * dt, -integral_limit, integral_limit)
    unclamped = kp * error + ki * candidate
    saturated_further = (unclamped.abs() > velocity_limit) & (
        torch.sign(unclamped) == torch.sign(error)
    )
    accepted = torch.where(saturated_further, integral, candidate)
    velocity = torch.clamp(kp * error + ki * accepted, -velocity_limit, velocity_limit)
    return PiStep(velocity=velocity, integral=accepted)


def quaternion_error_vector(
    current_wxyz: torch.Tensor, target_wxyz: torch.Tensor
) -> torch.Tensor:
    """Small-angle-compatible orientation error from current to target in world axes."""
    if current_wxyz.shape != target_wxyz.shape or current_wxyz.shape[-1] != 4:
        raise ValueError("quaternions must have matching (..., 4) shapes")
    cw, cx, cy, cz = current_wxyz.unbind(-1)
    tw, tx, ty, tz = target_wxyz.unbind(-1)
    w = tw * cw + tx * cx + ty * cy + tz * cz
    x = -tw * cx + tx * cw - ty * cz + tz * cy
    y = -tw * cy + tx * cz + ty * cw - tz * cx
    z = -tw * cz - tx * cy + ty * cx + tz * cw
    vec = torch.stack((x, y, z), dim=-1)
    return 2.0 * torch.where(w.unsqueeze(-1) < 0.0, -vec, vec)


def coefficient_of_determination(expected: torch.Tensor, measured: torch.Tensor) -> float:
    """Compute R^2 with explicit rejection of a constant expected vector."""
    if expected.shape != measured.shape or expected.numel() < 2:
        raise ValueError("expected and measured must be aligned and contain at least two values")
    total = ((expected - expected.mean()) ** 2).sum()
    if float(total) <= 0.0:
        raise ValueError("R^2 is undefined for constant expected values")
    return float(1.0 - ((measured - expected) ** 2).sum() / total)


__all__ = [
    "PiStep",
    "coefficient_of_determination",
    "damped_least_squares",
    "expected_normal_reaction",
    "interval_normal_force",
    "pi_force_step",
    "quaternion_error_vector",
]
