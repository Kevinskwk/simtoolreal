"""Validation and tensor scoring for trajectory-conditioned grasp adjustment."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch


SCENARIO_KINDS = ("nominal", "fixed_pose_hard")


def _quat_mul(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = first.unbind(-1)
    w2, x2, y2, z2 = second.unbind(-1)
    return torch.stack((
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ), dim=-1)


def _quat_inv(value: torch.Tensor) -> torch.Tensor:
    result = value.clone()
    result[..., 1:] *= -1.0
    return result / value.square().sum(-1, keepdim=True).clamp_min(1.0e-12)


def _quat_apply(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    pure = torch.cat((torch.zeros_like(vector[..., :1]), vector), dim=-1)
    return _quat_mul(_quat_mul(quat, pure), _quat_inv(quat))[..., 1:]


def _quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    half = 0.5 * angle
    return torch.cat((torch.cos(half).unsqueeze(-1), axis * torch.sin(half).unsqueeze(-1)), -1)


def orbit_palm_tool_about_screw_axis(
    palm_to_tool_pos: torch.Tensor,
    palm_to_tool_quat: torch.Tensor,
    angle_rad: torch.Tensor,
    screw_axis_tool: torch.Tensor,
    screw_pivot_tool: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Orbit the palm around a tool-frame screw axis and return target T_palm_tool."""
    if palm_to_tool_pos.ndim != 2 or palm_to_tool_pos.shape[-1] != 3:
        raise ValueError("palm_to_tool_pos must have shape (N, 3)")
    count = palm_to_tool_pos.shape[0]
    if palm_to_tool_quat.shape != (count, 4) or angle_rad.shape != (count,):
        raise ValueError("Allen-key orbit inputs have incompatible shapes")
    for name, value in (
        ("screw_axis_tool", screw_axis_tool),
        ("screw_pivot_tool", screw_pivot_tool),
    ):
        if value.shape not in ((3,), (count, 3)):
            raise ValueError(f"{name} must have shape (3,) or (N, 3)")
    axis = screw_axis_tool.expand(count, -1) if screw_axis_tool.ndim == 1 else screw_axis_tool
    pivot = screw_pivot_tool.expand(count, -1) if screw_pivot_tool.ndim == 1 else screw_pivot_tool
    axis = torch.nn.functional.normalize(axis, dim=-1)
    tool_to_palm_quat = _quat_inv(palm_to_tool_quat)
    tool_to_palm_pos = _quat_apply(tool_to_palm_quat, -palm_to_tool_pos)
    orbit_quat = _quat_from_angle_axis(angle_rad, axis)
    target_tool_to_palm_pos = pivot + _quat_apply(orbit_quat, tool_to_palm_pos - pivot)
    target_tool_to_palm_quat = _quat_mul(orbit_quat, tool_to_palm_quat)
    target_palm_to_tool_quat = _quat_inv(target_tool_to_palm_quat)
    target_palm_to_tool_pos = _quat_apply(
        target_palm_to_tool_quat, -target_tool_to_palm_pos
    )
    return target_palm_to_tool_pos, target_palm_to_tool_quat


def screw_axis_orbit_errors(
    current_palm_to_tool_pos: torch.Tensor,
    current_palm_to_tool_quat: torch.Tensor,
    target_palm_to_tool_pos: torch.Tensor,
    target_palm_to_tool_quat: torch.Tensor,
    screw_axis_tool: torch.Tensor,
    screw_pivot_tool: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return signed orbital, palm-position, and palm-orientation errors."""
    count = current_palm_to_tool_pos.shape[0]
    axis = screw_axis_tool.expand(count, -1) if screw_axis_tool.ndim == 1 else screw_axis_tool
    pivot = screw_pivot_tool.expand(count, -1) if screw_pivot_tool.ndim == 1 else screw_pivot_tool
    axis = torch.nn.functional.normalize(axis, dim=-1)

    def tool_to_palm(pos, quat):
        inverse = _quat_inv(quat)
        return _quat_apply(inverse, -pos), inverse

    current_pos, current_quat = tool_to_palm(
        current_palm_to_tool_pos, current_palm_to_tool_quat
    )
    target_pos, target_quat = tool_to_palm(
        target_palm_to_tool_pos, target_palm_to_tool_quat
    )
    current_radial = current_pos - pivot
    target_radial = target_pos - pivot
    current_radial -= (current_radial * axis).sum(-1, keepdim=True) * axis
    target_radial -= (target_radial * axis).sum(-1, keepdim=True) * axis
    current_unit = torch.nn.functional.normalize(current_radial, dim=-1)
    target_unit = torch.nn.functional.normalize(target_radial, dim=-1)
    sine = (axis * torch.linalg.cross(current_unit, target_unit, dim=-1)).sum(-1)
    cosine = (current_unit * target_unit).sum(-1).clamp(-1.0, 1.0)
    orbit_error = torch.atan2(sine, cosine)
    position_error = torch.linalg.vector_norm(current_pos - target_pos, dim=-1)
    alignment = torch.abs((current_quat * target_quat).sum(-1)).clamp(0.0, 1.0)
    orientation_error = 2.0 * torch.acos(alignment)
    return orbit_error, position_error, orientation_error


def load_adjustment_scenarios(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"adjustment scenario file does not exist: {path}")
    payload = json.loads(path.read_text())
    schema = int(payload.get("schema_version", -1))
    if schema not in (1, 2) or payload.get("kind") != "simtoolreal_adjustment_scenarios":
        raise ValueError(f"unsupported adjustment scenario schema: {path}")
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("adjustment scenario file contains no scenarios")
    for index, scenario in enumerate(scenarios):
        if scenario.get("scenario_kind") not in SCENARIO_KINDS:
            raise ValueError(f"scenario {index} has an unknown kind")
        for name, size in (
            ("joint_pos_canonical", 29), ("joint_targets_canonical", 29),
            ("object_pos_local", 3), ("object_quat_wxyz", 4),
            ("future_delta", 6), ("future_twist", 6),
        ):
            values = scenario.get(name)
            if not isinstance(values, list) or len(values) != size:
                raise ValueError(f"scenario {index} {name} must have length {size}")
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"scenario {index} {name} contains non-finite values")
        source = int(scenario.get("source_entry_index", -1))
        if source < 0:
            raise ValueError(f"scenario {index} has an invalid source entry")
        if schema == 2 and int(scenario.get("asset_index", -1)) < 0:
            raise ValueError(f"scenario {index} has an invalid asset index")
        quat_norm = sum(float(value) ** 2 for value in scenario["object_quat_wxyz"]) ** 0.5
        if abs(quat_norm - 1.0) > 1.0e-3:
            raise ValueError(f"scenario {index} object quaternion is not normalized")
    return payload


def arm_controllability_metrics(
    *,
    arm_position: torch.Tensor,
    arm_lower: torch.Tensor,
    arm_upper: torch.Tensor,
    palm_jacobian: torch.Tensor,
    future_twist: torch.Tensor,
    arm_velocity_limits: torch.Tensor,
    damping: float,
) -> dict[str, torch.Tensor]:
    """Compute differentiable configuration and future-motion diagnostics."""
    if arm_position.ndim != 2 or arm_position.shape[-1] != 7:
        raise ValueError("arm_position must have shape (N, 7)")
    n = arm_position.shape[0]
    if palm_jacobian.shape != (n, 6, 7) or future_twist.shape != (n, 6):
        raise ValueError("palm_jacobian or future_twist has an invalid shape")
    for name, value in (("arm_lower", arm_lower), ("arm_upper", arm_upper)):
        if value.shape not in ((7,), (n, 7)):
            raise ValueError(f"{name} must have shape (7,) or (N, 7)")
    if arm_velocity_limits.shape not in ((7,), (n, 7)):
        raise ValueError("arm_velocity_limits must have shape (7,) or (N, 7)")
    tensors = (arm_position, arm_lower, arm_upper, palm_jacobian, future_twist, arm_velocity_limits)
    if not all(torch.isfinite(value).all() for value in tensors):
        raise ValueError("controllability inputs contain NaN or Inf")
    if damping <= 0.0 or bool((arm_velocity_limits <= 0.0).any()):
        raise ValueError("damping and velocity limits must be positive")

    joint_margin = torch.minimum(arm_position - arm_lower, arm_upper - arm_position).min(-1).values
    singular_values = torch.linalg.svdvals(palm_jacobian)
    sigma_min = singular_values[:, -1]
    condition = singular_values[:, 0] / sigma_min.clamp_min(1.0e-8)
    regularizer = float(damping) ** 2 * torch.eye(
        6, device=palm_jacobian.device, dtype=palm_jacobian.dtype
    ).expand(n, -1, -1)
    solved = torch.linalg.solve(
        palm_jacobian @ palm_jacobian.transpose(-1, -2) + regularizer,
        future_twist.unsqueeze(-1),
    )
    qdot = (palm_jacobian.transpose(-1, -2) @ solved).squeeze(-1)
    velocity_ratio = (qdot.abs() / arm_velocity_limits).max(-1).values
    return {
        "joint_margin_rad": joint_margin,
        "minimum_singular_value": sigma_min,
        "condition_number": condition,
        "future_velocity_ratio": velocity_ratio,
    }


def controllability_score(
    metrics: dict[str, torch.Tensor], *, joint_margin_target_rad: float,
    singular_value_target: float, condition_number_limit: float,
) -> torch.Tensor:
    """Map explicit arm criteria to a bounded geometric-mean score."""
    if joint_margin_target_rad <= 0.0 or singular_value_target <= 0.0 or condition_number_limit <= 0.0:
        raise ValueError("controllability score thresholds must be positive")
    joint = (metrics["joint_margin_rad"] / joint_margin_target_rad).clamp(0.0, 1.0)
    singular = (metrics["minimum_singular_value"] / singular_value_target).clamp(0.0, 1.0)
    condition = (condition_number_limit / metrics["condition_number"].clamp_min(1.0)).clamp(0.0, 1.0)
    motion = torch.exp(-torch.clamp(metrics["future_velocity_ratio"] - 1.0, min=0.0))
    return (joint * singular * condition * motion).clamp_min(0.0).pow(0.25)


def adjustment_reward_terms(
    *, score: torch.Tensor, previous_score: torch.Tensor,
    tool_position_error_m: torch.Tensor, tool_rotation_error_rad: torch.Tensor,
    grasp_position_error_m: torch.Tensor, grasp_rotation_error_rad: torch.Tensor,
    action_delta_sq_mean: torch.Tensor, grasp_retained: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Dense shaping terms; task rollout success is added in the later refinement stage."""
    shape = score.shape
    values = (
        previous_score, tool_position_error_m, tool_rotation_error_rad,
        grasp_position_error_m, grasp_rotation_error_rad, action_delta_sq_mean,
    )
    if any(value.shape != shape for value in values) or grasp_retained.shape != shape:
        raise ValueError("adjustment reward inputs must share shape (N,)")
    return {
        "score": score,
        "score_improvement": score - previous_score,
        "tool_position": -tool_position_error_m,
        "tool_rotation": -tool_rotation_error_rad,
        "grasp_position": -grasp_position_error_m,
        "grasp_rotation": -grasp_rotation_error_rad,
        "action_rate": -action_delta_sq_mean,
        "grasp_loss": -(~grasp_retained).to(score.dtype),
    }


__all__ = [
    "SCENARIO_KINDS", "adjustment_reward_terms", "arm_controllability_metrics",
    "controllability_score", "load_adjustment_scenarios",
    "orbit_palm_tool_about_screw_axis", "screw_axis_orbit_errors",
]
