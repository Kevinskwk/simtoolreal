"""Validation and tensor scoring for trajectory-conditioned grasp adjustment."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch


SCENARIO_KINDS = ("nominal", "fixed_pose_hard")


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
]
