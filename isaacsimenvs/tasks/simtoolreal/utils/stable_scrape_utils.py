"""Pure tensor helpers for phased stable-contact scraping."""

from __future__ import annotations

import torch

ACQUISITION_PHASE = 0
CONTACT_APPROACH_PHASE = 1
SCRAPE_PHASE = 2
NUM_STABLE_SCRAPE_PHASES = 3


def phase_one_hot(phase: torch.Tensor) -> torch.Tensor:
    """Return a validated three-way phase encoding."""
    if phase.dtype not in (torch.int32, torch.int64):
        raise ValueError("phase must use an integer dtype")
    if bool(((phase < 0) | (phase >= NUM_STABLE_SCRAPE_PHASES)).any()):
        raise ValueError("phase contains an unknown stable-scrape phase")
    return torch.nn.functional.one_hot(phase, NUM_STABLE_SCRAPE_PHASES).float()


def quaternion_distance_rad(current: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Shortest angular distance between batched wxyz unit quaternions."""
    if current.shape != reference.shape or current.shape[-1] != 4:
        raise ValueError("quaternion tensors must have matching (..., 4) shapes")
    dot = torch.abs((current * reference).sum(dim=-1)).clamp(0.0, 1.0)
    return 2.0 * torch.acos(dot)


def advance_reflected_path(
    offset: torch.Tensor,
    direction_sign: torch.Tensor,
    *,
    distance: float,
    half_length: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance a scalar path and reflect overshoot at either endpoint."""
    if offset.shape != direction_sign.shape:
        raise ValueError("offset and direction_sign must have matching shapes")
    if distance < 0.0 or half_length <= 0.0:
        raise ValueError("distance must be non-negative and half_length positive")
    if distance > 2.0 * half_length:
        raise ValueError("one update may not traverse more than the full path")
    proposed = offset + direction_sign * float(distance)
    above = proposed > float(half_length)
    below = proposed < -float(half_length)
    proposed = torch.where(above, 2.0 * float(half_length) - proposed, proposed)
    proposed = torch.where(below, -2.0 * float(half_length) - proposed, proposed)
    sign = torch.where(above | below, -direction_sign, direction_sign)
    return proposed, sign


def consecutive_counter(condition: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
    """Count consecutive true samples and reset to zero on false."""
    if condition.dtype != torch.bool or condition.shape != previous.shape:
        raise ValueError("condition must be boolean and match previous")
    return torch.where(condition, previous + 1, torch.zeros_like(previous))


def stable_scrape_reward_terms(
    *, phase: torch.Tensor, pose_error_m: torch.Tensor, pose_sigma_m: float,
    edge_score: torch.Tensor, persistent_contact: torch.Tensor,
    support_count: torch.Tensor, relative_linear_speed: torch.Tensor,
    relative_angular_speed: torch.Tensor, action_delta_sq_mean: torch.Tensor,
    tool_acceleration: torch.Tensor, normal_force_n: torch.Tensor,
    soft_force_limit_n: float,
) -> dict[str, torch.Tensor]:
    """Compute normalized post-grasp rewards; force is safety-only."""
    active = (phase != ACQUISITION_PHASE).to(pose_error_m.dtype)
    grasp_valid = (support_count >= 2).to(pose_error_m.dtype)
    tracking = (
        torch.exp(-pose_error_m / max(float(pose_sigma_m), 1.0e-6))
        * active
        * grasp_valid
    )
    edge = edge_score * active * grasp_valid
    contact = persistent_contact.to(pose_error_m.dtype) * active * grasp_valid
    support = torch.clamp(support_count.to(pose_error_m.dtype) / 2.0, 0.0, 1.0) * active
    slip = -torch.clamp(relative_linear_speed - 0.03, min=0.0) * active
    spin = -torch.clamp(relative_angular_speed - 1.0, min=0.0) * active
    action_rate = -action_delta_sq_mean * active
    acceleration = -torch.clamp(tool_acceleration - 2.0, min=0.0) * active
    over_force = -torch.clamp(normal_force_n - float(soft_force_limit_n), min=0.0).square() * active
    return {"tracking": tracking, "edge": edge, "contact": contact,
            "support": support, "slip": slip, "spin": spin,
            "action_rate": action_rate, "acceleration": acceleration,
            "over_force": over_force}


__all__ = ["ACQUISITION_PHASE", "CONTACT_APPROACH_PHASE", "SCRAPE_PHASE",
           "NUM_STABLE_SCRAPE_PHASES", "phase_one_hot", "quaternion_distance_rad",
           "advance_reflected_path", "consecutive_counter", "stable_scrape_reward_terms"]
