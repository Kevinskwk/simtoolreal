"""Pure analysis helpers for the virtual pose-offset evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StabilityLimits:
    """Thresholds for a trailing tool-motion stability window."""

    position_m: float = 0.001
    orientation_deg: float = 2.0
    linear_speed_mps: float = 0.005
    angular_speed_radps: float = 0.1


def quaternion_angle_deg(quaternions_wxyz: np.ndarray, reference_wxyz: np.ndarray) -> np.ndarray:
    """Return sign-invariant quaternion angular distance from one reference."""
    quaternions = np.asarray(quaternions_wxyz, dtype=np.float64)
    reference = np.asarray(reference_wxyz, dtype=np.float64)
    if quaternions.ndim != 2 or quaternions.shape[1] != 4 or reference.shape != (4,):
        raise ValueError("expected quaternions with shape (N, 4) and reference with shape (4,)")
    norms = np.linalg.norm(quaternions, axis=1)
    reference_norm = np.linalg.norm(reference)
    if np.any(norms <= 0.0) or reference_norm <= 0.0:
        raise ValueError("quaternions must have non-zero norm")
    normalized = quaternions / norms[:, None]
    reference = reference / reference_norm
    dots = np.clip(np.abs(normalized @ reference), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dots))


def pose_window_is_stable(
    positions_m: np.ndarray,
    quaternions_wxyz: np.ndarray,
    linear_speeds_mps: np.ndarray,
    angular_speeds_radps: np.ndarray,
    limits: StabilityLimits = StabilityLimits(),
) -> tuple[bool, dict[str, float]]:
    """Evaluate tool stability relative to the final sample in a fixed window."""
    positions = np.asarray(positions_m, dtype=np.float64)
    quaternions = np.asarray(quaternions_wxyz, dtype=np.float64)
    linear_speeds = np.asarray(linear_speeds_mps, dtype=np.float64)
    angular_speeds = np.asarray(angular_speeds_radps, dtype=np.float64)
    n = positions.shape[0]
    if (
        positions.shape != (n, 3)
        or quaternions.shape != (n, 4)
        or linear_speeds.shape != (n,)
        or angular_speeds.shape != (n,)
        or n == 0
    ):
        raise ValueError("stability inputs have incompatible shapes")
    values = np.concatenate(
        (positions.reshape(-1), quaternions.reshape(-1), linear_speeds, angular_speeds)
    )
    if not np.isfinite(values).all():
        raise ValueError("stability inputs contain NaN or Inf")

    position_deviation = float(np.linalg.norm(positions - positions[-1], axis=1).max())
    orientation_deviation = float(
        quaternion_angle_deg(quaternions, quaternions[-1]).max()
    )
    maximum_linear_speed = float(linear_speeds.max())
    maximum_angular_speed = float(angular_speeds.max())
    metrics = {
        "position_deviation_m": position_deviation,
        "orientation_deviation_deg": orientation_deviation,
        "maximum_linear_speed_mps": maximum_linear_speed,
        "maximum_angular_speed_radps": maximum_angular_speed,
    }
    stable = (
        position_deviation <= limits.position_m
        and orientation_deviation <= limits.orientation_deg
        and maximum_linear_speed <= limits.linear_speed_mps
        and maximum_angular_speed <= limits.angular_speed_radps
    )
    return stable, metrics


def rankdata(values: np.ndarray) -> np.ndarray:
    """Return average ranks, including deterministic handling of ties."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("rankdata expects a one-dimensional array")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman correlation without requiring scipy at test time."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or x.size < 2:
        raise ValueError("Spearman inputs must be matching one-dimensional arrays")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Spearman inputs contain NaN or Inf")
    rx = rankdata(x)
    ry = rankdata(y)
    if np.std(rx) == 0.0 or np.std(ry) == 0.0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def select_force_calibration(
    offsets_m: np.ndarray,
    steady_forces_n: np.ndarray,
    stable: np.ndarray,
    *,
    target_force_n: float = 4.0,
    maximum_force_n: float = 20.0,
) -> dict[str, float | int | bool]:
    """Choose the stable, safe offset nearest a target and report bracketing."""
    offsets = np.asarray(offsets_m, dtype=np.float64)
    forces = np.asarray(steady_forces_n, dtype=np.float64)
    stable_mask = np.asarray(stable, dtype=bool)
    if offsets.ndim != 1 or forces.shape != offsets.shape or stable_mask.shape != offsets.shape:
        raise ValueError("calibration arrays must be matching one-dimensional arrays")
    valid = stable_mask & np.isfinite(offsets) & np.isfinite(forces) & (forces <= maximum_force_n)
    if not np.any(valid):
        raise ValueError("no stable, finite, safe force-offset samples are available")
    indices = np.flatnonzero(valid)
    selected_index = int(indices[np.argmin(np.abs(forces[indices] - target_force_n))])
    valid_forces = forces[indices]
    bracketed = bool(valid_forces.min() <= target_force_n <= valid_forces.max())
    return {
        "selected_index": selected_index,
        "offset_m": float(offsets[selected_index]),
        "virtual_depth_m": float(-offsets[selected_index]),
        "steady_force_n": float(forces[selected_index]),
        "absolute_error_n": float(abs(forces[selected_index] - target_force_n)),
        "bracketed": bracketed,
        "valid_sample_count": int(indices.size),
    }


def longest_false_run(mask: np.ndarray) -> int:
    """Return the longest contiguous run for which a boolean mask is false."""
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError("mask must be one-dimensional")
    longest = current = 0
    for value in values:
        if value:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def update_persistent_violation(
    value: float,
    *,
    threshold: float,
    previous_steps: int,
    required_steps: int,
    immediate_threshold: float | None = None,
) -> tuple[int, bool]:
    """Update a consecutive-threshold gate with an optional immediate ceiling."""
    if not np.isfinite(value):
        raise ValueError("violation value must be finite")
    if threshold <= 0.0 or required_steps <= 0 or previous_steps < 0:
        raise ValueError("violation thresholds/counts must be positive")
    if immediate_threshold is not None and immediate_threshold <= threshold:
        raise ValueError("immediate_threshold must exceed threshold")
    next_steps = previous_steps + 1 if value > threshold else 0
    violated = next_steps >= required_steps
    if immediate_threshold is not None and value > immediate_threshold:
        violated = True
    return next_steps, violated


def transition_force_metrics(
    force_n: np.ndarray,
    contact: np.ndarray,
    *,
    control_dt_s: float,
    target_force_n: float = 4.0,
    settle_tolerance_n: float = 1.0,
    settle_window_steps: int = 30,
    steady_window_steps: int = 30,
) -> dict[str, float | bool]:
    """Summarize contact continuity and target-force behavior for one transition."""
    force = np.asarray(force_n, dtype=np.float64)
    contact_mask = np.asarray(contact, dtype=bool)
    if force.ndim != 1 or contact_mask.shape != force.shape or force.size == 0:
        raise ValueError("transition arrays must be non-empty and have matching shapes")
    if not np.isfinite(force).all() or control_dt_s <= 0.0:
        raise ValueError("transition force must be finite and control_dt_s positive")

    settle_index: int | None = None
    within = np.abs(force - target_force_n) <= settle_tolerance_n
    if settle_window_steps > 0 and force.size >= settle_window_steps:
        for start in range(force.size - settle_window_steps + 1):
            if bool(within[start : start + settle_window_steps].all()):
                settle_index = start
                break
    steady = force[-min(steady_window_steps, force.size) :]
    return {
        "contact_maintenance_ratio": float(contact_mask.mean()),
        "longest_contact_loss_s": float(longest_false_run(contact_mask) * control_dt_s),
        "minimum_force_n": float(force.min()),
        "maximum_force_n": float(force.max()),
        "force_dip_from_target_n": float(max(0.0, target_force_n - force.min())),
        "force_overshoot_n": float(max(0.0, force.max() - target_force_n)),
        "steady_force_mean_n": float(steady.mean()),
        "steady_force_mae_n": float(np.abs(steady - target_force_n).mean()),
        "settled": settle_index is not None,
        "settling_time_s": (
            float(settle_index * control_dt_s) if settle_index is not None else float("nan")
        ),
    }


__all__ = [
    "StabilityLimits",
    "longest_false_run",
    "pose_window_is_stable",
    "quaternion_angle_deg",
    "select_force_calibration",
    "spearman_correlation",
    "transition_force_metrics",
    "update_persistent_violation",
]
