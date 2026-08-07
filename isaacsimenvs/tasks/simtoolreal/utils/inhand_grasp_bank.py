"""Validation helpers for compliant in-hand grasp snapshots."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch


SCHEMA_VERSION = 1
JOINT_COUNT = 29


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_vector(entry: dict, name: str, size: int) -> list[float]:
    value = entry.get(name)
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"grasp entry {name} must contain {size} values")
    tensor = torch.tensor(value, dtype=torch.float32)
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"grasp entry {name} contains NaN or Inf")
    return [float(item) for item in value]


def validate_grasp_bank(payload: dict, *, minimum_entries: int = 1) -> dict:
    if int(minimum_entries) <= 0:
        raise ValueError("minimum_entries must be positive")
    if not isinstance(payload, dict):
        raise ValueError("grasp bank must be a JSON object")
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported grasp-bank schema {payload.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    if payload.get("tool_type") != "spatula":
        raise ValueError("V1 in-hand grasp bank must use tool_type='spatula'")
    for name in ("asset_sha256", "source_checkpoint_sha256"):
        value = payload.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value.lower())
        ):
            raise ValueError(f"grasp bank {name} must be a SHA-256 hex digest")
    if float(payload.get("policy_coefficient_id", float("nan"))) != 0.0:
        raise ValueError("grasp bank policy_coefficient_id must be 0.0")
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) < int(minimum_entries):
        raise ValueError(
            f"grasp bank requires at least {minimum_entries} entries, got "
            f"{len(entries) if isinstance(entries, list) else 'invalid'}"
        )
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"grasp entry {index} must be an object")
        for name, size in (
            ("joint_pos_canonical", JOINT_COUNT),
            ("joint_vel_canonical", JOINT_COUNT),
            ("joint_targets_canonical", JOINT_COUNT),
            ("last_action_canonical", JOINT_COUNT),
            ("object_pos_local", 3),
            ("object_quat_wxyz", 4),
            ("object_velocity", 6),
            ("palm_to_tool_pos", 3),
            ("palm_to_tool_quat_wxyz", 4),
        ):
            _finite_vector(entry, name, size)
        for name in ("object_quat_wxyz", "palm_to_tool_quat_wxyz"):
            norm = torch.linalg.vector_norm(torch.tensor(entry[name], dtype=torch.float32))
            if not bool(torch.isclose(norm, torch.tensor(1.0), atol=1.0e-3)):
                raise ValueError(f"grasp entry {index} {name} is not normalized")
        metrics = entry.get("verification")
        if not isinstance(metrics, dict):
            raise ValueError(f"grasp entry {index} has no verification metrics")
        for name in (
            "edge_clearance_m", "table_force_n", "hold_drift_m",
            "hold_rotation_deg",
        ):
            value = float(metrics.get(name, float("nan")))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"grasp entry {index} verification {name} must be finite and non-negative"
                )
        if int(metrics.get("support_count", 0)) < 2:
            raise ValueError(f"grasp entry {index} has insufficient fingertip support")
        if int(metrics.get("stable_steps", 0)) < 15:
            raise ValueError(f"grasp entry {index} has an insufficient stability window")
        if int(metrics.get("hold_steps", 0)) < 120:
            raise ValueError(f"grasp entry {index} has an insufficient compliant hold window")
        if float(metrics.get("edge_clearance_m", -1.0)) < 0.03:
            raise ValueError(f"grasp entry {index} was not verified clear of the table")
        if float(metrics.get("hold_drift_m", float("inf"))) > 0.005:
            raise ValueError(f"grasp entry {index} failed the hold-drift limit")
        if float(metrics.get("hold_rotation_deg", float("inf"))) > 2.0:
            raise ValueError(f"grasp entry {index} failed the hold-rotation limit")
        if float(metrics.get("table_force_n", float("inf"))) >= 0.1:
            raise ValueError(f"grasp entry {index} was touching the table")
    return payload


def load_grasp_bank(path: str | Path, *, minimum_entries: int = 1) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"in-hand grasp bank does not exist: {path}. "
            "Run scripts/collect_inhand_grasp_bank.py first."
        )
    with path.open() as stream:
        payload = json.load(stream)
    return validate_grasp_bank(payload, minimum_entries=minimum_entries)


def collision_box_corners(bounds: torch.Tensor) -> torch.Tensor:
    """Return eight local AABB corners for (..., 6) bounds."""
    if bounds.shape[-1] != 6 or not torch.isfinite(bounds).all():
        raise ValueError("collision bounds must be finite (..., 6) tensors")
    lo, hi = bounds[..., :3], bounds[..., 3:]
    if bool((hi < lo).any()):
        raise ValueError("collision-bound maxima must not be below minima")
    selectors = torch.tensor(
        [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)],
        device=bounds.device,
        dtype=torch.bool,
    )
    return torch.where(selectors, hi.unsqueeze(-2), lo.unsqueeze(-2))


def table_root_z_for_lowest_clearance(
    points_w: torch.Tensor,
    table_normal_w: torch.Tensor,
    env_origins: torch.Tensor,
    clearance_m: torch.Tensor,
    *,
    table_half_height_m: float,
) -> torch.Tensor:
    """Solve table-root local z below the lowest supplied geometry point."""
    if points_w.ndim != 3 or points_w.shape[-1] != 3:
        raise ValueError("geometry points must have shape (N, M, 3)")
    if table_normal_w.shape != points_w.shape[:1] + (3,):
        raise ValueError("table normal must have shape (N, 3)")
    if env_origins.shape != table_normal_w.shape or clearance_m.shape != points_w.shape[:1]:
        raise ValueError("env origins or clearance shape is invalid")
    normal_z = table_normal_w[:, 2]
    if bool((normal_z <= 0.5).any()):
        raise ValueError("table normal is too steep to solve a stable root height")
    lowest_projection = (
        (points_w - env_origins.unsqueeze(1)) * table_normal_w.unsqueeze(1)
    ).sum(dim=-1).min(dim=-1).values
    return (
        lowest_projection
        - float(table_half_height_m)
        - clearance_m
    ) / normal_z


__all__ = [
    "SCHEMA_VERSION", "collision_box_corners", "load_grasp_bank", "sha256_file",
    "table_root_z_for_lowest_clearance", "validate_grasp_bank",
]
