"""Validation helpers for compliant in-hand grasp snapshots."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch


SCHEMA_VERSION = 2
MULTI_ASSET_SCHEMA_VERSION = 3
JOINT_COUNT = 29
SUPPORTED_TOOL_TYPES = frozenset(
    ("hammer", "screwdriver", "eraser", "spatula", "marker", "brush")
)


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
    if int(payload.get("schema_version", -1)) == MULTI_ASSET_SCHEMA_VERSION:
        return validate_multi_asset_grasp_bank(
            payload, minimum_entries_per_asset=minimum_entries
        )
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported grasp-bank schema {payload.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    tool_type = payload.get("tool_type")
    if tool_type not in SUPPORTED_TOOL_TYPES:
        raise ValueError(
            "V2 in-hand grasp bank tool_type must be one of "
            f"{sorted(SUPPORTED_TOOL_TYPES)}, got {tool_type!r}"
        )
    object_name = payload.get("object_name")
    # Banks created before multi-tool collection did not carry object_name.
    if object_name is None and tool_type != "eraser":
        raise ValueError("non-eraser grasp banks must specify object_name")
    if object_name is not None and (
        not isinstance(object_name, str) or not object_name.strip()
    ):
        raise ValueError("grasp bank object_name must be a non-empty string")
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
    tactile_fraction = float(payload.get("tactile_rich_fraction_min", float("nan")))
    tactile_min_fingers = int(payload.get("tactile_min_fingers", 0))
    if not math.isfinite(tactile_fraction) or not 0.0 <= tactile_fraction <= 1.0:
        raise ValueError("grasp bank tactile_rich_fraction_min must be in [0, 1]")
    if tactile_min_fingers < 1 or tactile_min_fingers > 5:
        raise ValueError("grasp bank tactile_min_fingers must be in [1, 5]")
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) < int(minimum_entries):
        raise ValueError(
            f"grasp bank requires at least {minimum_entries} entries, got "
            f"{len(entries) if isinstance(entries, list) else 'invalid'}"
        )
    has_joint_limits = any(
        name in payload
        for name in (
            "joint_lower_canonical",
            "joint_upper_canonical",
            "joint_limit_tolerance_rad",
        )
    )
    joint_lower = joint_upper = None
    joint_limit_tolerance = None
    if has_joint_limits:
        if not all(
            name in payload
            for name in (
                "joint_lower_canonical",
                "joint_upper_canonical",
                "joint_limit_tolerance_rad",
            )
        ):
            raise ValueError("grasp bank joint-limit metadata is incomplete")
        joint_lower = torch.tensor(
            _finite_vector(payload, "joint_lower_canonical", JOINT_COUNT)
        )
        joint_upper = torch.tensor(
            _finite_vector(payload, "joint_upper_canonical", JOINT_COUNT)
        )
        if bool((joint_upper <= joint_lower).any()):
            raise ValueError("grasp bank canonical joint limits are not ordered")
        joint_limit_tolerance = float(payload["joint_limit_tolerance_rad"])
        if not math.isfinite(joint_limit_tolerance) or not 0.0 <= joint_limit_tolerance <= 0.01:
            raise ValueError("grasp bank joint_limit_tolerance_rad must be in [0, 0.01]")

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
            ("reference_contact_quat_wxyz", 4),
        ):
            _finite_vector(entry, name, size)
        for name in (
            "object_quat_wxyz", "palm_to_tool_quat_wxyz",
            "reference_contact_quat_wxyz",
        ):
            norm = torch.linalg.vector_norm(torch.tensor(entry[name], dtype=torch.float32))
            if not bool(torch.isclose(norm, torch.tensor(1.0), atol=1.0e-3)):
                raise ValueError(f"grasp entry {index} {name} is not normalized")
        metrics = entry.get("verification")
        if not isinstance(metrics, dict):
            raise ValueError(f"grasp entry {index} has no verification metrics")
        for name in (
            "edge_clearance_m", "table_force_n", "hold_drift_m",
            "hold_rotation_deg", "pickup_orientation_error_deg",
            "tactile_contact_area_mean", "tactile_depth_mean",
            "tactile_depth_max",
        ):
            value = float(metrics.get(name, float("nan")))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"grasp entry {index} verification {name} must be finite and non-negative"
                )
        if int(metrics.get("support_count", 0)) < 2:
            raise ValueError(f"grasp entry {index} has insufficient fingertip support")
        if int(metrics.get("tactile_finger_count_min", -1)) < 0:
            raise ValueError(f"grasp entry {index} has invalid tactile finger count")
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
        if float(metrics.get("pickup_orientation_error_deg", float("inf"))) > 15.0:
            raise ValueError(f"grasp entry {index} did not track its edge-contact orientation")
        if float(metrics.get("table_force_n", float("inf"))) >= 0.1:
            raise ValueError(f"grasp entry {index} was touching the table")
        if has_joint_limits:
            joint_pos = torch.tensor(entry["joint_pos_canonical"])
            violation = torch.maximum(
                (joint_lower - joint_pos).clamp_min(0.0),
                (joint_pos - joint_upper).clamp_min(0.0),
            ).max()
            if float(violation.item()) > joint_limit_tolerance:
                raise ValueError(
                    f"grasp entry {index} exceeds canonical joint limits: "
                    f"violation={float(violation.item()):.7f}, "
                    f"tolerance={joint_limit_tolerance:.7f}"
                )
            recorded_violation = float(
                metrics.get("joint_limit_violation_max_rad", float("nan"))
            )
            if (
                not math.isfinite(recorded_violation)
                or recorded_violation < 0.0
                or recorded_violation > joint_limit_tolerance
            ):
                raise ValueError(
                    f"grasp entry {index} has invalid hold-window joint-limit verification"
                )
        yaw = float(entry.get("reference_edge_yaw_rad", float("nan")))
        tilt = float(entry.get("reference_edge_tilt_rad", float("nan")))
        if not math.isfinite(yaw) or not math.isfinite(tilt):
            raise ValueError(f"grasp entry {index} has non-finite edge orientation")
        if abs(yaw) > math.pi or not 0.0 < tilt < 0.5 * math.pi:
            raise ValueError(f"grasp entry {index} edge orientation is out of range")
    tactile_rich_count = sum(
        int(entry["verification"]["tactile_finger_count_min"])
        >= tactile_min_fingers
        for entry in entries
    )
    actual_fraction = tactile_rich_count / len(entries)
    if actual_fraction + 1.0e-12 < tactile_fraction:
        raise ValueError(
            "grasp bank tactile-rich fraction is below its declared minimum: "
            f"actual={actual_fraction:.3f}, required={tactile_fraction:.3f}"
        )
    return payload


def validate_multi_asset_grasp_bank(
    payload: dict, *, minimum_entries_per_asset: int = 1
) -> dict:
    """Validate a strict cache of matched grasps for procedural assets."""
    if int(minimum_entries_per_asset) <= 0:
        raise ValueError("minimum_entries_per_asset must be positive")
    if payload.get("kind") != "simtoolreal_multi_asset_grasp_cache":
        raise ValueError("V3 grasp cache has an invalid kind")
    assets = payload.get("assets")
    if not isinstance(assets, list) or not assets:
        raise ValueError("V3 grasp cache must contain a non-empty assets list")
    indices: set[int] = set()
    hashes: set[str] = set()
    common = {
        "schema_version": SCHEMA_VERSION,
        "source_checkpoint_sha256": payload.get("source_checkpoint_sha256"),
        "policy_coefficient_id": payload.get("policy_coefficient_id"),
        "tactile_rich_fraction_min": payload.get("tactile_rich_fraction_min", 0.0),
        "tactile_min_fingers": payload.get("tactile_min_fingers", 1),
        "control_dt_s": payload.get("control_dt_s"),
    }
    for optional in (
        "joint_lower_canonical", "joint_upper_canonical", "joint_limit_tolerance_rad"
    ):
        if optional in payload:
            common[optional] = payload[optional]
    for position, asset in enumerate(assets):
        if not isinstance(asset, dict):
            raise ValueError(f"V3 grasp-cache asset {position} must be an object")
        index = int(asset.get("asset_index", -1))
        if index < 0 or index in indices:
            raise ValueError(f"V3 grasp-cache asset index {index} is invalid or duplicated")
        indices.add(index)
        digest = asset.get("asset_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or digest in hashes:
            raise ValueError(f"V3 grasp-cache asset {index} has an invalid or duplicate hash")
        hashes.add(digest)
        scale = asset.get("object_scale")
        if (
            not isinstance(scale, list) or len(scale) != 3
            or not all(math.isfinite(float(value)) and float(value) > 0.0 for value in scale)
        ):
            raise ValueError(f"V3 grasp-cache asset {index} has an invalid object scale")
        single = dict(common)
        single.update({
            "tool_type": asset.get("tool_type"),
            "object_name": asset.get("object_name", f"procedural_{index:04d}"),
            "asset_sha256": digest,
            "entries": asset.get("entries"),
        })
        validate_grasp_bank(single, minimum_entries=minimum_entries_per_asset)
    expected = list(range(len(assets)))
    if sorted(indices) != expected:
        raise ValueError(
            "V3 grasp-cache asset indices must be contiguous and match generated-pool order"
        )
    return payload


def flatten_multi_asset_grasp_bank(payload: dict) -> tuple[list[dict], list[int]]:
    """Return entries and their generated-pool asset indices."""
    validate_multi_asset_grasp_bank(payload)
    entries: list[dict] = []
    asset_indices: list[int] = []
    for asset in payload["assets"]:
        entries.extend(asset["entries"])
        asset_indices.extend([int(asset["asset_index"])] * len(asset["entries"]))
    return entries, asset_indices


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


def merge_grasp_banks(payloads: list[dict]) -> dict:
    """Merge compatible banks while removing byte-identical grasp entries."""
    if not payloads:
        raise ValueError("at least one grasp bank is required for merging")
    validated = [validate_grasp_bank(payload) for payload in payloads]
    reference = validated[0]
    identity_fields = (
        "schema_version",
        "tool_type",
        "object_name",
        "asset_sha256",
        "source_checkpoint_sha256",
        "policy_coefficient_id",
        "tactile_rich_fraction_min",
        "tactile_min_fingers",
        "control_dt_s",
        "joint_lower_canonical",
        "joint_upper_canonical",
        "joint_limit_tolerance_rad",
    )
    for bank_index, payload in enumerate(validated[1:], start=1):
        for field in identity_fields:
            if payload.get(field) != reference.get(field):
                raise ValueError(
                    f"grasp bank {bank_index} has incompatible {field}: "
                    f"{payload.get(field)!r} != {reference.get(field)!r}"
                )

    merged = dict(reference)
    entries: list[dict] = []
    fingerprints: set[str] = set()
    duplicate_count = 0
    seeds: list[int] = []
    for payload in validated:
        if "seed" in payload:
            seeds.append(int(payload["seed"]))
        for entry in payload["entries"]:
            fingerprint = json.dumps(entry, sort_keys=True, separators=(",", ":"))
            if fingerprint in fingerprints:
                duplicate_count += 1
                continue
            fingerprints.add(fingerprint)
            entries.append(entry)
    merged["entries"] = entries
    merged["seeds"] = sorted(set(seeds))
    merged["merged_bank_count"] = len(validated)
    merged["duplicates_removed"] = duplicate_count
    return validate_grasp_bank(merged)


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
    "MULTI_ASSET_SCHEMA_VERSION", "SCHEMA_VERSION", "collision_box_corners",
    "flatten_multi_asset_grasp_bank", "load_grasp_bank", "sha256_file",
    "table_root_z_for_lowest_clearance", "validate_grasp_bank",
    "validate_multi_asset_grasp_bank",
]
