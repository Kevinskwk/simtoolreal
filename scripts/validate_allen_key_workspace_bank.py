#!/usr/bin/env python3
"""Validate rollout metadata and every workspace/pair curriculum stage."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
TIERS = ("easy", "support", "broad")
PROBABILITIES = (
    (0.80, 0.20, 0.00),
    (0.65, 0.30, 0.05),
    (0.50, 0.40, 0.10),
    (0.40, 0.40, 0.20),
    (0.30, 0.40, 0.30),
)
TRANSLATION_MAX = (0.060, 0.060, 0.075, 0.075, 0.090)
ROTATION_MAX = (60.0, 60.0, 80.0, 80.0, 100.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grasp-bank", type=Path,
        default=ROOT / "assets/grasp_banks/allen_key_rollout_adjustment_v1.json",
    )
    parser.add_argument("--minimum-quality-improvement", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.grasp_bank.read_text())
    entries = payload.get("entries", [])
    if len(entries) < 18:
        raise RuntimeError("workspace bank requires at least 18 validated grasps")
    count = len(entries)
    tiers = np.empty(count, dtype=np.int64)
    quality = np.empty(count, dtype=np.float64)
    centers = np.empty((count, 3), dtype=np.float64)
    rotations = []
    declared = np.zeros((count, count), dtype=bool)
    for index, entry in enumerate(entries):
        verification = entry.get("verification", {})
        tier_name = verification.get("rollout_workspace_tier")
        if tier_name not in TIERS:
            raise RuntimeError(f"entry {index} has invalid workspace tier {tier_name!r}")
        tiers[index] = TIERS.index(tier_name)
        quality[index] = float(verification.get("rollout_functional_quality", math.nan))
        if not math.isfinite(quality[index]):
            raise RuntimeError(f"entry {index} has non-finite rollout quality")
        position = np.asarray(entry["palm_to_tool_pos"], dtype=np.float64)
        quaternion = np.asarray(entry["palm_to_tool_quat_wxyz"], dtype=np.float64)
        rotation = Rotation.from_quat(
            (quaternion[1], quaternion[2], quaternion[3], quaternion[0])
        )
        centers[index] = -rotation.inv().apply(position)
        rotations.append(rotation)
        target_ids = verification.get("valid_target_ids", [])
        for target_id in target_ids:
            target_id = int(target_id)
            if not 0 <= target_id < count or target_id == index:
                raise RuntimeError(f"entry {index} declares invalid target {target_id}")
            declared[index, target_id] = True
    translation = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)
    rotation = np.zeros((count, count), dtype=np.float64)
    for source in range(count):
        for target in range(count):
            rotation[source, target] = math.degrees(
                float((rotations[source].inv() * rotations[target]).magnitude())
            )
    improvement = quality[None, :] - quality[:, None]
    reports = []
    for stage, (probabilities, translation_max, rotation_max) in enumerate(zip(
        PROBABILITIES, TRANSLATION_MAX, ROTATION_MAX, strict=True
    )):
        active = (
            declared & (translation <= translation_max) & (rotation <= rotation_max)
            & (improvement >= float(args.minimum_quality_improvement))
        )
        available = active.any(axis=1)
        tier_counts = {
            tier_name: int((available & (tiers == tier)).sum())
            for tier, tier_name in enumerate(TIERS)
        }
        for tier, probability in enumerate(probabilities):
            if probability > 0.0 and tier_counts[TIERS[tier]] == 0:
                raise RuntimeError(
                    f"stage {stage} requests tier {TIERS[tier]} without a valid start"
                )
        pair_count = int(active.sum())
        if pair_count == 0:
            raise RuntimeError(f"stage {stage} has no improved target pairs")
        reports.append({
            "stage": stage,
            "available_starts": int(available.sum()),
            "active_pairs": pair_count,
            "available_by_tier": tier_counts,
        })
    print(
        f"[pass] {count} rollout-validated grasps; tiers="
        f"{dict((name, int((tiers == i).sum())) for i, name in enumerate(TIERS))}; "
        f"curriculum={reports}"
    )


if __name__ == "__main__":
    main()
