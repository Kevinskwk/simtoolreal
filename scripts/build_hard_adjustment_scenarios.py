#!/usr/bin/env python3
"""Build matched fixed-tool-pose hard and recovery arm contexts."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
ROBOT_URDF = ROOT / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
DEFAULT_OBJECT_URDF = ROOT / "assets/urdf/objects/eraser_tactile_canonical.urdf"
SPHERES = ROOT / "baselines/assets/sharpa_spheres.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ge = load_module(
    "hard_adjustment_grasp_evaluator",
    ROOT / "isaacsimenvs/tasks/simtoolreal/utils/grasp_evaluator.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grasp-bank", type=Path,
        default=ROOT / "assets/grasp_banks/eraser_canonical_v2.json",
    )
    parser.add_argument("--object-urdf", type=Path, default=DEFAULT_OBJECT_URDF)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--entries", type=int, default=64)
    parser.add_argument(
        "--entries-per-asset", type=int, default=0,
        help="For V3 multi-asset caches, build this many source grasps per asset.",
    )
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--candidates", type=int, default=256)
    parser.add_argument("--nullspace-step-rad", type=float, default=0.04)
    parser.add_argument("--ik-position-tolerance-m", type=float, default=1.0e-3)
    parser.add_argument("--ik-rotation-tolerance-deg", type=float, default=1.0)
    parser.add_argument("--near-limit-min-margin-rad", type=float, default=0.005)
    parser.add_argument("--near-limit-max-margin-rad", type=float, default=0.04)
    parser.add_argument("--near-limit-target-margin-rad", type=float, default=0.01)
    parser.add_argument("--near-singular-max-sigma", type=float, default=0.10)
    parser.add_argument("--minimum-self-clearance-m", type=float, default=-5.0e-4)
    parser.add_argument("--recovery-min-score", type=float, default=0.90)
    parser.add_argument("--hard-max-score", type=float, default=0.80)
    parser.add_argument("--future-attempts", type=int, default=32)
    parser.add_argument("--dls-damping", type=float, default=0.03)
    parser.add_argument("--arm-velocity-limit-rad-s", type=float, default=10.0)
    parser.add_argument("--joint-margin-target-rad", type=float, default=0.10)
    parser.add_argument("--singular-value-target", type=float, default=0.10)
    parser.add_argument("--condition-number-limit", type=float, default=20.0)
    parser.add_argument("--future-horizon-s", type=float, default=2.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pose_delta(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    translation = float(np.linalg.norm(first[:3, 3] - second[:3, 3]))
    rotation = math.degrees(float(np.linalg.norm(
        Rotation.from_matrix(first[:3, :3] @ second[:3, :3].T).as_rotvec()
    )))
    return translation, rotation


def candidate_metrics(kinematics, q: np.ndarray, hand: np.ndarray, base: np.ndarray) -> tuple:
    palm, jacobian, links = kinematics.palm_fk_jacobian(q, hand, base)
    singular = np.linalg.svd(jacobian, compute_uv=False)
    margin = float(np.minimum(q - kinematics.arm_lower, kinematics.arm_upper - q).min())
    return (
        palm, jacobian, links, margin, float(singular[-1]),
        float(singular[0] / max(singular[-1], 1e-12)),
    )


def valid_candidate(
    *, kinematics, q, hand, base, target_palm, original_tool,
    minimum_self_clearance_m, ik_position_tolerance_m, ik_rotation_tolerance_deg,
) -> tuple | None:
    palm, jacobian, links, margin, sigma, condition = candidate_metrics(
        kinematics, q, hand, base
    )
    palm_translation, palm_rotation = pose_delta(palm, target_palm)
    if (palm_translation > ik_position_tolerance_m
            or palm_rotation > ik_rotation_tolerance_deg):
        return None
    clearance = ge.sphere_self_clearance(kinematics, links, SPHERES)
    if clearance < minimum_self_clearance_m:
        return None
    position = original_tool[:3, 3]
    if not (-0.75 <= position[0] <= 0.75 and -0.45 <= position[1] <= 0.85 and 0.20 <= position[2] <= 1.20):
        return None
    return (
        q.copy(), original_tool.copy(), margin, sigma, condition, clearance,
        palm_translation, palm_rotation, jacobian,
    )


def project_to_fixed_palm(
    seed: np.ndarray, *, kinematics, hand: np.ndarray, base: np.ndarray,
    target_palm: np.ndarray,
) -> np.ndarray | None:
    q = np.clip(
        seed, kinematics.arm_lower + 1.0e-5, kinematics.arm_upper - 1.0e-5
    ).copy()
    for _ in range(20):
        palm, jacobian, _ = kinematics.palm_fk_jacobian(q, hand, base)
        translation = target_palm[:3, 3] - palm[:3, 3]
        rotation = Rotation.from_matrix(
            target_palm[:3, :3] @ palm[:3, :3].T
        ).as_rotvec()
        error = np.concatenate((translation, rotation))
        if np.linalg.norm(translation) <= 2.5e-4 and math.degrees(
            np.linalg.norm(rotation)
        ) <= 0.25:
            return q
        regularizer = 1.0e-6 * np.eye(6)
        correction = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + regularizer, error
        )
        correction_norm = float(np.linalg.norm(correction))
        if correction_norm > 0.20:
            correction *= 0.20 / correction_norm
        q = np.clip(
            q + correction,
            kinematics.arm_lower + 1.0e-5,
            kinematics.arm_upper - 1.0e-5,
        )
    return None


def fixed_pose_candidates(*, entry: dict, kinematics, base, args) -> tuple[tuple, list[tuple]]:
    original_q = np.asarray(entry["joint_pos_canonical"][:7], dtype=np.float64)
    hand = np.asarray(entry["joint_pos_canonical"][7:], dtype=np.float64)
    original_tool = ge.pose_matrix(
        np.asarray(entry["object_pos_local"]), np.asarray(entry["object_quat_wxyz"])
    )
    target_palm, _, _ = kinematics.palm_fk_jacobian(original_q, hand, base)
    validation = dict(
        kinematics=kinematics, hand=hand, base=base, target_palm=target_palm,
        original_tool=original_tool,
        minimum_self_clearance_m=args.minimum_self_clearance_m,
        ik_position_tolerance_m=args.ik_position_tolerance_m,
        ik_rotation_tolerance_deg=args.ik_rotation_tolerance_deg,
    )
    recovery = valid_candidate(q=original_q, **validation)
    if recovery is None:
        raise RuntimeError("verified recovery state failed fixed-pose geometric validation")

    candidates: list[tuple] = []
    steps_per_direction = max(1, args.candidates // 2)
    for direction in (-1.0, 1.0):
        q = original_q.copy()
        previous_null = None
        for _ in range(steps_per_direction):
            _, jacobian, _, _, _, _ = candidate_metrics(kinematics, q, hand, base)
            null = np.linalg.svd(jacobian, full_matrices=True)[2][-1]
            if previous_null is not None and float(np.dot(null, previous_null)) < 0.0:
                null = -null
            previous_null = null
            projected = project_to_fixed_palm(
                q + direction * args.nullspace_step_rad * null,
                kinematics=kinematics, hand=hand, base=base, target_palm=target_palm,
            )
            if projected is None or np.linalg.norm(projected - q) < 1.0e-5:
                break
            q = projected
            candidate = valid_candidate(q=q, **validation)
            if candidate is not None and np.linalg.norm(q - original_q) >= 0.10:
                candidates.append(candidate)
    if not candidates:
        raise RuntimeError("no collision-free alternate IK state preserves the fixed palm/tool pose")
    return recovery, candidates


def controllability_metrics(selected: tuple, future_twist: np.ndarray, args) -> dict:
    _, _, margin, sigma, condition, _, _, _, jacobian = selected
    regularizer = args.dls_damping ** 2 * np.eye(6)
    solved = np.linalg.solve(jacobian @ jacobian.T + regularizer, future_twist)
    qdot = jacobian.T @ solved
    velocity_ratio = float(np.max(np.abs(qdot) / args.arm_velocity_limit_rad_s))
    joint_factor = np.clip(margin / args.joint_margin_target_rad, 0.0, 1.0)
    singular_factor = np.clip(sigma / args.singular_value_target, 0.0, 1.0)
    condition_factor = np.clip(args.condition_number_limit / max(condition, 1.0), 0.0, 1.0)
    motion_factor = math.exp(-max(velocity_ratio - 1.0, 0.0))
    score = float((joint_factor * singular_factor * condition_factor * motion_factor) ** 0.25)
    return {
        "score": score,
        "joint_margin_rad": float(margin),
        "minimum_jacobian_singular_value": float(sigma),
        "jacobian_condition_number": float(condition),
        "future_velocity_ratio": velocity_ratio,
    }


def sample_hard_candidate(
    *, entry: dict, recovery: tuple, candidates: list[tuple], rng, args,
) -> tuple:
    palm_to_tool = ge.pose_matrix(
        np.asarray(entry["palm_to_tool_pos"]), np.asarray(entry["palm_to_tool_quat_wxyz"])
    )
    eligible = [
        item for item in candidates
        if args.near_limit_min_margin_rad <= item[2] <= args.near_limit_max_margin_rad
    ]
    if not eligible:
        best = min(item[2] for item in candidates)
        raise RuntimeError(
            f"no fixed-pose hard candidate has margin in "
            f"[{args.near_limit_min_margin_rad:.5f}, "
            f"{args.near_limit_max_margin_rad:.5f}] rad; minimum={best:.5f} rad"
        )
    ordered = sorted(
        eligible, key=lambda item: abs(item[2] - args.near_limit_target_margin_rad)
    )

    for _ in range(args.future_attempts):
        future_delta, future_twist = future_motion(
            rng, args.future_horizon_s, recovery[1], palm_to_tool
        )
        twist = np.asarray(future_twist, dtype=np.float64)
        recovery_metrics = controllability_metrics(recovery, twist, args)
        if recovery_metrics["score"] < args.recovery_min_score:
            continue
        for selected in ordered:
            hard_metrics = controllability_metrics(selected, twist, args)
            if hard_metrics["score"] <= args.hard_max_score:
                return selected, future_delta, future_twist, hard_metrics, recovery_metrics
    raise RuntimeError(
        "no future motion separates a recoverable state "
        f"(score>={args.recovery_min_score:.2f}) from a hard state "
        f"(score<={args.hard_max_score:.2f})"
    )


def future_motion(
    rng: np.random.Generator, horizon_s: float, tool: np.ndarray,
    palm_to_tool: np.ndarray,
) -> tuple[list[float], list[float]]:
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    translation = direction * rng.uniform(0.04, 0.10)
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    rotation = axis * rng.uniform(math.radians(5.0), math.radians(20.0))
    delta = np.concatenate((translation, rotation))
    angular_velocity = rotation / horizon_s
    palm = tool @ np.linalg.inv(palm_to_tool)
    palm_linear_velocity = (
        translation / horizon_s
        + np.cross(angular_velocity, palm[:3, 3] - tool[:3, 3])
    )
    palm_twist = np.concatenate((palm_linear_velocity, angular_velocity))
    return delta.tolist(), palm_twist.tolist()


def make_scenario(
    kind: str, source_index: int, entry: dict, selected: tuple,
    future_delta: list[float], future_twist: list[float], initial_score_metrics: dict,
    recovery: tuple, recovery_score_metrics: dict, asset_index: int = 0,
) -> dict:
    q, tool, margin, sigma, condition, clearance, palm_translation, palm_rotation, _ = selected
    position, quaternion = ge.matrix_pose(tool)
    joint_pos = np.asarray(entry["joint_pos_canonical"], dtype=np.float64).copy()
    joint_targets = np.asarray(entry["joint_targets_canonical"], dtype=np.float64).copy()
    joint_pos[:7] = q
    joint_targets[:7] = q
    recovery_joint_pos = np.asarray(entry["joint_pos_canonical"], dtype=np.float64).copy()
    recovery_joint_targets = np.asarray(
        entry["joint_targets_canonical"], dtype=np.float64
    ).copy()
    recovery_joint_pos[:7] = recovery[0]
    recovery_joint_targets[:7] = recovery[0]
    return {
        "source_entry_index": source_index,
        "asset_index": asset_index,
        "scenario_kind": kind,
        "joint_pos_canonical": joint_pos.tolist(),
        "joint_targets_canonical": joint_targets.tolist(),
        "object_pos_local": position.tolist(),
        "object_quat_wxyz": quaternion.tolist(),
        "future_delta": future_delta,
        "future_twist": future_twist,
        "recovery_joint_pos_canonical": recovery_joint_pos.tolist(),
        "recovery_joint_targets_canonical": recovery_joint_targets.tolist(),
        "recovery_metrics": recovery_score_metrics,
        "initial_metrics": {
            **initial_score_metrics,
            "minimum_self_clearance_m": clearance,
            "palm_translation_from_recovery_m": palm_translation,
            "palm_rotation_from_recovery_deg": palm_rotation,
            "arm_distance_from_recovery_rad": float(np.linalg.norm(q - recovery[0])),
        },
    }


def main() -> None:
    args = parse_args()
    if args.entries <= 0 or args.candidates <= 0 or args.future_horizon_s <= 0.0:
        raise ValueError("entries, candidates, and future horizon must be positive")
    if args.nullspace_step_rad <= 0.0 or args.future_attempts <= 0:
        raise ValueError("nullspace step and future attempts must be positive")
    if not 0.0 < args.near_limit_min_margin_rad <= args.near_limit_target_margin_rad:
        raise ValueError("near-limit minimum margin must be in (0, target margin]")
    if args.near_limit_target_margin_rad > args.near_limit_max_margin_rad:
        raise ValueError("near-limit target margin must not exceed maximum margin")
    if not 0.0 < args.hard_max_score < args.recovery_min_score <= 1.0:
        raise ValueError("scores must satisfy 0 < hard-max < recovery-min <= 1")
    for path in (args.grasp_bank, ROBOT_URDF, SPHERES):
        if not path.is_file():
            raise FileNotFoundError(path)
    bank = json.loads(args.grasp_bank.read_text())
    multi_asset = int(bank.get("schema_version", -1)) == 3
    if multi_asset:
        if args.entries_per_asset <= 0:
            raise ValueError("V3 caches require --entries-per-asset")
        source_records = []
        source_index = 0
        for asset in bank.get("assets", []):
            entries = asset.get("entries", [])
            if len(entries) < args.entries_per_asset:
                raise ValueError(
                    f"asset {asset.get('asset_index')} has {len(entries)} entries; "
                    f"requested={args.entries_per_asset}"
                )
            for entry in entries:
                source_records.append((source_index, int(asset["asset_index"]), entry))
                source_index += 1
    else:
        if not args.object_urdf.is_file():
            raise FileNotFoundError(args.object_urdf)
        entries = bank.get("entries", [])
        if args.entries > len(entries):
            raise ValueError(f"requested {args.entries} entries from bank of size {len(entries)}")
        source_records = [(index, 0, entry) for index, entry in enumerate(entries)]
    output = args.output or (
        ROOT / "outputs/adjustment_scenarios"
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{bank.get('object_name', 'tool')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    rng = np.random.default_rng(args.seed)
    candidate_ids = rng.permutation(len(source_records))
    kinematics = ge.UrdfKinematics(ROBOT_URDF)
    base = np.eye(4)
    base[1, 3] = 0.8
    scenarios = []
    selected_ids = []
    rejected = []
    asset_count = len(bank.get("assets", [])) if multi_asset else 1
    target_count = (
        asset_count * args.entries_per_asset if multi_asset else args.entries
    )
    selected_per_asset = {asset_index: 0 for asset_index in range(asset_count)}
    for record_index in candidate_ids:
        if len(selected_ids) >= target_count:
            break
        source_index, asset_index, entry = source_records[int(record_index)]
        if multi_asset and selected_per_asset[asset_index] >= args.entries_per_asset:
            continue
        try:
            recovery, fixed_candidates = fixed_pose_candidates(
                entry=entry, kinematics=kinematics, base=base, args=args
            )
            hard = sample_hard_candidate(
                entry=entry, recovery=recovery,
                candidates=fixed_candidates, rng=rng, args=args,
            )
        except RuntimeError as exc:
            rejected.append((int(source_index), str(exc)))
            print(f"[reject] source={source_index} reason={exc}", flush=True)
            continue
        hard_selected, future_delta, future_twist, hard_metrics, recovery_metrics = hard
        nominal_metrics = controllability_metrics(
            recovery, np.asarray(future_twist, dtype=np.float64), args
        )
        scenarios.append(make_scenario(
            "nominal", int(source_index), entry, recovery, future_delta, future_twist,
            nominal_metrics, recovery, recovery_metrics, asset_index=asset_index,
        ))
        scenarios.append(make_scenario(
            "fixed_pose_hard", int(source_index), entry, hard_selected,
            future_delta, future_twist, hard_metrics, recovery, recovery_metrics,
            asset_index=asset_index,
        ))
        selected_ids.append(int(source_index))
        selected_per_asset[asset_index] += 1
        print(f"[scenario] grasp={len(selected_ids)}/{target_count} source={source_index}", flush=True)
    if len(selected_ids) != target_count:
        missing = {
            asset_index: args.entries_per_asset - selected
            for asset_index, selected in selected_per_asset.items()
            if selected < args.entries_per_asset
        }
        raise RuntimeError(
            f"only {len(selected_ids)}/{target_count} grasp entries produced both "
            f"scenario kinds; rejected={len(rejected)}; missing_per_asset={missing}"
        )
    payload = {
        "schema_version": 2 if multi_asset else 1,
        "kind": "simtoolreal_adjustment_scenarios",
        "generation_mode": "fixed_pose_recoverable_v2",
        "object_name": bank.get("object_name", "eraser"),
        "grasp_bank": str(args.grasp_bank.resolve()),
        "grasp_bank_sha256": sha256(args.grasp_bank),
        "object_urdf": str(args.object_urdf.resolve()) if not multi_asset else None,
        "object_urdf_sha256": sha256(args.object_urdf) if not multi_asset else None,
        "asset_sha256": (
            [asset["asset_sha256"] for asset in bank["assets"]]
            if multi_asset else [sha256(args.object_urdf)]
        ),
        "robot_urdf_sha256": sha256(ROBOT_URDF),
        "seed": args.seed,
        "source_entry_ids": selected_ids,
        "rejected_sources": [
            {"source_entry_index": source, "reason": reason}
            for source, reason in rejected
        ],
        "config": vars(args) | {"grasp_bank": str(args.grasp_bank), "object_urdf": str(args.object_urdf), "output": str(output)},
        "provenance": {"code_commit": subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()},
        "scenarios": scenarios,
    }
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"[output] {output.resolve()} scenarios={len(scenarios)}", flush=True)


if __name__ == "__main__":
    main()
