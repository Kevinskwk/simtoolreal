#!/usr/bin/env python3
"""Evaluate eraser grasp-bank entries on reproducible future scrape trajectories."""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOT_URDF = REPO_ROOT / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
ERASER_URDF = REPO_ROOT / "assets/urdf/objects/eraser_tactile_canonical.urdf"
SPHERES = REPO_ROOT / "baselines/assets/sharpa_spheres.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evaluator = load_module(
    "grasp_evaluator_pilot",
    REPO_ROOT / "isaacsimenvs/tasks/simtoolreal/utils/grasp_evaluator.py",
)
scrape_utils = load_module(
    "scrape_pose_utils_pilot",
    REPO_ROOT / "isaacsimenvs/tasks/simtoolreal/utils/scrape_pose_utils.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grasp-bank", type=Path,
        default=REPO_ROOT / "assets/grasp_banks/eraser_canonical_v2.json",
    )
    parser.add_argument("--entries", type=int, default=16)
    parser.add_argument("--trajectories-per-grasp", type=int, default=8)
    parser.add_argument("--contact-keyframes", type=int, default=8)
    parser.add_argument("--transition-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--dt", type=float, default=1.0 / 60.0)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def validate_args(args: argparse.Namespace, bank: dict) -> None:
    for name in ("entries", "trajectories_per_grasp", "contact_keyframes", "transition_steps"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.entries > len(bank["entries"]):
        raise ValueError(f"requested {args.entries} entries from a {len(bank['entries'])}-entry bank")
    if args.contact_keyframes < 2 or args.transition_steps < 2:
        raise ValueError("pilot needs at least two keyframes and two transition steps")
    if not math.isfinite(args.dt) or args.dt <= 0.0:
        raise ValueError("--dt must be positive")


def normalized(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 1e-9:
        raise ValueError("cannot normalize a zero or non-finite vector")
    return vector / norm


def edge_contact_pose(
    *, table_top: np.ndarray, table_normal: np.ndarray, yaw: float, tilt: float,
    offset_xy: np.ndarray, contact_edge_local: np.ndarray,
) -> np.ndarray:
    candidate = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    edge = normalized(candidate - np.dot(candidate, table_normal) * table_normal)
    forward = normalized(np.cross(edge, table_normal))
    local_x = math.cos(tilt) * forward - math.sin(tilt) * table_normal
    local_z = math.sin(tilt) * forward + math.cos(tilt) * table_normal
    rotation = np.column_stack((local_x, edge, local_z))
    anchor = table_top + forward * offset_xy[0] + edge * offset_xy[1]
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = anchor - rotation @ contact_edge_local
    return transform


def interpolate_poses(keyframes: list[np.ndarray], transition_steps: int) -> np.ndarray:
    poses: list[np.ndarray] = [keyframes[0]]
    fractions = np.linspace(0.0, 1.0, transition_steps + 1)[1:]
    for start, end in zip(keyframes[:-1], keyframes[1:], strict=True):
        slerp = Slerp(
            [0.0, 1.0], Rotation.from_matrix(np.stack((start[:3, :3], end[:3, :3])))
        )
        rotations = slerp(fractions).as_matrix()
        positions = (
            (1.0 - fractions[:, None]) * start[:3, 3]
            + fractions[:, None] * end[:3, 3]
        )
        for position, rotation in zip(positions, rotations, strict=True):
            pose = np.eye(4)
            pose[:3, :3] = rotation
            pose[:3, 3] = position
            poses.append(pose)
    return np.stack(poses)


def sample_trajectory(
    entry: dict, rng: np.random.Generator, bounds: tuple[float, ...],
    contact_keyframes: int, transition_steps: int,
) -> tuple[np.ndarray, dict]:
    roll, pitch = rng.uniform(-math.radians(8.0), math.radians(8.0), size=2)
    table_rotation = Rotation.from_euler("xy", [roll, pitch]).as_matrix()
    table_normal = table_rotation[:, 2]
    table_root = np.array([0.0, 0.0, rng.uniform(0.36, 0.38)])
    table_top = table_root + table_normal * scrape_utils.TABLE_HALF_HEIGHT
    yaw = float(entry["reference_edge_yaw_rad"]) + rng.uniform(
        -math.radians(5.0), math.radians(5.0)
    )
    reference_tilt = float(entry["reference_edge_tilt_rad"])
    x_tip, y_center, z_contact = bounds[3], 0.5 * (bounds[1] + bounds[4]), bounds[2]
    contact_edge_local = np.array([x_tip, y_center, z_contact])
    offset = rng.uniform(-0.025, 0.025, size=2)
    keyframes = [evaluator.pose_matrix(
        np.asarray(entry["object_pos_local"]), np.asarray(entry["object_quat_wxyz"])
    )]
    contact_poses: list[np.ndarray] = []
    for _ in range(contact_keyframes):
        offset = np.clip(offset + rng.uniform(-0.012, 0.012, size=2), -0.04, 0.04)
        tilt = np.clip(
            reference_tilt + rng.uniform(-math.radians(5.0), math.radians(5.0)),
            math.radians(10.0), math.radians(70.0),
        )
        contact_poses.append(edge_contact_pose(
            table_top=table_top,
            table_normal=table_normal,
            yaw=yaw,
            tilt=float(tilt),
            offset_xy=offset,
            contact_edge_local=contact_edge_local,
        ))
    keyframes.extend(contact_poses)
    trajectory = interpolate_poses(keyframes, transition_steps)
    metadata = {
        "table_root_position": table_root.tolist(),
        "table_quaternion_wxyz": evaluator.matrix_pose(
            np.block([[table_rotation, table_root[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]])
        )[1].tolist(),
        "table_top_point": table_top.tolist(),
        "table_normal": table_normal.tolist(),
        "table_rotation_matrix": table_rotation.tolist(),
        "edge_yaw_rad": yaw,
        "contact_edge_local": contact_edge_local.tolist(),
        "contact_keyframe_poses": [
            [*evaluator.matrix_pose(pose)[0], *evaluator.matrix_pose(pose)[1]]
            for pose in contact_poses
        ],
    }
    return trajectory, metadata


def main() -> None:
    args = parse_args()
    if not args.grasp_bank.is_file():
        raise FileNotFoundError(f"grasp bank does not exist: {args.grasp_bank}")
    bank = json.loads(args.grasp_bank.read_text())
    validate_args(args, bank)
    output_dir = args.output_dir or (
        REPO_ROOT / "outputs/grasp_evaluator"
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_eraser_pilot"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    bounds = scrape_utils.load_urdf_collision_bounds(ERASER_URDF)
    kinematics = evaluator.UrdfKinematics(ROBOT_URDF)
    base = np.eye(4)
    base[1, 3] = 0.8
    rng = np.random.default_rng(args.seed)
    selected_ids = np.sort(rng.choice(len(bank["entries"]), size=args.entries, replace=False))
    results: list[dict] = []
    for ordinal, entry_id in enumerate(selected_ids, start=1):
        entry = bank["entries"][int(entry_id)]
        palm_to_tool = evaluator.pose_matrix(
            np.asarray(entry["palm_to_tool_pos"]),
            np.asarray(entry["palm_to_tool_quat_wxyz"]),
        )
        for trajectory_id in range(args.trajectories_per_grasp):
            trajectory, metadata = sample_trajectory(
                entry, rng, bounds, args.contact_keyframes, args.transition_steps
            )
            evaluation = evaluator.evaluate_grasp_trajectory(
                kinematics=kinematics,
                sphere_path=SPHERES,
                joint_positions=np.asarray(entry["joint_pos_canonical"]),
                palm_to_tool=palm_to_tool,
                tool_trajectory=trajectory,
                dt=args.dt,
                table_transform=np.block([
                    [np.asarray(metadata["table_rotation_matrix"]), np.asarray(metadata["table_root_position"])[:, None]],
                    [np.zeros((1, 3)), np.ones((1, 1))],
                ]),
                table_extent=np.array([0.475, 0.4, 0.3]),
                tool_bounds=bounds,
                base_transform=base,
            )
            tool_poses = []
            for pose in trajectory:
                position, quaternion = evaluator.matrix_pose(pose)
                tool_poses.append([*position.tolist(), *quaternion.tolist()])
            results.append({
                "grasp_entry_id": int(entry_id),
                "grasp_fingerprint": evaluator.grasp_fingerprint(entry),
                "trajectory_id": trajectory_id,
                "trajectory_seed": args.seed,
                "trajectory": metadata,
                "tool_poses_wxyz": tool_poses,
                "computed": evaluation.to_dict(),
            })
        print(f"[computed] grasp={ordinal}/{len(selected_ids)} bank_id={entry_id}", flush=True)
    gate_counts: dict[str, int] = {}
    for result in results:
        for name, passed in result["computed"]["gates"].items():
            gate_counts[name] = gate_counts.get(name, 0) + int(bool(passed))
    payload = {
        "schema_version": 1,
        "kind": "eraser_grasp_evaluator_pilot",
        "config": {
            "grasp_bank": str(args.grasp_bank.resolve()),
            "entries": args.entries,
            "trajectories_per_grasp": args.trajectories_per_grasp,
            "contact_keyframes": args.contact_keyframes,
            "transition_steps": args.transition_steps,
            "seed": args.seed,
            "dt": args.dt,
            "thresholds": evaluator.GraspEvaluatorThresholds().__dict__,
        },
        "selected_entry_ids": selected_ids.tolist(),
        "computed_gate_pass_counts": gate_counts,
        "result_count": len(results),
        "provenance": {
            "code_commit": subprocess.run(
                ("git", "rev-parse", "HEAD"), cwd=REPO_ROOT,
                check=True, capture_output=True, text=True,
            ).stdout.strip(),
        },
        "results": results,
    }
    output_path = output_dir / "pilot_results.json"
    output_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"[summary] results={len(results)} gate_pass_counts={gate_counts}", flush=True)
    print(f"[output] {output_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
