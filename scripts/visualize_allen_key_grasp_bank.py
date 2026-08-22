#!/usr/bin/env python3
"""Render every Allen-key bank grasp and every valid directed start/goal pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grasp-bank", type=Path,
        default=ROOT / "assets/grasp_banks/allen_key_manipulation_v2.json",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "outputs/allen_key_grasp_bank_visualization",
    )
    parser.add_argument("--settle-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    return parser.parse_args()


ARGS = parse_args()
APP = AppLauncher(ARGS).app


import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_inv  # noqa: E402

import isaacsimenvs  # noqa: E402,F401
from isaacsimenvs.tasks.simtoolreal.pose_viewer import (  # noqa: E402
    build_pose_viewer_html,
    capture_pose_viewer_frame,
    object_urdf_for_env,
    table_urdf_for_env,
    workpiece_urdf_for_env,
)
from isaacsimenvs.tasks.simtoolreal.simtoolreal_tacmap_env_cfg import (  # noqa: E402
    SimToolRealAllenKeyAdjustmentEnvCfg,
)


TASK_ID = "Isaacsimenvs-SimToolReal-AllenKey-Adjustment-Direct-v0"


def target_palm_pose(inner, source_id: int, target_id: int) -> np.ndarray:
    target_relative_pos = inner._inhand_bank_relative_pos[target_id : target_id + 1]
    target_relative_quat = inner._inhand_bank_relative_quat[target_id : target_id + 1]
    tool_to_palm_quat = quat_inv(target_relative_quat)
    tool_to_palm_pos = quat_apply(tool_to_palm_quat, -target_relative_pos)
    palm_pos, palm_quat = combine_frame_transforms(
        inner.object.data.root_pos_w[source_id : source_id + 1],
        inner.object.data.root_quat_w[source_id : source_id + 1],
        tool_to_palm_pos,
        tool_to_palm_quat,
    )
    origin = inner.scene.env_origins[source_id]
    quat = palm_quat[0].detach().cpu().numpy()
    pose = np.empty(7, dtype=np.float32)
    pose[:3] = (palm_pos[0] - origin).detach().cpu().numpy()
    pose[3:] = quat[[1, 2, 3, 0]]
    return pose


def write_view(inner, frames: list[dict], path: Path) -> None:
    object_text, object_path = object_urdf_for_env(inner, 0)
    table_text, table_path = table_urdf_for_env(inner, 0)
    workpiece_text, workpiece_path = workpiece_urdf_for_env(inner)
    path.write_text(build_pose_viewer_html(
        frames=frames,
        object_urdf_text=object_text,
        table_urdf_text=table_text,
        workpiece_urdf_text=workpiece_text,
        object_urdf_path=object_path,
        table_urdf_path=table_path,
        workpiece_urdf_path=workpiece_path,
    ))


def main() -> None:
    payload = json.loads(ARGS.grasp_bank.read_text())
    entries = payload.get("entries", [])
    if payload.get("tool_type") != "allen_key" or len(entries) < 2:
        raise ValueError("visualization requires an Allen-key bank with at least two entries")
    if ARGS.settle_steps < 0:
        raise ValueError("settle-steps must be non-negative")

    cfg = SimToolRealAllenKeyAdjustmentEnvCfg()
    cfg.seed = int(ARGS.seed)
    cfg.scene.num_envs = len(entries)
    cfg.grasp_bank_path = str(ARGS.grasp_bank.resolve())
    cfg.adjustment_curriculum_min_eligible_count = 1_000_000
    # Preserve the canonical bank poses so each entry is directly comparable.
    cfg.allen_reset_yaw_range_stages_deg = (0.0,) * len(
        cfg.adjustment_target_rotation_deg
    )
    env = gym.make(TASK_ID, cfg=cfg)
    ARGS.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        inner = env.unwrapped
        env_ids = torch.arange(len(entries), device=inner.device)
        inner._restore_inhand_state(env_ids, env_ids)
        inner.episode_length_buf.zero_()
        inner._replay_target_lab_order = inner._inhand_bank_joint_targets[
            :, inner._perm_canon_to_lab
        ]
        actions = inner._inhand_bank_last_action.clone()
        for _ in range(int(ARGS.settle_steps)):
            _, reward, terminated, truncated, _ = env.step(actions)
            if not bool(torch.isfinite(reward).all()):
                raise RuntimeError("non-finite reward while settling bank visualization")
            if bool(terminated.any()) or bool(truncated.any()):
                raise RuntimeError("episode ended while settling bank visualization")

        entry_frames = []
        entry_index = []
        for entry_id in range(len(entries)):
            frame = capture_pose_viewer_frame(inner, entry_id)
            frame["target_palm_pose"] = target_palm_pose(inner, entry_id, entry_id)
            entry_frames.append(frame)
            entry_index.append({"frame": entry_id, "grasp_id": entry_id})

        pair_frames = []
        pair_index = []
        for source_id in range(len(entries)):
            targets = torch.nonzero(
                inner._allen_target_pair_valid[source_id], as_tuple=False
            ).squeeze(-1).tolist()
            for target_id in targets:
                frame = capture_pose_viewer_frame(inner, source_id)
                frame["target_palm_pose"] = target_palm_pose(
                    inner, source_id, int(target_id)
                )
                pair_index.append({
                    "frame": len(pair_frames),
                    "start_grasp_id": source_id,
                    "goal_grasp_id": int(target_id),
                    "translation_m": float(
                        inner._allen_pair_translation_m[source_id, target_id].item()
                    ),
                    "rotation_deg": float(
                        inner._allen_pair_rotation_deg[source_id, target_id].item()
                    ),
                })
                pair_frames.append(frame)

        entries_path = ARGS.output_dir / "allen_key_bank_entries.html"
        pairs_path = ARGS.output_dir / "allen_key_start_goal_pairs.html"
        index_path = ARGS.output_dir / "frame_index.json"
        write_view(inner, entry_frames, entries_path)
        write_view(inner, pair_frames, pairs_path)
        index_path.write_text(json.dumps({
            "legend": {
                "articulated_hand": "initial/start grasp",
                "cyan_marker": "current palm frame",
                "magenta_marker": "goal palm frame",
            },
            "entries": entry_index,
            "start_goal_pairs": pair_index,
        }, indent=2))
        print(
            f"[pass] rendered {len(entry_frames)} bank entries and "
            f"{len(pair_frames)} directed start/goal pairs\n"
            f"  entries: {entries_path.resolve()}\n"
            f"  pairs:   {pairs_path.resolve()}\n"
            f"  index:   {index_path.resolve()}",
            flush=True,
        )
    finally:
        env.close()
        APP.close()


if __name__ == "__main__":
    main()
