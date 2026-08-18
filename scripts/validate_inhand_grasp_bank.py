#!/usr/bin/env python3
"""Replay in-hand grasp-bank states and fail if compliant holds are unreliable."""

from __future__ import annotations

import argparse
import json
import math
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resets", type=int, default=1000)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--hold-steps", type=int, default=30)
    parser.add_argument("--minimum-pass-rate", type=float, default=0.95)
    parser.add_argument(
        "--grasp-bank", type=Path,
        default=Path("assets/grasp_banks/eraser_canonical_v2.json"),
    )
    parser.add_argument("--write-filtered-bank", type=Path)
    parser.add_argument("--minimum-filtered-entries", type=int, default=64)
    parser.add_argument("--curriculum-stage", type=int, default=0)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    args = parser.parse_args()
    args.enable_cameras = False
    return args


ARGS = parse_args()
APP = AppLauncher(ARGS).app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402

import isaacsimenvs  # noqa: E402,F401
from isaacsimenvs.tasks.simtoolreal.simtoolreal_tacmap_env_cfg import (  # noqa: E402
    SimToolRealInHandStableScrapeEnvCfg,
)
from isaacsimenvs.tasks.simtoolreal.utils.inhand_grasp_bank import (  # noqa: E402
    sha256_file,
    validate_grasp_bank,
)
from dextoolbench.objects import NAME_TO_OBJECT  # noqa: E402


TASK_ID = "Isaacsimenvs-SimToolReal-Stable-Scrape-InHand-Direct-v0"
REPO_ROOT = Path(__file__).resolve().parents[1]
OBJECT_TO_CATEGORY = {
    "mallet_hammer": "hammer", "claw_hammer": "hammer",
    "long_screwdriver": "screwdriver", "short_screwdriver": "screwdriver",
    "handle_eraser": "eraser", "flat_eraser": "eraser",
    "flat_spatula": "spatula", "spoon_spatula": "spatula",
    "sharpie_marker": "marker", "staples_marker": "marker",
    "red_brush": "brush", "blue_brush": "brush",
}


def validate_args() -> None:
    if ARGS.resets <= 0 or ARGS.num_envs <= 0 or ARGS.hold_steps <= 0:
        raise ValueError("resets, num-envs, and hold-steps must be positive")
    if not 0.0 < ARGS.minimum_pass_rate <= 1.0:
        raise ValueError("minimum-pass-rate must be in (0, 1]")
    if ARGS.minimum_filtered_entries <= 0:
        raise ValueError("minimum-filtered-entries must be positive")
    if ARGS.curriculum_stage < 0:
        raise ValueError("curriculum-stage must be non-negative")


def palm_tool_relative(inner) -> tuple[torch.Tensor, torch.Tensor]:
    return subtract_frame_transforms(
        inner.robot.data.body_link_pos_w[:, inner._palm_body_id],
        inner.robot.data.body_link_quat_w[:, inner._palm_body_id],
        inner.object.data.root_pos_w,
        inner.object.data.root_quat_w,
    )


def hold_action(inner) -> torch.Tensor:
    targets = inner._cur_targets[:, inner._perm_lab_to_canon]
    action = torch.zeros(inner.num_envs, inner.cfg.action_space, device=inner.device)
    action[:, 7:] = 2.0 * (
        targets[:, 7:] - inner._joint_lower_canon[7:]
    ) / (
        inner._joint_upper_canon[7:] - inner._joint_lower_canon[7:]
    ) - 1.0
    return action.clamp(-1.0, 1.0)


def run() -> None:
    validate_args()
    if not ARGS.grasp_bank.is_file():
        raise FileNotFoundError(f"grasp bank does not exist: {ARGS.grasp_bank}")
    payload = validate_grasp_bank(json.loads(ARGS.grasp_bank.read_text()))
    object_name = payload.get("object_name", "eraser_canonical")
    if object_name == "eraser_canonical":
        object_urdf = REPO_ROOT / "assets/urdf/objects/eraser_tactile_canonical.urdf"
        object_scale = (2.9373215824170767, 0.5126639800346792, 1.2951200580119278)
    else:
        if object_name not in NAME_TO_OBJECT:
            raise ValueError(f"unknown DexToolBench object_name {object_name!r}")
        expected_category = OBJECT_TO_CATEGORY[object_name]
        if payload["tool_type"] != expected_category:
            raise ValueError(
                f"bank tool_type {payload['tool_type']!r} does not match "
                f"{object_name!r} category {expected_category!r}"
            )
        tool = NAME_TO_OBJECT[object_name]
        object_urdf = tool.decomposed_urdf_path
        object_scale = tool.scale
    if sha256_file(object_urdf) != payload["asset_sha256"]:
        raise RuntimeError(f"bank asset hash does not match {object_urdf}")
    checkpoint = Path(payload.get("source_checkpoint", ""))
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"bank source checkpoint is unavailable: {checkpoint}"
        )
    if sha256_file(checkpoint) != payload["source_checkpoint_sha256"]:
        raise RuntimeError(f"bank checkpoint hash does not match {checkpoint}")

    cfg = SimToolRealInHandStableScrapeEnvCfg()
    cfg.grasp_bank_path = str(ARGS.grasp_bank.resolve())
    cfg.grasp_bank_source_checkpoint_path = str(checkpoint.resolve())
    cfg.grasp_bank_min_entries = 1
    cfg.assets.handle_head_types = (payload["tool_type"],)
    cfg.assets.object_urdf = str(object_urdf)
    cfg.assets.object_scale = tuple(float(value) for value in object_scale)
    object_root_name = "object_root" if object_name == "eraser_canonical" else object_name
    cfg.tool_table_contact_sensor_prim_path = (
        f"/World/envs/env_.*/Object/{object_root_name}"
    )
    cfg.scene.num_envs = min(int(ARGS.num_envs), int(ARGS.resets))
    env = gym.make(TASK_ID, cfg=cfg)
    inner = env.unwrapped
    if ARGS.curriculum_stage >= len(inner.cfg.inhand_table_angle_stages_deg):
        raise ValueError(
            f"curriculum-stage {ARGS.curriculum_stage} is out of range for "
            f"{len(inner.cfg.inhand_table_angle_stages_deg)} stages"
        )
    inner._inhand_curriculum_stage = int(ARGS.curriculum_stage)

    attempted = 0
    passed = 0
    maximum_drift = 0.0
    maximum_rotation = 0.0
    maximum_force = 0.0
    minimum_initial_clearance = float("inf")
    minimum_target_clearance = float("inf")
    attempts_by_entry = torch.zeros(inner._inhand_bank_size, dtype=torch.long)
    passes_by_entry = torch.zeros(inner._inhand_bank_size, dtype=torch.long)

    for batch_index in range(math.ceil(ARGS.resets / inner.num_envs)):
        env.reset()
        active_count = min(inner.num_envs, ARGS.resets - attempted)
        active = torch.arange(active_count, device=inner.device)
        # Cover every bank entry before repeating any. On later passes, rotate
        # the entry-to-slot assignment so cloned-environment artifacts cannot be
        # hidden by one fixed pairing.
        sample_ids = torch.arange(
            attempted, attempted + active_count, device=inner.device
        )
        replay_cycle = torch.div(
            sample_ids, inner._inhand_bank_size, rounding_mode="floor"
        )
        bank_ids = (
            sample_ids % inner._inhand_bank_size + 37 * replay_cycle
        ) % inner._inhand_bank_size
        inner._restore_inhand_state(active, bank_ids)
        reference_pos = inner._stable_relative_pos.clone()
        reference_quat = inner._stable_relative_quat.clone()
        failed = torch.zeros(active_count, dtype=torch.bool, device=inner.device)
        batch_drift = torch.zeros(active_count, device=inner.device)
        batch_rotation = torch.zeros(active_count, device=inner.device)
        batch_min_support = torch.full(
            (active_count,), 5, dtype=torch.long, device=inner.device
        )

        minimum_initial_clearance = min(
            minimum_initial_clearance,
            float(inner._inhand_initial_min_box_clearance[active].min().item()),
        )
        minimum_target_clearance = min(
            minimum_target_clearance,
            float(inner._inhand_target_min_box_clearance[active].min().item()),
        )

        for _ in range(int(ARGS.hold_steps)):
            _, _, terminated, truncated, _ = env.step(hold_action(inner))
            relative_pos, relative_quat = palm_tool_relative(inner)
            drift = torch.linalg.vector_norm(
                relative_pos[active] - reference_pos[active], dim=-1
            )
            rotation = torch.rad2deg(2.0 * torch.acos(
                torch.abs(
                    (relative_quat[active] * reference_quat[active]).sum(-1)
                ).clamp(0.0, 1.0)
            ))
            support = (inner._curr_fingertip_distances[active] < 0.12).sum(-1)
            force = inner._scrape_table_normal_force_interval[active]
            if not bool(torch.isfinite(force).all()):
                raise RuntimeError("contact sensor produced NaN or Inf during replay")
            batch_drift = torch.maximum(batch_drift, drift)
            batch_rotation = torch.maximum(batch_rotation, rotation)
            batch_min_support = torch.minimum(batch_min_support, support)
            failed |= terminated[active] | truncated[active]
            failed |= force > float(inner.cfg.hard_contact_normal_force_limit)
            maximum_force = max(maximum_force, float(force.max().item()))

        failed |= batch_drift > 0.005
        failed |= batch_rotation > 2.0
        failed |= batch_min_support < 2
        passed += int((~failed).sum().item())
        bank_ids_cpu = bank_ids.cpu()
        attempts_by_entry.scatter_add_(
            0, bank_ids_cpu, torch.ones_like(bank_ids_cpu, dtype=torch.long)
        )
        passes_by_entry.scatter_add_(
            0, bank_ids_cpu, (~failed).long().cpu()
        )
        attempted += active_count
        maximum_drift = max(maximum_drift, float(batch_drift.max().item()))
        maximum_rotation = max(maximum_rotation, float(batch_rotation.max().item()))
        print(f"[replay] attempted={attempted} passed={passed}", flush=True)

    robust_entry_ids = (
        (attempts_by_entry > 0) & (attempts_by_entry == passes_by_entry)
    ).nonzero(as_tuple=False).squeeze(-1).tolist()
    env.close()
    pass_rate = passed / attempted
    print(
        "[summary] "
        f"pass_rate={pass_rate:.4f} ({passed}/{attempted}) "
        f"max_drift_m={maximum_drift:.6f} "
        f"max_rotation_deg={maximum_rotation:.3f} "
        f"max_table_force_n={maximum_force:.3f} "
        f"min_initial_clearance_m={minimum_initial_clearance:.6f} "
        f"min_target_clearance_m={minimum_target_clearance:.6f}",
        flush=True,
    )
    print(
        f"[summary] robust_entries={len(robust_entry_ids)}/{inner._inhand_bank_size}",
        flush=True,
    )
    if ARGS.write_filtered_bank is not None:
        if len(robust_entry_ids) < int(ARGS.minimum_filtered_entries):
            raise RuntimeError(
                f"only {len(robust_entry_ids)} replay-robust entries are available; "
                f"need {ARGS.minimum_filtered_entries}"
            )
        payload = json.loads(ARGS.grasp_bank.read_text())
        tactile_min = int(payload["tactile_min_fingers"])
        required_count = math.ceil(
            float(payload["tactile_rich_fraction_min"])
            * int(ARGS.minimum_filtered_entries)
        )
        tactile_ids = [
            index for index in robust_entry_ids
            if int(payload["entries"][index]["verification"]["tactile_finger_count_min"])
            >= tactile_min
        ]
        if len(tactile_ids) < required_count:
            raise RuntimeError(
                f"only {len(tactile_ids)} replay-robust tactile-rich entries are "
                f"available; need {required_count}"
            )
        selected_ids = tactile_ids[:required_count]
        selected_set = set(selected_ids)
        selected_ids.extend(
            index for index in robust_entry_ids if index not in selected_set
        )
        selected_ids = selected_ids[: int(ARGS.minimum_filtered_entries)]
        payload["entries"] = [payload["entries"][index] for index in selected_ids]
        validate_grasp_bank(
            payload, minimum_entries=int(ARGS.minimum_filtered_entries)
        )
        ARGS.write_filtered_bank.parent.mkdir(parents=True, exist_ok=True)
        ARGS.write_filtered_bank.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n"
        )
        print(f"[output] {ARGS.write_filtered_bank.resolve()}", flush=True)
    if ARGS.write_filtered_bank is None and pass_rate < float(ARGS.minimum_pass_rate):
        raise RuntimeError(
            f"grasp-bank replay pass rate {pass_rate:.4f} is below "
            f"required {ARGS.minimum_pass_rate:.4f}"
        )


if __name__ == "__main__":
    try:
        run()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        APP.close()
