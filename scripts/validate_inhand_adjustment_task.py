#!/usr/bin/env python3
"""Run a short finite-value smoke test of the in-hand adjustment task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasp-bank", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--num-envs", type=int, default=12)
    parser.add_argument("--curriculum-stage", type=int, default=0)
    parser.add_argument("--screwdriver-axial", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    return parser.parse_args()


ARGS = parse_args()
APP = AppLauncher(ARGS).app


import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaacsimenvs  # noqa: E402,F401
from isaacsimenvs.tasks.simtoolreal.simtoolreal_tacmap_env_cfg import (  # noqa: E402
    SimToolRealInHandAdjustmentEnvCfg,
    SimToolRealScrewdriverAxialAdjustmentEnvCfg,
)
from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import compute_obs_dim  # noqa: E402


def main() -> None:
    if ARGS.steps <= 0 or ARGS.num_envs <= 0:
        raise ValueError("steps and num-envs must be positive")
    bank_payload = json.loads(ARGS.grasp_bank.read_text())
    cfg = (
        SimToolRealScrewdriverAxialAdjustmentEnvCfg()
        if ARGS.screwdriver_axial else SimToolRealInHandAdjustmentEnvCfg()
    )
    cfg.grasp_bank_path = str(ARGS.grasp_bank.resolve())
    cfg.scene.num_envs = ARGS.num_envs
    if int(bank_payload["schema_version"]) == 3:
        cfg.assets.object_urdf = ""
        procedural = bank_payload.get("procedural", {})
        cfg.assets.handle_head_types = tuple(procedural.get(
            "tool_types", ("hammer", "screwdriver", "marker", "spatula", "eraser", "brush")
        ))
        cfg.assets.num_assets_per_type = int(procedural.get("assets_per_distribution", 20))
        cfg.assets.procedural_asset_seed = int(procedural.get("asset_seed", 42))
        cfg.assets.shuffle_assets = True
        cfg.assets.object_pool_limit = 0
        cfg.grasp_bank_min_entries = 2
    else:
        cfg.grasp_bank_min_entries = 1
    cfg.enable_tool_table_contact_sensor = False
    task_id = (
        "Isaacsimenvs-SimToolReal-Screwdriver-Axial-Adjustment-Direct-v0"
        if ARGS.screwdriver_axial
        else "Isaacsimenvs-SimToolReal-InHand-Adjustment-Direct-v0"
    )
    env = gym.make(task_id, cfg=cfg)
    try:
        stage_count = len(cfg.adjustment_finger_perturb_fractions)
        if not 0 <= ARGS.curriculum_stage < stage_count:
            raise ValueError(f"curriculum stage must be in [0, {stage_count})")
        env.unwrapped._adjustment_curriculum_stage = int(ARGS.curriculum_stage)
        observation, _ = env.reset()
        expected_policy_shape = (ARGS.num_envs, compute_obs_dim(cfg.obs.obs_list))
        expected_critic_shape = (ARGS.num_envs, compute_obs_dim(cfg.obs.state_list))
        if observation["policy"].shape != expected_policy_shape:
            raise RuntimeError(f"unexpected policy observation shape: {observation['policy'].shape}")
        if observation["critic"].shape != expected_critic_shape:
            raise RuntimeError(f"unexpected critic observation shape: {observation['critic'].shape}")
        inner = env.unwrapped
        for step in range(ARGS.steps):
            actions = inner._stable_previous_action.clone()
            observation, reward, terminated, truncated, _ = env.step(actions)
            tensors = (observation["policy"], observation["critic"], reward)
            if not all(torch.isfinite(value).all() for value in tensors):
                raise RuntimeError(f"non-finite task output at step {step}")
        print(
            f"[pass] envs={ARGS.num_envs} steps={ARGS.steps} "
            f"stage={ARGS.curriculum_stage} "
            f"obs={tuple(observation['policy'].shape)} "
            f"relative_position_error_m="
            f"{inner._adjustment_relative_position_error.mean().item():.4f} "
            f"relative_rotation_error_deg="
            f"{torch.rad2deg(inner._adjustment_relative_rotation_error).mean().item():.2f} "
            f"target_axial_m={inner._adjustment_target_axial_translation.mean().item():.4f} "
            f"target_perpendicular_m="
            f"{inner._adjustment_target_perpendicular_translation.mean().item():.4f} "
            f"terminated={int(terminated.sum().item())} "
            f"truncated={int(truncated.sum().item())}",
            flush=True,
        )
    finally:
        env.close()
        APP.close()


if __name__ == "__main__":
    main()
