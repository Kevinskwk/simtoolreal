#!/usr/bin/env python3
"""Smoke-test, audit tactile signal, or evaluate fixed-grasp force RL."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "tactile-audit", "evaluate"), default="smoke")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument(
        "--policy-config",
        default="",
        help="Resolved Hydra config; inferred from the checkpoint run when omitted.",
    )
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="")
    parser.add_argument("--minimum-tactile-std", type=float, default=1.0e-4)
    parser.add_argument("--minimum-predictive-r2", type=float, default=0.1)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True)
    return parser.parse_args()


ARGS = parse_args()
APP = AppLauncher(ARGS).app


import gymnasium as gym  # noqa: E402
import isaacsimenvs  # noqa: E402,F401
from gym import spaces  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from rl_games.common import env_configurations  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaacsimenvs.tasks.simtoolreal.simtoolreal_tacmap_env_cfg import (  # noqa: E402
    SimToolRealFixedGraspNormalForceEnvCfg,
)


TASK_ID = "Isaacsimenvs-SimToolReal-TacMap-FixedGrasp-NormalForce-Direct-v0"


def infer_resolved_policy_config(checkpoint: Path) -> Path:
    if ARGS.policy_config:
        path = Path(ARGS.policy_config).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Policy config does not exist: {path}")
        return path
    for parent in checkpoint.parents:
        candidate = parent / ".hydra/config.yaml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not infer .hydra/config.yaml from checkpoint; pass --policy-config."
    )


class CheckpointPlayer:
    """Minimal rl_games player using the exact resolved training configuration."""

    def __init__(self, env, checkpoint: Path) -> None:
        inner = env.unwrapped
        self.num_observations = int(inner.cfg.observation_space)
        self.num_actions = 1
        self.num_envs = inner.num_envs
        self.device = str(inner.device)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_observations,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_actions,), dtype=np.float32
        )
        self.set_env_state = lambda *args, **kwargs: None

        config_path = infer_resolved_policy_config(checkpoint)
        resolved = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        if "agent" in resolved:
            config = resolved["agent"]
        elif "train" in resolved:
            config = resolved["train"]
        elif "params" in resolved:
            config = resolved
        else:
            raise ValueError(
                f"Resolved policy config has no agent/train/params section: {config_path}"
            )
        config["params"]["config"]["device"] = self.device
        config["params"]["config"]["device_name"] = self.device

        checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        checkpoint_data = checkpoint_data.get(0, checkpoint_data)
        model = checkpoint_data.get("model")
        if not isinstance(model, dict):
            raise RuntimeError(f"Checkpoint has no model state dict: {checkpoint}")
        group_counts = {
            int(value.shape[0])
            for key, value in model.items()
            if key.endswith(("extra_params", "sigma")) and value.ndim >= 2
        }
        if len(group_counts) != 1:
            raise RuntimeError(
                "Could not infer one exploration-coefficient group count; "
                f"found {sorted(group_counts)}"
            )
        config["params"]["config"]["expl_coef_num_ids"] = group_counts.pop()

        env_configurations.register(
            "rlgpu", {"env_creator": lambda **kwargs: self, "vecenv_type": "RLGPU"}
        )
        runner = Runner()
        runner.load(config)
        self.player = runner.create_player()
        self.player.init_rnn()
        self.player.has_batch_dimension = True
        self.player.restore(str(checkpoint))

    def get_action(self, observation: torch.Tensor) -> torch.Tensor:
        expected = (self.num_envs, self.num_observations)
        if observation.shape != expected:
            raise RuntimeError(
                f"Policy observation must have shape {expected}, got {tuple(observation.shape)}"
            )
        coefficient = torch.zeros(self.num_envs, 1, device=self.device)
        action = self.player.get_action(
            torch.cat((observation, coefficient), dim=-1), is_deterministic=True
        ).reshape(self.num_envs, self.num_actions)
        if not torch.isfinite(action).all():
            raise RuntimeError("Checkpoint policy produced NaN or Inf actions.")
        return action

    def reset(self) -> None:
        self.player.reset()


def make_env(feedback_mode: str):
    cfg = SimToolRealFixedGraspNormalForceEnvCfg()
    cfg.seed = ARGS.seed
    cfg.scene.num_envs = ARGS.num_envs
    cfg.feedback_mode = feedback_mode
    cfg.enable_vbts = feedback_mode == "tactile"
    print(f"[fixed-force] creating {ARGS.num_envs} envs in {feedback_mode} mode", flush=True)
    try:
        env = gym.make(TASK_ID, cfg=cfg)
    except BaseException as exc:
        print(
            f"[fixed-force] environment construction failed with {type(exc).__name__}: {exc!r}",
            flush=True,
        )
        raise
    print("[fixed-force] environment construction complete", flush=True)
    return env


def scripted_action(step: int, num_envs: int, device: torch.device) -> torch.Tensor:
    period = 300
    phase = step % period
    value = 1.0 if phase < 240 else -1.0
    return torch.full((num_envs, 1), value, device=device)


def smoke(env) -> dict:
    inner = env.unwrapped
    env.reset()
    forces = []
    offsets = []
    table_heights = []
    table_tilts_deg = []
    max_drift = 0.0
    for step in range(ARGS.steps):
        _, _, _, _, _ = env.step(scripted_action(step, inner.num_envs, inner.device))
        force = inner._scrape_table_normal_force.detach().cpu()
        if not torch.isfinite(force).all():
            raise RuntimeError("Smoke test observed non-finite contact force.")
        forces.append(float(force.mean()))
        offsets.append(float(inner._normal_offset.mean()))
        table_heights.extend(inner._table_z_per_env.detach().cpu().tolist())
        table_normal = inner._table_normal().detach().cpu()
        table_tilts_deg.extend(
            torch.rad2deg(torch.acos(table_normal[:, 2].clamp(-1.0, 1.0))).tolist()
        )
        max_drift = max(max_drift, float(inner._fixed_grasp_drift.max()))
    force_span = float(np.max(forces) - np.min(forces))
    table_height_span = float(np.max(table_heights) - np.min(table_heights))
    maximum_table_tilt = float(np.max(table_tilts_deg))
    if max_drift > float(inner.cfg.fixed_grasp_max_drift_m):
        raise RuntimeError(f"Fixed-grasp drift {max_drift:.6f} m exceeded limit.")
    if force_span < 2.0:
        raise RuntimeError(f"Scripted normal action produced only {force_span:.3f} N force span.")
    if inner.num_envs >= 8 and table_height_span < 0.005:
        raise RuntimeError(
            f"Table height randomization produced only {table_height_span:.4f} m span."
        )
    if inner.num_envs >= 8 and maximum_table_tilt < 1.0:
        raise RuntimeError(
            f"Table-angle randomization produced only {maximum_table_tilt:.3f} deg tilt."
        )
    return {
        "passed": True,
        "force_span_n": force_span,
        "maximum_drift_m": max_drift,
        "offset_min_m": float(np.min(offsets)),
        "offset_max_m": float(np.max(offsets)),
        "table_height_min_m": float(np.min(table_heights)),
        "table_height_max_m": float(np.max(table_heights)),
        "maximum_table_tilt_deg": maximum_table_tilt,
    }


def held_out_linear_r2(features: torch.Tensor, force: torch.Tensor) -> float:
    split = max(2, int(0.8 * features.shape[0]))
    train_x = torch.cat((features[:split], torch.ones(split, 1)), dim=-1)
    test_x = torch.cat(
        (features[split:], torch.ones(features.shape[0] - split, 1)), dim=-1
    )
    if test_x.shape[0] < 2:
        raise RuntimeError("Tactile audit needs at least three samples.")
    weights = torch.linalg.lstsq(train_x, force[:split].unsqueeze(-1)).solution
    prediction = (test_x @ weights).squeeze(-1)
    expected = force[split:]
    denominator = ((expected - expected.mean()) ** 2).sum()
    if float(denominator) <= 1.0e-12:
        return float("-inf")
    return float(1.0 - ((prediction - expected) ** 2).sum() / denominator)


def tactile_audit(env) -> dict:
    inner = env.unwrapped
    obs, _ = env.reset()
    tactile_rows = []
    force_rows = []
    for step in range(ARGS.steps):
        obs, _, _, _, _ = env.step(scripted_action(step, inner.num_envs, inner.device))
        tactile_rows.append(obs["policy"][:, 4:].detach().cpu())
        force_rows.append(inner._scrape_table_normal_force.detach().cpu())
    tactile = torch.cat(tactile_rows).float()
    force = torch.cat(force_rows).float()
    channel_std = tactile.std(dim=0)
    active = channel_std > float(ARGS.minimum_tactile_std)
    if not bool(active.any()):
        return {
            "passed": False,
            "active_channels": 0,
            "maximum_channel_std": float(channel_std.max()),
            "held_out_linear_r2": None,
            "force_span_n": float(force.max() - force.min()),
            "failure": "no tactile channel exceeded the variance threshold",
        }
    reduced = tactile[:, active]
    if reduced.shape[1] > 32:
        keep = torch.topk(channel_std[active], k=32).indices
        reduced = reduced[:, keep]
    r2 = held_out_linear_r2(reduced, force)
    result = {
        "passed": r2 >= float(ARGS.minimum_predictive_r2),
        "active_channels": int(active.sum()),
        "maximum_channel_std": float(channel_std.max()),
        "held_out_linear_r2": r2,
        "force_span_n": float(force.max() - force.min()),
    }
    if not result["passed"]:
        result["failure"] = (
            f"held-out force-prediction R2={r2:.4f} is below "
            f"{ARGS.minimum_predictive_r2:.4f}"
        )
    return result


def evaluate(env) -> dict:
    if not ARGS.checkpoint:
        raise ValueError("--checkpoint is required in evaluate mode")
    inner = env.unwrapped
    checkpoint = Path(ARGS.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    player = CheckpointPlayer(env, checkpoint)
    obs, _ = env.reset()
    player.reset()
    bucket_errors = [[] for _ in range(4)]
    within = 0
    over = 0
    count = 0
    for _ in range(ARGS.steps):
        action = player.get_action(obs["policy"])
        obs, _, _, _, _ = env.step(action.to(inner.device))
        force = inner._scrape_table_normal_force
        target = inner._scrape_target_contact_normal_force
        error = torch.abs(force - target)
        bucket = torch.clamp(target.floor().long() - 2, 0, 3)
        for index in range(4):
            values = error[bucket == index]
            if values.numel():
                bucket_errors[index].extend(values.detach().cpu().tolist())
        within += int((error <= 1.0).sum())
        over += int((force > inner.cfg.max_contact_normal_force).sum())
        count += inner.num_envs
    maes = [float(np.mean(values)) if values else float("nan") for values in bucket_errors]
    result = {
        "passed": all(value <= 1.0 for value in maes)
        and within / count >= 0.8
        and over / count < 0.01,
        "target_bin_mae_n": dict(zip(("2-3", "3-4", "4-5", "5-6"), maes)),
        "within_1n_ratio": within / count,
        "over_force_ratio": over / count,
    }
    return result


def main() -> int:
    feedback = "tactile" if ARGS.mode == "tactile-audit" else "force"
    env = make_env(feedback)
    try:
        if ARGS.mode == "smoke":
            result = smoke(env)
        elif ARGS.mode == "tactile-audit":
            result = tactile_audit(env)
        else:
            result = evaluate(env)
    finally:
        env.close()
    payload = json.dumps(result, indent=2)
    print(payload)
    if ARGS.output:
        output = Path(ARGS.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except BaseException:
        traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    if exit_code != 0:
        os._exit(exit_code)
    APP.close()
