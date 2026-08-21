#!/usr/bin/env python3
"""Compare pair-filtered finger-tool impulses with compact TacMap signals."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
UTILS_PATH = (
    REPO_ROOT
    / "isaacsimenvs/tasks/simtoolreal/utils/wrench_tactile_observability.py"
)
SPEC = importlib.util.spec_from_file_location("wrench_impulse_gate_offline", UTILS_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load utilities from {UTILS_PATH}")
UTILS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = UTILS
SPEC.loader.exec_module(UTILS)

FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs/wrench_tactile_observability/impulse_analysis",
    )
    return parser.parse_args()


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.std(x) < 1.0e-12 or np.std(y) < 1.0e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def load_episodes(data_dir: Path) -> list[dict]:
    shards = sorted(data_dir.resolve().glob("shard_*.pt"))
    if not shards:
        raise FileNotFoundError(f"no shard_*.pt files under {data_dir}")
    episodes = []
    for path in shards:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != 2:
            raise ValueError(f"{path} is not a schema-v2 impulse shard")
        for episode in payload.get("episodes", []):
            UTILS.validate_impulse_episode(episode)
            episodes.append(episode)
    if not episodes:
        raise RuntimeError("dataset contains no episodes")
    return episodes


def centered(episodes: list[dict], key: str, transform) -> np.ndarray:
    chunks = []
    for episode in episodes:
        value = transform(episode[key].numpy())
        chunks.append(value - value.mean(axis=0, keepdims=True))
    return np.concatenate(chunks, axis=0)


def main() -> None:
    args = parse_args()
    episodes = load_episodes(args.data_dir)
    output_dir = args.output_root.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)

    normal = np.concatenate(
        [np.linalg.norm(ep["finger_normal_impulse_palm_ns"].numpy(), axis=-1) for ep in episodes]
    )
    tangential = np.concatenate(
        [np.linalg.norm(ep["finger_tangential_impulse_palm_ns"].numpy(), axis=-1) for ep in episodes]
    )
    tactile = np.concatenate([ep["tactile_compact"].numpy() for ep in episodes])
    normal_centered = centered(
        episodes, "finger_normal_impulse_palm_ns", lambda x: np.linalg.norm(x, axis=-1)
    )
    tangential_centered = centered(
        episodes, "finger_tangential_impulse_palm_ns", lambda x: np.linalg.norm(x, axis=-1)
    )
    tactile_centered = centered(episodes, "tactile_compact", lambda x: x)

    normal_depth_corr = [
        correlation(normal_centered[:, i], tactile_centered[:, i, 1]) for i in range(5)
    ]
    normal_area_corr = [
        correlation(normal_centered[:, i], tactile_centered[:, i, 0]) for i in range(5)
    ]
    tangent_depth_corr = [
        correlation(tangential_centered[:, i], tactile_centered[:, i, 1]) for i in range(5)
    ]
    threshold = float(episodes[0]["metadata"]["impulse_contact_threshold_ns"])
    active_fraction = (normal > threshold).mean(axis=0)
    static_tactile_fraction = np.mean(
        np.stack([ep["tactile_compact"].std(0).mean((0, 1)) < 1.0e-6 for ep in episodes])
    )

    episode_variance = [
        float(torch.linalg.vector_norm(ep["finger_normal_impulse_palm_ns"], dim=-1).std())
        for ep in episodes
    ]
    representative = episodes[int(np.argmax(episode_variance))]
    rep_normal = torch.linalg.vector_norm(
        representative["finger_normal_impulse_palm_ns"], dim=-1
    ).numpy()
    rep_tangent = torch.linalg.vector_norm(
        representative["finger_tangential_impulse_palm_ns"], dim=-1
    ).numpy()
    rep_tactile = representative["tactile_compact"][:, :, 1].numpy()
    finger_id = int(np.argmax(rep_normal.std(0)))
    control_dt = float(representative["metadata"]["control_dt_s"])
    time_s = np.arange(rep_normal.shape[0]) * control_dt

    def standardize(value: np.ndarray) -> np.ndarray:
        return (value - value.mean()) / max(float(value.std()), 1.0e-12)

    colors = ("#315A7D", "#C55A11", "#5B8C5A")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0))
    x = np.arange(5)
    axes[0, 0].bar(x - 0.18, normal.mean(0), 0.36, label="Normal", color=colors[0])
    axes[0, 0].bar(x + 0.18, tangential.mean(0), 0.36, label="Tangential", color=colors[1])
    axes[0, 0].set_ylabel("Mean impulse (N s)")
    axes[0, 0].set_title("Pair-filtered finger-tool contact")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].bar(x - 0.24, normal_depth_corr, 0.24, label="Normal vs depth", color=colors[0])
    axes[0, 1].bar(x, normal_area_corr, 0.24, label="Normal vs area", color=colors[2])
    axes[0, 1].bar(x + 0.24, tangent_depth_corr, 0.24, label="Tangential vs depth", color=colors[1])
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set_ylabel("Within-episode Pearson r")
    axes[0, 1].set_title("Impulse-TacMap association")
    axes[0, 1].legend(frameon=False, fontsize=9)

    axes[1, 0].bar(x, active_fraction, color=colors[0])
    axes[1, 0].set_ylim(0.0, 1.0)
    axes[1, 0].set_ylabel("Fraction of frames")
    axes[1, 0].set_title(f"Physical contact (> {threshold:.0e} N s)")

    axes[1, 1].plot(time_s, standardize(rep_normal[:, finger_id]), label="Normal impulse", color=colors[0])
    axes[1, 1].plot(time_s, standardize(rep_tangent[:, finger_id]), label="Tangential impulse", color=colors[1])
    axes[1, 1].plot(time_s, standardize(rep_tactile[:, finger_id]), label="TacMap depth", color=colors[2])
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Standardized signal")
    axes[1, 1].set_title(f"Representative trace: {FINGERS[finger_id]}")
    axes[1, 1].legend(frameon=False, fontsize=9)

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
    for axis in axes[:, 0]:
        axis.set_xticks(x, FINGERS)
    axes[0, 1].set_xticks(x, FINGERS)
    fig.suptitle("Finger contact impulse vs geometric TacMap", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / "finger_impulse_tacmap_diagnostics.png", dpi=220)
    fig.savefig(output_dir / "finger_impulse_tacmap_diagnostics.pdf")
    plt.close(fig)

    summary = {
        "created_at": datetime.now().astimezone().isoformat(),
        "data_dir": str(args.data_dir.resolve()),
        "episodes": len(episodes),
        "frames": int(normal.shape[0]),
        "finger_order": list(FINGERS),
        "mean_normal_impulse_ns": normal.mean(0).tolist(),
        "mean_tangential_impulse_ns": tangential.mean(0).tolist(),
        "physical_contact_fraction": active_fraction.tolist(),
        "within_episode_normal_depth_correlation": normal_depth_corr,
        "within_episode_normal_area_correlation": normal_area_corr,
        "within_episode_tangential_depth_correlation": tangent_depth_corr,
        "static_tactile_episode_fraction": float(static_tactile_fraction),
        "representative_episode": representative["metadata"]["episode_id"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    print(f"[output] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
