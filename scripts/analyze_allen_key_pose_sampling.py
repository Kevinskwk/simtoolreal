#!/usr/bin/env python3
"""Summarize workspace and yaw dependence of an Allen-key turning sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def wilson(
    successes: int, count: int, z: float = 1.96
) -> dict[str, float | int | None]:
    if count == 0:
        return {"count": 0, "successes": 0, "rate": None,
                "ci95_low": None, "ci95_high": None}
    rate = successes / count
    denominator = 1.0 + z * z / count
    center = (rate + z * z / (2.0 * count)) / denominator
    half_width = z * math.sqrt(
        rate * (1.0 - rate) / count + z * z / (4.0 * count * count)
    ) / denominator
    return {
        "count": count,
        "successes": successes,
        "rate": rate,
        "ci95_low": center - half_width,
        "ci95_high": center + half_width,
    }


def summarize_mask(mask: np.ndarray, acquired: np.ndarray,
                   all_turns: np.ndarray) -> dict[str, object]:
    return {
        "sample_fraction": float(mask.mean()),
        "acquisition": wilson(int(acquired[mask].sum()), int(mask.sum())),
        "all_turns": wilson(int(all_turns[mask].sum()), int(mask.sum())),
        "all_turns_given_acquisition": wilson(
            int(all_turns[mask].sum()), int(acquired[mask].sum())
        ),
    }


def binned_rates(values: np.ndarray, edges: np.ndarray, acquired: np.ndarray,
                 all_turns: np.ndarray) -> list[dict[str, object]]:
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (values >= low) & (values < high)
        rows.append({
            "low": float(low), "high": float(high),
            **summarize_mask(mask, acquired, all_turns),
        })
    return rows


def rate_grid(x: np.ndarray, y: np.ndarray, x_edges: np.ndarray,
              y_edges: np.ndarray, outcome: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts, _, _ = np.histogram2d(y, x, bins=(y_edges, x_edges))
    successes, _, _ = np.histogram2d(y, x, bins=(y_edges, x_edges), weights=outcome)
    rates = np.divide(successes, counts, out=np.full_like(successes, np.nan), where=counts > 0)
    return rates, counts


def main() -> None:
    args = parse_args()
    csv_path = args.run_dir / "pose_outcomes.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    with csv_path.open(newline="") as file:
        records = list(csv.DictReader(file))
    if not records:
        raise RuntimeError(f"no pose outcomes in {csv_path}")

    def column(name: str) -> np.ndarray:
        return np.asarray([float(row[name]) for row in records], dtype=np.float64)

    socket_z = column("socket_z_m")
    yaw = column("yaw_deg")
    handle_x = column("handle_center_x_m")
    handle_y = column("handle_center_y_m")
    acquired = column("acquired").astype(bool)
    all_turns = column("all_turns_success").astype(bool)

    z_edges = np.arange(0.37, 0.771, 0.05)
    yaw_edges = np.arange(-180.0, 180.1, 30.0)
    handle_y_edges = np.arange(-0.325, 0.326, 0.065)
    easy_yaw = ((yaw >= -180.0) & (yaw < -120.0)) | (yaw >= 150.0)
    summary = {
        "source": str(csv_path.resolve()),
        "num_samples": len(records),
        "overall": summarize_mask(np.ones(len(records), dtype=bool), acquired, all_turns),
        "bins": {
            "socket_z_m": binned_rates(socket_z, z_edges, acquired, all_turns),
            "yaw_deg": binned_rates(yaw, yaw_edges, acquired, all_turns),
            "handle_center_y_m": binned_rates(
                handle_y, handle_y_edges, acquired, all_turns
            ),
        },
        "curriculum_regions": {
            "support_workspace": summarize_mask(
                (handle_x >= -0.13) & (handle_x <= 0.325)
                & (handle_y >= -0.065) & (handle_y <= 0.195)
                & (socket_z >= 0.37) & (socket_z <= 0.67),
                acquired, all_turns,
            ),
            "easy_joint": summarize_mask(
                easy_yaw & (handle_y >= -0.065) & (handle_y <= 0.195)
                & (socket_z >= 0.37) & (socket_z <= 0.62),
                acquired, all_turns,
            ),
            "hard_yaw": summarize_mask(
                (yaw >= 60.0) & (yaw < 120.0)
                & (socket_z >= 0.37) & (socket_z <= 0.67),
                acquired, all_turns,
            ),
        },
    }
    (args.run_dir / "pose_sampling_analysis.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, values, edges, label in (
        (axes[0, 0], yaw, yaw_edges, "Allen-key yaw (deg)"),
        (axes[0, 1], socket_z, z_edges, "Socket height (m)"),
        (axes[1, 0], handle_y, handle_y_edges, "Handle-center y (m)"),
    ):
        centers = (edges[:-1] + edges[1:]) / 2.0
        width = np.diff(edges) * 0.84
        acq_rate = []
        turn_rate = []
        for low, high in zip(edges[:-1], edges[1:]):
            mask = (values >= low) & (values < high)
            acq_rate.append(acquired[mask].mean() if mask.any() else np.nan)
            turn_rate.append(all_turns[mask].mean() if mask.any() else np.nan)
        ax.bar(centers, acq_rate, width=width, color="#4C78A8", alpha=0.8,
               label="acquired")
        ax.plot(centers, turn_rate, "o-", color="#D55E00", linewidth=2.2,
                label="completed 30/60/90 deg")
        ax.set(xlabel=label, ylabel="Fraction", ylim=(0.0, 0.75))
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=9)

    joint_y_edges = np.arange(-0.325, 0.326, 0.065)
    rates, _ = rate_grid(yaw, handle_y, yaw_edges, joint_y_edges, all_turns)
    image = axes[1, 1].imshow(
        rates, origin="lower", aspect="auto", vmin=0.0, vmax=0.30,
        extent=(yaw_edges[0], yaw_edges[-1], joint_y_edges[0], joint_y_edges[-1]),
        cmap="magma",
    )
    axes[1, 1].set(xlabel="Allen-key yaw (deg)", ylabel="Handle-center y (m)",
                   title="All-turn completion rate")
    fig.colorbar(image, ax=axes[1, 1], label="Fraction")
    fig.suptitle(f"Pretrained Allen-key gate, n={len(records):,}", fontsize=15)
    for suffix in ("png", "pdf"):
        fig.savefig(args.run_dir / f"pose_sampling_analysis.{suffix}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
