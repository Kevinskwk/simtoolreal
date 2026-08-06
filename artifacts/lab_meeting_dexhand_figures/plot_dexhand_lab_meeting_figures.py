#!/usr/bin/env python3
"""Build slide-ready figures from the recorded DexHand RL experiments."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


OUT_DIR = Path(__file__).resolve().parent
REPO = OUT_DIR.parents[1]

TACTILE_EVENTS = (
    REPO
    / "outputs/2026-07-20/17-51-49/0_simtoolreal_sapg/summaries/"
    "events.out.tfevents.1784541609.showlab-0703"
)
TACTILE_RESUME_EVENTS = (
    REPO
    / "outputs/2026-07-22/08-24-28/0_simtoolreal_sapg/summaries/"
    "events.out.tfevents.1784680342.showlab-0703"
)
NO_TACTILE_EVENTS = (
    REPO
    / "outputs/2026-07-21/11-48-11/0_simtoolreal_sapg/summaries/"
    "events.out.tfevents.1784606083.showlab-0703"
)
TACTILE_OVERRIDES = REPO / "outputs/2026-07-20/17-51-49/.hydra/overrides.yaml"
NO_TACTILE_OVERRIDES = REPO / "outputs/2026-07-21/11-48-11/.hydra/overrides.yaml"

FAILED_LIFT = REPO / "outputs/contact_force_controllability/20260722_212208/summary.json"
CONTROLLER_SOURCE = REPO / "scripts/validate_contact_force_controllability.py"

PASSIVE_DWELL = REPO / "outputs/contact_force_stability/frozen_targets_default.json"
FORCE_SUMMARY = REPO / "outputs/contact_force_controllability/20260722_220045/summary.json"
FORCE_CSV = REPO / "outputs/contact_force_controllability/20260722_220045/held_dls.csv"
DYNAMIC_120 = REPO / "outputs/contact_force_controllability/20260723_175652/summary.json"
DYNAMIC_240 = REPO / "outputs/contact_force_controllability/20260723_175506/summary.json"

TEAL = "#007C83"
ORANGE = "#D97706"
BLUE = "#276FBF"
RED = "#C43D3D"
GREEN = "#3A7D44"
INK = "#20252B"
MID = "#66717E"
GRID = "#D8DDE3"
LIGHT = "#F2F4F6"


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required experiment file is missing: {path}")
    with path.open() as stream:
        return json.load(stream)


def assert_close(actual: float, expected: float, tolerance: float, label: str) -> None:
    if not np.isfinite(actual) or abs(actual - expected) > tolerance:
        raise ValueError(
            f"{label} mismatch: expected {expected} +/- {tolerance}, found {actual}"
        )


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": MID,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 15,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "text.color": INK,
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "legend.fontsize": 11,
            "legend.frameon": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.75,
            "lines.linewidth": 2.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_png(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_scalars(path: Path, tags: tuple[str, ...]) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Required TensorBoard history is missing: {path}")
    # Reservoir sampling preserves a readable raw trace without retaining every
    # scalar from these multi-gigastep runs.
    event_file = EventAccumulator(str(path), size_guidance={"scalars": 10_000})
    event_file.Reload()
    available = set(event_file.Tags()["scalars"])
    missing = sorted(set(tags) - available)
    if missing:
        raise KeyError(f"Missing TensorBoard metrics in {path}: {missing}")
    result = {}
    for tag in tags:
        events = event_file.Scalars(tag)
        if len(events) < 100:
            raise ValueError(f"Too few values for {tag} in {path}: {len(events)}")
        result[tag] = np.asarray([(event.step, event.value) for event in events], dtype=float)
    return result


def merge_history(first: dict[str, np.ndarray], second: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    merged = {}
    for tag in first:
        values = np.concatenate((first[tag], second[tag]))
        order = np.argsort(values[:, 0], kind="stable")
        values = values[order]
        _, unique_indices = np.unique(values[:, 0], return_index=True)
        merged[tag] = values[np.sort(unique_indices)]
    return merged


def smooth_to_grid(
    values: np.ndarray, maximum_step: float, grid_size: int = 1200, window: int = 61
) -> tuple[np.ndarray, np.ndarray]:
    values = values[values[:, 0] <= maximum_step]
    grid = np.linspace(0.0, maximum_step, grid_size)
    interpolated = np.interp(grid, values[:, 0], values[:, 1])
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(interpolated, (window // 2, window // 2), mode="edge")
    return grid, np.convolve(padded, kernel, mode="valid")


def validate_matched_configs() -> None:
    tactile = TACTILE_OVERRIDES.read_text()
    no_tactile = NO_TACTILE_OVERRIDES.read_text()
    shared = (
        "env.contact_force_filter_alpha=0.2",
        "env.enable_tool_table_contact_force_reward=true",
        "env.scene.num_envs=12288",
        "agent.params.seed=42",
        "agent.params.config.minibatch_size=98304",
        "agent.params.config.learning_rate=5e-5",
    )
    for setting in shared:
        if setting not in tactile or setting not in no_tactile:
            raise ValueError(f"Matched-run setting is absent: {setting}")
    if "env.include_tacmap_in_policy=true" not in tactile:
        raise ValueError("Tactile continuation does not include tacmap in the actor input")
    if "env.include_tacmap_in_policy=false" not in no_tactile:
        raise ValueError("No-tactile continuation unexpectedly includes tacmap")
    target_obs = "scrape_target_contact_normal_force"
    if target_obs not in tactile or target_obs not in no_tactile:
        raise ValueError("Force target is absent from one of the matched actor observations")


def make_matched_comparison() -> None:
    validate_matched_configs()
    tags = (
        "curriculum/contact_force_success_mean",
        "curriculum/contact_force_threshold",
        "scrape_pose/edge_contact_error_mean",
    )
    tactile = merge_history(
        load_scalars(TACTILE_EVENTS, tags),
        load_scalars(TACTILE_RESUME_EVENTS, tags),
    )
    no_tactile = load_scalars(NO_TACTILE_EVENTS, tags)
    maximum_step = min(
        tactile[tags[0]][-1, 0],
        no_tactile[tags[0]][-1, 0],
    )
    if maximum_step < 5.7e9:
        raise ValueError(f"Matched continuation is unexpectedly short: {maximum_step} steps")

    fig, axes = plt.subplots(2, 1, figsize=(10.0, 5.625), sharex=True)
    fig.subplots_adjust(left=0.10, right=0.89, top=0.82, bottom=0.14, hspace=0.18)
    fig.suptitle("Tactile input did not improve force tracking", fontsize=20, weight="bold", y=0.97)
    fig.text(
        0.5,
        0.905,
        "Matched single-seed continuations; curves are descriptive, not significance estimates",
        ha="center",
        color=MID,
        fontsize=11.5,
    )

    run_specs = (("Tactile", tactile, TEAL), ("No tactile", no_tactile, ORANGE))
    force_tag = "curriculum/contact_force_success_mean"
    for label, history, color in run_specs:
        raw = history[force_tag]
        raw = raw[raw[:, 0] <= maximum_step]
        axes[0].plot(raw[:, 0] / 1e9, raw[:, 1], color=color, alpha=0.11, linewidth=0.8)
        grid, smoothed = smooth_to_grid(history[force_tag], maximum_step)
        axes[0].plot(grid / 1e9, smoothed, color=color, label=label)
    axes[0].set_ylabel("Force success rate")
    axes[0].set_ylim(0.60, 0.84)
    axes[0].legend(loc="upper right", ncol=2)
    axes[0].set_title("Contact-force curriculum stalled after one tolerance update", loc="left")

    threshold_ax = axes[0].twinx()
    threshold_ax.grid(False)
    for _, history, color in run_specs:
        threshold = history["curriculum/contact_force_threshold"]
        threshold = threshold[threshold[:, 0] <= maximum_step]
        threshold_ax.step(
            threshold[:, 0] / 1e9,
            threshold[:, 1],
            where="post",
            color=color,
            alpha=0.55,
            linewidth=1.3,
            linestyle="--",
        )
    threshold_ax.set_ylim(3.48, 4.12)
    threshold_ax.set_yticks([3.6, 4.0])
    threshold_ax.set_ylabel("Tolerance (N)", color=MID)
    threshold_ax.tick_params(axis="y", colors=MID)

    edge_tag = "scrape_pose/edge_contact_error_mean"
    for label, history, color in run_specs:
        raw = history[edge_tag]
        raw = raw[raw[:, 0] <= maximum_step]
        axes[1].plot(raw[:, 0] / 1e9, raw[:, 1] * 1000.0, color=color, alpha=0.10, linewidth=0.8)
        grid, smoothed = smooth_to_grid(history[edge_tag], maximum_step)
        axes[1].plot(grid / 1e9, smoothed * 1000.0, color=color)
    axes[1].set_title("Geometric edge contact improved in both variants", loc="left")
    axes[1].set_ylabel("Edge error (mm)")
    axes[1].set_xlabel("Environment steps (billions)")
    axes[1].set_xlim(0.0, maximum_step / 1e9)
    axes[1].set_ylim(bottom=0.0)
    save_figure(fig, "matched_tactile_vs_no_tactile")


def draw_lift_state(
    ax: plt.Axes,
    hand_height: float,
    tool_height: float,
    title: str,
    command_arrow: bool = False,
    separated: bool = False,
) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((0.02, 0.08), 0.96, 0.12, facecolor="#BFC7CF", edgecolor=MID))
    ax.add_patch(
        Polygon(
            [
                (0.43, tool_height + 0.03),
                (0.78, tool_height + 0.03),
                (0.82, tool_height + 0.08),
                (0.47, tool_height + 0.08),
            ],
            closed=True,
            facecolor=BLUE,
            edgecolor=INK,
            linewidth=1.2,
        )
    )
    ax.add_patch(
        Rectangle((0.30, hand_height), 0.24, 0.18, facecolor="#E6B17E", edgecolor=INK, linewidth=1.2)
    )
    for offset in (0.00, 0.065, 0.13):
        ax.add_patch(
            Rectangle(
                (0.34 + offset, hand_height - 0.14),
                0.045,
                0.15,
                facecolor="#E6B17E",
                edgecolor=INK,
                linewidth=1.0,
            )
        )
    if command_arrow:
        ax.add_patch(
            FancyArrowPatch(
                (0.18, 0.34),
                (0.18, 0.69),
                arrowstyle="-|>",
                mutation_scale=15,
                linewidth=2.2,
                color=TEAL,
            )
        )
        ax.text(0.14, 0.52, "+40 mm", rotation=90, ha="center", va="center", color=TEAL, weight="bold")
    if separated:
        ax.plot([0.55, 0.55], [tool_height + 0.11, hand_height - 0.03], color=RED, linestyle=":")
        ax.text(0.59, 0.43, "relative\ndrift", color=RED, va="center", fontsize=10)
    ax.set_title(title, fontsize=12.5, pad=5)


def make_lift_validation() -> None:
    summary = load_json(FAILED_LIFT)
    failed = summary["phases"]["held-dls"]
    if failed.get("passed") is not False:
        raise ValueError("The commanded-lift source is not a failed held-tool test")
    match = re.search(
        r"tool lift=([0-9.]+) m, tool-palm drift=([0-9.]+) m", failed["message"]
    )
    if not match:
        raise ValueError("Failed-lift summary does not contain measured lift and drift")
    tool_lift_m, drift_m = map(float, match.groups())
    source = CONTROLLER_SOURCE.read_text()
    command_match = re.search(r"contact_palm_pos \+ ([0-9.]+) \* normal", source)
    if not command_match:
        raise ValueError("Could not recover the commanded lift from the validator source")
    commanded_m = float(command_match.group(1))
    assert_close(commanded_m, 0.04, 1e-9, "commanded palm lift")
    assert_close(tool_lift_m, 0.0, 5e-5, "measured tool lift")
    assert_close(drift_m, 0.0369, 5e-5, "tool-palm drift")

    fig = plt.figure(figsize=(10.0, 5.625))
    grid = fig.add_gridspec(2, 4, height_ratios=[1.7, 1.0], left=0.05, right=0.98, top=0.80, bottom=0.12)
    fig.suptitle("Fingertip-near was not a mechanically stable grasp", fontsize=20, weight="bold", y=0.97)
    fig.text(
        0.5,
        0.89,
        "Commanded-lift challenge after policy acquisition",
        ha="center",
        color=MID,
        fontsize=12,
    )
    states = (
        (0.39, 0.20, "1  Candidate detected", False, False),
        (0.39, 0.20, "2  Lift commanded", True, False),
        (0.61, 0.20, "3  Separation grows", False, True),
        (0.66, 0.20, "4  Tool stays on table", False, True),
    )
    for index, state in enumerate(states):
        draw_lift_state(fig.add_subplot(grid[0, index]), *state)

    ax = fig.add_subplot(grid[1, :])
    labels = ("Palm lift target", "Measured tool lift", "Tool-palm drift")
    values_mm = np.asarray((commanded_m, tool_lift_m, drift_m)) * 1000.0
    bars = ax.barh(labels, values_mm, color=(TEAL, RED, ORANGE), height=0.54)
    ax.set_xlim(0, 45)
    ax.set_xlabel("Normal displacement (mm)")
    ax.set_title("Endpoint measurement: the tool failed to follow the lift", loc="left")
    for bar, value in zip(bars, values_mm):
        ax.text(value + 0.8, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center", weight="bold")
    ax.invert_yaxis()
    fig.text(
        0.98,
        0.025,
        "State sequence reconstructed from the logged command and endpoint; no camera frames were recorded.",
        ha="right",
        color=MID,
        fontsize=9.5,
    )
    save_figure(fig, "commanded_lift_grasp_validation")


def read_pi_csv(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required PI trace is missing: {path}")
    stages: dict[int, list[tuple[float, float]]] = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["phase"] != "held-dls-pi":
                continue
            stage = int(row["stage"])
            stages.setdefault(stage, []).append(
                (float(row["target_force_n"]), float(row["measured_force_n"]))
            )
    if set(stages) != {0, 1, 2} or any(len(values) != 600 for values in stages.values()):
        raise ValueError(f"Unexpected PI trace stages or lengths in {path}")
    return {
        stage: (
            np.asarray([value[0] for value in values]),
            np.asarray([value[1] for value in values]),
        )
        for stage, values in stages.items()
    }


def read_contact_onset(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Required contact-onset trace is missing: {path}")
    time_s = []
    force_n = []
    equilibrium_n = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if (
                row["protocol"] == "acquisition"
                and row["repetition"] == "0"
                and float(row["initial_speed_mps"]) == 0.0
            ):
                time_s.append(float(row["time_s"]))
                force_n.append(float(row["measured_force_n"]))
                equilibrium_n.append(float(row["equilibrium_reaction_n"]))
    if len(time_s) != 180:
        raise ValueError(f"Expected 180 contact-onset samples in {path}, found {len(time_s)}")
    time_s = np.asarray(time_s)
    force_n = np.asarray(force_n)
    onset = np.flatnonzero(force_n > 0.01)
    if onset.size == 0:
        raise ValueError(f"No contact onset found in {path}")
    relative_ms = (time_s - time_s[onset[0]]) * 1000.0
    return relative_ms, force_n, float(np.median(equilibrium_n))


def make_passive_dwell_stability_simple() -> None:
    dwell = load_json(PASSIVE_DWELL)
    if dwell.get("passed") is not True:
        raise ValueError("Passive-dwell source did not pass")
    values_mn = np.asarray(
        (
            dwell["raw_force_std_n"]["median"] * 1000.0,
            dwell["raw_force_range_n"]["median"] * 1000.0,
        )
    )
    assert_close(float(values_mn[0]), 0.8457, 5e-4, "passive-dwell median std")
    assert_close(float(values_mn[1]), 4.3731, 5e-4, "passive-dwell median range")

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    fig.subplots_adjust(left=0.15, right=0.96, top=0.72, bottom=0.17)
    fig.suptitle("Passive dwell force is stable", fontsize=19, weight="bold", y=0.97)
    fig.text(
        0.5,
        0.82,
        f"Median contact force: {dwell['raw_force_mean_n']['median']:.2f} N"
        f"   |   Contact retained: {100 * dwell['contact_retention_ratio']['median']:.0f}%",
        ha="center",
        fontsize=12,
        color=MID,
    )
    bars = ax.bar(("Standard deviation", "Peak-to-peak range"), values_mn, color=(TEAL, BLUE), width=0.58)
    ax.set_ylabel("Median within-dwell variation (mN)")
    ax.set_ylim(0, 5.2)
    ax.grid(axis="x", visible=False)
    for bar, value in zip(bars, values_mn):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.12,
            f"{value:.2f} mN",
            ha="center",
            va="bottom",
            fontsize=13,
            weight="bold",
        )
    save_png(fig, "passive_dwell_stability")


def make_pi_target_force_tracking_simple() -> None:
    summary = load_json(FORCE_SUMMARY)
    trace = read_pi_csv(FORCE_CSV)
    held = summary["phases"]["held-dls"]
    control_hz = float(summary["metadata"]["control_hz"])
    expected_maes = (0.368, 0.671, 0.266)
    colors = (TEAL, ORANGE, BLUE)

    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.83, bottom=0.16)
    elapsed = 0.0
    for stage, color, expected_mae, target_summary in zip(
        (0, 1, 2), colors, expected_maes, held["pi_targets"]
    ):
        target, measured = trace[stage]
        assert_close(float(target_summary["mae_n"]), expected_mae, 5e-4, f"stage {stage} PI MAE")
        time = elapsed + np.arange(measured.size) / control_hz
        ax.plot(time, measured, color=color, linewidth=2.0)
        ax.plot(time, target, color=INK, linewidth=1.5, linestyle="--")
        ax.text(
            elapsed + 5.0,
            float(target[0]) + 0.48,
            f"{target[0]:.0f} N target\nMAE {target_summary['mae_n']:.3f} N",
            ha="center",
            color=color,
            fontsize=11.5,
            weight="bold",
        )
        elapsed = time[-1] + 1.0 / control_hz
    ax.set_title("PI target-force tracking", fontsize=19, pad=16)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normal force (N)")
    ax.set_xlim(0, elapsed)
    ax.set_ylim(-0.2, 7.3)
    ax.plot([], [], color=INK, linestyle="--", label="Target")
    ax.legend(loc="lower right")
    save_png(fig, "pi_target_force_tracking")


def make_contact_onset_timestep_simple() -> None:
    dynamic_120 = load_json(DYNAMIC_120)
    dynamic_240 = load_json(DYNAMIC_240)
    time_120, force_120, equilibrium_120 = read_contact_onset(
        REPO / "outputs/contact_force_controllability/20260723_175652/bare_dynamics.csv"
    )
    time_240, force_240, equilibrium_240 = read_contact_onset(
        REPO / "outputs/contact_force_controllability/20260723_175506/bare_dynamics.csv"
    )
    if dynamic_120["metadata"]["physics_hz"] != 120 or dynamic_240["metadata"]["physics_hz"] != 240:
        raise ValueError("Contact-onset sources do not have the expected physics rates")
    assert_close(float(force_120.max()), 1.1958, 5e-4, "120 Hz onset peak")
    assert_close(float(force_240.max()), 2.2117, 5e-4, "240 Hz onset peak")
    assert_close(equilibrium_120, equilibrium_240, 1e-8, "equilibrium reaction")

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    fig.subplots_adjust(left=0.12, right=0.96, top=0.82, bottom=0.17)
    window_120 = (time_120 >= -25.0) & (time_120 <= 100.0)
    window_240 = (time_240 >= -25.0) & (time_240 <= 100.0)
    ax.plot(
        time_120[window_120],
        force_120[window_120],
        color=TEAL,
        marker="o",
        markersize=4,
        label="120 Hz  (Δt = 8.33 ms)",
    )
    ax.plot(
        time_240[window_240],
        force_240[window_240],
        color=ORANGE,
        marker="o",
        markersize=3.5,
        label="240 Hz  (Δt = 4.17 ms)",
    )
    ax.axhline(equilibrium_120, color=MID, linestyle="--", linewidth=1.4, label="Static equilibrium")
    ax.axvline(0.0, color=INK, linestyle=":", linewidth=1.2)
    ax.scatter((time_120[np.argmax(force_120)],), (force_120.max(),), color=TEAL, s=45, zorder=5)
    ax.scatter((time_240[np.argmax(force_240)],), (force_240.max(),), color=ORANGE, s=45, zorder=5)
    ax.text(
        time_120[np.argmax(force_120)] + 4,
        force_120.max(),
        f"{force_120.max():.2f} N",
        color=TEAL,
        va="center",
        weight="bold",
    )
    ax.text(
        time_240[np.argmax(force_240)] + 4,
        force_240.max(),
        f"{force_240.max():.2f} N",
        color=ORANGE,
        va="center",
        weight="bold",
    )
    ax.set_title("Raw contact-onset force depends on simulation timestep", fontsize=18, pad=16)
    ax.set_xlabel("Time from first contact sample (ms)")
    ax.set_ylabel("Measured normal force (N)")
    ax.set_xlim(-25, 100)
    ax.set_ylim(-0.05, 2.55)
    ax.legend(loc="upper right")
    save_png(fig, "contact_onset_force_transient_across_simulation_timesteps")


def make_force_validation() -> None:
    dwell = load_json(PASSIVE_DWELL)
    force = load_json(FORCE_SUMMARY)
    dynamic_120 = load_json(DYNAMIC_120)
    dynamic_240 = load_json(DYNAMIC_240)
    held = force["phases"]["held-dls"]
    pi_trace = read_pi_csv(FORCE_CSV)

    rho = float(held["monotonic"]["spearman_rho"])
    span = float(held["monotonic"]["force_span_n"])
    maes = np.asarray([entry["mae_n"] for entry in held["pi_targets"]])
    assert_close(rho, 0.994, 5e-4, "displacement-force Spearman rho")
    assert_close(span, 4.91, 0.005, "displacement-force span")
    for actual, expected, target in zip(maes, (0.368, 0.671, 0.266), (2, 4, 6)):
        assert_close(float(actual), expected, 5e-4, f"{target} N PI MAE")
    if dwell.get("passed") is not True:
        raise ValueError("Passive-dwell source did not pass")
    residual_p95 = float(dynamic_120["phases"]["bare-dynamics"]["step"]["force_balance_p95_n"])
    max_hysteresis = max(
        float(entry["hysteresis_mae_n"])
        for entry in dynamic_120["phases"]["bare-dynamics"]["ramps"]
    )
    assert_close(residual_p95, 0.012, 5e-5, "120 Hz force-balance p95")
    if max_hysteresis >= 0.0015:
        raise ValueError(f"120 Hz ramp hysteresis is not below 0.0015 N: {max_hysteresis}")
    if dynamic_120["metadata"]["physics_hz"] != 120 or dynamic_240["metadata"]["physics_hz"] != 240:
        raise ValueError("Dynamic comparison does not contain the expected 120/240 Hz runs")

    fig = plt.figure(figsize=(12.0, 6.75))
    grid = fig.add_gridspec(1, 3, left=0.06, right=0.98, top=0.78, bottom=0.17, wspace=0.28)
    fig.suptitle("Isaac Sim contact force is controllable when averaged by control interval", fontsize=20, weight="bold", y=0.97)
    fig.text(
        0.5,
        0.895,
        "Passive stability, fixed-grasp displacement response, and closed-loop force tracking",
        ha="center",
        color=MID,
        fontsize=12,
    )

    ax = fig.add_subplot(grid[0, 0])
    stability_mn = np.asarray(
        (
            dwell["raw_force_std_n"]["median"] * 1000.0,
            dwell["raw_force_range_n"]["median"] * 1000.0,
        )
    )
    bars = ax.bar(("Std.", "Peak-to-peak"), stability_mn, color=(TEAL, BLUE), width=0.58)
    ax.set_title("A  Passive dwell is stable", loc="left")
    ax.set_ylabel("Median within-dwell variation (mN)")
    ax.set_ylim(0, 5.5)
    for bar, value in zip(bars, stability_mn):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.16, f"{value:.2f}", ha="center", weight="bold")
    ax.text(
        0.04,
        0.93,
        f"Median force  {dwell['raw_force_mean_n']['median']:.2f} N\n"
        f"Contact retained  {100.0 * dwell['contact_retention_ratio']['median']:.0f}%",
        transform=ax.transAxes,
        va="top",
        fontsize=11.5,
        bbox={"facecolor": "white", "edgecolor": GRID, "boxstyle": "round,pad=0.35"},
    )

    inset = ax.inset_axes([0.43, 0.37, 0.52, 0.25])
    peak_120 = np.mean(
        [entry["peak_force_mean_n"] for entry in dynamic_120["phases"]["bare-dynamics"]["acquisition"]]
    )
    peak_240 = np.mean(
        [entry["peak_force_mean_n"] for entry in dynamic_240["phases"]["bare-dynamics"]["acquisition"]]
    )
    inset.bar(("120", "240"), (peak_120, peak_240), color=(TEAL, ORANGE), width=0.55)
    inset.set_title("Raw impact peak (N)", fontsize=9.5, pad=2)
    inset.set_xticks((0, 1), ("120 Hz", "240 Hz"))
    inset.set_ylabel("Peak", fontsize=8.5)
    inset.tick_params(labelsize=8.5)
    inset.grid(axis="x", visible=False)

    ax = fig.add_subplot(grid[0, 1])
    displacement_mm = -1000.0 * np.asarray(held["offsets_m"], dtype=float)
    measured_n = np.asarray(held["steady_offset_forces_n"], dtype=float)
    ax.plot(displacement_mm, measured_n, color=BLUE, marker="o", markersize=6)
    ax.axvline(0, color=MID, linewidth=1.0, linestyle=":")
    ax.set_title("B  Displacement controls force", loc="left")
    ax.set_xlabel("Normal displacement into table (mm)")
    ax.set_ylabel("Steady measured force (N)")
    ax.set_ylim(-0.2, 5.5)
    ax.text(
        0.05,
        0.93,
        rf"$\rho$ = {rho:.3f}" + f"\nForce span = {span:.2f} N",
        transform=ax.transAxes,
        va="top",
        fontsize=12,
        weight="bold",
    )

    ax = fig.add_subplot(grid[0, 2])
    colors = (TEAL, ORANGE, BLUE)
    elapsed = 0.0
    for stage, color, target_summary in zip((0, 1, 2), colors, held["pi_targets"]):
        target, measured = pi_trace[stage]
        time = elapsed + np.arange(measured.size) / float(force["metadata"]["control_hz"])
        ax.plot(time, measured, color=color, linewidth=1.4)
        ax.plot(time, target, color=INK, linewidth=1.0, linestyle="--")
        midpoint = elapsed + 5.0
        ax.text(
            midpoint,
            float(target[0]) + 0.50,
            f"{target[0]:.0f} N\nMAE {target_summary['mae_n']:.3f}",
            ha="center",
            fontsize=10.5,
            color=color,
            weight="bold",
        )
        elapsed = time[-1] + 1.0 / float(force["metadata"]["control_hz"])
        if stage < 2:
            ax.axvline(elapsed, color=GRID, linewidth=1.0)
    ax.set_title("C  PI tracks 2, 4, 6 N", loc="left")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Measured force (N)")
    ax.set_xlim(0, elapsed)
    ax.set_ylim(-0.3, 7.4)

    fig.text(
        0.06,
        0.075,
        f"120 Hz dynamic gate: force-balance residual p95 = {residual_p95:.3f} N; "
        f"ramp hysteresis ≤ {max_hysteresis:.4f} N.",
        fontsize=11.5,
        weight="bold",
        color=GREEN,
    )
    fig.text(
        0.06,
        0.025,
        "Conclusion: interval-averaged force/impulse is sufficiently valid for control; raw impact peaks are timestep-dependent.",
        fontsize=12.5,
        weight="bold",
    )
    save_figure(fig, "force_measurement_validation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--simple-only",
        action="store_true",
        help="Render only the three standalone validation PNGs.",
    )
    args = parser.parse_args()
    configure_style()
    if not args.simple_only:
        make_matched_comparison()
        make_lift_validation()
        make_force_validation()
    make_passive_dwell_stability_simple()
    make_pi_target_force_tracking_simple()
    make_contact_onset_timestep_simple()
    print("Verified sources and wrote:")
    if not args.simple_only:
        for stem in (
            "matched_tactile_vs_no_tactile",
            "commanded_lift_grasp_validation",
            "force_measurement_validation",
        ):
            print(f"  {OUT_DIR / (stem + '.png')}")
            print(f"  {OUT_DIR / (stem + '.pdf')}")
    for stem in (
        "passive_dwell_stability",
        "pi_target_force_tracking",
        "contact_onset_force_transient_across_simulation_timesteps",
    ):
        print(f"  {OUT_DIR / (stem + '.png')}")


if __name__ == "__main__":
    main()
