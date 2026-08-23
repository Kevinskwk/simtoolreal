from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "isolated_rlgames_utils", ROOT / "isaacsimenvs/utils/rlgames_utils.py"
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load rlgames_utils.py")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
EnvStatsAlgoObserver = MODULE.EnvStatsAlgoObserver


class RecordingWriter:
    def __init__(self):
        self.values: dict[str, tuple[float, int]] = {}

    def add_scalar(self, key: str, value: float, frame: int) -> None:
        self.values[key] = (float(value), int(frame))


def test_observer_separates_rollout_and_episode_reward_means() -> None:
    observer = EnvStatsAlgoObserver()
    writer = RecordingWriter()
    observer.after_init(SimpleNamespace(writer=writer, games_to_track=16))

    for step, reward in enumerate((1.0, 3.0)):
        done = torch.tensor([0], dtype=torch.long)
        if step == 1:
            done = torch.tensor([0], dtype=torch.long)
        else:
            done = torch.empty(0, dtype=torch.long)
        observer.process_infos(
            {
                "reward": {"total_reward": reward},
                "diagnostic": float(10 + step),
                "episode_cumulative": {
                    "total_reward": torch.tensor([reward]),
                    "episode_step_count": torch.ones(1),
                    "phase/adjustment_reward_sum": torch.tensor([reward]),
                    "phase/adjustment_step_count": torch.ones(1),
                },
            },
            done,
        )

    observer.after_print_stats(frame=32, epoch_num=2, total_time=0.0)

    assert writer.values["rollout_step_mean/reward/total_reward"] == pytest.approx((2.0, 32))
    assert writer.values["diagnostic"] == pytest.approx((10.5, 32))
    assert writer.values["reward/total_reward"] == pytest.approx((2.0, 32))
    assert writer.values["episode_phase/adjustment_reward_mean"] == pytest.approx((2.0, 32))
    assert observer.direct_info_sum == {}
    assert observer.direct_info_count == {}


def test_observer_rejects_nonfinite_direct_metric() -> None:
    observer = EnvStatsAlgoObserver()
    observer.after_init(SimpleNamespace(writer=RecordingWriter(), games_to_track=16))

    with pytest.raises(RuntimeError, match="not finite"):
        observer.process_infos(
            {"reward": {"total_reward": float("nan")}},
            torch.empty(0, dtype=torch.long),
        )


def _record_finished_phase_metrics(
    observer: EnvStatsAlgoObserver,
    *,
    reward_sums: list[float],
    phase_step_counts: list[float],
) -> None:
    num_episodes = len(reward_sums)
    observer.process_infos(
        {
            "episode_cumulative": {
                "episode_step_count": torch.full((num_episodes,), 10.0),
                "phase/final_hold_reward_sum": torch.tensor(reward_sums),
                "phase/final_hold_step_count": torch.tensor(phase_step_counts),
            }
        },
        torch.arange(num_episodes),
    )


def test_observer_averages_phase_reward_only_over_episodes_that_entered_phase() -> None:
    observer = EnvStatsAlgoObserver()
    writer = RecordingWriter()
    observer.after_init(SimpleNamespace(writer=writer, games_to_track=16))
    _record_finished_phase_metrics(
        observer,
        reward_sums=[6.0, 0.0],
        phase_step_counts=[3.0, 0.0],
    )

    observer.after_print_stats(frame=64, epoch_num=4, total_time=0.0)

    assert writer.values["episode_phase/final_hold_reward_mean"] == pytest.approx((2.0, 64))


def test_observer_skips_phase_reward_when_no_episode_entered_phase() -> None:
    observer = EnvStatsAlgoObserver()
    writer = RecordingWriter()
    observer.after_init(SimpleNamespace(writer=writer, games_to_track=16))
    _record_finished_phase_metrics(
        observer,
        reward_sums=[0.0, 0.0],
        phase_step_counts=[0.0, 0.0],
    )

    observer.after_print_stats(frame=64, epoch_num=4, total_time=0.0)

    assert "episode_phase/final_hold_reward_mean" not in writer.values


def test_observer_rejects_phase_reward_without_phase_steps() -> None:
    observer = EnvStatsAlgoObserver()
    observer.after_init(SimpleNamespace(writer=RecordingWriter(), games_to_track=16))
    _record_finished_phase_metrics(
        observer,
        reward_sums=[1.0, 0.0],
        phase_step_counts=[0.0, 0.0],
    )

    with pytest.raises(RuntimeError, match="accumulated reward without any phase steps"):
        observer.after_print_stats(frame=64, epoch_num=4, total_time=0.0)


@pytest.mark.parametrize(
    ("reward_sums", "phase_step_counts"),
    [
        ([float("nan")], [1.0]),
        ([0.0], [float("inf")]),
        ([0.0], [-1.0]),
    ],
)
def test_observer_rejects_invalid_phase_metrics(
    reward_sums: list[float], phase_step_counts: list[float]
) -> None:
    observer = EnvStatsAlgoObserver()
    observer.after_init(SimpleNamespace(writer=RecordingWriter(), games_to_track=16))
    _record_finished_phase_metrics(
        observer,
        reward_sums=reward_sums,
        phase_step_counts=phase_step_counts,
    )

    with pytest.raises(RuntimeError, match="phase 'final_hold'"):
        observer.after_print_stats(frame=64, epoch_num=4, total_time=0.0)
