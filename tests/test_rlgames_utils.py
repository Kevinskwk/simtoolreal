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
