from pathlib import Path
import importlib.util
from types import SimpleNamespace

import torch


def _load_logging_utils():
    path = Path(__file__).resolve().parents[1] / "isaacsimenvs/tasks/simtoolreal/utils/logging_utils.py"
    spec = importlib.util.spec_from_file_location("logging_utils_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_log_step_metrics_adds_compact_curriculum_and_reward_component_means():
    log_step_metrics = _load_logging_utils().log_step_metrics
    env = SimpleNamespace(
        cfg=SimpleNamespace(
            termination=SimpleNamespace(
                max_consecutive_successes=2,
            )
        ),
        _successes=torch.tensor([0, 2, 3]),
        _termination_reasons={"fall": torch.tensor([False, True, False])},
        _reward_terms={
            "base_rew": torch.tensor([1.0, 2.0, 3.0]),
            "total_reward": torch.tensor([2.0, 4.0, 6.0]),
        },
        _prev_episode_successes=torch.tensor([0, 1, 2]),
        _current_success_tolerance=0.025,
        _curriculum_success_mean=1.5,
        extras={},
    )

    log_step_metrics(env)

    assert env.extras["curriculum/success_tolerance"] == 0.025
    assert env.extras["curriculum/current_success_tolerance"] == 0.025
    assert env.extras["curriculum/success_mean"] == 1.5
    assert env.extras["reward/base_rew"].item() == 2.0
    assert env.extras["reward/total_reward"].item() == 4.0
