from pathlib import Path
import importlib.util
import sys

import numpy as np


def _load_training_module():
    path = Path(__file__).resolve().parents[1] / "scripts/train_tactile_observability_probe.py"
    spec = importlib.util.spec_from_file_location("tactile_probe_training_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


training = _load_training_module()


def test_histogram_cluster_bootstrap_is_finite_and_deterministic():
    rng = np.random.default_rng(4)
    episode_ids = np.repeat(np.asarray([f"episode-{index}" for index in range(12)]), 40)
    values = {"contact_mode_episode_id": episode_ids}
    for head_index, name in enumerate(training.PRIMARY_HEADS):
        target = rng.random(episode_ids.size) < (0.15 + 0.05 * head_index)
        score = np.clip(0.15 + 0.65 * target + rng.normal(0.0, 0.15, target.size), 0.0, 1.0)
        values[f"{name}_episode_id"] = episode_ids
        values[f"{name}_target"] = target.astype(np.float32)
        values[f"{name}_score"] = score.astype(np.float32)

    first = training.bootstrap_primary_auprc(values, repetitions=40, score_bins=64)
    second = training.bootstrap_primary_auprc(values, repetitions=40, score_bins=64)
    assert np.all(np.isfinite(first))
    assert first == second
    assert 0.0 <= first[0] <= first[1] <= 1.0
