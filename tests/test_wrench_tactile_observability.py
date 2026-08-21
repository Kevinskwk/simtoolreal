from pathlib import Path
import importlib.util
import sys

import torch


def load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "isaacsimenvs/tasks/simtoolreal/utils/wrench_tactile_observability.py"
    )
    spec = importlib.util.spec_from_file_location("wrench_observability_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


utils = load_module()


def make_episode(length: int = 12) -> dict:
    return {
        "proprio_state": torch.randn(length, 71),
        "tactile_compact": torch.rand(length, 5, 5),
        "commanded_force_palm_n": torch.zeros(length, 3),
        "commanded_torque_palm_nm": torch.zeros(length, 3),
        "wrench_mode": torch.arange(length) % len(utils.WRENCH_MODE_NAMES),
        "valid": torch.ones(length, dtype=torch.bool),
        "metadata": {"episode_id": "episode-0", "split_group": "grasp-bank-0"},
    }


def test_episode_contract_and_window_exclude_wrench_from_inputs(tmp_path):
    episode = make_episode()
    assert utils.validate_episode(episode) == 12
    shard = tmp_path / "shard_00000.pt"
    torch.save({"episodes": [episode]}, shard)
    split = utils.stable_split("grasp-bank-0", 0)
    refs = utils.catalog([shard])
    dataset = utils.WrenchWindowDataset(refs, split, history=5)
    sample = dataset[0]
    assert sample["state"].shape == (5, 71)
    assert sample["compact"].shape == (5, 5, 5)
    assert set(sample) == {"state", "compact", "force", "torque", "mode"}


def test_matched_probe_shapes():
    batch, history = 4, 5
    state = torch.randn(batch, history, 71)
    compact = torch.rand(batch, history, 5, 5)
    for variant in ("state", "compact", "state_compact"):
        model = utils.WrenchProbe(71, variant)
        logits, wrench = model(state, compact)
        assert logits.shape == (batch, len(utils.WRENCH_MODE_NAMES))
        assert wrench.shape == (batch, 6)


def test_impulse_episode_contract_and_probe_shapes():
    episode = make_episode()
    episode["finger_normal_impulse_palm_ns"] = torch.randn(12, 5, 3)
    episode["finger_tangential_impulse_palm_ns"] = torch.randn(12, 5, 3)
    assert utils.validate_impulse_episode(episode) == 12

    state = torch.randn(4, 5, 71)
    compact = torch.rand(4, 5, 5, 5)
    impulse = torch.randn(4, 5, 5, 6)
    for variant in (
        "normal_impulse", "tangential_impulse", "impulse", "impulse_compact"
    ):
        model = utils.WrenchProbe(71, variant)
        logits, wrench = model(state, compact, impulse)
        assert logits.shape == (4, len(utils.WRENCH_MODE_NAMES))
        assert wrench.shape == (4, 6)


def test_impulse_contract_rejects_nonfinite_measurements():
    episode = make_episode()
    episode["finger_normal_impulse_palm_ns"] = torch.zeros(12, 5, 3)
    episode["finger_tangential_impulse_palm_ns"] = torch.zeros(12, 5, 3)
    episode["finger_normal_impulse_palm_ns"][0, 0, 0] = float("nan")
    try:
        utils.validate_impulse_episode(episode)
    except ValueError as exc:
        assert "NaN or Inf" in str(exc)
    else:
        raise AssertionError("non-finite impulse measurement was accepted")
