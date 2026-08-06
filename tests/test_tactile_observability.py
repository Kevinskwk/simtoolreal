from pathlib import Path
import importlib.util
import sys

import pytest
import torch


def _load_utils():
    path = (
        Path(__file__).resolve().parents[1]
        / "isaacsimenvs/tasks/simtoolreal/utils/tactile_observability.py"
    )
    spec = importlib.util.spec_from_file_location("tactile_observability_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


utils = _load_utils()


def make_episode(length: int = 48) -> dict:
    force = torch.zeros(length)
    force[10:25] = 2.0
    force[31:40] = 2.0
    edge = torch.full((length,), 0.02)
    edge[10:25] = 0.002
    edge[31:40] = 0.002
    position = torch.zeros(length, 3)
    position[20:, 0] = torch.arange(length - 20) * 0.002
    quaternion = torch.zeros(length, 4)
    quaternion[:, 0] = 1.0
    return {
        "actor_state": torch.zeros(length, 140),
        "deployable_geometry": torch.zeros(length, 13),
        "oracle_geometry": torch.stack((edge, edge, torch.zeros(length)), -1),
        "tactile_compact": torch.zeros(length, 5, 5),
        "tactile_raw": torch.zeros(length, 5, 12, 12, dtype=torch.uint8),
        "force_interval_n": force,
        "edge_error_m": edge,
        "palm_tool_pos": position,
        "palm_tool_quat": quaternion,
        "fingertip_count": torch.full((length,), 3, dtype=torch.int8),
        "object_fallen": torch.zeros(length, dtype=torch.bool),
        "grasp_reference_pos": torch.zeros(3),
        "grasp_reference_quat": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "metadata": {"episode_id": "episode-0", "tool_id": "tool-0"},
    }


def test_episode_contract_rejects_missing_and_nonfinite_data():
    episode = make_episode()
    assert utils.validate_episode(episode) == 48
    del episode["tactile_raw"]
    with pytest.raises(ValueError, match="missing required"):
        utils.validate_episode(episode)
    episode = make_episode()
    episode["actor_state"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        utils.validate_episode(episode)


def test_labels_use_persistent_contact_and_future_censoring():
    labels = utils.derive_labels(make_episode())
    assert torch.all(labels["contact_mode"][10:25] == 1)
    assert labels["contact_mode"][0] == 0
    assert labels["onset"][5] == 1
    assert labels["loss"][20] == 1
    assert not labels["onset_eligible"][-1]
    assert not labels["loss_eligible"][-1]
    assert labels["slip"].sum() > 0
    assert labels["instability"].sum() > 0


def test_quaternion_angle_is_sign_invariant():
    a = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert utils.quaternion_angle_rad(a, -a).item() == pytest.approx(0.0)


def test_probe_shapes_for_all_input_encoders():
    batch, history = 4, 5
    state = torch.randn(batch, history, 140)
    compact = torch.randn(batch, history, 5, 5)
    raw = torch.randint(0, 256, (batch, history, 5, 12, 12), dtype=torch.uint8)
    for variant in ("state", "compact", "state_compact", "state_raw"):
        output = utils.ObservabilityProbe(variant=variant)(state, compact, raw)
        assert output["contact_mode"].shape == (batch, 3)
        assert output["loss"].shape == (batch,)
        assert output["relative_speed"].shape == (batch, 2)


def test_window_dataset_preserves_temporal_order(tmp_path):
    episode = make_episode()
    episode["actor_state"][:, 0] = torch.arange(episode["actor_state"].shape[0])
    # Ensure this tool hashes into a train split without weakening production splitting.
    index = 0
    while utils.assign_split(episode["metadata"], seed=0) != "train":
        index += 1
        episode["metadata"]["tool_id"] = f"tool-{index}"
    shard = tmp_path / "shard_00000.pt"
    torch.save({"schema_version": 1, "episodes": [episode]}, shard)
    dataset = utils.EpisodeWindowDataset([shard], "train", history=5)
    sample = dataset[0]
    assert sample["state"].shape == (5, 140)
    assert sample["state_geometry"].shape == (5, 153)
    assert torch.equal(sample["state"][:, 0], torch.arange(5).float())
    assert sample["compact"].shape == (5, 5, 5)
    assert sample["raw"].shape == (5, 5, 12, 12)


def test_hard_loss_uses_persistent_fingertip_loss():
    episode = make_episode()
    episode["fingertip_count"][20:35] = 1
    labels = utils.derive_labels(episode)
    assert not labels["hard_loss"][20]
    assert labels["hard_loss"][34]


def test_catalog_cache_skips_label_recomputation(tmp_path, monkeypatch):
    episode = make_episode()
    index = 0
    while utils.assign_split(episode["metadata"], seed=0) != "train":
        index += 1
        episode["metadata"]["tool_id"] = f"tool-{index}"
    shard = tmp_path / "shard_00000.pt"
    cache_dir = tmp_path / "cache"
    torch.save({"schema_version": 1, "episodes": [episode]}, shard)

    first = utils.EpisodeCatalog(
        [shard], cache_dir=cache_dir, preload=False
    )
    assert len(first.records) == 1
    assert len(list(cache_dir.glob("catalog_*.pt"))) == 1

    def fail_if_called(*args, **kwargs):
        raise AssertionError("derive_labels should not run on a catalog cache hit")

    monkeypatch.setattr(utils, "derive_labels", fail_if_called)
    cached = utils.EpisodeCatalog(
        [shard], cache_dir=cache_dir, preload=False
    )
    dataset = utils.EpisodeWindowDataset(
        [shard], "train", history=5, catalog=cached
    )
    assert dataset[0]["state"].shape == (5, 140)
