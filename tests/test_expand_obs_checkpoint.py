from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "rl_games"))

from rl_games.torch_runner import _expand_obs_checkpoint_weights


class _DummyModel:
    def __init__(self, state):
        self._state = state

    def state_dict(self):
        return self._state


class _DummyAgent:
    def __init__(self, target_state):
        self.model = _DummyModel(target_state)


def test_expand_obs_checkpoint_weights_moves_sapg_embedding_and_zeroes_tactile_columns():
    old_obs_dim = 140
    tactile_dim = 75
    tool_contact_dim = 10
    added_obs_dim = tactile_dim + tool_contact_dim
    embed_dim = 32
    hidden4 = 16
    new_obs_dim = old_obs_dim + added_obs_dim

    source_rnn = torch.arange(hidden4 * (old_obs_dim + embed_dim), dtype=torch.float32).reshape(
        hidden4, old_obs_dim + embed_dim
    )
    source_extra = torch.arange(6 * embed_dim, dtype=torch.float32).reshape(6, embed_dim)
    source_sigma = torch.arange(6 * 29, dtype=torch.float32).reshape(6, 29)
    source_model = {
        "running_mean_std.running_mean": torch.arange(old_obs_dim, dtype=torch.float32),
        "running_mean_std.running_var": torch.arange(old_obs_dim, dtype=torch.float32) + 10.0,
        "running_mean_std.count": torch.tensor(123.0),
        "a2c_network.rnn.rnn.weight_ih_l0": source_rnn,
        "a2c_network.extra_params": source_extra,
        "a2c_network.sigma": source_sigma,
    }
    target_model = {
        "running_mean_std.running_mean": torch.empty(new_obs_dim),
        "running_mean_std.running_var": torch.empty(new_obs_dim),
        "running_mean_std.count": torch.tensor(0.0),
        "a2c_network.rnn.rnn.weight_ih_l0": torch.full(
            (hidden4, new_obs_dim + embed_dim), -999.0
        ),
        "a2c_network.extra_params": torch.full((3, embed_dim), -999.0),
        "a2c_network.sigma": torch.full((3, 29), -999.0),
    }
    weights = {"model": source_model}

    expanded = _expand_obs_checkpoint_weights(_DummyAgent(target_model), weights)
    model = expanded["model"]

    assert model["running_mean_std.running_mean"].shape == (new_obs_dim,)
    assert model["running_mean_std.running_var"].shape == (new_obs_dim,)
    assert torch.equal(model["running_mean_std.running_mean"][:old_obs_dim], source_model["running_mean_std.running_mean"])
    assert torch.equal(model["running_mean_std.running_var"][:old_obs_dim], source_model["running_mean_std.running_var"])
    assert torch.all(model["running_mean_std.running_mean"][old_obs_dim:] == 0.0)
    assert torch.all(model["running_mean_std.running_var"][old_obs_dim:] == 1.0)

    rnn = model["a2c_network.rnn.rnn.weight_ih_l0"]
    assert rnn.shape == (hidden4, new_obs_dim + embed_dim)
    assert torch.equal(rnn[:, :old_obs_dim], source_rnn[:, :old_obs_dim])
    assert torch.all(rnn[:, old_obs_dim:new_obs_dim] == 0.0)
    assert torch.equal(rnn[:, new_obs_dim:], source_rnn[:, old_obs_dim:])
    assert torch.equal(model["a2c_network.extra_params"], source_extra[:3])
    assert torch.equal(model["a2c_network.sigma"], source_sigma[:3])

    assert weights["model"]["a2c_network.rnn.rnn.weight_ih_l0"].shape == source_rnn.shape
    assert weights["model"]["a2c_network.extra_params"].shape == source_extra.shape
