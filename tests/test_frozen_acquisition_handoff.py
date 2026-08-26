from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rl_games"))

from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.common.a2c_common import A2CBase


class _FrozenModel:
    def __call__(self, inputs):
        batch = inputs["obs"].shape[0]
        return {
            "mus": torch.full((batch, 2), 0.75),
            "rnn_states": [torch.ones(1, batch, 3)],
        }


def test_frozen_action_is_used_only_in_acquisition_and_states_are_separated():
    agent = A2CAgent.__new__(A2CAgent)
    agent.frozen_acquisition_enabled = True
    agent.frozen_acquisition_cfg = {
        "observation_dim": 2, "phase_offset": 2, "coefficient_id": 50.0,
    }
    agent.intr_reward_coef_embd = None
    agent.frozen_acquisition_model = _FrozenModel()
    agent.frozen_acquisition_rnn_states = [torch.zeros(1, 2, 3)]
    agent.is_rnn = True
    obs = {"obs": torch.tensor([[1.0, 2.0, 1.0, 0.0, 0.0],
                                [3.0, 4.0, 0.0, 1.0, 0.0]])}
    result = {"actions": torch.zeros(2, 2), "rnn_states": [torch.ones(1, 2, 3)]}

    result, mask = A2CAgent.select_rollout_actions(agent, obs, result)

    assert torch.equal(mask, torch.tensor([0.0, 1.0]))
    assert torch.all(result["actions"][0] == 0.75)
    assert torch.all(result["actions"][1] == 0.0)
    assert torch.all(result["rnn_states"][0][:, 0] == 0.0)
    assert torch.all(agent.frozen_acquisition_rnn_states[0][:, 1] == 0.0)


def test_frozen_phase_can_be_read_from_privileged_state_without_actor_suffix():
    agent = A2CAgent.__new__(A2CAgent)
    agent.frozen_acquisition_enabled = True
    agent.frozen_acquisition_cfg = {
        "observation_dim": 2,
        "phase_source": "states",
        "phase_offset": 1,
        "coefficient_id": 0.0,
    }
    agent.intr_reward_coef_embd = None
    agent.frozen_acquisition_model = _FrozenModel()
    agent.frozen_acquisition_rnn_states = [torch.zeros(1, 2, 3)]
    agent.is_rnn = True
    obs = {
        "obs": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "states": torch.tensor([
            [9.0, 1.0, 0.0, 0.0],
            [9.0, 0.0, 1.0, 0.0],
        ]),
    }
    result = {"actions": torch.zeros(2, 2), "rnn_states": [torch.ones(1, 2, 3)]}

    result, mask = A2CAgent.select_rollout_actions(agent, obs, result)

    assert torch.equal(mask, torch.tensor([0.0, 1.0]))
    assert torch.all(result["actions"][0] == 0.75)
    assert torch.all(result["actions"][1] == 0.0)


def test_masked_gae_breaks_credit_assignment_before_handoff():
    stub = type("Stub", (), {"horizon_length": 3, "gamma": 1.0, "tau": 1.0})()
    rewards = torch.ones(3, 1, 1)
    values = torch.zeros_like(rewards)
    masks = torch.tensor([[0.0], [1.0], [1.0]])
    advantages = A2CBase.discount_values_masks(
        stub, torch.zeros(1), torch.zeros(1, 1), torch.zeros(3, 1),
        values, rewards, masks,
    )
    assert advantages[0].item() == 0.0
    assert advantages[1].item() == 2.0
    assert advantages[2].item() == 1.0
