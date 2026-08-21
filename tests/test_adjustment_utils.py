from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch
import json


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "isaacsimenvs/tasks/simtoolreal/utils/adjustment_utils.py"
spec = importlib.util.spec_from_file_location("adjustment_utils_test", PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_controllability_metrics_identify_margin_and_directional_speed():
    jacobian = torch.zeros(1, 6, 7)
    jacobian[0, :, :6] = torch.eye(6)
    metrics = module.arm_controllability_metrics(
        arm_position=torch.tensor([[0.95, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        arm_lower=torch.full((7,), -1.0),
        arm_upper=torch.full((7,), 1.0),
        palm_jacobian=jacobian,
        future_twist=torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        arm_velocity_limits=torch.ones(7),
        damping=1.0e-3,
    )
    assert metrics["joint_margin_rad"].item() == pytest.approx(0.05)
    assert metrics["minimum_singular_value"].item() == pytest.approx(1.0)
    assert metrics["future_velocity_ratio"].item() == pytest.approx(2.0, rel=1.0e-4)


def test_controllability_score_penalizes_each_failed_criterion():
    good = {
        "joint_margin_rad": torch.tensor([0.2]),
        "minimum_singular_value": torch.tensor([0.2]),
        "condition_number": torch.tensor([5.0]),
        "future_velocity_ratio": torch.tensor([0.5]),
    }
    bad = dict(good)
    bad["joint_margin_rad"] = torch.tensor([0.0])
    kwargs = dict(
        joint_margin_target_rad=0.1,
        singular_value_target=0.1,
        condition_number_limit=20.0,
    )
    assert module.controllability_score(good, **kwargs).item() == pytest.approx(1.0)
    assert module.controllability_score(bad, **kwargs).item() == pytest.approx(0.0)


def test_adjustment_reward_improvement_is_potential_difference():
    terms = module.adjustment_reward_terms(
        score=torch.tensor([0.7]), previous_score=torch.tensor([0.4]),
        tool_position_error_m=torch.tensor([0.0]), tool_rotation_error_rad=torch.tensor([0.0]),
        grasp_position_error_m=torch.tensor([0.0]), grasp_rotation_error_rad=torch.tensor([0.0]),
        action_delta_sq_mean=torch.tensor([0.0]), grasp_retained=torch.tensor([True]),
    )
    assert terms["score_improvement"].item() == pytest.approx(0.3)


def test_v2_scenarios_require_asset_index(tmp_path):
    payload = {
        "schema_version": 2,
        "kind": "simtoolreal_adjustment_scenarios",
        "scenarios": [{
            "source_entry_index": 0,
            "scenario_kind": "nominal",
            "joint_pos_canonical": [0.0] * 29,
            "joint_targets_canonical": [0.0] * 29,
            "object_pos_local": [0.0] * 3,
            "object_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            "future_delta": [0.0] * 6,
            "future_twist": [0.0] * 6,
        }],
    }
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="asset index"):
        module.load_adjustment_scenarios(path)
    payload["scenarios"][0]["asset_index"] = 0
    path.write_text(json.dumps(payload))
    assert module.load_adjustment_scenarios(path)["schema_version"] == 2


def test_allen_target_orbits_screw_axis_not_long_handle_axis():
    # T_palm_tool identity with the palm displaced from the screw pivot in tool X.
    current_pos = torch.tensor([[-0.10, 0.0, 0.0]])
    current_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    angle = torch.tensor([torch.pi / 2])
    target_pos, target_quat = module.orbit_palm_tool_about_screw_axis(
        current_pos,
        current_quat,
        angle,
        torch.tensor([0.0, 0.0, -1.0]),
        torch.zeros(3),
    )
    orbit, position, orientation = module.screw_axis_orbit_errors(
        current_pos,
        current_quat,
        target_pos,
        target_quat,
        torch.tensor([0.0, 0.0, -1.0]),
        torch.zeros(3),
    )
    assert orbit.abs().item() == pytest.approx(torch.pi / 2, rel=1e-5)
    assert position.item() == pytest.approx(2 ** 0.5 * 0.10, rel=1e-5)
    assert orientation.item() == pytest.approx(torch.pi / 2, rel=1e-5)
    # An orbit about local X would leave this point unchanged, so Y motion is
    # the regression guard against restoring screwdriver handle-axis behavior.
    inverse_target_pos = module._quat_apply(module._quat_inv(target_quat), -target_pos)
    assert abs(inverse_target_pos[0, 1].item()) > 0.09


def test_allen_zero_orbit_is_identity():
    position = torch.tensor([[0.02, -0.03, 0.18]])
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    target_pos, target_quat = module.orbit_palm_tool_about_screw_axis(
        position, quat, torch.zeros(1),
        torch.tensor([0.0, 0.0, -1.0]), torch.tensor([0.06, 0.0, -0.03]),
    )
    assert torch.allclose(target_pos, position, atol=1e-6)
    assert torch.allclose(target_quat, quat, atol=1e-6)
