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


def test_palm_keypoint_error_couples_translation_and_rotation():
    position = torch.zeros(1, 3)
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    keypoints = torch.tensor([
        [0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [-0.05, 0.0, 0.0],
    ])
    translated = module.palm_keypoint_error(
        position, identity, torch.tensor([[0.01, 0.0, 0.0]]), identity, keypoints
    )
    rotated_quat = torch.tensor([[2 ** -0.5, 0.0, 0.0, 2 ** -0.5]])
    rotated = module.palm_keypoint_error(
        position, identity, position, rotated_quat, keypoints
    )
    assert translated.item() == pytest.approx(0.01, rel=1e-5)
    assert rotated.item() == pytest.approx(2 ** 0.5 * 0.05, rel=1e-5)


def test_in_place_palm_rotation_preserves_grasp_center_in_tool_frame():
    position = torch.tensor([[0.03, -0.02, 0.10]])
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    target_pos, target_quat = module.rotate_palm_about_tool_axis_in_place(
        position,
        quat,
        torch.tensor([torch.pi / 3]),
        torch.tensor([0.0, 0.0, -1.0]),
    )
    current_center = module._quat_apply(module._quat_inv(quat), -position)
    target_center = module._quat_apply(module._quat_inv(target_quat), -target_pos)
    assert torch.allclose(target_center, current_center, atol=1e-6)
    alignment = torch.abs((quat * target_quat).sum(-1)).clamp(0.0, 1.0)
    assert (2.0 * torch.acos(alignment)).item() == pytest.approx(torch.pi / 3, rel=1e-5)


def test_allen_workspace_tiers_match_measured_curriculum_regions():
    tiers = module.allen_workspace_tier(
        torch.tensor([0.10, 0.10, -0.20]),
        torch.tensor([0.05, 0.05, -0.20]),
        torch.tensor([0.50, 0.65, 0.50]),
        torch.tensor([-150.0, -30.0, 90.0]),
    )
    assert tiers.tolist() == [0, 1, 2]


def test_allen_workspace_weights_preserve_tier_probability_mass():
    tiers = torch.tensor([0, 0, 1, 2, 2, 2])
    weights = module.allen_workspace_sampling_weights(tiers, (0.6, 0.3, 0.1))
    assert weights[tiers == 0].sum().item() == pytest.approx(0.6)
    assert weights[tiers == 1].sum().item() == pytest.approx(0.3)
    assert weights[tiers == 2].sum().item() == pytest.approx(0.1)
    with pytest.raises(ValueError, match="absent tier"):
        module.allen_workspace_sampling_weights(torch.tensor([0, 1]), (0.5, 0.4, 0.1))


def test_allen_pair_curriculum_only_removes_prevalidated_edges():
    valid = torch.tensor([[False, True, True], [True, False, True], [True, True, False]])
    translation = torch.tensor([
        [0.0, 0.03, 0.07], [0.03, 0.0, 0.04], [0.07, 0.04, 0.0]
    ])
    rotation = torch.tensor([
        [0.0, 20.0, 30.0], [20.0, 0.0, 70.0], [30.0, 70.0, 0.0]
    ])
    result = module.allen_pair_curriculum_mask(
        valid, translation, rotation,
        maximum_translation_m=0.06, maximum_rotation_deg=60.0,
    )
    assert result.tolist() == [
        [False, True, False], [True, False, False], [False, False, False]
    ]
