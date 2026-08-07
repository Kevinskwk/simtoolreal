from pathlib import Path
import importlib.util

import torch


PATH = Path(__file__).resolve().parents[1] / "isaacsimenvs/tasks/simtoolreal/utils/stable_scrape_utils.py"
spec = importlib.util.spec_from_file_location("stable_scrape_utils", PATH)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


def test_reflected_path_preserves_bounds_and_reverses():
    offset, sign = utils.advance_reflected_path(
        torch.tensor([0.039, -0.039]), torch.tensor([1.0, -1.0]),
        distance=0.003, half_length=0.04,
    )
    assert torch.allclose(offset, torch.tensor([0.038, -0.038]))
    assert torch.equal(sign, torch.tensor([-1.0, 1.0]))


def test_phase_one_hot_rejects_unknown_phase():
    assert torch.equal(
        utils.phase_one_hot(torch.tensor([0, 1, 2])), torch.eye(3)
    )
    try:
        utils.phase_one_hot(torch.tensor([3]))
    except ValueError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("unknown phase must fail")


def test_acquisition_has_no_reward_and_force_is_safety_only():
    terms = utils.stable_scrape_reward_terms(
        phase=torch.tensor([0, 2]), pose_error_m=torch.tensor([0.0, 0.0]),
        pose_sigma_m=0.03, edge_score=torch.ones(2),
        persistent_contact=torch.ones(2, dtype=torch.bool),
        support_count=torch.tensor([2, 2]),
        grasp_retained=torch.ones(2, dtype=torch.bool),
        relative_linear_speed=torch.zeros(2),
        relative_angular_speed=torch.zeros(2), action_delta_sq_mean=torch.zeros(2),
        tool_acceleration=torch.zeros(2), normal_force_n=torch.tensor([30.0, 15.0]),
        soft_force_limit_n=12.0,
    )
    assert all(value[0].item() == 0.0 for value in terms.values())
    assert terms["tracking"][1].item() == 1.0
    assert terms["over_force"][1].item() == -9.0


def test_contact_reward_requires_retained_grasp_not_only_nearby_fingers():
    terms = utils.stable_scrape_reward_terms(
        phase=torch.tensor([2]), pose_error_m=torch.tensor([0.0]), pose_sigma_m=0.03,
        edge_score=torch.ones(1), persistent_contact=torch.ones(1, dtype=torch.bool),
        support_count=torch.tensor([5]),
        grasp_retained=torch.zeros(1, dtype=torch.bool),
        relative_linear_speed=torch.zeros(1),
        relative_angular_speed=torch.zeros(1), action_delta_sq_mean=torch.zeros(1),
        tool_acceleration=torch.zeros(1), normal_force_n=torch.zeros(1),
        soft_force_limit_n=12.0,
    )
    assert terms["contact"].item() == 0.0
    assert terms["edge"].item() == 0.0
    assert terms["tracking"].item() == 0.0
    assert terms["support"].item() == 0.0


def test_contact_rewards_are_scaled_by_target_pose_accuracy():
    terms = utils.stable_scrape_reward_terms(
        phase=torch.tensor([2]),
        pose_error_m=torch.tensor([0.03 * torch.log(torch.tensor(2.0))]),
        pose_sigma_m=0.03, edge_score=torch.ones(1),
        persistent_contact=torch.ones(1, dtype=torch.bool),
        support_count=torch.tensor([3]),
        grasp_retained=torch.ones(1, dtype=torch.bool),
        relative_linear_speed=torch.zeros(1),
        relative_angular_speed=torch.zeros(1), action_delta_sq_mean=torch.zeros(1),
        tool_acceleration=torch.zeros(1), normal_force_n=torch.zeros(1),
        soft_force_limit_n=12.0,
    )
    assert torch.allclose(terms["tracking"], torch.tensor([0.5]))
    assert torch.allclose(terms["edge"], torch.tensor([0.5]))
    assert torch.allclose(terms["contact"], torch.tensor([0.5]))
