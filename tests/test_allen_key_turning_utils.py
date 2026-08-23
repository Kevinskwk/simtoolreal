from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
PATH = (
    ROOT
    / "isaacsimenvs/tasks/simtoolreal/utils/allen_key_turning_utils.py"
)
SPEC = importlib.util.spec_from_file_location("allen_key_turning_utils_test", PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_unwrapped_angle_crosses_pi_without_jump():
    previous = torch.tensor([math.radians(179.0), math.radians(-179.0)])
    current = torch.tensor([math.radians(-179.0), math.radians(179.0)])
    cumulative, delta = module.update_unwrapped_angle(
        previous, current, torch.zeros(2)
    )
    assert torch.rad2deg(delta).tolist() == pytest.approx([2.0, -2.0], abs=1e-4)
    assert torch.allclose(cumulative, delta)


def test_turn_goal_keeps_screw_pivot_fixed():
    initial_position = torch.tensor([[0.1, -0.2, 0.5]])
    initial_quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    pivot_tool = torch.tensor([[0.192, 0.0, -0.03]])
    axis_tool = torch.tensor([[0.0, 0.0, -1.0]])
    goal_position, goal_quaternion = module.turn_goal_pose(
        initial_position,
        initial_quaternion,
        pivot_tool,
        axis_tool,
        torch.tensor([math.pi / 2.0]),
    )
    initial_pivot = initial_position + module.quat_apply(initial_quaternion, pivot_tool)
    goal_pivot = goal_position + module.quat_apply(goal_quaternion, pivot_tool)
    assert torch.allclose(initial_pivot, goal_pivot, atol=1e-6)


def test_finger_effort_penalty_ignores_normal_grip_and_penalizes_excess():
    torque = torch.tensor([[0.2, 0.7, 1.0]])
    limits = torch.ones_like(torque)
    penalty, maximum, saturation = module.finger_effort_soft_penalty(
        torque, limits, 0.7
    )
    assert penalty.item() == pytest.approx(0.03)
    assert maximum.item() == pytest.approx(1.0)
    assert saturation.item() == pytest.approx(1.0 / 3.0)


def test_curriculum_requires_both_acquisition_and_conditional_turning():
    ready, acquisition, conditional = module.turning_curriculum_ready(
        700,
        1000,
        450,
        minimum_episodes=1000,
        acquisition_threshold=0.6,
        conditional_turn_threshold=0.6,
    )
    assert ready
    assert acquisition == pytest.approx(0.7)
    assert conditional == pytest.approx(450 / 700)
    not_ready, _, _ = module.turning_curriculum_ready(
        500,
        1000,
        450,
        minimum_episodes=1000,
        acquisition_threshold=0.6,
        conditional_turn_threshold=0.6,
    )
    assert not not_ready


def test_generated_urdf_preserves_requested_dimensions(tmp_path):
    mesh = tmp_path / "unit.obj"
    mesh.write_text("v 0 0 0\n")
    text = module.allen_key_urdf_text(
        long_handle_length_m=0.264,
        handle_across_flats_m=0.030,
        short_leg_length_m=0.060,
        elbow_x_m=0.192,
        mesh_path=mesh,
    )
    assert 'scale="0.2640000' in text
    assert 'xyz="0.1920000 0 -0.0300000"' in text
    assert str(mesh) in text
