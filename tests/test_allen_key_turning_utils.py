from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import xml.etree.ElementTree as ET

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


def test_stick_slip_friction_has_no_constant_torque_at_rest():
    torque, stuck, restuck, clipped = module.stick_slip_torsional_friction(
        torch.zeros(2),
        torch.zeros(2),
        torch.ones(2, dtype=torch.bool),
        kinetic_limit_nm=0.08,
        static_to_kinetic_ratio=1.5,
        stiction_stiffness_nm_per_rad=1.0,
        damping_nm_per_radps=0.05,
        maximum_abs_torque_nm=0.16,
        kinetic_transition_speed_radps=0.05,
        restick_speed_radps=0.03,
    )
    assert torch.equal(torque, torch.zeros(2))
    assert bool(stuck.all())
    assert not bool(restuck.any())
    assert not bool(clipped.any())


def test_stick_slip_friction_reacts_then_breaks_away():
    torque, stuck, _, clipped = module.stick_slip_torsional_friction(
        torch.tensor([0.02, 0.20]),
        torch.zeros(2),
        torch.ones(2, dtype=torch.bool),
        kinetic_limit_nm=0.08,
        static_to_kinetic_ratio=1.5,
        stiction_stiffness_nm_per_rad=1.0,
        damping_nm_per_radps=0.05,
        maximum_abs_torque_nm=0.16,
        kinetic_transition_speed_radps=0.05,
        restick_speed_radps=0.03,
    )
    assert torque[0].item() == pytest.approx(-0.02)
    assert bool(stuck[0])
    assert torque[1].item() == pytest.approx(-0.08)
    assert not bool(stuck[1])
    assert not bool(clipped.any())


def test_slipping_friction_opposes_motion_and_resticks_near_zero():
    torque, stuck, restuck, clipped = module.stick_slip_torsional_friction(
        torch.tensor([0.2, 0.2]),
        torch.tensor([1.0, 0.01]),
        torch.zeros(2, dtype=torch.bool),
        kinetic_limit_nm=0.08,
        static_to_kinetic_ratio=1.5,
        stiction_stiffness_nm_per_rad=1.0,
        damping_nm_per_radps=0.05,
        maximum_abs_torque_nm=0.16,
        kinetic_transition_speed_radps=0.05,
        restick_speed_radps=0.03,
    )
    assert torque[0].item() < -0.08
    assert not bool(stuck[0])
    assert bool(stuck[1]) and bool(restuck[1])
    assert abs(torque[1].item()) <= 0.12
    assert not bool(clipped.any())


def test_stick_slip_friction_supports_per_environment_coefficients():
    torque, stuck, _, clipped = module.stick_slip_torsional_friction(
        torch.tensor([0.2, 0.2]),
        torch.tensor([1.0, 1.0]),
        torch.zeros(2, dtype=torch.bool),
        kinetic_limit_nm=torch.tensor([0.02, 0.08]),
        static_to_kinetic_ratio=1.5,
        stiction_stiffness_nm_per_rad=1.0,
        damping_nm_per_radps=torch.tensor([0.01, 0.04]),
        maximum_abs_torque_nm=torch.tensor([0.04, 0.16]),
        kinetic_transition_speed_radps=0.05,
        restick_speed_radps=0.03,
    )
    assert not bool(stuck.any())
    assert torque.tolist() == pytest.approx([-0.03, -0.12], abs=1e-6)
    assert not bool(clipped.any())


def test_stick_slip_friction_caps_large_damping_transient():
    torque, stuck, _, clipped = module.stick_slip_torsional_friction(
        torch.tensor([0.2]),
        torch.tensor([20.0]),
        torch.zeros(1, dtype=torch.bool),
        kinetic_limit_nm=0.08,
        static_to_kinetic_ratio=1.5,
        stiction_stiffness_nm_per_rad=1.0,
        damping_nm_per_radps=0.6,
        maximum_abs_torque_nm=0.16,
        kinetic_transition_speed_radps=0.05,
        restick_speed_radps=0.03,
    )
    assert not bool(stuck[0])
    assert bool(clipped[0])
    assert torque.item() == pytest.approx(-0.16)


def test_stick_slip_friction_rejects_mismatched_coefficient_shape():
    with pytest.raises(ValueError, match="kinetic_limit_nm must be scalar or match"):
        module.stick_slip_torsional_friction(
            torch.zeros(2),
            torch.zeros(2),
            torch.ones(2, dtype=torch.bool),
            kinetic_limit_nm=torch.zeros(3),
            static_to_kinetic_ratio=1.5,
            stiction_stiffness_nm_per_rad=1.0,
            damping_nm_per_radps=0.05,
            maximum_abs_torque_nm=0.16,
            kinetic_transition_speed_radps=0.05,
            restick_speed_radps=0.03,
        )


def test_finger_effort_penalty_ignores_normal_grip_and_penalizes_excess():
    torque = torch.tensor([[0.2, 0.7, 1.0]])
    limits = torch.ones_like(torque)
    penalty, maximum, saturation = module.finger_effort_soft_penalty(
        torque, limits, 0.7
    )
    assert penalty.item() == pytest.approx(0.03)
    assert maximum.item() == pytest.approx(1.0)
    assert saturation.item() == pytest.approx(1.0 / 3.0)


def test_loaded_grasp_requires_palm_multifinger_contact_and_low_slip():
    quality, valid = module.loaded_grasp_quality(
        torch.tensor([1.0, 0.0, 0.5, 1.0]),
        torch.tensor([True, False, True, True]),
        torch.tensor([2, 5, 1, 3]),
        torch.tensor([0.0, 0.0, 0.0, 0.09]),
        torch.tensor([0.0, 0.0, 0.0, 0.0]),
        minimum_contact_fingers=2,
        maximum_relative_linear_speed_mps=0.08,
        maximum_relative_angular_speed_radps=2.0,
    )
    assert valid.tolist() == [True, False, False, False]
    assert quality[0].item() == pytest.approx(1.0)
    assert quality[1].item() == pytest.approx(0.0)
    assert quality[2].item() == pytest.approx(0.25)
    assert quality[3].item() == pytest.approx(0.0)


def test_progress_gate_blocks_ungrasped_gain_but_keeps_regression_penalty():
    gated = module.gate_positive_progress(
        torch.tensor([1.0, 1.0, -1.0, -1.0]),
        torch.tensor([0.0, 0.5, 0.0, 1.0]),
    )
    assert gated.tolist() == pytest.approx([0.0, 0.5, -1.0, -1.0])


def test_grasp_confirmation_requires_consecutive_valid_steps_and_fires_once():
    count = torch.zeros(2, dtype=torch.long)
    confirmed = torch.zeros(2, dtype=torch.bool)
    sequence = ([True, True], [True, False], [True, True])
    events = []
    for values in sequence:
        count, just_confirmed = module.update_consecutive_grasp_hold(
            torch.tensor(values), count, confirmed, required_steps=3
        )
        confirmed |= just_confirmed
        events.append(just_confirmed.tolist())
    assert events == [[False, False], [False, False], [True, False]]
    assert count.tolist() == [3, 1]
    count, just_confirmed = module.update_consecutive_grasp_hold(
        torch.tensor([True, True]), count, confirmed, required_steps=3
    )
    assert not bool(just_confirmed.any())


def test_grasp_hold_resets_immediately_after_invalid_step():
    count, just_confirmed = module.update_consecutive_grasp_hold(
        torch.tensor([False]),
        torch.tensor([19], dtype=torch.long),
        torch.tensor([False]),
        required_steps=20,
    )
    assert count.item() == 0
    assert not bool(just_confirmed.item())


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
    root = ET.fromstring(text)
    grip_visual = next(
        visual for visual in root.findall(".//visual")
        if visual.find("material").get("name") == "grip"
    )
    values = [
        float(value)
        for value in grip_visual.find("geometry/mesh").get("scale").split()
    ]
    assert values == pytest.approx([0.264, 0.030 / math.sqrt(3.0), 0.030 / math.sqrt(3.0)])
