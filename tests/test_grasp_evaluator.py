from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "isaacsimenvs/tasks/simtoolreal/utils/grasp_evaluator.py"
spec = importlib.util.spec_from_file_location("grasp_evaluator", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_pose_matrix_round_trip():
    position = np.array([0.2, -0.4, 0.7])
    quaternion = np.array([0.9238795325, 0.0, 0.3826834324, 0.0])
    transform = module.pose_matrix(position, quaternion)
    recovered_position, recovered_quaternion = module.matrix_pose(transform)
    assert np.allclose(recovered_position, position)
    assert abs(np.dot(recovered_quaternion, quaternion)) == pytest.approx(1.0)


def test_urdf_fk_jacobian_matches_finite_difference():
    kinematics = module.UrdfKinematics(
        REPO_ROOT / "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
    )
    arm = np.array([-1.5, 1.3, -0.2, 1.2, -0.1, 1.5, 1.3])
    hand = np.zeros(22)
    base = np.eye(4)
    base[1, 3] = 0.8
    palm, jacobian, _ = kinematics.palm_fk_jacobian(arm, hand, base)
    epsilon = 1e-6
    finite = np.zeros((3, 7))
    for index in range(7):
        shifted = arm.copy()
        shifted[index] += epsilon
        shifted_palm, _, _ = kinematics.palm_fk_jacobian(shifted, hand, base)
        finite[:, index] = (shifted_palm[:3, 3] - palm[:3, 3]) / epsilon
    assert np.allclose(jacobian[:3], finite, atol=2e-6)


def test_tool_box_clearance_detects_penetration():
    bounds = (-0.1, -0.02, -0.01, 0.1, 0.02, 0.01)
    pose = np.eye(4)
    pose[2, 3] = 0.009
    clearance = module.tool_box_table_clearance(
        pose, bounds, np.zeros(3), np.array([0.0, 0.0, 1.0])
    )
    assert clearance == pytest.approx(-0.001)


def test_grasp_fingerprint_is_order_independent():
    assert module.grasp_fingerprint({"a": 1, "b": 2}) == module.grasp_fingerprint({"b": 2, "a": 1})


def test_functional_edge_clearance_uses_palm_in_tool_frame():
    palm_to_tool = np.eye(4)
    palm_to_tool[0, 3] = 0.04
    bounds = (-0.1, -0.02, -0.01, 0.1, 0.02, 0.01)
    assert module.bounds_edge_clearance(palm_to_tool, bounds) == pytest.approx(0.14)


def test_functional_edge_threshold_is_explicit():
    thresholds = module.GraspEvaluatorThresholds()
    assert thresholds.functional_edge_clearance_m == pytest.approx(0.02)


def test_arm_controllability_thresholds_are_explicit():
    thresholds = module.GraspEvaluatorThresholds()
    assert thresholds.arm_joint_margin_rad == pytest.approx(0.05)
    assert thresholds.minimum_jacobian_singular_value == pytest.approx(0.10)
    assert thresholds.maximum_jacobian_condition_number == pytest.approx(20.0)


@pytest.mark.parametrize(
    ("overrides", "failed_gate"),
    [
        ({"minimum_joint_margin_rad": 0.049}, "arm_joint_margin"),
        ({"minimum_jacobian_singular_value": 0.099}, "arm_singularity"),
        ({"maximum_jacobian_condition_number": 20.01}, "arm_singularity"),
        ({"maximum_arm_velocity_ratio": 1.01}, "arm_velocity"),
    ],
)
def test_arm_controllability_gates_reject_hard_cases(overrides, failed_gate):
    values = {
        "initial_joint_violation_rad": 0.0,
        "minimum_joint_margin_rad": 0.1,
        "minimum_jacobian_singular_value": 0.2,
        "maximum_jacobian_condition_number": 10.0,
        "maximum_arm_velocity_ratio": 0.5,
    }
    values.update(overrides)
    gates = module.arm_controllability_gates(
        **values, thresholds=module.GraspEvaluatorThresholds()
    )
    assert not gates[failed_gate]
    assert all(passed for name, passed in gates.items() if name != failed_gate)


def test_joint_limit_validity_is_distinct_from_joint_margin():
    gates = module.arm_controllability_gates(
        initial_joint_violation_rad=0.0,
        minimum_joint_margin_rad=0.0,
        minimum_jacobian_singular_value=0.2,
        maximum_jacobian_condition_number=10.0,
        maximum_arm_velocity_ratio=0.5,
        thresholds=module.GraspEvaluatorThresholds(),
    )
    assert gates["joint_limits"]
    assert not gates["arm_joint_margin"]
