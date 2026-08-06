from pathlib import Path
import importlib.util
import sys

import numpy as np
import pytest


def _load_utils():
    path = (
        Path(__file__).resolve().parents[1]
        / "isaacsimenvs/tasks/simtoolreal/utils/virtual_pose_offset_eval.py"
    )
    spec = importlib.util.spec_from_file_location("virtual_pose_offset_eval_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


utils = _load_utils()


def test_pose_window_stability_checks_motion_and_velocity():
    positions = np.zeros((30, 3))
    positions[:, 2] = np.linspace(0.0, 0.0005, 30)
    quaternions = np.tile([1.0, 0.0, 0.0, 0.0], (30, 1))
    linear_speed = np.full(30, 0.004)
    angular_speed = np.full(30, 0.05)

    stable, metrics = utils.pose_window_is_stable(
        positions, quaternions, linear_speed, angular_speed
    )

    assert stable
    assert metrics["position_deviation_m"] == pytest.approx(0.0005)
    linear_speed[-1] = 0.006
    assert not utils.pose_window_is_stable(
        positions, quaternions, linear_speed, angular_speed
    )[0]


def test_pose_window_stability_is_quaternion_sign_invariant():
    positions = np.zeros((2, 3))
    quaternions = np.asarray([[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]])
    stable, metrics = utils.pose_window_is_stable(
        positions, quaternions, np.zeros(2), np.zeros(2)
    )
    assert stable
    assert metrics["orientation_deviation_deg"] == pytest.approx(0.0)


def test_force_calibration_selects_nearest_safe_stable_sample_and_bracketing():
    result = utils.select_force_calibration(
        np.asarray([0.001, 0.0, -0.001, -0.002]),
        np.asarray([0.0, 1.0, 4.4, 22.0]),
        np.asarray([True, True, True, True]),
    )
    assert result["selected_index"] == 2
    assert result["offset_m"] == pytest.approx(-0.001)
    assert result["bracketed"]


def test_force_calibration_reports_unbracketed_and_rejects_no_valid_samples():
    result = utils.select_force_calibration(
        np.asarray([0.001, 0.0]),
        np.asarray([0.0, 1.0]),
        np.asarray([True, True]),
    )
    assert not result["bracketed"]
    with pytest.raises(ValueError, match="no stable"):
        utils.select_force_calibration(
            np.asarray([0.001]), np.asarray([2.0]), np.asarray([False])
        )


def test_spearman_supports_ties_and_detects_monotonic_response():
    assert utils.spearman_correlation(
        np.asarray([0.0, 1.0, 2.0, 3.0]),
        np.asarray([1.0, 1.0, 2.0, 4.0]),
    ) > 0.9


def test_transition_metrics_measure_contact_loss_and_settling():
    force = np.asarray([0.0, 2.0] + [4.2] * 30)
    contact = np.asarray([False, False] + [True] * 30)

    result = utils.transition_force_metrics(
        force, contact, control_dt_s=1.0 / 60.0
    )

    assert result["contact_maintenance_ratio"] == pytest.approx(30.0 / 32.0)
    assert result["longest_contact_loss_s"] == pytest.approx(2.0 / 60.0)
    assert result["settled"]
    assert result["settling_time_s"] == pytest.approx(2.0 / 60.0)
    assert result["steady_force_mae_n"] == pytest.approx(0.2)


def test_persistent_safety_gate_ignores_one_step_spike_but_rejects_sustained_force():
    count, failed = utils.update_persistent_violation(
        69.0,
        threshold=20.0,
        previous_steps=0,
        required_steps=3,
        immediate_threshold=100.0,
    )
    assert count == 1
    assert not failed

    count, failed = utils.update_persistent_violation(
        25.0,
        threshold=20.0,
        previous_steps=count,
        required_steps=3,
        immediate_threshold=100.0,
    )
    assert count == 2
    assert not failed
    count, failed = utils.update_persistent_violation(
        21.0,
        threshold=20.0,
        previous_steps=count,
        required_steps=3,
        immediate_threshold=100.0,
    )
    assert count == 3
    assert failed


def test_persistent_safety_gate_resets_and_has_immediate_ceiling():
    count, failed = utils.update_persistent_violation(
        5.0,
        threshold=20.0,
        previous_steps=2,
        required_steps=3,
        immediate_threshold=100.0,
    )
    assert count == 0
    assert not failed
    count, failed = utils.update_persistent_violation(
        101.0,
        threshold=20.0,
        previous_steps=0,
        required_steps=3,
        immediate_threshold=100.0,
    )
    assert count == 1
    assert failed
