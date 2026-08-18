from pathlib import Path
import importlib.util
import sys

import pytest


PATH = Path(__file__).resolve().parents[1] / "isaacsimenvs/tasks/simtoolreal/utils/grasp_stability.py"
spec = importlib.util.spec_from_file_location("grasp_stability", PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_static_failure_during_settle_has_zero_stable_fraction():
    assert module.static_stable_fraction(failed=True, failure_step=-3, ramp_steps=120) == 0.0
    assert module.static_stable_fraction(failed=False, failure_step=-1, ramp_steps=120) == 1.0


def test_task_gate_reports_every_failed_requirement():
    gates = module.task_stability_gates(
        computed_feasible=False,
        grasp_survived=True,
        final_position_error_m=0.04,
        final_orientation_error_deg=3.0,
        geometric_contact_ratio=0.9,
        sensed_contact_ratio=0.2,
        position_tolerance_m=0.03,
        orientation_tolerance_deg=15.0,
        contact_ratio_threshold=0.5,
    )
    assert not gates.passed
    assert gates.failure_reasons == ["computed_infeasible", "tracking_failed", "contact_failed"]


def test_task_gate_accepts_joint_success():
    gates = module.task_stability_gates(
        computed_feasible=True,
        grasp_survived=True,
        final_position_error_m=0.01,
        final_orientation_error_deg=4.0,
        geometric_contact_ratio=0.8,
        sensed_contact_ratio=0.6,
        position_tolerance_m=0.03,
        orientation_tolerance_deg=15.0,
        contact_ratio_threshold=0.5,
    )
    assert gates.passed
    assert gates.failure_reasons == []


def test_task_gate_validates_thresholds():
    with pytest.raises(ValueError, match="contact ratio"):
        module.task_stability_gates(
            computed_feasible=True,
            grasp_survived=True,
            final_position_error_m=0.0,
            final_orientation_error_deg=0.0,
            geometric_contact_ratio=1.0,
            sensed_contact_ratio=1.0,
            position_tolerance_m=0.03,
            orientation_tolerance_deg=15.0,
            contact_ratio_threshold=1.1,
        )
