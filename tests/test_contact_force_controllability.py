from pathlib import Path
import importlib.util
import sys

import torch


def _load_utils():
    path = Path(__file__).resolve().parents[1] / "isaacsimenvs/tasks/simtoolreal/utils/contact_force_controllability.py"
    spec = importlib.util.spec_from_file_location("contact_force_controllability_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_utils = _load_utils()
coefficient_of_determination = _utils.coefficient_of_determination
damped_least_squares = _utils.damped_least_squares
expected_normal_reaction = _utils.expected_normal_reaction
interval_normal_force = _utils.interval_normal_force
pi_force_step = _utils.pi_force_step
quaternion_error_vector = _utils.quaternion_error_vector



def test_interval_normal_force_averages_history_and_sums_pairs():
    force = torch.zeros(1, 2, 1, 2, 3)
    force[0, 0, 0, :, 2] = torch.tensor([2.0, 4.0])
    force[0, 1, 0, :, 2] = torch.tensor([6.0, 8.0])
    normal = torch.tensor([[0.0, 0.0, 1.0]])
    assert torch.allclose(interval_normal_force(force, normal), torch.tensor([10.0]))


def test_expected_reaction_includes_gravity_and_applied_load():
    result = expected_normal_reaction(
        torch.tensor([2.0]),
        torch.tensor([0.0, 0.0, -9.81]),
        torch.tensor([[0.0, 0.0, 1.0]]),
        torch.tensor([4.0]),
    )
    assert torch.allclose(result, torch.tensor([23.62]))


def test_damped_least_squares_has_expected_direction_and_shape():
    jacobian = torch.eye(3).unsqueeze(0)
    twist = torch.tensor([[1.0, -2.0, 0.5]])
    qdot = damped_least_squares(jacobian, twist, damping=1.0e-3)
    assert qdot.shape == (1, 3)
    assert torch.allclose(qdot, twist, atol=1.0e-5)


def test_pi_force_step_clamps_and_prevents_windup():
    step = pi_force_step(
        torch.tensor([100.0]),
        torch.tensor([0.0]),
        dt=0.1,
        kp=1.0,
        ki=1.0,
        velocity_limit=2.0,
        integral_limit=5.0,
    )
    assert step.velocity.item() == 2.0
    assert step.integral.item() == 0.0


def test_quaternion_error_is_zero_for_equal_orientation():
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(quaternion_error_vector(quat, quat), torch.zeros(1, 3))


def test_r2_is_one_for_exact_measurements():
    values = torch.tensor([1.0, 2.0, 3.0])
    assert coefficient_of_determination(values, values) == 1.0
