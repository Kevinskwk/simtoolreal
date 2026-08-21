import numpy as np
import pytest

from dextoolbench.eval_interactive_isaacgym import (
    euler_deg_to_quat_xyzw,
    quat_xyzw_to_euler_deg,
)
from dextoolbench.eval_interactive_isaacsim import _parse_policy_arg


@pytest.mark.parametrize(
    "angles",
    [
        (0.0, 0.0, 0.0),
        (30.0, -20.0, 75.0),
        (-120.0, 40.0, -90.0),
    ],
)
def test_goal_euler_quaternion_round_trip(angles):
    quat = euler_deg_to_quat_xyzw(angles)
    recovered = quat_xyzw_to_euler_deg(quat)
    quat_recovered = euler_deg_to_quat_xyzw(recovered)
    assert np.isclose(abs(np.dot(quat, quat_recovered)), 1.0, atol=1.0e-7)


def test_parse_policy_uses_default_config():
    name, spec = _parse_policy_arg("Scrape=outputs/run/model.pth", "base.yaml")
    assert name == "Scrape"
    assert spec.checkpoint_path == "outputs/run/model.pth"
    assert spec.config_path == "base.yaml"


def test_parse_policy_accepts_explicit_config():
    name, spec = _parse_policy_arg(
        "Scrape=outputs/run/model.pth::outputs/run/config.yaml", "base.yaml"
    )
    assert name == "Scrape"
    assert spec.config_path == "outputs/run/config.yaml"


@pytest.mark.parametrize("value", ["missing-equals", "=model.pth", "Name="])
def test_parse_policy_rejects_invalid_values(value):
    with pytest.raises(Exception):
        _parse_policy_arg(value, "base.yaml")
