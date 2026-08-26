from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack((
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ), dim=-1)


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    vector_quaternion = torch.cat((torch.zeros_like(v[..., :1]), v), dim=-1)
    conjugate = q.clone()
    conjugate[..., 1:] *= -1.0
    return _quat_mul(_quat_mul(q, vector_quaternion), conjugate)[..., 1:]


math_module = types.ModuleType("isaaclab.utils.math")
math_module.quat_apply = _quat_apply
math_module.quat_mul = _quat_mul
sys.modules.setdefault("isaaclab", types.ModuleType("isaaclab"))
sys.modules.setdefault("isaaclab.utils", types.ModuleType("isaaclab.utils"))
sys.modules["isaaclab.utils.math"] = math_module


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "isaacsimenvs/tasks/simtoolreal/utils/palm_geometry.py"
SPEC = importlib.util.spec_from_file_location("palm_geometry_test", PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_identity_wrist_maps_to_physical_palm_mesh_center():
    wrist_position = torch.tensor([[1.0, 2.0, 3.0]])
    wrist_quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    palm_position, palm_quaternion = module.palm_center_pose_from_merged_wrist(
        wrist_position, wrist_quaternion
    )
    expected_offset = torch.tensor([module.PALM_CENTER_IN_IIWA_LINK_7_M])
    expected_quaternion = torch.tensor([
        module.PALM_LINK_QUAT_IN_IIWA_LINK_7_WXYZ
    ])
    assert torch.allclose(palm_position, wrist_position + expected_offset)
    assert torch.allclose(palm_quaternion, expected_quaternion)
    assert torch.linalg.vector_norm(expected_offset).item() == pytest.approx(
        0.13821, abs=1.0e-5
    )


def test_palm_pose_helper_rejects_mismatched_batches():
    with pytest.raises(ValueError, match="shapes do not match"):
        module.palm_center_pose_from_merged_wrist(
            torch.zeros(2, 3), torch.zeros(1, 4)
        )
