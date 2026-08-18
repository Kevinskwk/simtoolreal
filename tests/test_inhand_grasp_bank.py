from pathlib import Path
import copy
import importlib.util

import pytest
import torch


PATH = (
    Path(__file__).resolve().parents[1]
    / "isaacsimenvs/tasks/simtoolreal/utils/inhand_grasp_bank.py"
)
spec = importlib.util.spec_from_file_location("inhand_grasp_bank", PATH)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


def valid_entry():
    return {
        "joint_pos_canonical": [0.0] * 29,
        "joint_vel_canonical": [0.0] * 29,
        "joint_targets_canonical": [0.0] * 29,
        "last_action_canonical": [0.0] * 29,
        "object_pos_local": [0.0, 0.0, 0.7],
        "object_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        "object_velocity": [0.0] * 6,
        "palm_to_tool_pos": [0.0, 0.0, 0.2],
        "palm_to_tool_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        "reference_contact_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        "reference_edge_yaw_rad": 0.0,
        "reference_edge_tilt_rad": 0.25,
        "verification": {
            "support_count": 2,
            "edge_clearance_m": 0.04,
            "table_force_n": 0.0,
            "stable_steps": 15,
            "hold_steps": 120,
            "hold_drift_m": 0.004,
            "hold_rotation_deg": 1.0,
            "pickup_orientation_error_deg": 5.0,
            "tactile_finger_count_min": 2,
            "tactile_contact_area_mean": 0.01,
            "tactile_depth_mean": 0.1,
            "tactile_depth_max": 0.2,
        },
    }


def valid_bank():
    return {
        "schema_version": 2,
        "tool_type": "eraser",
        "object_name": "eraser_canonical",
        "asset_sha256": "a" * 64,
        "source_checkpoint_sha256": "b" * 64,
        "policy_coefficient_id": 0.0,
        "tactile_rich_fraction_min": 0.75,
        "tactile_min_fingers": 2,
        "entries": [valid_entry()],
    }


def test_grasp_bank_requires_verified_compliant_grasp():
    assert utils.validate_grasp_bank(valid_bank())["tool_type"] == "eraser"
    payload = valid_bank()
    payload["entries"][0]["verification"]["support_count"] = 1
    with pytest.raises(ValueError, match="support"):
        utils.validate_grasp_bank(payload)


@pytest.mark.parametrize("tool_type", sorted(utils.SUPPORTED_TOOL_TYPES))
def test_grasp_bank_accepts_all_supported_tool_categories(tool_type):
    payload = valid_bank()
    payload["tool_type"] = tool_type
    payload["object_name"] = f"test_{tool_type}"
    assert utils.validate_grasp_bank(payload)["tool_type"] == tool_type


def test_grasp_bank_requires_object_name_for_new_non_eraser_banks():
    payload = valid_bank()
    payload["tool_type"] = "hammer"
    payload.pop("object_name")
    with pytest.raises(ValueError, match="object_name"):
        utils.validate_grasp_bank(payload)


def test_legacy_canonical_eraser_bank_without_object_name_is_accepted():
    payload = valid_bank()
    payload.pop("object_name")
    assert utils.validate_grasp_bank(payload)["tool_type"] == "eraser"


def test_merge_grasp_banks_combines_seeds_and_removes_exact_duplicates():
    first = valid_bank()
    first["seed"] = 1
    second = copy.deepcopy(first)
    second["seed"] = 2
    second["entries"][0]["object_pos_local"][0] = 0.1
    duplicate = copy.deepcopy(first)

    merged = utils.merge_grasp_banks([first, second, duplicate])

    assert len(merged["entries"]) == 2
    assert merged["seeds"] == [1, 2]
    assert merged["merged_bank_count"] == 3
    assert merged["duplicates_removed"] == 1


def test_merge_grasp_banks_rejects_incompatible_assets():
    first = valid_bank()
    second = copy.deepcopy(first)
    second["asset_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="asset_sha256"):
        utils.merge_grasp_banks([first, second])


def test_grasp_bank_enforces_embedded_joint_limits_and_hold_verification():
    payload = valid_bank()
    payload["joint_lower_canonical"] = [-2.0] * 29
    payload["joint_upper_canonical"] = [2.0] * 29
    payload["joint_limit_tolerance_rad"] = 5.0e-4
    payload["entries"][0]["verification"][
        "joint_limit_violation_max_rad"
    ] = 0.0
    utils.validate_grasp_bank(payload)

    payload["entries"][0]["joint_pos_canonical"][22] = -2.01
    with pytest.raises(ValueError, match="canonical joint limits"):
        utils.validate_grasp_bank(payload)


def test_grasp_bank_rejects_incomplete_joint_limit_metadata():
    payload = valid_bank()
    payload["joint_lower_canonical"] = [-2.0] * 29
    with pytest.raises(ValueError, match="metadata is incomplete"):
        utils.validate_grasp_bank(payload)


def test_grasp_bank_rejects_wrong_policy_coefficient_and_fixed_joint_like_bad_hold():
    payload = valid_bank()
    payload["policy_coefficient_id"] = 50.0
    with pytest.raises(ValueError, match="0.0"):
        utils.validate_grasp_bank(payload)
    payload = valid_bank()
    payload["entries"][0]["verification"]["hold_drift_m"] = 0.006
    with pytest.raises(ValueError, match="hold-drift"):
        utils.validate_grasp_bank(payload)


def test_grasp_bank_rejects_nonfinite_metrics_and_short_verification():
    payload = valid_bank()
    payload["entries"][0]["verification"]["table_force_n"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        utils.validate_grasp_bank(payload)
    payload = valid_bank()
    payload["entries"][0]["verification"]["hold_steps"] = 30
    with pytest.raises(ValueError, match="hold window"):
        utils.validate_grasp_bank(payload)


def test_grasp_bank_enforces_tactile_rich_fraction():
    payload = valid_bank()
    payload["entries"][0]["verification"]["tactile_finger_count_min"] = 1
    with pytest.raises(ValueError, match="tactile-rich fraction"):
        utils.validate_grasp_bank(payload)


def test_collision_box_corners_cover_all_extrema():
    bounds = torch.tensor([[-1.0, -2.0, -3.0, 4.0, 5.0, 6.0]])
    corners = utils.collision_box_corners(bounds)
    assert corners.shape == (1, 8, 3)
    assert torch.equal(corners.amin(dim=1), bounds[:, :3])
    assert torch.equal(corners.amax(dim=1), bounds[:, 3:])


def test_table_height_solver_produces_requested_lowest_clearance():
    points = torch.tensor([
        [[0.2, -0.1, 0.8], [0.3, 0.0, 0.85]],
        [[1.2, 0.4, 0.75], [1.1, 0.6, 0.9]],
    ])
    normal = torch.nn.functional.normalize(
        torch.tensor([[0.1, -0.05, 1.0], [-0.08, 0.12, 1.0]]), dim=-1
    )
    origins = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0]])
    requested = torch.tensor([0.05, 0.04])
    local_z = utils.table_root_z_for_lowest_clearance(
        points, normal, origins, requested, table_half_height_m=0.15
    )
    table_root = origins.clone()
    table_root[:, 2] += local_z
    table_top = table_root + normal * 0.15
    actual = ((points - table_top.unsqueeze(1)) * normal.unsqueeze(1)).sum(-1).min(-1).values
    assert torch.allclose(actual, requested, atol=1.0e-6)
