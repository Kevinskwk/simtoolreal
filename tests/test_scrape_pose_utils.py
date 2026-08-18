from pathlib import Path
import importlib.util

import pytest
import torch


def _load_scrape_pose_utils():
    path = Path(__file__).resolve().parents[1] / "isaacsimenvs/tasks/simtoolreal/utils/scrape_pose_utils.py"
    spec = importlib.util.spec_from_file_location("scrape_pose_utils_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_utils = _load_scrape_pose_utils()
quat_apply_wxyz = _utils.quat_apply_wxyz
sample_edge_contact_goal_pose = _utils.sample_edge_contact_goal_pose
load_urdf_collision_bounds = _utils.load_urdf_collision_bounds
edge_contact_points_w = _utils.edge_contact_points_w
edge_tilt_from_pose = _utils.edge_tilt_from_pose
contact_force_reward = _utils.contact_force_reward
conditional_success_rate = _utils.conditional_success_rate
contact_force_onset_gate = _utils.contact_force_onset_gate
edge_contact_reward = _utils.edge_contact_reward
table_top_state = _utils.table_top_state


def test_edge_contact_goal_places_anchor_on_tilted_table_plane():
    table_pos = torch.tensor([[0.0, 0.0, 0.38]])
    # Small pitch-like quaternion around y.
    angle = torch.tensor(0.2)
    table_quat = torch.tensor([[torch.cos(angle / 2), 0.0, torch.sin(angle / 2), 0.0]])
    x_tip = torch.tensor([0.23])
    y_center = torch.tensor([0.0])
    z_contact = torch.tensor([-0.04])

    _, _, edge_anchor, _ = sample_edge_contact_goal_pose(
        table_pos_w=table_pos,
        table_quat_wxyz=table_quat,
        x_tip=x_tip,
        y_center=y_center,
        z_contact=z_contact,
        xy_half_range=(0.0, 0.0),
        edge_yaw_range_rad=0.0,
        tilt_range_rad=(0.25, 0.25),
        device=torch.device("cpu"),
    )
    top, normal = table_top_state(table_pos, table_quat)
    signed_dist = ((edge_anchor - top) * normal).sum(dim=-1)

    assert torch.allclose(signed_dist, torch.zeros_like(signed_dist), atol=1e-6)


def test_edge_contact_goal_keeps_conservative_tool_box_above_table():
    table_pos = torch.tensor([[0.0, 0.0, 0.38]])
    table_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    x_tip = torch.tensor([0.23])
    y_center = torch.tensor([0.0])
    z_contact = torch.tensor([-0.04])

    goal_pos, goal_quat, _, _ = sample_edge_contact_goal_pose(
        table_pos_w=table_pos,
        table_quat_wxyz=table_quat,
        x_tip=x_tip,
        y_center=y_center,
        z_contact=z_contact,
        xy_half_range=(0.0, 0.0),
        edge_yaw_range_rad=0.0,
        tilt_range_rad=(0.25, 0.25),
        device=torch.device("cpu"),
    )
    top, normal = table_top_state(table_pos, table_quat)
    x_min = -0.08
    x_max = x_tip.item()
    y = 0.04
    z = abs(z_contact.item())
    corners = torch.tensor(
        [[x, yy, zz] for x in (x_min, x_max) for yy in (-y, y) for zz in (-z, z)],
        dtype=torch.float32,
    )
    corners_w = goal_pos + quat_apply_wxyz(goal_quat.expand(corners.shape[0], -1), corners)
    signed_dist = ((corners_w - top) * normal).sum(dim=-1)

    assert signed_dist.min().item() >= -1e-6
    assert signed_dist.min().item() < 1e-5


def test_load_urdf_collision_bounds_includes_head_offset(tmp_path):
    urdf = tmp_path / "tool.urdf"
    urdf.write_text(
        '<?xml version="1.0"?>\n<robot name="tool">\n  <link name="object_root">\n    <collision>\n      <origin xyz="0 0 0" rpy="0 0 0"/>\n      <geometry><box size="0.2 0.02 0.02"/></geometry>\n    </collision>\n    <collision>\n      <origin xyz="0.15 0 0" rpy="0 0 0"/>\n      <geometry><box size="0.1 0.06 0.04"/></geometry>\n    </collision>\n  </link>\n</robot>\n'
    )

    bounds = load_urdf_collision_bounds(urdf)

    assert bounds == (-0.1, -0.03, -0.02, 0.2, 0.03, 0.02)


def test_load_urdf_collision_bounds_supports_scaled_meshes(tmp_path):
    mesh = tmp_path / "part.obj"
    mesh.write_text(
        "v -1 -2 -3\n"
        "v 1 -2 -3\n"
        "v -1 2 -3\n"
        "v -1 -2 3\n"
        "f 1 2 3\n"
        "f 1 2 4\n"
    )
    urdf = tmp_path / "mesh_tool.urdf"
    urdf.write_text(
        '<?xml version="1.0"?>\n<robot name="tool"><link name="tool">'
        '<collision><origin xyz="0.5 1.0 -0.5" rpy="0 0 0"/>'
        '<geometry><mesh filename="part.obj" scale="0.1 0.2 0.3"/></geometry>'
        '</collision></link></robot>\n'
    )

    bounds = load_urdf_collision_bounds(urdf)

    assert bounds == pytest.approx((0.4, 0.6, -1.4, 0.6, 1.4, 0.4))


def test_resampling_with_prior_edge_yaw_keeps_same_contact_edge_direction():
    table_pos = torch.tensor([[0.0, 0.0, 0.38]])
    table_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    kwargs = dict(
        table_pos_w=table_pos,
        table_quat_wxyz=table_quat,
        x_tip=torch.tensor([0.23]),
        y_center=torch.tensor([0.0]),
        z_contact=torch.tensor([-0.04]),
        xy_half_range=(0.0, 0.0),
        edge_yaw_range_rad=0.5,
        tilt_range_rad=(0.25, 0.25),
        device=torch.device("cpu"),
    )
    _, q0, _, yaw = sample_edge_contact_goal_pose(**kwargs)
    _, q1, _, yaw_reused = sample_edge_contact_goal_pose(**kwargs, edge_yaw=yaw)

    y_axis0 = quat_apply_wxyz(q0, torch.tensor([[0.0, 1.0, 0.0]]))
    y_axis1 = quat_apply_wxyz(q1, torch.tensor([[0.0, 1.0, 0.0]]))
    assert torch.allclose(yaw, yaw_reused)
    assert torch.allclose(y_axis0, y_axis1, atol=1e-6)


def test_explicit_edge_tilt_is_preserved_and_recoverable():
    table_pos = torch.tensor([[0.0, 0.0, 0.38], [0.0, 0.0, 0.38]])
    table_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
    yaw = torch.tensor([-0.3, 0.4])
    tilt = torch.tensor([0.12, 0.31])
    _, quat, _, _ = sample_edge_contact_goal_pose(
        table_pos_w=table_pos,
        table_quat_wxyz=table_quat,
        x_tip=torch.tensor([0.23, 0.23]),
        y_center=torch.zeros(2),
        z_contact=torch.tensor([-0.04, -0.04]),
        xy_half_range=(0.0, 0.0),
        edge_yaw_range_rad=0.0,
        tilt_range_rad=(0.0, 0.0),
        device=torch.device("cpu"),
        edge_yaw=yaw,
        edge_tilt=tilt,
    )
    normal = torch.tensor([[0.0, 0.0, 1.0]]).repeat(2, 1)
    recovered = edge_tilt_from_pose(quat, normal, yaw)
    assert torch.allclose(recovered, tilt, atol=1.0e-6)


def test_edge_contact_reward_scores_selected_edge_distance():
    table_pos = torch.tensor([[0.0, 0.0, 0.38], [0.0, 0.0, 0.38]])
    table_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    x_tip = torch.tensor([0.1, 0.1])
    y_min = torch.tensor([-0.02, -0.02])
    y_max = torch.tensor([0.02, 0.02])
    z_contact = torch.tensor([-0.01, -0.01])
    object_pos = torch.tensor([[-0.1, 0.0, 0.54], [-0.1, 0.0, 0.59]])
    object_quat = table_quat.clone()

    points = edge_contact_points_w(object_pos, object_quat, x_tip, y_min, y_max, z_contact)
    reward, error = edge_contact_reward(points, table_pos, table_quat, distance_sigma=0.01)

    assert error[0].item() < 1e-6
    assert reward[0].item() > 0.99
    assert error[1].item() > 0.04
    assert reward[1].item() < 0.02


def test_contact_force_reward_prefers_target_force_and_reports_overforce():
    force = torch.tensor([4.0, 0.0, 24.0])
    reward, over_force = contact_force_reward(
        force, target_force=4.0, force_sigma=4.0, max_force=20.0
    )

    assert reward[0].item() == 1.0
    assert reward[0].item() > reward[1].item()
    assert over_force[0].item() == 0.0
    assert over_force[2].item() == 4.0


def test_edge_contact_reward_sigma_tightening_is_stricter():
    table_pos = torch.tensor([[0.0, 0.0, 0.38]])
    table_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    top, _ = table_top_state(table_pos, table_quat)
    edge_points = top[:, None, :].repeat(1, 3, 1)
    edge_points[:, :, 2] += 0.01

    loose_reward, loose_error = edge_contact_reward(
        edge_points, table_pos, table_quat, distance_sigma=0.03
    )
    tight_reward, tight_error = edge_contact_reward(
        edge_points, table_pos, table_quat, distance_sigma=0.005
    )

    assert torch.allclose(loose_error, tight_error)
    assert loose_reward.item() > tight_reward.item()


def test_contact_force_reward_sigma_tightening_is_stricter():
    force = torch.tensor([0.0, 4.0])
    loose_reward, _ = contact_force_reward(
        force, target_force=4.0, force_sigma=8.0, max_force=20.0
    )
    tight_reward, _ = contact_force_reward(
        force, target_force=4.0, force_sigma=2.0, max_force=20.0
    )

    assert loose_reward[1].item() == tight_reward[1].item() == 1.0
    assert loose_reward[0].item() > tight_reward[0].item()


def test_contact_force_reward_supports_per_env_targets():
    force = torch.tensor([2.0, 4.0, 6.0])
    target = torch.tensor([2.0, 6.0, 4.0])

    reward, over_force = contact_force_reward(
        force, target_force=target, force_sigma=2.0, max_force=20.0
    )

    assert reward[0].item() == 1.0
    assert reward[0].item() > reward[1].item()
    assert torch.all(over_force == 0.0)


def test_contact_force_reward_huber_is_robust_to_large_errors():
    force = torch.tensor([4.0, 8.0])
    l1_reward, _ = contact_force_reward(
        force, target_force=4.0, force_sigma=2.0, max_force=20.0
    )
    huber_reward, _ = contact_force_reward(
        force,
        target_force=4.0,
        force_sigma=2.0,
        max_force=20.0,
        huber_delta_n=1.0,
    )

    assert l1_reward[0].item() == huber_reward[0].item() == 1.0
    assert huber_reward[1].item() > l1_reward[1].item()


def test_contact_force_onset_gate_requires_persistence_and_resets():
    age = torch.zeros(2, dtype=torch.long)
    ramps = []
    for force in (
        torch.tensor([0.0, 1.0]),
        torch.tensor([1.0, 1.0]),
        torch.tensor([1.0, 1.0]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([1.0, 1.0]),
        torch.tensor([1.0, 1.0]),
    ):
        age, ramp, _ = contact_force_onset_gate(
            force,
            age,
            contact_threshold_n=0.1,
            grace_steps=2,
            ramp_steps=2,
        )
        ramps.append(ramp.clone())

    assert torch.equal(age, torch.tensor([5, 2]))
    assert torch.allclose(ramps[0], torch.tensor([0.0, 0.0]))
    assert torch.allclose(ramps[2], torch.tensor([0.0, 0.5]))
    assert torch.allclose(ramps[3], torch.tensor([0.5, 0.0]))
    assert torch.allclose(ramps[-1], torch.tensor([1.0, 0.0]))


def test_conditional_success_rate_uses_only_eligible_samples():
    eligible = torch.tensor([True, True, False, False, True])
    success = torch.tensor([True, False, False, False, True])

    rate, count, has_support = conditional_success_rate(
        success, eligible, min_eligible_count=3
    )

    assert count.item() == 3
    assert torch.isclose(rate, torch.tensor(2.0 / 3.0))
    assert has_support.item()


def test_conditional_success_rate_reports_insufficient_support_without_hiding_rate():
    eligible = torch.tensor([True, False, False])
    success = torch.tensor([True, False, False])

    rate, count, has_support = conditional_success_rate(
        success, eligible, min_eligible_count=2
    )

    assert count.item() == 1
    assert rate.item() == 1.0
    assert not has_support.item()


def test_conditional_success_rate_rejects_success_outside_eligibility():
    eligible = torch.tensor([False, True])
    success = torch.tensor([True, False])

    try:
        conditional_success_rate(success, eligible, min_eligible_count=1)
    except ValueError as exc:
        assert "not eligible" in str(exc)
    else:
        raise AssertionError("Expected success outside eligibility to fail.")
