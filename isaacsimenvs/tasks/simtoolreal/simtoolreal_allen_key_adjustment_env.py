"""Palm-supported Allen-key adjustment around an engaged screw axis."""

from __future__ import annotations

import math

import torch
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_mul

from .simtoolreal_inhand_adjustment_env import SimToolRealInHandAdjustmentEnv
from .simtoolreal_tacmap_env_cfg import SimToolRealAllenKeyAdjustmentEnvCfg
from .utils.adjustment_utils import (
    palm_keypoint_error,
    rotate_palm_about_tool_axis_in_place,
    screw_axis_orbit_errors,
)
from .utils.logging_utils import log_step_metrics
from .utils.obs_utils import compute_intermediate_values
from .utils.scrape_pose_utils import TABLE_HALF_HEIGHT
from .utils.stable_scrape_utils import consecutive_counter, quaternion_distance_rad


class SimToolRealAllenKeyAdjustmentEnv(SimToolRealInHandAdjustmentEnv):
    """Adjust a whole-palm grasp while the Allen key remains socket-supported."""

    cfg: SimToolRealAllenKeyAdjustmentEnvCfg

    def __init__(self, cfg: SimToolRealAllenKeyAdjustmentEnvCfg, render_mode=None, **kwargs):
        self._allen_ready = False
        self._validate_allen_cfg(cfg)
        super().__init__(cfg, render_mode, **kwargs)
        n, device = self.num_envs, self.device
        self._allen_current_palm_tool_obs = torch.zeros(n, 7, device=device)
        self._allen_target_palm_tool_obs = torch.zeros(n, 7, device=device)
        self._allen_target_error_obs = torch.zeros(n, 5, device=device)
        self._allen_geometry_obs = torch.zeros(n, 9, device=device)
        geometry = torch.tensor(
            (0.264, 0.06, 0.010, *cfg.allen_screw_axis_tool, *cfg.allen_screw_pivot_tool_m),
            device=device,
        )
        self._allen_geometry_obs[:] = geometry
        self._allen_socket_state_obs = torch.zeros(n, 4, device=device)
        self._allen_phase_obs = torch.zeros(n, 3, device=device)
        self._allen_validity_obs = torch.zeros(n, 7, device=device)
        self._allen_orbit_error = torch.zeros(n, device=device)
        self._allen_position_error = torch.zeros(n, device=device)
        self._allen_orientation_error = torch.zeros(n, device=device)
        self._allen_socket_lateral_error = torch.zeros(n, device=device)
        self._allen_socket_insertion_error = torch.zeros(n, device=device)
        self._allen_socket_tilt_error = torch.zeros(n, device=device)
        self._allen_palm_force_n = torch.zeros(n, device=device)
        self._allen_palm_contact = torch.zeros(n, dtype=torch.bool, device=device)
        self._allen_fingertip_force_n = torch.zeros(n, 5, device=device)
        self._allen_fingertip_contact_count = torch.zeros(n, dtype=torch.long, device=device)
        self._allen_contact_quality = torch.zeros(n, device=device)
        self._allen_flexion_closure = torch.zeros(n, device=device)
        self._allen_final_grasp_valid = torch.zeros(n, dtype=torch.bool, device=device)
        self._allen_palm_keypoint_error = torch.zeros(n, device=device)
        self._allen_socket_valid = torch.zeros(n, dtype=torch.bool, device=device)
        self._allen_combined_valid = torch.zeros(n, dtype=torch.bool, device=device)
        self._allen_hold_count = torch.zeros(n, dtype=torch.long, device=device)
        self._allen_succeeded = torch.zeros(n, dtype=torch.bool, device=device)
        self._allen_just_succeeded = torch.zeros(n, dtype=torch.bool, device=device)
        self._allen_previous_pose_potential = torch.zeros(n, device=device)
        self._allen_fixture_tool_pos = torch.zeros(n, 3, device=device)
        self._allen_fixture_tool_quat = torch.zeros(n, 4, device=device)
        self._allen_fixture_tool_quat[:, 0] = 1.0
        self._allen_reset_yaw_rad = torch.zeros(n, device=device)
        self._allen_curriculum_eligible = 0
        self._allen_curriculum_successes = 0
        self._allen_curriculum_success_mean = 0.0
        self._allen_closure_bias_active = torch.zeros(n, dtype=torch.bool, device=device)
        self._allen_flexion_joint_ids = torch.tensor([
            index for index, name in enumerate(self.robot.data.joint_names)
            if name.endswith(("_FE", "_PIP", "_DIP", "_IP"))
            or name == "left_pinky_CMC"
        ], dtype=torch.long, device=device)
        if self._allen_flexion_joint_ids.numel() == 0:
            raise RuntimeError("Allen-key closure could not identify any hand flexion joints")
        self._allen_ready = True
        self._reset_idx(torch.arange(n, device=device))

    @staticmethod
    def _validate_allen_cfg(cfg: SimToolRealAllenKeyAdjustmentEnvCfg) -> None:
        phase_steps = (
            int(cfg.allen_adjustment_steps) + int(cfg.allen_closure_steps)
            + int(cfg.allen_release_steps)
        )
        if phase_steps != 480:
            raise ValueError("Allen-key adjustment, closure, and release phases must total 480 steps")
        if not 0 < int(cfg.allen_success_hold_steps) <= int(cfg.allen_release_steps):
            raise ValueError("Allen-key successful hold length is invalid")
        if tuple(cfg.allen_screw_axis_tool) != (0.0, 0.0, -1.0):
            raise ValueError("the canonical Allen-key screw axis must be local -Z")
        yaw_ranges = tuple(float(value) for value in cfg.allen_reset_yaw_range_stages_deg)
        if len(yaw_ranges) != len(cfg.adjustment_target_rotation_deg):
            raise ValueError("Allen-key reset yaw curriculum length is inconsistent")
        if any(not math.isfinite(value) or not 0.0 <= value <= 45.0 for value in yaw_ranges):
            raise ValueError("Allen-key reset yaw ranges must be finite and in [0, 45]")
        target_low, target_high = map(float, cfg.allen_target_orbit_range_deg)
        if not 0.0 < target_low <= target_high <= 60.0:
            raise ValueError("Allen-key target orbit range must be within prevalidated (0, 60] degrees")
        sigma = tuple(float(value) for value in cfg.allen_pose_sigma_stages_m)
        if len(sigma) != len(cfg.adjustment_target_rotation_deg):
            raise ValueError("Allen-key pose sigma curriculum length is inconsistent")
        if any(value <= 0.0 or not math.isfinite(value) for value in sigma):
            raise ValueError("Allen-key pose sigmas must be finite and positive")
        if any(later >= earlier for earlier, later in zip(sigma, sigma[1:])):
            raise ValueError("Allen-key pose sigma curriculum must strictly tighten")
        if len(cfg.allen_palm_keypoints_m) < 4:
            raise ValueError("Allen-key pose reward requires at least four palm keypoints")
        for name in (
            "allen_socket_lateral_tolerance_m", "allen_socket_insertion_tolerance_m",
            "allen_socket_tilt_tolerance_deg", "allen_palm_contact_threshold_n",
            "allen_tool_position_tolerance_m", "allen_tool_rotation_tolerance_deg",
            "allen_fingertip_contact_threshold_n",
            "allen_release_linear_speed_tolerance_mps",
            "allen_release_angular_speed_tolerance_radps",
            "allen_closure_flexion_fraction",
            "allen_min_flexion_closure_fraction",
        ):
            if not math.isfinite(float(getattr(cfg, name))) or float(getattr(cfg, name)) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("allen_closure_flexion_fraction", "allen_min_flexion_closure_fraction"):
            if float(getattr(cfg, name)) > 1.0:
                raise ValueError(f"{name} must not exceed one")
        if int(cfg.allen_fingertip_contact_quality_saturation_count) <= 0:
            raise ValueError("Allen-key fingertip contact saturation count must be positive")
        for name in ("allen_release_challenge_force_n", "allen_release_challenge_torque_nm"):
            value = float(getattr(cfg, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    def _setup_scene(self) -> None:
        super()._setup_scene()
        if not hasattr(self, "workpiece"):
            raise RuntimeError("Allen-key task requires cfg.assets.workpiece_urdf")
        try:
            self._palm_tool_contact_sensor = ContactSensor(ContactSensorCfg(
                prim_path=self.cfg.palm_tool_contact_sensor_prim_path,
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                track_pose=False,
                track_contact_points=False,
                track_friction_forces=False,
                track_air_time=False,
                filter_prim_paths_expr=list(self.cfg.palm_tool_contact_sensor_filter_paths),
            ))
            self.scene.sensors["palm_tool_contact_sensor"] = self._palm_tool_contact_sensor
        except Exception as exc:  # pragma: no cover - requires Kit runtime.
            raise RuntimeError(
                f"Allen-key palm-tool ContactSensor could not be created: {exc!r}"
            ) from exc

    def _sample_target_relationship(
        self, env_ids: torch.Tensor, relative_pos: torch.Tensor,
        relative_quat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        low, high = self.cfg.allen_target_orbit_range_deg
        angle = torch.empty(relative_pos.shape[0], device=self.device).uniform_(
            math.radians(float(low)), math.radians(float(high))
        )
        target_pos, target_quat = rotate_palm_about_tool_axis_in_place(
            relative_pos, relative_quat, angle,
            torch.tensor(self.cfg.allen_screw_axis_tool, device=self.device),
        )
        return target_pos, target_quat, angle.abs(), torch.zeros_like(angle)

    def _sample_nearby_contact_target(
        self, env_ids: torch.Tensor, bank_ids: torch.Tensor,
        object_quat: torch.Tensor, table_pos: torch.Tensor,
        table_quat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Bypass scrape-edge sampling; the Allen target is a palm transform."""
        del table_pos, table_quat
        object_pos = self._inhand_bank_object_pos[bank_ids] + self.scene.env_origins[env_ids]
        zeros = torch.zeros(env_ids.numel(), device=self.device)
        return object_pos, object_quat, object_pos.clone(), zeros, zeros

    def _restore_inhand_state(self, env_ids: torch.Tensor, bank_ids=None) -> None:
        super()._restore_inhand_state(env_ids, bank_ids)
        if not hasattr(self, "workpiece"):
            return
        count = env_ids.numel()
        if getattr(self, "_allen_ready", False):
            self._randomize_engaged_yaw(env_ids)
        tool_pos = self.object.data.root_pos_w[env_ids]
        tool_quat = self.object.data.root_quat_w[env_ids]
        offset = torch.tensor(
            self.cfg.allen_workpiece_from_tool_m, device=self.device
        ).expand(count, -1)
        workpiece_pos = tool_pos + quat_apply(tool_quat, offset)
        workpiece_quat = tool_quat.clone()
        self.workpiece.write_root_pose_to_sim(
            torch.cat((workpiece_pos, workpiece_quat), dim=-1), env_ids=env_ids
        )
        self.workpiece.write_root_velocity_to_sim(
            torch.zeros(count, 6, device=self.device), env_ids=env_ids
        )
        normal = quat_apply(
            workpiece_quat,
            torch.tensor((0.0, 0.0, 1.0), device=self.device).expand(count, -1),
        )
        table_pos = workpiece_pos - normal * TABLE_HALF_HEIGHT
        self.table.write_root_pose_to_sim(
            torch.cat((table_pos, workpiece_quat), dim=-1), env_ids=env_ids
        )
        self._table_z_per_env[env_ids] = table_pos[:, 2] - self.scene.env_origins[env_ids, 2]
        self._table_quat_wxyz_per_env[env_ids] = workpiece_quat
        if getattr(self, "_allen_ready", False):
            self._allen_hold_count[env_ids] = 0
            self._allen_succeeded[env_ids] = False
            self._allen_just_succeeded[env_ids] = False
            self._allen_previous_pose_potential[env_ids] = 0.0
            self._allen_fixture_tool_pos[env_ids] = self.object.data.root_pos_w[env_ids]
            self._allen_fixture_tool_quat[env_ids] = self.object.data.root_quat_w[env_ids]

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        super()._pre_physics_step(actions)
        closure_start = int(self.cfg.allen_adjustment_steps)
        release_start = int(self.cfg.allen_adjustment_steps) + int(self.cfg.allen_closure_steps)
        pose_sigma = float(self.cfg.allen_pose_sigma_stages_m[
            self._adjustment_curriculum_stage
        ])
        closure_bias = (
            (self.episode_length_buf >= closure_start)
            & (self._allen_palm_keypoint_error <= 2.0 * pose_sigma)
        )
        self._allen_closure_bias_active.copy_(closure_bias)
        if bool(closure_bias.any()):
            ids = self._allen_flexion_joint_ids
            upper = self.robot.data.joint_pos_limits[:, ids, 1]
            fraction = float(self.cfg.allen_closure_flexion_fraction)
            tightened = self._cur_targets[:, ids] + fraction * (
                upper - self._cur_targets[:, ids]
            )
            self._cur_targets[:, ids] = torch.where(
                closure_bias[:, None], tightened, self._cur_targets[:, ids]
            )
            self._prev_targets[:, ids] = self._cur_targets[:, ids]
        release = self.episode_length_buf >= release_start
        local_force = torch.zeros(self.num_envs, 3, device=self.device)
        local_force[:, 0] = float(self.cfg.allen_release_challenge_force_n)
        local_torque = torch.zeros_like(local_force)
        local_torque[:, 1] = float(self.cfg.allen_release_challenge_torque_nm)
        force = quat_apply(self.object.data.root_quat_w, local_force) * release[:, None]
        torque = quat_apply(self.object.data.root_quat_w, local_torque) * release[:, None]
        self.object.set_external_force_and_torque(
            force[:, None, :], torque[:, None, :], is_global=True
        )

    def _apply_action(self) -> None:
        super()._apply_action()
        release_start = int(self.cfg.allen_adjustment_steps) + int(self.cfg.allen_closure_steps)
        fixture_ids = torch.nonzero(
            self.episode_length_buf < release_start, as_tuple=False
        ).squeeze(-1)
        if fixture_ids.numel() == 0:
            return
        self.object.write_root_pose_to_sim(
            torch.cat((
                self._allen_fixture_tool_pos[fixture_ids],
                self._allen_fixture_tool_quat[fixture_ids],
            ), dim=-1),
            env_ids=fixture_ids,
        )
        self.object.write_root_velocity_to_sim(
            torch.zeros(fixture_ids.numel(), 6, device=self.device), env_ids=fixture_ids
        )

    def _randomize_engaged_yaw(self, env_ids: torch.Tensor) -> None:
        """Rotate the grasped key and its world targets about the robot base Z axis."""
        count = env_ids.numel()
        limit = math.radians(float(
            self.cfg.allen_reset_yaw_range_stages_deg[self._adjustment_curriculum_stage]
        ))
        arm_joint = int(self._arm_joint_ids[0])
        current_targets = self._cur_targets[env_ids, arm_joint]
        low = torch.maximum(
            torch.zeros_like(current_targets),
            self._arm_lower[env_ids, 0] - current_targets,
        )
        high = torch.minimum(
            torch.full_like(current_targets, limit),
            self._arm_upper[env_ids, 0] - current_targets,
        )
        if bool((low > high).any()):
            raise RuntimeError("Allen-key reset yaw has no valid first-joint interval")
        yaw = low + torch.rand(count, device=self.device) * (high - low)
        self._allen_reset_yaw_rad[env_ids] = yaw

        z_axis = torch.zeros(count, 3, device=self.device)
        z_axis[:, 2] = 1.0
        yaw_quat = quat_from_angle_axis(yaw, z_axis)
        robot_base = self.robot.data.root_pos_w[env_ids]

        def rotate_position(position: torch.Tensor) -> torch.Tensor:
            return robot_base + quat_apply(yaw_quat, position - robot_base)

        joint_pos = self._cur_targets[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        joint_pos[:, arm_joint] += yaw
        self._cur_targets[env_ids, arm_joint] += yaw
        self._prev_targets[env_ids, arm_joint] += yaw
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.set_joint_position_target(self._cur_targets[env_ids], env_ids=env_ids)

        tool_pos = rotate_position(self.object.data.root_pos_w[env_ids])
        tool_quat = quat_mul(yaw_quat, self.object.data.root_quat_w[env_ids])
        self.object.write_root_pose_to_sim(
            torch.cat((tool_pos, tool_quat), dim=-1), env_ids=env_ids
        )
        self.object.write_root_velocity_to_sim(
            torch.zeros(count, 6, device=self.device), env_ids=env_ids
        )

        target_palm_pos = rotate_position(self._adjustment_target_palm_pos_w[env_ids])
        target_palm_quat = quat_mul(
            yaw_quat, self._adjustment_target_palm_quat_w[env_ids]
        )
        self._adjustment_target_palm_pos_w[env_ids] = target_palm_pos
        self._adjustment_target_palm_quat_w[env_ids] = target_palm_quat
        self._adjustment_initial_tool_pos[env_ids] = tool_pos
        self._adjustment_initial_tool_quat[env_ids] = tool_quat
        self._write_goal(env_ids, tool_pos, tool_quat, tool_pos)

    def _read_palm_contact(self) -> torch.Tensor:
        data = getattr(self._palm_tool_contact_sensor, "data", None)
        matrix = None if data is None else getattr(data, "force_matrix_w", None)
        if matrix is None:
            raise RuntimeError(
                "Allen-key palm-tool ContactSensor has no pair-filtered force_matrix_w"
            )
        if matrix.shape[0] != self.num_envs or matrix.shape[-1] != 3:
            raise RuntimeError(
                f"Allen-key palm contact matrix has invalid shape {tuple(matrix.shape)}"
            )
        force = torch.linalg.vector_norm(matrix.reshape(self.num_envs, -1, 3), dim=-1).sum(-1)
        if not bool(torch.isfinite(force).all()):
            raise RuntimeError("Allen-key palm-tool contact contains NaN or Inf")
        self._allen_palm_force_n.copy_(force)
        return force >= float(self.cfg.allen_palm_contact_threshold_n)

    def _read_fingertip_contacts(self) -> torch.Tensor:
        sensors = getattr(self, "_fingertip_tool_contact_sensors", None)
        if sensors is None or len(sensors) != 5:
            raise RuntimeError("Allen-key task requires exactly five fingertip-tool sensors")
        forces = []
        for sensor_id, sensor in enumerate(sensors):
            data = getattr(sensor, "data", None)
            matrix = None if data is None else getattr(data, "force_matrix_w", None)
            if matrix is None or matrix.shape[0] != self.num_envs or matrix.shape[-1] != 3:
                shape = None if matrix is None else tuple(matrix.shape)
                raise RuntimeError(
                    f"Allen-key fingertip sensor {sensor_id} has invalid force matrix {shape}"
                )
            force = torch.linalg.vector_norm(
                matrix.reshape(self.num_envs, -1, 3), dim=-1
            ).sum(-1)
            if not bool(torch.isfinite(force).all()):
                raise RuntimeError(f"Allen-key fingertip sensor {sensor_id} contains NaN or Inf")
            forces.append(force)
        self._allen_fingertip_force_n.copy_(torch.stack(forces, dim=-1))
        return self._allen_fingertip_force_n >= float(
            self.cfg.allen_fingertip_contact_threshold_n
        )

    def _update_allen_metrics(self) -> None:
        axis_tool = torch.tensor(self.cfg.allen_screw_axis_tool, device=self.device)
        pivot_tool = torch.tensor(self.cfg.allen_screw_pivot_tool_m, device=self.device)
        orbit, position, orientation = screw_axis_orbit_errors(
            self._stable_relative_pos, self._stable_relative_quat,
            self._adjustment_target_relative_pos, self._adjustment_target_relative_quat,
            axis_tool, pivot_tool,
        )
        self._allen_orbit_error.copy_(orbit.abs())
        self._allen_position_error.copy_(position)
        self._allen_orientation_error.copy_(orientation)
        current_quat = self._stable_relative_quat * torch.where(
            self._stable_relative_quat[:, :1] < 0, -1.0, 1.0
        )
        target_quat = self._adjustment_target_relative_quat * torch.where(
            self._adjustment_target_relative_quat[:, :1] < 0, -1.0, 1.0
        )
        self._allen_current_palm_tool_obs[:, :3] = self._stable_relative_pos / 0.25
        self._allen_current_palm_tool_obs[:, 3:] = current_quat
        self._allen_target_palm_tool_obs[:, :3] = self._adjustment_target_relative_pos / 0.25
        self._allen_target_palm_tool_obs[:, 3:] = target_quat

        self._allen_palm_keypoint_error.copy_(palm_keypoint_error(
            self._stable_relative_pos,
            self._stable_relative_quat,
            self._adjustment_target_relative_pos,
            self._adjustment_target_relative_quat,
            torch.tensor(
                self.cfg.allen_palm_keypoints_m,
                device=self.device,
                dtype=self._stable_relative_pos.dtype,
            ),
        ))

        tool_pos = self.object.data.root_pos_w
        tool_quat = self.object.data.root_quat_w
        expected_workpiece = tool_pos + quat_apply(
            tool_quat,
            torch.tensor(self.cfg.allen_workpiece_from_tool_m, device=self.device).expand(
                self.num_envs, -1
            ),
        )
        axis_world = quat_apply(tool_quat, axis_tool.expand(self.num_envs, -1))
        delta = self.workpiece.data.root_pos_w - expected_workpiece
        axial = (delta * axis_world).sum(-1)
        lateral = torch.linalg.vector_norm(delta - axial.unsqueeze(-1) * axis_world, dim=-1)
        workpiece_axis = quat_apply(
            self.workpiece.data.root_quat_w, axis_tool.expand(self.num_envs, -1)
        )
        tilt = torch.acos((axis_world * workpiece_axis).sum(-1).clamp(-1.0, 1.0))
        self._allen_socket_lateral_error.copy_(lateral)
        self._allen_socket_insertion_error.copy_(axial.abs())
        self._allen_socket_tilt_error.copy_(tilt)
        self._allen_palm_contact.copy_(self._read_palm_contact())
        fingertip_contact = self._read_fingertip_contacts()
        self._allen_fingertip_contact_count.copy_(fingertip_contact.sum(-1))
        fingertip_force_quality = (
            self._allen_fingertip_contact_count.float()
            / float(self.cfg.allen_fingertip_contact_quality_saturation_count)
        ).clamp(0.0, 1.0)
        flexion_ids = self._allen_flexion_joint_ids
        limits = self.robot.data.joint_pos_limits[:, flexion_ids]
        closure = (
            (self.robot.data.joint_pos[:, flexion_ids] - limits[:, :, 0])
            / (limits[:, :, 1] - limits[:, :, 0]).clamp_min(1.0e-6)
        ).mean(-1).clamp(0.0, 1.0)
        self._allen_flexion_closure.copy_(closure)
        self._allen_contact_quality.copy_(
            0.40 * closure + 0.25 * fingertip_force_quality
            + 0.35 * self._allen_palm_contact.float()
        )
        self._allen_final_grasp_valid.copy_(
            (closure >= float(self.cfg.allen_min_flexion_closure_fraction))
            & self._allen_palm_contact
        )
        self._allen_socket_valid.copy_(
            (lateral <= float(self.cfg.allen_socket_lateral_tolerance_m))
            & (axial.abs() <= float(self.cfg.allen_socket_insertion_tolerance_m))
            & (tilt <= math.radians(float(self.cfg.allen_socket_tilt_tolerance_deg)))
        )

        pose_sigma = float(self.cfg.allen_pose_sigma_stages_m[
            self._adjustment_curriculum_stage
        ])
        target_valid = self._allen_palm_keypoint_error <= pose_sigma
        tool_valid = (
            self._adjustment_tool_position_error <= float(self.cfg.allen_tool_position_tolerance_m)
        ) & (
            self._adjustment_tool_rotation_error
            <= math.radians(float(self.cfg.allen_tool_rotation_tolerance_deg))
        )
        motion_valid = (
            self._stable_relative_linear_speed
            <= float(self.cfg.allen_release_linear_speed_tolerance_mps)
        ) & (
            self._stable_relative_angular_speed
            <= float(self.cfg.allen_release_angular_speed_tolerance_radps)
        )
        self._allen_combined_valid.copy_(
            target_valid & self._allen_final_grasp_valid & tool_valid
            & self._allen_socket_valid & motion_valid
        )
        closure_start = int(self.cfg.allen_adjustment_steps)
        release_start = closure_start + int(self.cfg.allen_closure_steps)
        adjustment_phase = self.episode_length_buf < closure_start
        closure_phase = (
            (self.episode_length_buf >= closure_start)
            & (self.episode_length_buf < release_start)
        )
        release_phase = self.episode_length_buf >= release_start
        self._allen_hold_count.copy_(consecutive_counter(
            release_phase & self._allen_combined_valid, self._allen_hold_count
        ))
        self._allen_phase_obs[:, 0] = adjustment_phase.float()
        self._allen_phase_obs[:, 1] = closure_phase.float()
        self._allen_phase_obs[:, 2] = release_phase.float()
        self._allen_socket_state_obs[:, 0] = lateral / float(self.cfg.allen_socket_lateral_tolerance_m)
        self._allen_socket_state_obs[:, 1] = axial / float(self.cfg.allen_socket_insertion_tolerance_m)
        self._allen_socket_state_obs[:, 2] = tilt / math.radians(float(self.cfg.allen_socket_tilt_tolerance_deg))
        self._allen_socket_state_obs[:, 3] = self._allen_palm_contact.float()
        self._allen_target_error_obs[:, 0] = orbit / math.pi
        self._allen_target_error_obs[:, 1] = position / 0.05
        self._allen_target_error_obs[:, 2] = orientation / math.pi
        self._allen_target_error_obs[:, 3] = self._adjustment_tool_position_error / 0.05
        self._allen_target_error_obs[:, 4] = self._adjustment_tool_rotation_error / math.pi
        self._allen_validity_obs[:] = torch.stack((
            target_valid,
            self._allen_flexion_closure >= float(
                self.cfg.allen_min_flexion_closure_fraction
            ),
            self._allen_palm_contact,
            self._allen_final_grasp_valid,
            self._allen_socket_valid,
            tool_valid,
            motion_valid,
        ), dim=-1).float()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._frame_counter += 1
        compute_intermediate_values(self)
        self._stable_support_count.copy_((
            self._curr_fingertip_distances
            < float(self.cfg.acquisition_max_fingertip_distance_m)
        ).sum(dim=-1))
        self._update_relative_motion()
        self._update_adjustment_metrics()
        self._update_allen_metrics()
        truncated = self.episode_length_buf >= self.max_episode_length
        endpoint_success = (
            truncated
            & self._allen_combined_valid
            & (self._allen_hold_count >= int(self.cfg.allen_success_hold_steps))
        )
        self._allen_just_succeeded.copy_(endpoint_success & ~self._allen_succeeded)
        self._allen_succeeded |= endpoint_success
        self._is_success.copy_(endpoint_success)
        completed_count = int(truncated.sum().item())
        if completed_count:
            self._allen_curriculum_eligible += completed_count
            self._allen_curriculum_successes += int(endpoint_success.sum().item())
            minimum = int(self.cfg.adjustment_curriculum_min_eligible_count)
            if self._allen_curriculum_eligible >= minimum:
                self._allen_curriculum_success_mean = (
                    self._allen_curriculum_successes / self._allen_curriculum_eligible
                )
                last = len(self.cfg.allen_pose_sigma_stages_m) - 1
                if (
                    self._adjustment_curriculum_stage < last
                    and self._allen_curriculum_success_mean
                    >= float(self.cfg.adjustment_curriculum_success_threshold)
                ):
                    self._adjustment_curriculum_stage += 1
                    self._adjustment_curriculum_updates += 1
                self._allen_curriculum_eligible = 0
                self._allen_curriculum_successes = 0
        self._termination_reasons = {
            "allen_endpoint_success": endpoint_success,
            "timeout": truncated,
        }
        return torch.zeros_like(truncated), truncated

    def _get_rewards(self) -> torch.Tensor:
        pose_sigma = float(self.cfg.allen_pose_sigma_stages_m[
            self._adjustment_curriculum_stage
        ])
        pose_potential = torch.exp(-self._allen_palm_keypoint_error / pose_sigma)
        progress = pose_potential - self._allen_previous_pose_potential
        progress = torch.where(
            self.episode_length_buf <= 1, torch.zeros_like(progress), progress
        )
        self._allen_previous_pose_potential.copy_(pose_potential)
        closure_start = int(self.cfg.allen_adjustment_steps)
        release_start = closure_start + int(self.cfg.allen_closure_steps)
        closure_or_release = self.episode_length_buf >= closure_start
        release_phase = self.episode_length_buf >= release_start
        near_target = self._allen_palm_keypoint_error <= 2.0 * pose_sigma
        weighted = {
            "palm_keypoint_target_rew": 8.0 * pose_potential,
            "palm_keypoint_progress_rew": 2.0 * progress,
            # Contact is deliberately gated to closure/release and proximity to
            # the target. The policy may fully release during adjustment.
            "closure_contact_rew": 2.0 * (
                closure_or_release & near_target
            ).float() * self._allen_contact_quality,
            "release_tool_position_penalty": -release_phase.float() * 0.5 * (
                self._adjustment_tool_position_error
                / float(self.cfg.allen_tool_position_tolerance_m)
            ).clamp(0.0, 2.0),
            "release_tool_rotation_penalty": -release_phase.float() * 0.5 * (
                self._adjustment_tool_rotation_error
                / math.radians(float(self.cfg.allen_tool_rotation_tolerance_deg))
            ).clamp(0.0, 2.0),
            "release_socket_penalty": -release_phase.float() * (
                ~self._allen_socket_valid
            ).float(),
            "action_rate_penalty": -0.01 * self._stable_action_delta_sq_mean,
            "valid_release_hold_rew": 3.0 * (
                release_phase & self._allen_combined_valid
            ).float(),
            "final_hold_bonus": 25.0 * self._allen_just_succeeded.float(),
        }
        reward = torch.stack(tuple(weighted.values())).sum(0)
        self._reward_terms = {**weighted, "total_reward": reward}
        self.extras.update({f"reward/{name}": value.mean() for name, value in weighted.items()})
        self.extras.update({
            "allen/orbit_error_mean_deg": torch.rad2deg(self._allen_orbit_error).mean(),
            "allen/palm_position_error_mean_m": self._allen_position_error.mean(),
            "allen/palm_orientation_error_mean_deg": torch.rad2deg(
                self._allen_orientation_error
            ).mean(),
            "allen/palm_keypoint_error_mean_m": self._allen_palm_keypoint_error.mean(),
            "allen/world_palm_position_error_mean_m": (
                self._adjustment_target_palm_position_error.mean()
            ),
            "allen/world_palm_rotation_error_mean_deg": torch.rad2deg(
                self._adjustment_target_palm_rotation_error
            ).mean(),
            "allen/target_position_valid_ratio": self._allen_validity_obs[:, 0].mean(),
            "allen/target_orientation_valid_ratio": self._allen_validity_obs[:, 1].mean(),
            "allen/socket_lateral_error_mean_m": self._allen_socket_lateral_error.mean(),
            "allen/socket_insertion_error_mean_m": self._allen_socket_insertion_error.mean(),
            "allen/socket_tilt_error_mean_deg": torch.rad2deg(
                self._allen_socket_tilt_error
            ).mean(),
            "allen/palm_contact_ratio": self._allen_palm_contact.float().mean(),
            "allen/palm_force_mean_n": self._allen_palm_force_n.mean(),
            "allen/fingertip_contact_count_mean": (
                self._allen_fingertip_contact_count.float().mean()
            ),
            "allen/contact_quality_mean": self._allen_contact_quality.mean(),
            "allen/flexion_closure_mean": self._allen_flexion_closure.mean(),
            "allen/final_grasp_valid_ratio": self._allen_final_grasp_valid.float().mean(),
            "allen/closure_bias_active_ratio": self._allen_closure_bias_active.float().mean(),
            "allen/reset_yaw_mean_deg": torch.rad2deg(self._allen_reset_yaw_rad).mean(),
            "allen/socket_valid_ratio": self._allen_socket_valid.float().mean(),
            "allen/combined_valid_ratio": self._allen_combined_valid.float().mean(),
            "allen/hold_count_mean": self._allen_hold_count.float().mean(),
            "allen/endpoint_success_ratio": self._allen_succeeded.float().mean(),
            "curriculum/allen_stage": self._adjustment_curriculum_stage,
            "curriculum/allen_success_mean": self._allen_curriculum_success_mean,
            "curriculum/allen_palm_pose_sigma_m": pose_sigma,
            "curriculum/allen_target_orbit_min_deg": self.cfg.allen_target_orbit_range_deg[0],
            "curriculum/allen_target_orbit_max_deg": self.cfg.allen_target_orbit_range_deg[1],
            "curriculum/allen_reset_yaw_range_deg": (
                self.cfg.allen_reset_yaw_range_stages_deg[
                    self._adjustment_curriculum_stage
                ]
            ),
        })
        log_step_metrics(self)
        return reward


__all__ = ["SimToolRealAllenKeyAdjustmentEnv"]
