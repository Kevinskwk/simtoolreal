"""End-to-end acquisition and resistance-loaded Allen-key turning."""

from __future__ import annotations

import math

import torch
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.math import (
    quat_apply,
    quat_from_angle_axis,
    subtract_frame_transforms,
)

from .simtoolreal_tacmap_env import SimToolRealTacMapEnv
from .simtoolreal_tacmap_env_cfg import SimToolRealAllenKeyTurningEnvCfg
from .utils.action_utils import apply_action_pipeline
from .utils.allen_key_turning_utils import (
    finger_effort_soft_penalty,
    turn_goal_pose,
    turning_curriculum_ready,
    update_unwrapped_angle,
    yaw_from_quaternion,
)
from .utils.logging_utils import log_step_metrics
from .utils.obs_utils import compute_intermediate_values


class SimToolRealAllenKeyTurningEnv(SimToolRealTacMapEnv):
    """Acquire an engaged Allen key and complete a loaded 360-degree turn."""

    cfg: SimToolRealAllenKeyTurningEnvCfg

    def __init__(
        self,
        cfg: SimToolRealAllenKeyTurningEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        self._turn_ready = False
        self._validate_cfg(cfg)
        super().__init__(cfg, render_mode, **kwargs)
        n, device = self.num_envs, self.device

        palm_position = self.robot.data.body_link_pos_w[:, self._palm_body_id]
        fingertip_position = self.robot.data.body_link_pos_w[:, self._fingertip_body_ids]
        self._turn_default_hand_points_local = torch.cat(
            (palm_position[:, None, :], fingertip_position), dim=1
        ) - self.scene.env_origins[:, None, :]
        arm_body_ids = [
            index for index, name in enumerate(self.robot.data.body_names)
            if name.startswith("iiwa14_link_")
        ]
        if not arm_body_ids:
            raise RuntimeError("arm body geometry is unavailable for reset screening")
        self._turn_default_arm_points_local = (
            self.robot.data.body_link_pos_w[:, arm_body_ids]
            - self.scene.env_origins[:, None, :]
        )
        if not bool(torch.isfinite(self._turn_default_hand_points_local).all()) or not bool(
            torch.isfinite(self._turn_default_arm_points_local).all()
        ):
            raise RuntimeError("robot geometry is non-finite during reset screening")

        self._turn_phase = torch.zeros(n, dtype=torch.long, device=device)
        self._turn_direction = torch.ones(n, device=device)
        self._turn_initial_tool_pos = torch.zeros(n, 3, device=device)
        self._turn_initial_tool_quat = torch.zeros(n, 4, device=device)
        self._turn_initial_tool_quat[:, 0] = 1.0
        self._turn_pivot_w = torch.zeros(n, 3, device=device)
        self._turn_initial_yaw = torch.zeros(n, device=device)
        self._turn_previous_yaw = torch.zeros(n, device=device)
        self._turn_cumulative_angle = torch.zeros(n, device=device)
        self._turn_angle_delta = torch.zeros(n, device=device)
        self._turn_target_angle = torch.zeros(n, device=device)
        self._turn_angle_error = torch.zeros(n, device=device)
        self._turn_subgoal_index = torch.zeros(n, dtype=torch.long, device=device)
        self._turn_subgoal_hold = torch.zeros(n, dtype=torch.long, device=device)
        self._turn_final_hold = torch.zeros(n, dtype=torch.long, device=device)
        self._turn_acquisition_hold = torch.zeros(n, dtype=torch.long, device=device)
        self._turn_acquired = torch.zeros(n, dtype=torch.bool, device=device)
        self._turn_just_acquired = torch.zeros(n, dtype=torch.bool, device=device)
        self._turn_just_subgoal = torch.zeros(n, dtype=torch.bool, device=device)
        self._turn_just_succeeded = torch.zeros(n, dtype=torch.bool, device=device)
        self._turn_success = torch.zeros(n, dtype=torch.bool, device=device)
        self._turn_previous_subgoal_potential = torch.zeros(n, device=device)
        self._turn_subgoal_potential_progress = torch.zeros(n, device=device)
        self._turn_previous_fingertip_distance = torch.zeros(n, 5, device=device)
        self._turn_fingertip_approach = torch.zeros(n, device=device)

        self._turn_fingertip_force_n = torch.zeros(n, 5, device=device)
        self._turn_fingertip_contact = torch.zeros(n, 5, dtype=torch.bool, device=device)
        self._turn_palm_force_n = torch.zeros(n, device=device)
        self._turn_palm_contact = torch.zeros(n, dtype=torch.bool, device=device)
        self._turn_stable_grasp = torch.zeros(n, dtype=torch.bool, device=device)
        self._turn_relative_linear_speed = torch.zeros(n, device=device)
        self._turn_relative_angular_speed = torch.zeros(n, device=device)

        self._turn_constraint_position_error = torch.zeros(n, device=device)
        self._turn_constraint_tilt_error = torch.zeros(n, device=device)
        self._turn_effort_penalty = torch.zeros(n, device=device)
        self._turn_effort_max_ratio = torch.zeros(n, device=device)
        self._turn_effort_saturation = torch.zeros(n, device=device)
        self._turn_effort_mean_ratio = torch.zeros(n, device=device)
        self._turn_previous_actions = torch.zeros(n, cfg.action_space, device=device)
        self._turn_action_delta_sq = torch.zeros(n, device=device)
        self._turn_previous_relative_pos = torch.zeros(n, 3, device=device)
        self._turn_previous_relative_quat = torch.zeros(n, 4, device=device)
        self._turn_previous_relative_quat[:, 0] = 1.0
        self._turn_relative_initialized = torch.zeros(n, dtype=torch.bool, device=device)
        self._turn_cumulative_palm_tool_translation = torch.zeros(n, device=device)
        self._turn_cumulative_palm_tool_rotation = torch.zeros(n, device=device)
        self._turn_previous_contact_code = torch.zeros(n, dtype=torch.long, device=device)
        self._turn_contact_topology_changes = torch.zeros(n, dtype=torch.long, device=device)
        self._turn_previous_stable_grasp = torch.zeros(n, dtype=torch.bool, device=device)
        self._turn_contact_losses = torch.zeros(n, dtype=torch.long, device=device)
        self._turn_contact_reacquisitions = torch.zeros(n, dtype=torch.long, device=device)
        self._turn_arm_joint_margin = torch.zeros(n, device=device)

        self._turn_state_obs = torch.zeros(n, 8, device=device)
        self._turn_geometry_obs = torch.zeros(n, 3, device=device)
        self._turn_grasp_obs = torch.zeros(n, 8, device=device)
        self._turn_effort_obs = torch.zeros(n, 4, device=device)

        self._turn_curriculum_stage = 0
        self._turn_curriculum_completed = 0
        self._turn_curriculum_acquired = 0
        self._turn_curriculum_successes = 0
        self._turn_curriculum_updates = 0
        self._turn_curriculum_acquisition_rate = 0.0
        self._turn_curriculum_conditional_success_rate = 0.0

        lengths = torch.tensor(cfg.assets.allen_key_lengths_m, device=device)
        if lengths.numel() != len(self._object_urdf_paths):
            raise RuntimeError("Allen-key scene asset count does not match configured lengths")
        self._turn_handle_length = lengths[self._object_asset_index_per_env]
        self._turn_geometry_obs[:, 0] = self._turn_handle_length / 0.1
        self._turn_geometry_obs[:, 1] = (
            float(cfg.assets.allen_key_handle_across_flats_m) / 0.1
        )
        self._turn_geometry_obs[:, 2] = (
            float(cfg.assets.allen_key_short_leg_length_m) / 0.1
        )
        self._turn_ready = True
        self._reset_idx(torch.arange(n, device=device))

    @staticmethod
    def _validate_cfg(cfg: SimToolRealAllenKeyTurningEnvCfg) -> None:
        if tuple(cfg.allen_turn_screw_axis_tool) != (0.0, 0.0, -1.0):
            raise ValueError("Allen-key turning currently requires canonical local -Z screw axis")
        if int(cfg.allen_turn_goal_count) * float(cfg.allen_turn_goal_increment_deg) != 360.0:
            raise ValueError("Allen-key subgoals must sum exactly to 360 degrees")
        stage_count = len(cfg.allen_turn_resistance_fractions)
        if stage_count < 2:
            raise ValueError("Allen-key turning curriculum requires at least two stages")
        if len(cfg.allen_turn_xy_half_range_stages_m) != stage_count:
            raise ValueError("Allen-key XY curriculum length is inconsistent")
        if len(cfg.allen_turn_z_range_stages_m) != stage_count:
            raise ValueError("Allen-key Z curriculum length is inconsistent")
        fractions = tuple(float(value) for value in cfg.allen_turn_resistance_fractions)
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("Allen-key resistance fractions must be finite and in [0, 1]")
        if any(later < earlier for earlier, later in zip(fractions, fractions[1:])):
            raise ValueError("Allen-key resistance fractions must be non-decreasing")
        if bool(cfg.allen_turn_require_calibrated_load) and max(fractions) > 0.0:
            calibrated = float(cfg.allen_turn_calibrated_torque_nm)
            if not math.isfinite(calibrated) or calibrated <= 0.0:
                raise ValueError(
                    "Allen-key turning requires a positive calibrated torque; run "
                    "scripts/calibrate_allen_key_resistance.py and pass its recommendation"
                )
        for low, high in cfg.allen_turn_z_range_stages_m:
            if not (math.isfinite(low) and math.isfinite(high) and 0.0 < low < high):
                raise ValueError("Allen-key Z curriculum contains an invalid interval")
        if not cfg.assets.allen_key_lengths_m:
            raise ValueError("Allen-key turning requires a nonempty physical length pool")
        if float(cfg.allen_turn_initial_hand_clearance_m) < 0.0:
            raise ValueError("Allen-key initial hand clearance must be non-negative")
        if float(cfg.allen_turn_initial_arm_clearance_m) <= 0.0:
            raise ValueError("Allen-key initial arm clearance must be positive")
        if float(cfg.allen_turn_fixture_damping_nm_per_radps) <= 0.0:
            raise ValueError("Allen-key fixture damping must be positive")
        if float(cfg.allen_turn_resistance_transition_speed_radps) <= 0.0:
            raise ValueError("Allen-key resistance transition speed must be positive")
        if int(cfg.allen_turn_initial_sampling_max_attempts) <= 0:
            raise ValueError("Allen-key initial sampling attempts must be positive")

    def _setup_scene(self) -> None:
        super()._setup_scene()
        if not hasattr(self, "workpiece"):
            raise RuntimeError("Allen-key turning requires a socket workpiece")
        if not bool(self.cfg.enable_palm_tool_contact_sensor):
            raise RuntimeError("Allen-key turning requires its palm-tool contact sensor")
        try:
            self._turn_palm_contact_sensor = ContactSensor(ContactSensorCfg(
                prim_path=self.cfg.palm_tool_contact_sensor_prim_path,
                update_period=0.0,
                history_length=0,
                debug_vis=False,
                track_pose=False,
                track_contact_points=False,
                track_friction_forces=True,
                track_air_time=False,
                filter_prim_paths_expr=list(self.cfg.palm_tool_contact_sensor_filter_paths),
            ))
        except Exception as exc:
            raise RuntimeError(
                f"Allen-key palm-tool ContactSensor creation failed: {exc!r}"
            ) from exc
        self.scene.sensors["allen_turn_palm_tool_contact"] = self._turn_palm_contact_sensor

    def _current_resistance_torque_nm(self) -> float:
        return float(self.cfg.allen_turn_calibrated_torque_nm) * float(
            self.cfg.allen_turn_resistance_fractions[self._turn_curriculum_stage]
        )

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        super()._reset_idx(env_ids)
        if not getattr(self, "_turn_ready", False):
            return
        count = env_ids.numel()
        stage = self._turn_curriculum_stage
        xy_limit = float(self.cfg.allen_turn_xy_half_range_stages_m[stage])
        z_low, z_high = (
            float(value) for value in self.cfg.allen_turn_z_range_stages_m[stage]
        )
        xy, z, yaw = self._sample_clear_initial_pose(
            env_ids, xy_limit=xy_limit, z_low=z_low, z_high=z_high
        )
        z_axis = torch.zeros(count, 3, device=self.device)
        z_axis[:, 2] = 1.0
        quaternion = quat_from_angle_axis(yaw, z_axis)
        pivot = self.scene.env_origins[env_ids] + torch.cat((xy, z[:, None]), dim=-1)
        pivot_tool = torch.tensor(
            self.cfg.allen_turn_screw_pivot_tool_m, device=self.device
        ).expand(count, -1)
        position = pivot - quat_apply(quaternion, pivot_tool)
        self.object.write_root_pose_to_sim(
            torch.cat((position, quaternion), dim=-1), env_ids=env_ids
        )
        self.object.write_root_velocity_to_sim(
            torch.zeros(count, 6, device=self.device), env_ids=env_ids
        )
        self._object_init_z[env_ids] = position[:, 2] - self.scene.env_origins[env_ids, 2]

        socket_offset = torch.tensor(
            self.cfg.allen_turn_socket_root_from_pivot_m, device=self.device
        ).expand(count, -1)
        socket_position = pivot + socket_offset
        self.workpiece.write_root_pose_to_sim(
            torch.cat((socket_position, quaternion), dim=-1), env_ids=env_ids
        )
        self.workpiece.write_root_velocity_to_sim(
            torch.zeros(count, 6, device=self.device), env_ids=env_ids
        )
        table_position = self.scene.env_origins[env_ids].clone()
        table_position[:, 2] -= float(self.cfg.allen_turn_hidden_table_offset_m)
        identity = torch.zeros(count, 4, device=self.device)
        identity[:, 0] = 1.0
        self.table.write_root_pose_to_sim(
            torch.cat((table_position, identity), dim=-1), env_ids=env_ids
        )
        self.table.write_root_velocity_to_sim(
            torch.zeros(count, 6, device=self.device), env_ids=env_ids
        )

        direction = torch.where(
            torch.rand(count, device=self.device) < 0.5,
            -torch.ones(count, device=self.device),
            torch.ones(count, device=self.device),
        )
        self._turn_phase[env_ids] = 0
        self._turn_direction[env_ids] = direction
        self._turn_initial_tool_pos[env_ids] = position
        self._turn_initial_tool_quat[env_ids] = quaternion
        self._turn_pivot_w[env_ids] = pivot
        self._turn_initial_yaw[env_ids] = yaw
        self._turn_previous_yaw[env_ids] = yaw
        self._turn_cumulative_angle[env_ids] = 0.0
        self._turn_angle_delta[env_ids] = 0.0
        self._turn_target_angle[env_ids] = 0.0
        self._turn_angle_error[env_ids] = 0.0
        self._turn_subgoal_index[env_ids] = 0
        self._turn_subgoal_hold[env_ids] = 0
        self._turn_final_hold[env_ids] = 0
        self._turn_acquisition_hold[env_ids] = 0
        self._turn_acquired[env_ids] = False
        self._turn_just_acquired[env_ids] = False
        self._turn_just_subgoal[env_ids] = False
        self._turn_just_succeeded[env_ids] = False
        self._turn_success[env_ids] = False
        self._turn_previous_subgoal_potential[env_ids] = 0.0
        self._turn_subgoal_potential_progress[env_ids] = 0.0
        self._turn_previous_fingertip_distance[env_ids] = 0.0
        self._turn_fingertip_approach[env_ids] = 0.0
        self._turn_effort_penalty[env_ids] = 0.0
        self._turn_previous_actions[env_ids] = 0.0
        self._turn_action_delta_sq[env_ids] = 0.0
        self._turn_relative_initialized[env_ids] = False
        self._turn_cumulative_palm_tool_translation[env_ids] = 0.0
        self._turn_cumulative_palm_tool_rotation[env_ids] = 0.0
        self._turn_previous_contact_code[env_ids] = 0
        self._turn_contact_topology_changes[env_ids] = 0
        self._turn_previous_stable_grasp[env_ids] = False
        self._turn_contact_losses[env_ids] = 0
        self._turn_contact_reacquisitions[env_ids] = 0
        self._turn_arm_joint_margin[env_ids] = 0.0
        self._write_turn_goal(env_ids, torch.zeros(count, device=self.device))

    def _sample_clear_initial_pose(
        self,
        env_ids: torch.Tensor,
        *,
        xy_limit: float,
        z_low: float,
        z_high: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reject either tool segment intersecting the default hand or arm."""
        count = env_ids.numel()
        xy = torch.zeros(count, 2, device=self.device)
        z = torch.zeros(count, device=self.device)
        yaw = torch.zeros(count, device=self.device)
        pending = torch.arange(count, device=self.device)
        hand_points = self._turn_default_hand_points_local[env_ids]
        arm_points = self._turn_default_arm_points_local[env_ids]
        lengths = self._turn_handle_length[env_ids]
        clearance = (
            0.5 * float(self.cfg.assets.allen_key_handle_across_flats_m)
            + float(self.cfg.allen_turn_initial_hand_clearance_m)
        )
        for _ in range(int(self.cfg.allen_turn_initial_sampling_max_attempts)):
            if pending.numel() == 0:
                break
            sampled_xy = torch.empty(
                pending.numel(), 2, device=self.device
            ).uniform_(-xy_limit, xy_limit)
            sampled_z = torch.empty(pending.numel(), device=self.device).uniform_(
                z_low, z_high
            )
            sampled_yaw = torch.empty(pending.numel(), device=self.device).uniform_(
                -math.pi, math.pi
            )
            direction = torch.stack((
                torch.cos(sampled_yaw), torch.sin(sampled_yaw),
                torch.zeros_like(sampled_yaw),
            ), dim=-1)
            pivot_local = torch.cat((sampled_xy, sampled_z[:, None]), dim=-1)
            # The long segment lies 3 cm above the screw pivot and extends
            # backward from the elbow along local -X.
            near = pivot_local.clone()
            near[:, 2] += 0.03
            far = near - direction * lengths[pending, None]
            short_far = pivot_local.clone()
            short_far[:, 2] += float(self.cfg.assets.allen_key_short_leg_length_m)

            def minimum_segment_distance(
                points: torch.Tensor, start: torch.Tensor, end: torch.Tensor
            ) -> torch.Tensor:
                segment = end - start
                segment_sq = segment.square().sum(-1).clamp_min(1.0e-12)
                point_delta = points - start[:, None, :]
                fraction = (
                    (point_delta * segment[:, None, :]).sum(-1)
                    / segment_sq[:, None]
                ).clamp(0.0, 1.0)
                closest = start[:, None, :] + fraction[:, :, None] * segment[:, None, :]
                return torch.linalg.vector_norm(points - closest, dim=-1).amin(-1)

            pending_hand = hand_points[pending]
            pending_arm = arm_points[pending]
            hand_distance = torch.minimum(
                minimum_segment_distance(pending_hand, near, far),
                minimum_segment_distance(pending_hand, pivot_local, short_far),
            )
            arm_distance = torch.minimum(
                minimum_segment_distance(pending_arm, near, far),
                minimum_segment_distance(pending_arm, pivot_local, short_far),
            )
            valid = (hand_distance >= clearance) & (
                arm_distance >= float(self.cfg.allen_turn_initial_arm_clearance_m)
            )
            accepted = pending[valid]
            xy[accepted] = sampled_xy[valid]
            z[accepted] = sampled_z[valid]
            yaw[accepted] = sampled_yaw[valid]
            pending = pending[~valid]
        if pending.numel():
            raise RuntimeError(
                "Allen-key reset sampling could not clear the default robot for "
                f"{pending.numel()}/{count} environments after "
                f"{self.cfg.allen_turn_initial_sampling_max_attempts} attempts"
            )
        return xy, z, yaw

    def _write_turn_goal(self, env_ids: torch.Tensor, angle: torch.Tensor) -> None:
        pivot_tool = torch.tensor(
            self.cfg.allen_turn_screw_pivot_tool_m, device=self.device
        ).expand(env_ids.numel(), -1)
        axis_tool = torch.tensor(
            self.cfg.allen_turn_screw_axis_tool, device=self.device
        ).expand(env_ids.numel(), -1)
        position, quaternion = turn_goal_pose(
            self._turn_initial_tool_pos[env_ids],
            self._turn_initial_tool_quat[env_ids],
            pivot_tool,
            axis_tool,
            angle,
        )
        self.goal_viz.write_root_pose_to_sim(
            torch.cat((position, quaternion), dim=-1), env_ids=env_ids
        )
        self.goal_viz.write_root_velocity_to_sim(
            torch.zeros(env_ids.numel(), 6, device=self.device), env_ids=env_ids
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._turn_action_delta_sq.copy_(
            (actions - self._turn_previous_actions).square().mean(-1)
        )
        self._turn_previous_actions.copy_(actions)
        apply_action_pipeline(self, actions)
        active = self._turn_phase >= 1
        torque = torch.zeros(self.num_envs, 3, device=self.device)
        angular_velocity = self.object.data.root_ang_vel_w[:, 2]
        damping = (
            -float(self.cfg.allen_turn_fixture_damping_nm_per_radps)
            * angular_velocity
        )
        override = getattr(self, "_turn_external_torque_override_nm", None)
        if override is None:
            transition_speed = float(
                self.cfg.allen_turn_resistance_transition_speed_radps
            )
            resistance = -self._current_resistance_torque_nm() * torch.tanh(
                angular_velocity / transition_speed
            )
        else:
            if override.shape != (self.num_envs,):
                raise RuntimeError(
                    "Allen-key torque override must have shape "
                    f"({self.num_envs},), got {tuple(override.shape)}"
                )
            if not bool(torch.isfinite(override).all()):
                raise RuntimeError("Allen-key torque override contains NaN or Inf")
            resistance = override
        torque[:, 2] = (damping + resistance) * active.float()
        self.object.set_external_force_and_torque(
            torch.zeros(self.num_envs, 1, 3, device=self.device),
            torque[:, None, :],
            is_global=True,
        )

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._cur_targets)
        if not getattr(self, "_turn_ready", False):
            return
        env_ids = torch.arange(self.num_envs, device=self.device)
        yaw = yaw_from_quaternion(self.object.data.root_quat_w)
        z_axis = torch.zeros(self.num_envs, 3, device=self.device)
        z_axis[:, 2] = 1.0
        quaternion = quat_from_angle_axis(yaw, z_axis)
        pivot_tool = torch.tensor(
            self.cfg.allen_turn_screw_pivot_tool_m, device=self.device
        ).expand(self.num_envs, -1)
        position = self._turn_pivot_w - quat_apply(quaternion, pivot_tool)
        angular_velocity = torch.zeros(self.num_envs, 3, device=self.device)
        angular_velocity[:, 2] = self.object.data.root_ang_vel_w[:, 2]
        self.object.write_root_pose_to_sim(
            torch.cat((position, quaternion), dim=-1), env_ids=env_ids
        )
        self.object.write_root_velocity_to_sim(
            torch.cat((torch.zeros_like(angular_velocity), angular_velocity), dim=-1),
            env_ids=env_ids,
        )
        socket_position = self._turn_pivot_w + torch.tensor(
            self.cfg.allen_turn_socket_root_from_pivot_m, device=self.device
        )
        self.workpiece.write_root_pose_to_sim(
            torch.cat((socket_position, quaternion), dim=-1), env_ids=env_ids
        )
        self.workpiece.write_root_velocity_to_sim(
            torch.zeros(self.num_envs, 6, device=self.device), env_ids=env_ids
        )

    def _read_contacts(self) -> None:
        sensors = getattr(self, "_fingertip_tool_contact_sensors", None)
        if sensors is None or len(sensors) != 5:
            raise RuntimeError("Allen-key turning requires exactly five fingertip sensors")
        forces = []
        for sensor_id, sensor in enumerate(sensors):
            matrix = getattr(getattr(sensor, "data", None), "force_matrix_w", None)
            if matrix is None or matrix.shape[0] != self.num_envs or matrix.shape[-1] != 3:
                shape = None if matrix is None else tuple(matrix.shape)
                raise RuntimeError(
                    f"Allen-key fingertip sensor {sensor_id} has invalid matrix {shape}"
                )
            force = torch.linalg.vector_norm(
                matrix.reshape(self.num_envs, -1, 3), dim=-1
            ).sum(-1)
            if not bool(torch.isfinite(force).all()):
                raise RuntimeError(f"Allen-key fingertip sensor {sensor_id} is non-finite")
            forces.append(force)
        self._turn_fingertip_force_n.copy_(torch.stack(forces, dim=-1))
        self._turn_fingertip_contact.copy_(
            self._turn_fingertip_force_n
            >= float(self.cfg.allen_turn_contact_force_threshold_n)
        )

        matrix = getattr(
            getattr(self._turn_palm_contact_sensor, "data", None),
            "force_matrix_w",
            None,
        )
        if matrix is None or matrix.shape[0] != self.num_envs or matrix.shape[-1] != 3:
            shape = None if matrix is None else tuple(matrix.shape)
            raise RuntimeError(f"Allen-key palm sensor has invalid matrix {shape}")
        palm_force = torch.linalg.vector_norm(
            matrix.reshape(self.num_envs, -1, 3), dim=-1
        ).sum(-1)
        if not bool(torch.isfinite(palm_force).all()):
            raise RuntimeError("Allen-key palm sensor is non-finite")
        self._turn_palm_force_n.copy_(palm_force)
        self._turn_palm_contact.copy_(
            palm_force >= float(self.cfg.allen_turn_palm_contact_threshold_n)
        )

    def _update_metrics(self) -> None:
        compute_intermediate_values(self)
        self._read_contacts()
        current_yaw = yaw_from_quaternion(self.object.data.root_quat_w)
        _, yaw_delta = update_unwrapped_angle(
            self._turn_previous_yaw, current_yaw, self._turn_cumulative_angle
        )
        # The canonical screw axis is local -Z, opposite positive world yaw.
        cumulative = self._turn_cumulative_angle - yaw_delta
        delta = -yaw_delta
        self._turn_cumulative_angle.copy_(cumulative)
        self._turn_angle_delta.copy_(delta)
        self._turn_previous_yaw.copy_(current_yaw)

        pivot_tool = torch.tensor(
            self.cfg.allen_turn_screw_pivot_tool_m, device=self.device
        ).expand(self.num_envs, -1)
        current_pivot = self.object.data.root_pos_w + quat_apply(
            self.object.data.root_quat_w, pivot_tool
        )
        self._turn_constraint_position_error.copy_(
            torch.linalg.vector_norm(current_pivot - self._turn_pivot_w, dim=-1)
        )
        local_axis = torch.tensor(
            self.cfg.allen_turn_screw_axis_tool, device=self.device
        ).expand(self.num_envs, -1)
        axis_world = quat_apply(self.object.data.root_quat_w, local_axis)
        expected_axis = torch.zeros_like(axis_world)
        expected_axis[:, 2] = -1.0
        self._turn_constraint_tilt_error.copy_(
            torch.acos((axis_world * expected_axis).sum(-1).clamp(-1.0, 1.0))
        )

        palm_position = self.robot.data.body_link_pos_w[:, self._palm_body_id]
        palm_quaternion = self.robot.data.body_link_quat_w[:, self._palm_body_id]
        relative_position, relative_quaternion = subtract_frame_transforms(
            palm_position,
            palm_quaternion,
            self.object.data.root_pos_w,
            self.object.data.root_quat_w,
        )
        alignment = torch.abs(
            (relative_quaternion * self._turn_previous_relative_quat).sum(-1)
        ).clamp(0.0, 1.0)
        relative_rotation_delta = 2.0 * torch.acos(alignment)
        relative_translation_delta = torch.linalg.vector_norm(
            relative_position - self._turn_previous_relative_pos, dim=-1
        )
        initialized = self._turn_relative_initialized
        control_dt = float(self.step_dt)
        self._turn_relative_linear_speed.copy_(torch.where(
            initialized, relative_translation_delta / control_dt,
            torch.zeros_like(relative_translation_delta),
        ))
        self._turn_relative_angular_speed.copy_(torch.where(
            initialized, relative_rotation_delta / control_dt,
            torch.zeros_like(relative_rotation_delta),
        ))
        contact_support = self._turn_palm_contact | (
            self._turn_fingertip_contact.sum(-1)
            >= int(self.cfg.allen_turn_minimum_contact_fingers)
        )
        self._turn_stable_grasp.copy_(
            contact_support
            & (
                self._turn_relative_linear_speed
                <= float(self.cfg.allen_turn_max_relative_linear_speed_mps)
            )
            & (
                self._turn_relative_angular_speed
                <= float(self.cfg.allen_turn_max_relative_angular_speed_radps)
            )
        )

        diagnostic_active = self._turn_relative_initialized & self._turn_acquired
        self._turn_cumulative_palm_tool_translation += torch.where(
            diagnostic_active, relative_translation_delta, 0.0
        )
        self._turn_cumulative_palm_tool_rotation += torch.where(
            diagnostic_active, relative_rotation_delta, 0.0
        )
        self._turn_previous_relative_pos.copy_(relative_position)
        self._turn_previous_relative_quat.copy_(relative_quaternion)
        self._turn_relative_initialized.fill_(True)

        contact_code = self._turn_palm_contact.long()
        for finger_id in range(5):
            contact_code += self._turn_fingertip_contact[:, finger_id].long() << (finger_id + 1)
        topology_changed = self._turn_acquired & (
            contact_code != self._turn_previous_contact_code
        )
        self._turn_contact_topology_changes += topology_changed.long()
        contact_lost = (
            self._turn_acquired & self._turn_previous_stable_grasp
            & ~self._turn_stable_grasp
        )
        contact_reacquired = (
            self._turn_acquired & ~self._turn_previous_stable_grasp
            & self._turn_stable_grasp
        )
        self._turn_contact_losses += contact_lost.long()
        self._turn_contact_reacquisitions += contact_reacquired.long()
        self._turn_previous_contact_code.copy_(contact_code)
        self._turn_previous_stable_grasp.copy_(self._turn_stable_grasp)
        arm_position = self.robot.data.joint_pos[:, self._arm_joint_ids]
        arm_limits = self.robot.data.joint_pos_limits[:, self._arm_joint_ids]
        self._turn_arm_joint_margin.copy_(torch.minimum(
            arm_position - arm_limits[:, :, 0],
            arm_limits[:, :, 1] - arm_position,
        ).amin(-1))

        hand_torque = self.robot.data.applied_torque[:, self._hand_joint_ids]
        hand_limit = self.robot.data.joint_effort_limits[:, self._hand_joint_ids]
        penalty, maximum, saturation = finger_effort_soft_penalty(
            hand_torque,
            hand_limit,
            float(self.cfg.allen_turn_effort_soft_threshold_fraction),
        )
        ratio = hand_torque.abs() / hand_limit
        self._turn_effort_penalty.copy_(penalty)
        self._turn_effort_max_ratio.copy_(maximum)
        self._turn_effort_saturation.copy_(saturation)
        self._turn_effort_mean_ratio.copy_(ratio.mean(-1))

        self._turn_just_acquired.zero_()
        self._turn_just_subgoal.zero_()
        self._turn_just_succeeded.zero_()
        acquiring = self._turn_phase == 0
        self._turn_acquisition_hold.copy_(torch.where(
            acquiring & self._turn_stable_grasp,
            self._turn_acquisition_hold + 1,
            torch.zeros_like(self._turn_acquisition_hold),
        ))
        acquired_now = acquiring & (
            self._turn_acquisition_hold >= int(self.cfg.allen_turn_acquisition_hold_steps)
        )
        if bool(acquired_now.any()):
            ids = torch.nonzero(acquired_now, as_tuple=False).squeeze(-1)
            self._turn_phase[ids] = 1
            self._turn_acquired[ids] = True
            self._turn_just_acquired[ids] = True
            increment = math.radians(float(self.cfg.allen_turn_goal_increment_deg))
            self._turn_target_angle[ids] = self._turn_direction[ids] * increment
            self._turn_previous_subgoal_potential[ids] = 0.0
            self._write_turn_goal(ids, self._turn_target_angle[ids])

        turning = self._turn_phase == 1
        self._turn_angle_error.copy_(self._turn_target_angle - self._turn_cumulative_angle)
        tolerance = math.radians(float(self.cfg.allen_turn_goal_tolerance_deg))
        at_subgoal = turning & self._turn_stable_grasp & (
            self._turn_angle_error.abs() <= tolerance
        )
        self._turn_subgoal_hold.copy_(torch.where(
            at_subgoal,
            self._turn_subgoal_hold + 1,
            torch.zeros_like(self._turn_subgoal_hold),
        ))
        reached = turning & (
            self._turn_subgoal_hold >= int(self.cfg.allen_turn_goal_hold_steps)
        )
        if bool(reached.any()):
            ids = torch.nonzero(reached, as_tuple=False).squeeze(-1)
            self._turn_just_subgoal[ids] = True
            self._turn_subgoal_index[ids] += 1
            self._turn_subgoal_hold[ids] = 0
            final = self._turn_subgoal_index[ids] >= int(self.cfg.allen_turn_goal_count)
            final_ids = ids[final]
            continuing_ids = ids[~final]
            if continuing_ids.numel():
                target = (
                    self._turn_direction[continuing_ids]
                    * math.radians(float(self.cfg.allen_turn_goal_increment_deg))
                    * (self._turn_subgoal_index[continuing_ids].float() + 1.0)
                )
                self._turn_target_angle[continuing_ids] = target
                self._turn_previous_subgoal_potential[continuing_ids] = 0.0
                self._write_turn_goal(continuing_ids, target)
            if final_ids.numel():
                self._turn_phase[final_ids] = 2
                self._turn_final_hold[final_ids] = 0

        holding = self._turn_phase == 2
        final_target = self._turn_direction * 2.0 * math.pi
        final_valid = holding & self._turn_stable_grasp & (
            (final_target - self._turn_cumulative_angle).abs() <= tolerance
        )
        self._turn_final_hold.copy_(torch.where(
            final_valid,
            self._turn_final_hold + 1,
            torch.zeros_like(self._turn_final_hold),
        ))
        succeeded = holding & (
            self._turn_final_hold >= int(self.cfg.allen_turn_final_hold_steps)
        )
        self._turn_just_succeeded.copy_(succeeded & ~self._turn_success)
        self._turn_success |= succeeded

        phase_one_hot = torch.nn.functional.one_hot(
            self._turn_phase.clamp(0, 2), num_classes=3
        ).float()
        signed_progress = self._turn_direction * self._turn_cumulative_angle
        signed_target = self._turn_direction * self._turn_target_angle
        increment = math.radians(float(self.cfg.allen_turn_goal_increment_deg))
        self._turn_state_obs[:, :3] = phase_one_hot
        self._turn_state_obs[:, 3] = self._turn_direction
        self._turn_state_obs[:, 4] = signed_progress / (2.0 * math.pi)
        self._turn_state_obs[:, 5] = signed_target / (2.0 * math.pi)
        self._turn_state_obs[:, 6] = self._turn_direction * self._turn_angle_error / increment
        calibrated = max(float(self.cfg.allen_turn_calibrated_torque_nm), 1.0e-6)
        self._turn_state_obs[:, 7] = self._current_resistance_torque_nm() / calibrated
        self._turn_grasp_obs[:, 0] = self._turn_palm_contact.float()
        self._turn_grasp_obs[:, 1:6] = self._turn_fingertip_contact.float()
        self._turn_grasp_obs[:, 6] = self._turn_stable_grasp.float()
        self._turn_grasp_obs[:, 7] = self._turn_acquired.float()
        self._turn_effort_obs[:, 0] = self._turn_effort_mean_ratio
        self._turn_effort_obs[:, 1] = self._turn_effort_max_ratio
        self._turn_effort_obs[:, 2] = self._turn_effort_saturation
        self._turn_effort_obs[:, 3] = self._turn_effort_penalty

        current_distance = self._curr_fingertip_distances
        initialized = self._turn_previous_fingertip_distance.sum(-1) > 0.0
        approach = torch.clamp(
            self._turn_previous_fingertip_distance - current_distance,
            min=0.0,
            max=0.02,
        ).sum(-1)
        self._turn_fingertip_approach.copy_(
            torch.where(initialized & acquiring, approach, torch.zeros_like(approach))
        )
        self._turn_previous_fingertip_distance.copy_(current_distance)

        error = self._turn_angle_error.abs()
        potential = torch.exp(-error / max(tolerance, 1.0e-6))
        potential_progress = potential - self._turn_previous_subgoal_potential
        potential_progress = torch.where(turning, potential_progress, torch.zeros_like(potential))
        self._turn_subgoal_potential_progress.copy_(potential_progress)
        self._turn_previous_subgoal_potential.copy_(torch.where(
            turning, potential, self._turn_previous_subgoal_potential
        ))
        reset_potential = self._turn_just_subgoal & (self._turn_phase == 1)
        self._turn_previous_subgoal_potential[reset_potential] = 0.0

    def _update_curriculum(self, done: torch.Tensor) -> None:
        count = int(done.sum().item())
        if count == 0:
            return
        self._turn_curriculum_completed += count
        self._turn_curriculum_acquired += int((done & self._turn_acquired).sum().item())
        self._turn_curriculum_successes += int((done & self._turn_success).sum().item())
        minimum = int(self.cfg.allen_turn_curriculum_min_episodes)
        if self._turn_curriculum_completed < minimum:
            return
        ready, acquisition_rate, conditional_rate = turning_curriculum_ready(
            self._turn_curriculum_acquired,
            self._turn_curriculum_completed,
            self._turn_curriculum_successes,
            minimum_episodes=minimum,
            acquisition_threshold=float(
                self.cfg.allen_turn_curriculum_acquisition_threshold
            ),
            conditional_turn_threshold=float(
                self.cfg.allen_turn_curriculum_conditional_success_threshold
            ),
        )
        self._turn_curriculum_acquisition_rate = acquisition_rate
        self._turn_curriculum_conditional_success_rate = conditional_rate
        last = len(self.cfg.allen_turn_resistance_fractions) - 1
        if ready and self._turn_curriculum_stage < last:
            self._turn_curriculum_stage += 1
            self._turn_curriculum_updates += 1
        self._turn_curriculum_completed = 0
        self._turn_curriculum_acquired = 0
        self._turn_curriculum_successes = 0

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_metrics()
        acquisition_failure = (
            (self._turn_phase == 0)
            & (self.episode_length_buf >= int(self.cfg.allen_turn_acquisition_timeout_steps))
        )
        success = self._turn_success
        timeout = self.episode_length_buf >= int(self.cfg.termination.episode_length)
        terminated = acquisition_failure | success
        truncated = timeout & ~terminated
        done = terminated | truncated
        self._is_success.copy_(success)
        self._termination_reasons = {
            "allen_turn_success": success,
            "allen_turn_acquisition_failure": acquisition_failure,
            "timeout": truncated,
        }
        self._episode_final_terms = {
            "allen_turn_acquired": self._turn_acquired.float(),
            "allen_turn_subgoals_completed": self._turn_subgoal_index.float(),
            "allen_turn_full_success": self._turn_success.float(),
            "allen_turn_signed_progress_deg": torch.rad2deg(
                self._turn_direction * self._turn_cumulative_angle
            ),
            "allen_turn_effort_max_ratio": self._turn_effort_max_ratio,
            "allen_turn_palm_tool_translation_m": (
                self._turn_cumulative_palm_tool_translation
            ),
            "allen_turn_palm_tool_rotation_deg": torch.rad2deg(
                self._turn_cumulative_palm_tool_rotation
            ),
            "allen_turn_contact_topology_changes": (
                self._turn_contact_topology_changes.float()
            ),
            "allen_turn_contact_losses": self._turn_contact_losses.float(),
            "allen_turn_contact_reacquisitions": (
                self._turn_contact_reacquisitions.float()
            ),
            "allen_turn_arm_joint_margin_rad": self._turn_arm_joint_margin,
        }
        self._update_curriculum(done)
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        phase_turning = self._turn_phase >= 1
        increment = math.radians(float(self.cfg.allen_turn_goal_increment_deg))
        directed_delta = self._turn_direction * self._turn_angle_delta
        directed_progress = torch.clamp(
            directed_delta / increment, min=-0.25, max=0.25
        ) * phase_turning.float()
        constraint = (
            self._turn_constraint_position_error
            / float(self.cfg.allen_turn_constraint_position_tolerance_m)
        ).clamp(0.0, 2.0) + (
            self._turn_constraint_tilt_error
            / math.radians(float(self.cfg.allen_turn_constraint_tilt_tolerance_deg))
        ).clamp(0.0, 2.0)
        weighted = {
            "acquisition_approach_rew": (
                float(self.cfg.allen_turn_fingertip_approach_weight)
                * self._turn_fingertip_approach
            ),
            "acquisition_bonus": (
                float(self.cfg.allen_turn_acquisition_bonus)
                * self._turn_just_acquired.float()
            ),
            "turn_progress_rew": (
                float(self.cfg.allen_turn_angular_progress_weight) * directed_progress
            ),
            "subgoal_progress_rew": (
                float(self.cfg.allen_turn_subgoal_progress_weight)
                * self._turn_subgoal_potential_progress
            ),
            "subgoal_bonus": (
                float(self.cfg.allen_turn_subgoal_bonus)
                * self._turn_just_subgoal.float()
            ),
            "final_hold_rew": (
                float(self.cfg.allen_turn_final_hold_reward)
                * ((self._turn_phase == 2) & self._turn_stable_grasp).float()
            ),
            "final_success_bonus": (
                float(self.cfg.allen_turn_final_success_bonus)
                * self._turn_just_succeeded.float()
            ),
            "finger_effort_penalty": (
                -float(self.cfg.allen_turn_effort_penalty_weight)
                * self._turn_effort_penalty
            ),
            "constraint_penalty": (
                -float(self.cfg.allen_turn_constraint_penalty_weight) * constraint
            ),
            "action_rate_penalty": (
                -float(self.cfg.allen_turn_action_rate_penalty_weight)
                * self._turn_action_delta_sq
            ),
        }
        reward = torch.stack(tuple(weighted.values())).sum(0)
        if not bool(torch.isfinite(reward).all()):
            raise RuntimeError("Allen-key turning reward contains NaN or Inf")
        self._reward_terms = {**weighted, "total_reward": reward}
        self._episode_cumulative_terms = {}
        for name, mask in (
            ("acquisition", self._turn_phase == 0),
            ("turning", self._turn_phase == 1),
            ("final_hold", self._turn_phase == 2),
        ):
            self._episode_cumulative_terms[f"phase/{name}_reward_sum"] = reward * mask.float()
            self._episode_cumulative_terms[f"phase/{name}_step_count"] = mask.float()
        self.extras.update({
            "allen_turn/acquisition_ratio": self._turn_acquired.float().mean(),
            "allen_turn/stable_grasp_ratio": self._turn_stable_grasp.float().mean(),
            "allen_turn/subgoals_completed_mean": self._turn_subgoal_index.float().mean(),
            "allen_turn/full_turn_success_ratio": self._turn_success.float().mean(),
            "allen_turn/signed_progress_mean_deg": torch.rad2deg(
                self._turn_direction * self._turn_cumulative_angle
            ).mean(),
            "allen_turn/angle_error_mean_deg": torch.rad2deg(
                self._turn_angle_error.abs()
            ).mean(),
            "allen_turn/palm_contact_ratio": self._turn_palm_contact.float().mean(),
            "allen_turn/fingertip_contact_count_mean": (
                self._turn_fingertip_contact.float().sum(-1).mean()
            ),
            "allen_turn/finger_effort_mean_ratio": self._turn_effort_mean_ratio.mean(),
            "allen_turn/finger_effort_max_ratio": self._turn_effort_max_ratio.mean(),
            "allen_turn/finger_effort_saturation_ratio": self._turn_effort_saturation.mean(),
            "allen_turn/palm_tool_translation_cumulative_mean_m": (
                self._turn_cumulative_palm_tool_translation.mean()
            ),
            "allen_turn/palm_tool_rotation_cumulative_mean_deg": torch.rad2deg(
                self._turn_cumulative_palm_tool_rotation
            ).mean(),
            "allen_turn/contact_topology_changes_mean": (
                self._turn_contact_topology_changes.float().mean()
            ),
            "allen_turn/contact_losses_mean": self._turn_contact_losses.float().mean(),
            "allen_turn/contact_reacquisitions_mean": (
                self._turn_contact_reacquisitions.float().mean()
            ),
            "allen_turn/arm_joint_margin_mean_rad": self._turn_arm_joint_margin.mean(),
            "allen_turn/constraint_position_error_mean_m": (
                self._turn_constraint_position_error.mean()
            ),
            "allen_turn/constraint_tilt_error_mean_deg": torch.rad2deg(
                self._turn_constraint_tilt_error
            ).mean(),
            "curriculum/allen_turn_stage": self._turn_curriculum_stage,
            "curriculum/allen_turn_resistance_nm": self._current_resistance_torque_nm(),
            "curriculum/allen_turn_acquisition_rate": (
                self._turn_curriculum_acquisition_rate
            ),
            "curriculum/allen_turn_conditional_success_rate": (
                self._turn_curriculum_conditional_success_rate
            ),
            "curriculum/allen_turn_updates": self._turn_curriculum_updates,
        })
        log_step_metrics(self)
        return reward


__all__ = ["SimToolRealAllenKeyTurningEnv"]
