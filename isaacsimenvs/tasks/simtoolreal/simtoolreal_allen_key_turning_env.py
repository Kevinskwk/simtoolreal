"""Single-policy resistance-loaded Allen-key pose tracking."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, Sdf, Usd, UsdPhysics
from isaaclab.utils.math import (
    quat_apply,
    quat_from_angle_axis,
    subtract_frame_transforms,
)

from .simtoolreal_tacmap_env import SimToolRealTacMapEnv
from .simtoolreal_tacmap_env_cfg import SimToolRealAllenKeyTurningEnvCfg
from .utils.action_utils import apply_action_pipeline
from .utils.allen_key_turning_utils import (
    deep_grasp_quality,
    finger_effort_soft_penalty,
    gate_positive_progress,
    loaded_grasp_quality,
    stick_slip_torsional_friction,
    translational_force_imbalance_penalty,
    turn_goal_pose,
    update_consecutive_grasp_hold,
    update_unwrapped_angle,
    wrap_to_pi,
    yaw_from_quaternion,
)
from .utils.logging_utils import log_step_metrics
from .utils.obs_utils import compute_intermediate_values
from .utils.palm_geometry import palm_center_pose_from_merged_wrist
from .utils.reward_utils import compute_rewards
from .utils.reset_utils import _randomize_robot_dof_state


class SimToolRealAllenKeyTurningEnv(SimToolRealTacMapEnv):
    """Finetune SimToolReal for sustained, resistance-loaded Allen-key turning."""

    cfg: SimToolRealAllenKeyTurningEnvCfg

    def __init__(
        self,
        cfg: SimToolRealAllenKeyTurningEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        self._turn_ready = False
        self._turn_fixture_joints_created = False
        self._turn_fixture_robot_joint_pos: torch.Tensor | None = None
        self._validate_cfg(cfg)
        super().__init__(cfg, render_mode, **kwargs)
        n, device = self.num_envs, self.device

        palm_position, _ = self._current_palm_center_pose_w()
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
        self._turn_arm_body_ids = arm_body_ids
        self._turn_default_arm_points_local = (
            self.robot.data.body_link_pos_w[:, arm_body_ids]
            - self.scene.env_origins[:, None, :]
        )
        try:
            shoulder_body_id = self.robot.data.body_names.index("iiwa14_link_2")
        except ValueError as exc:
            raise RuntimeError("iiwa14_link_2 is unavailable for reach screening") from exc
        self._turn_shoulder_body_id = shoulder_body_id
        self._turn_default_shoulder_point_local = (
            self.robot.data.body_link_pos_w[:, shoulder_body_id]
            - self.scene.env_origins
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
        self._turn_just_subgoal = torch.zeros(n, dtype=torch.bool, device=device)
        self._turn_just_succeeded = torch.zeros(n, dtype=torch.bool, device=device)
        self._turn_success = torch.zeros(n, dtype=torch.bool, device=device)
        self._turn_previous_subgoal_potential = torch.zeros(n, device=device)
        self._turn_subgoal_potential_progress = torch.zeros(n, device=device)

        self._turn_finger_force_n = torch.zeros(n, 5, device=device)
        self._turn_finger_force_w = torch.zeros(n, 5, 3, device=device)
        self._turn_finger_contact = torch.zeros(n, 5, dtype=torch.bool, device=device)
        self._turn_proximal_finger_contact = torch.zeros(
            n, 5, dtype=torch.bool, device=device
        )
        self._turn_palm_force_n = torch.zeros(n, device=device)
        self._turn_contact_resultant_force_w = torch.zeros(n, 3, device=device)
        self._turn_contact_resultant_force_n = torch.zeros(n, device=device)
        self._turn_summed_contact_load_n = torch.zeros(n, device=device)
        self._turn_translational_force_imbalance = torch.zeros(n, device=device)
        self._turn_translational_force_penalty = torch.zeros(n, device=device)
        arm_table_sensor_count = len(cfg.allen_turn_arm_table_contact_prim_paths)
        self._turn_arm_table_contact_force_n = torch.zeros(
            n, arm_table_sensor_count, device=device
        )
        self._turn_arm_table_contact = torch.zeros(
            n, arm_table_sensor_count, dtype=torch.bool, device=device
        )
        self._turn_arm_table_contact_penalty = torch.zeros(n, device=device)
        self._turn_palm_contact = torch.zeros(n, dtype=torch.bool, device=device)
        self._turn_palm_supported = torch.zeros(n, dtype=torch.bool, device=device)
        self._turn_contact_support = torch.zeros(n, dtype=torch.bool, device=device)
        self._turn_stable_grasp = torch.zeros(n, dtype=torch.bool, device=device)
        self._turn_loaded_grasp = torch.zeros(n, dtype=torch.bool, device=device)
        self._turn_grasp_quality = torch.zeros(n, device=device)
        self._turn_max_grasp_quality = torch.zeros(n, device=device)
        self._turn_ever_loaded_grasp = torch.zeros(n, dtype=torch.bool, device=device)
        self._turn_just_loaded_grasp = torch.zeros(n, dtype=torch.bool, device=device)
        self._turn_grasp_hold_count = torch.zeros(n, dtype=torch.long, device=device)
        self._turn_deep_grasp = torch.zeros(n, dtype=torch.bool, device=device)
        self._turn_deep_grasp_confirmed = torch.zeros(
            n, dtype=torch.bool, device=device
        )
        self._turn_deep_grasp_hold_count = torch.zeros(
            n, dtype=torch.long, device=device
        )
        self._turn_deep_grasp_quality = torch.zeros(n, device=device)
        self._turn_deep_grasp_opposition_cosine = torch.ones(n, device=device)
        self._turn_max_deep_grasp_quality = torch.zeros(n, device=device)
        self._turn_relative_linear_speed = torch.zeros(n, device=device)
        self._turn_relative_angular_speed = torch.zeros(n, device=device)
        self._turn_palm_handle_distance = torch.zeros(n, device=device)
        self._turn_min_palm_handle_distance = torch.full(
            (n,), float("inf"), device=device
        )
        self._turn_max_finger_contact_count = torch.zeros(
            n, dtype=torch.long, device=device
        )
        self._turn_ever_palm_contact = torch.zeros(n, dtype=torch.bool, device=device)
        self._turn_previous_approach_potential = torch.zeros(n, device=device)
        self._turn_approach_initialized = torch.zeros(n, dtype=torch.bool, device=device)
        self._turn_approach_potential = torch.zeros(n, device=device)
        self._turn_approach_progress = torch.zeros(n, device=device)
        self._turn_initial_palm_handle_distance = torch.zeros(n, device=device)
        self._turn_initial_robot_resample_count = torch.zeros(
            n, dtype=torch.long, device=device
        )
        self._turn_qualified_progress = torch.zeros(n, device=device)
        self._turn_unqualified_positive_progress = torch.zeros(n, device=device)
        self._turn_final_hold_valid = torch.zeros(n, dtype=torch.bool, device=device)

        self._turn_constraint_position_error = torch.zeros(n, device=device)
        self._turn_constraint_tilt_error = torch.zeros(n, device=device)
        self._turn_effort_penalty = torch.zeros(n, device=device)
        self._turn_effort_max_ratio = torch.zeros(n, device=device)
        self._turn_effort_saturation = torch.zeros(n, device=device)
        self._turn_effort_mean_ratio = torch.zeros(n, device=device)
        self._turn_stiction_yaw = torch.zeros(n, device=device)
        self._turn_friction_stuck = torch.ones(n, dtype=torch.bool, device=device)
        self._turn_fixture_torque_nm = torch.zeros(n, device=device)
        self._turn_fixture_torque_clipped = torch.zeros(
            n, dtype=torch.bool, device=device
        )
        self._turn_fixture_torque_peak_nm = torch.zeros(n, device=device)
        self._turn_fixture_torque_clipped_steps = torch.zeros(
            n, dtype=torch.long, device=device
        )
        self._turn_coulomb_friction_nm = torch.zeros(n, device=device)
        self._turn_damping_nm_per_radps = torch.zeros(n, device=device)
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
        self._turn_initial_shoulder_to_handle_m = torch.zeros(n, device=device)

        self._turn_state_obs = torch.zeros(n, 8, device=device)
        self._turn_geometry_obs = torch.zeros(n, 3, device=device)
        self._turn_grasp_obs = torch.zeros(n, 8, device=device)
        self._turn_effort_obs = torch.zeros(n, 4, device=device)

        self._turn_curriculum_stage = 0
        self._turn_curriculum_completed = 0
        self._turn_curriculum_successes = 0
        self._turn_curriculum_updates = 0
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
        all_env_ids = torch.arange(n, device=device)
        self._reset_idx(all_env_ids)
        self._turn_fixture_robot_joint_pos = self.robot.data.joint_pos.clone()
        self.scene.write_data_to_sim()
        self._create_turn_fixture_joints()
        self._turn_fixture_joints_created = True

    @staticmethod
    def _validate_cfg(cfg: SimToolRealAllenKeyTurningEnvCfg) -> None:
        expected_table = (
            Path(__file__).resolve().parents[3]
            / "assets" / "urdf" / "table_allen_turning.urdf"
        ).resolve()
        configured_table = Path(cfg.assets.table_urdf).resolve()
        if configured_table != expected_table:
            raise ValueError(
                "Allen-key turning requires the physical turning table: "
                f"configured={configured_table}, expected={expected_table}"
            )
        if tuple(cfg.allen_turn_screw_axis_tool) != (0.0, 0.0, -1.0):
            raise ValueError("Allen-key turning currently requires canonical local -Z screw axis")
        total_rotation_deg = int(cfg.allen_turn_goal_count) * float(
            cfg.allen_turn_goal_increment_deg
        )
        if int(cfg.allen_turn_goal_count) <= 0 or float(
            cfg.allen_turn_goal_increment_deg
        ) <= 0.0:
            raise ValueError("Allen-key subgoal count and increment must be positive")
        if total_rotation_deg < 360.0:
            raise ValueError("Allen-key target must include at least one revolution")
        stage_count = len(cfg.allen_turn_friction_ranges_nm)
        if stage_count < 2:
            raise ValueError("Allen-key turning curriculum requires at least two stages")
        staged_fields = (
            cfg.allen_turn_handle_center_x_range_stages_m,
            cfg.allen_turn_handle_center_y_range_stages_m,
            cfg.allen_turn_easy_yaw_probability_stages,
            cfg.allen_turn_max_shoulder_to_handle_stages_m,
            cfg.allen_turn_max_initial_palm_handle_distance_stages_m,
            cfg.allen_turn_palm_support_distance_stages_m,
            cfg.allen_turn_damping_ranges_nm_per_radps,
            cfg.allen_turn_regularization_scale_stages,
        )
        if any(len(values) != stage_count for values in staged_fields):
            raise ValueError("Allen-key workspace curriculum length is inconsistent")
        if len(cfg.allen_turn_z_range_stages_m) != stage_count:
            raise ValueError("Allen-key Z curriculum length is inconsistent")
        friction_ranges = cfg.allen_turn_friction_ranges_nm
        for low, high in friction_ranges:
            if not (
                math.isfinite(low) and math.isfinite(high)
                and 0.0 <= low <= high
            ):
                raise ValueError("Allen-key friction ranges must be finite and non-negative")
        damping_ranges = cfg.allen_turn_damping_ranges_nm_per_radps
        for low, high in damping_ranges:
            if not (math.isfinite(low) and math.isfinite(high) and 0.0 <= low <= high):
                raise ValueError("Allen-key damping ranges must be finite and non-negative")
        if any(
            later[1] < earlier[1]
            for earlier, later in zip(friction_ranges, friction_ranges[1:])
        ):
            raise ValueError("Allen-key friction curriculum maxima must be non-decreasing")
        if any(
            later[1] < earlier[1]
            for earlier, later in zip(damping_ranges, damping_ranges[1:])
        ):
            raise ValueError("Allen-key damping curriculum maxima must be non-decreasing")
        for low, high in cfg.allen_turn_z_range_stages_m:
            if not (math.isfinite(low) and math.isfinite(high) and 0.0 < low < high):
                raise ValueError("Allen-key Z curriculum contains an invalid interval")
        for ranges in (
            cfg.allen_turn_handle_center_x_range_stages_m,
            cfg.allen_turn_handle_center_y_range_stages_m,
        ):
            for low, high in ranges:
                if not (math.isfinite(low) and math.isfinite(high) and low < high):
                    raise ValueError(
                        "Allen-key handle-center curriculum contains an invalid interval"
                    )
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in cfg.allen_turn_easy_yaw_probability_stages
        ):
            raise ValueError("Allen-key easy-yaw probabilities must lie in [0, 1]")
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in cfg.allen_turn_max_shoulder_to_handle_stages_m
        ):
            raise ValueError("Allen-key shoulder reach limits must be positive")
        initial_distance_limits = cfg.allen_turn_max_initial_palm_handle_distance_stages_m
        if any(not math.isfinite(value) or value <= 0.0 for value in initial_distance_limits):
            raise ValueError("Allen-key initial palm-handle limits must be positive")
        if any(
            later < earlier
            for earlier, later in zip(initial_distance_limits, initial_distance_limits[1:])
        ):
            raise ValueError("Allen-key initial palm-handle limits must be non-decreasing")
        if not cfg.assets.allen_key_lengths_m:
            raise ValueError("Allen-key turning requires a nonempty physical length pool")
        if float(cfg.allen_turn_initial_hand_clearance_m) < 0.0:
            raise ValueError("Allen-key initial hand clearance must be non-negative")
        if float(cfg.allen_turn_initial_arm_clearance_m) <= 0.0:
            raise ValueError("Allen-key initial arm clearance must be positive")
        if float(cfg.allen_turn_stiction_stiffness_nm_per_rad) <= 0.0:
            raise ValueError("Allen-key fixture stiction stiffness must be positive")
        if float(cfg.allen_turn_static_to_kinetic_friction_ratio) < 1.0:
            raise ValueError("Allen-key static friction must be at least kinetic friction")
        if float(cfg.allen_turn_max_fixture_torque_multiplier) < float(
            cfg.allen_turn_static_to_kinetic_friction_ratio
        ):
            raise ValueError(
                "Allen-key fixture torque cap must cover the static-friction limit"
            )
        if float(cfg.allen_turn_resistance_transition_speed_radps) <= 0.0:
            raise ValueError("Allen-key resistance transition speed must be positive")
        if float(cfg.allen_turn_friction_restick_speed_radps) <= 0.0:
            raise ValueError("Allen-key friction re-stick speed must be positive")
        if int(cfg.allen_turn_initial_sampling_max_attempts) <= 0:
            raise ValueError("Allen-key initial sampling attempts must be positive")
        if int(cfg.allen_turn_initial_robot_resampling_max_attempts) <= 0:
            raise ValueError("Allen-key robot reset sampling attempts must be positive")
        if int(cfg.allen_turn_loaded_grasp_minimum_contact_fingers) <= 0:
            raise ValueError("Allen-key loaded grasp requires at least one finger")
        if int(cfg.allen_turn_acquisition_hold_steps) <= 0:
            raise ValueError("Allen-key acquisition hold steps must be positive")
        if not 3 <= int(cfg.allen_turn_deep_grasp_minimum_contact_fingers) <= 5:
            raise ValueError("Allen-key deep grasp must require three to five fingers")
        if int(cfg.allen_turn_deep_grasp_hold_steps) <= 0:
            raise ValueError("Allen-key deep-grasp hold steps must be positive")
        if not -1.0 <= float(
            cfg.allen_turn_deep_grasp_maximum_opposition_cosine
        ) < 1.0:
            raise ValueError("Allen-key deep-grasp opposition cosine is invalid")
        if not 0.0 <= float(
            cfg.allen_turn_translational_force_soft_threshold_ratio
        ) < 1.0:
            raise ValueError("Allen-key force-imbalance threshold is invalid")
        arm_table_paths = tuple(cfg.allen_turn_arm_table_contact_prim_paths)
        if not bool(cfg.enable_arm_table_contact_sensor):
            raise ValueError("Allen-key turning requires arm-table contact sensors")
        if len(arm_table_paths) != 7 or len(set(arm_table_paths)) != 7:
            raise ValueError(
                "Allen-key arm-table contact paths must contain seven unique arm/wrist links"
            )
        if tuple(cfg.allen_turn_arm_table_contact_filter_paths) != (
            "/World/envs/env_.*/Table/box",
        ):
            raise ValueError("Allen-key arm-table filter must target the table body")
        if float(cfg.allen_turn_arm_table_contact_force_threshold_n) < 0.0:
            raise ValueError("Allen-key arm-table force threshold must be non-negative")
        if float(cfg.allen_turn_arm_table_contact_force_scale_n) <= 0.0:
            raise ValueError("Allen-key arm-table force scale must be positive")
        if float(cfg.allen_turn_table_height_m) <= 0.0:
            raise ValueError("Allen-key table height must be positive")
        finger_groups = tuple(cfg.allen_turn_finger_tool_contact_prim_paths)
        if len(finger_groups) != 5 or any(not group for group in finger_groups):
            raise ValueError(
                "Allen-key turning requires five nonempty finger contact-link groups"
            )
        contact_paths = [path for group in finger_groups for path in group]
        if len(contact_paths) != len(set(contact_paths)):
            raise ValueError("Allen-key finger contact-link paths must be unique")
        if bool(cfg.enable_fingertip_tool_contact_sensors):
            raise ValueError(
                "Allen-key turning uses all-link finger sensors; base fingertip sensors "
                "must be disabled"
            )
        if float(cfg.allen_turn_handle_approach_sigma_m) <= 0.0:
            raise ValueError("Allen-key handle approach sigma must be positive")
        if float(cfg.allen_turn_palm_support_soft_margin_m) <= 0.0:
            raise ValueError("Allen-key palm support soft margin must be positive")
        if any(
            not math.isfinite(float(value)) or float(value) <= 0.0
            for value in cfg.allen_turn_palm_support_distance_stages_m
        ):
            raise ValueError("Allen-key palm support distances must be finite and positive")
        nonnegative_reward_fields = (
            cfg.allen_turn_handle_approach_progress_weight,
            cfg.allen_turn_handle_proximity_penalty_weight,
            cfg.allen_turn_first_loaded_grasp_bonus,
            cfg.allen_turn_grasp_maintenance_reward_weight,
            cfg.allen_turn_deep_grasp_reward_weight,
            cfg.allen_turn_translational_force_penalty_weight,
            cfg.allen_turn_arm_table_contact_penalty_weight,
            cfg.allen_turn_subgoal_bonus,
            cfg.allen_turn_full_turn_bonus,
            cfg.allen_turn_effort_penalty_weight,
            cfg.allen_turn_action_rate_penalty_weight,
        )
        if any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in nonnegative_reward_fields
        ):
            raise ValueError("Allen-key grasp and effort reward weights must be finite and non-negative")
        regularization = tuple(
            float(value) for value in cfg.allen_turn_regularization_scale_stages
        )
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in regularization):
            raise ValueError("Allen-key regularization scales must lie in [0, 1]")
        if any(later < earlier for earlier, later in zip(regularization, regularization[1:])):
            raise ValueError("Allen-key regularization scales must be non-decreasing")

    def _setup_scene(self) -> None:
        super()._setup_scene()
        self._assert_table_collision_enabled()
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
        self._turn_finger_contact_sensors: list[list[ContactSensor]] = []
        filter_paths = list(self.cfg.palm_tool_contact_sensor_filter_paths)
        for finger_id, prim_paths in enumerate(
            self.cfg.allen_turn_finger_tool_contact_prim_paths
        ):
            finger_sensors: list[ContactSensor] = []
            for link_id, prim_path in enumerate(prim_paths):
                try:
                    sensor = ContactSensor(ContactSensorCfg(
                        prim_path=prim_path,
                        update_period=0.0,
                        history_length=0,
                        debug_vis=False,
                        track_pose=False,
                        track_contact_points=False,
                        track_friction_forces=True,
                        track_air_time=False,
                        filter_prim_paths_expr=filter_paths,
                        max_contact_data_count_per_prim=int(
                            self.cfg.fingertip_tool_contact_max_data_count
                        ),
                    ))
                except Exception as exc:
                    raise RuntimeError(
                        "Allen-key finger-tool ContactSensor creation failed for "
                        f"finger={finger_id}, link={link_id}, prim={prim_path!r}: {exc!r}"
                    ) from exc
                self.scene.sensors[
                    f"allen_turn_finger_{finger_id}_link_{link_id}_contact"
                ] = sensor
                finger_sensors.append(sensor)
            self._turn_finger_contact_sensors.append(finger_sensors)
        self._turn_arm_table_contact_sensors: list[ContactSensor] = []
        table_filters = list(self.cfg.allen_turn_arm_table_contact_filter_paths)
        for link_id, prim_path in enumerate(
            self.cfg.allen_turn_arm_table_contact_prim_paths
        ):
            try:
                sensor = ContactSensor(ContactSensorCfg(
                    prim_path=prim_path,
                    update_period=0.0,
                    history_length=0,
                    debug_vis=False,
                    track_pose=False,
                    track_contact_points=False,
                    track_friction_forces=False,
                    track_air_time=False,
                    filter_prim_paths_expr=table_filters,
                ))
            except Exception as exc:
                raise RuntimeError(
                    "Allen-key arm-table ContactSensor creation failed for "
                    f"link={link_id}, prim={prim_path!r}: {exc!r}"
                ) from exc
            self.scene.sensors[f"allen_turn_arm_{link_id}_table_contact"] = sensor
            self._turn_arm_table_contact_sensors.append(sensor)

    def _assert_table_collision_enabled(self) -> None:
        """Fail before simulation if the instantiated table has no active collider."""
        stage = get_current_stage()
        table_root = stage.GetPrimAtPath("/World/envs/env_0/Table")
        if not table_root.IsValid():
            raise RuntimeError("Allen-key table prim is missing in env_0")
        colliders = []
        for prim in Usd.PrimRange(table_root, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            if enabled is not False:
                colliders.append(str(prim.GetPath()))
        if not colliders:
            raise RuntimeError(
                "Allen-key table has no enabled collision prims after USD conversion"
            )
        self._turn_table_collider_paths = tuple(colliders)

    def _create_turn_fixture_joints(self) -> None:
        """Constrain every key to its socket with a physical screw-axis joint."""
        stage = get_current_stage()
        pivot_tool = tuple(float(value) for value in self.cfg.allen_turn_screw_pivot_tool_m)
        for env_id in range(self.cfg.scene.num_envs):
            root = f"/World/envs/env_{env_id}"
            workpiece_path = f"{root}/Workpiece/workpiece_root"
            tool_path = f"{root}/Object/object_root"
            for body_path in (workpiece_path, tool_path):
                if not stage.GetPrimAtPath(body_path).IsValid():
                    raise RuntimeError(
                        f"Allen-key fixture rigid body prim is missing: {body_path}"
                    )
            joint = UsdPhysics.RevoluteJoint.Define(
                stage, f"{root}/AllenKeyFixtureJoint"
            )
            joint.CreateExcludeFromArticulationAttr().Set(True)
            joint.CreateBody1Rel().SetTargets([Sdf.Path(tool_path)])
            # Body 0 is omitted intentionally, fixing this frame in world
            # space. The collision-disabled workpiece remains a visual socket;
            # tying the joint to its imported kinematic actor would retain the
            # actor's authored origin instead of its tensor-set pose.
            world_pivot = self._turn_pivot_w[env_id].tolist()
            yaw = float(self._turn_initial_yaw[env_id])
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*world_pivot))
            joint.CreateLocalRot0Attr().Set(Gf.Quatf(
                math.cos(0.5 * yaw), Gf.Vec3f(0.0, 0.0, math.sin(0.5 * yaw))
            ))
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(*pivot_tool))
            joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0)))
            joint.CreateAxisAttr().Set(UsdPhysics.Tokens.z)

    def _current_palm_center_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the physical palm mesh-center pose, not the merged wrist frame."""
        return palm_center_pose_from_merged_wrist(
            self.robot.data.body_link_pos_w[:, self._palm_body_id],
            self.robot.data.body_link_quat_w[:, self._palm_body_id],
        )

    def _current_handle_center_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the generated long-handle mesh-center pose."""
        handle_center_local = torch.zeros(self.num_envs, 3, device=self.device)
        handle_center_local[:, 0] = (
            float(self.cfg.assets.allen_key_elbow_x_m)
            - 0.5 * self._turn_handle_length
            + 0.25 * float(self.cfg.assets.allen_key_handle_across_flats_m)
        )
        handle_center = self.object.data.root_pos_w + quat_apply(
            self.object.data.root_quat_w, handle_center_local
        )
        return handle_center, self.object.data.root_quat_w

    def _sample_resistance_coefficients(self, env_ids: torch.Tensor) -> None:
        stage = self._turn_curriculum_stage
        friction_low, friction_high = self.cfg.allen_turn_friction_ranges_nm[stage]
        damping_low, damping_high = self.cfg.allen_turn_damping_ranges_nm_per_radps[stage]
        count = env_ids.numel()
        friction = torch.empty(count, device=self.device).uniform_(
            float(friction_low), float(friction_high)
        )
        damping = torch.empty(count, device=self.device).uniform_(
            float(damping_low), float(damping_high)
        )
        self._turn_coulomb_friction_nm[env_ids] = friction
        self._turn_damping_nm_per_radps[env_ids] = damping

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        super()._reset_idx(env_ids)
        if not getattr(self, "_turn_ready", False):
            return
        if self._turn_fixture_joints_created:
            if self._turn_fixture_robot_joint_pos is None:
                raise RuntimeError("Allen-key fixture reset state was not initialized")
            joint_pos = self._turn_fixture_robot_joint_pos[env_ids]
            joint_vel = torch.zeros_like(joint_pos)
            self.robot.write_joint_state_to_sim(
                joint_pos, joint_vel, env_ids=env_ids
            )
            self._cur_targets[env_ids] = joint_pos
            self._prev_targets[env_ids] = joint_pos
        count = env_ids.numel()
        stage = self._turn_curriculum_stage
        x_range = tuple(
            float(value)
            for value in self.cfg.allen_turn_handle_center_x_range_stages_m[stage]
        )
        y_range = tuple(
            float(value)
            for value in self.cfg.allen_turn_handle_center_y_range_stages_m[stage]
        )
        z_low, z_high = (
            float(value) for value in self.cfg.allen_turn_z_range_stages_m[stage]
        )
        xy = torch.zeros(count, 2, device=self.device)
        z = torch.zeros(count, device=self.device)
        yaw = torch.zeros(count, device=self.device)
        shoulder_reach = torch.zeros(count, device=self.device)
        palm_handle_distance = torch.zeros(count, device=self.device)
        robot_resample_count = torch.zeros(count, dtype=torch.long, device=self.device)
        if self._turn_fixture_joints_created:
            pivot = self._turn_pivot_w[env_ids].clone()
            yaw = self._turn_initial_yaw[env_ids].clone()
            local_pivot = pivot - self.scene.env_origins[env_ids]
            xy = local_pivot[:, :2]
            z = local_pivot[:, 2]
            # The paired robot state is restored above; retain the metrics
            # computed when that exact state and fixture pose were screened.
            shoulder_reach = self._turn_initial_shoulder_to_handle_m[env_ids].clone()
            palm_handle_distance = self._turn_initial_palm_handle_distance[env_ids].clone()
            pending = torch.empty(0, dtype=torch.long, device=self.device)
        else:
            pending = torch.arange(count, device=self.device)
        for robot_attempt in range(
            int(self.cfg.allen_turn_initial_robot_resampling_max_attempts)
        ):
            if pending.numel() == 0:
                break
            sampled = self._sample_clear_initial_pose(
                env_ids[pending],
                handle_center_x_range=x_range,
                handle_center_y_range=y_range,
                easy_yaw_probability=float(
                    self.cfg.allen_turn_easy_yaw_probability_stages[stage]
                ),
                maximum_shoulder_reach_m=float(
                    self.cfg.allen_turn_max_shoulder_to_handle_stages_m[stage]
                ),
                maximum_palm_handle_distance_m=float(
                    self.cfg.allen_turn_max_initial_palm_handle_distance_stages_m[stage]
                ),
                z_low=z_low,
                z_high=z_high,
            )
            sampled_xy, sampled_z, sampled_yaw, sampled_reach, sampled_distance, valid = sampled
            accepted = pending[valid]
            xy[accepted] = sampled_xy[valid]
            z[accepted] = sampled_z[valid]
            yaw[accepted] = sampled_yaw[valid]
            shoulder_reach[accepted] = sampled_reach[valid]
            palm_handle_distance[accepted] = sampled_distance[valid]
            pending = pending[~valid]
            if pending.numel() == 0:
                break
            if robot_attempt + 1 < int(
                self.cfg.allen_turn_initial_robot_resampling_max_attempts
            ):
                _randomize_robot_dof_state(self, env_ids[pending])
                robot_resample_count[pending] += 1
        if pending.numel():
            raise RuntimeError(
                "Allen-key reset could not sample an acquisition-reachable robot/tool "
                f"state for {pending.numel()}/{count} environments after "
                f"{self.cfg.allen_turn_initial_robot_resampling_max_attempts} robot "
                "reset attempts"
            )
        if self._turn_fixture_joints_created:
            # Standalone PhysX joints retain their instantiated kinematic
            # anchor. Keep each environment's initially randomized fixture
            # pose across resets instead of corrupting that anchor by
            # teleporting the socket.
            pivot = self._turn_pivot_w[env_ids].clone()
            yaw = self._turn_initial_yaw[env_ids].clone()
            local_pivot = pivot - self.scene.env_origins[env_ids]
            xy = local_pivot[:, :2]
            z = local_pivot[:, 2]
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
        table_position[:, 2] = (
            socket_position[:, 2]
            - 0.5 * float(self.cfg.allen_turn_table_height_m)
        )
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
        self._turn_phase[env_ids] = 1
        self._turn_direction[env_ids] = direction
        self._turn_initial_tool_pos[env_ids] = position
        self._turn_initial_tool_quat[env_ids] = quaternion
        self._turn_pivot_w[env_ids] = pivot
        self._turn_initial_yaw[env_ids] = yaw
        self._turn_previous_yaw[env_ids] = yaw
        self._turn_stiction_yaw[env_ids] = yaw
        self._turn_friction_stuck[env_ids] = True
        self._sample_resistance_coefficients(env_ids)
        self._turn_fixture_torque_nm[env_ids] = 0.0
        self._turn_fixture_torque_clipped[env_ids] = False
        self._turn_fixture_torque_peak_nm[env_ids] = 0.0
        self._turn_fixture_torque_clipped_steps[env_ids] = 0
        self._turn_cumulative_angle[env_ids] = 0.0
        self._turn_angle_delta[env_ids] = 0.0
        first_target = direction * math.radians(
            float(self.cfg.allen_turn_goal_increment_deg)
        )
        self._turn_target_angle[env_ids] = first_target
        self._turn_angle_error[env_ids] = 0.0
        self._turn_subgoal_index[env_ids] = 0
        self._turn_subgoal_hold[env_ids] = 0
        self._turn_final_hold[env_ids] = 0
        self._turn_just_subgoal[env_ids] = False
        self._turn_just_succeeded[env_ids] = False
        self._turn_success[env_ids] = False
        self._turn_previous_subgoal_potential[env_ids] = 0.0
        self._turn_subgoal_potential_progress[env_ids] = 0.0
        self._turn_contact_support[env_ids] = False
        self._turn_palm_supported[env_ids] = False
        self._turn_loaded_grasp[env_ids] = False
        self._turn_grasp_quality[env_ids] = 0.0
        self._turn_max_grasp_quality[env_ids] = 0.0
        self._turn_ever_loaded_grasp[env_ids] = False
        self._turn_just_loaded_grasp[env_ids] = False
        self._turn_grasp_hold_count[env_ids] = 0
        self._turn_deep_grasp[env_ids] = False
        self._turn_deep_grasp_confirmed[env_ids] = False
        self._turn_deep_grasp_hold_count[env_ids] = 0
        self._turn_deep_grasp_quality[env_ids] = 0.0
        self._turn_deep_grasp_opposition_cosine[env_ids] = 1.0
        self._turn_max_deep_grasp_quality[env_ids] = 0.0
        self._turn_contact_resultant_force_w[env_ids] = 0.0
        self._turn_contact_resultant_force_n[env_ids] = 0.0
        self._turn_summed_contact_load_n[env_ids] = 0.0
        self._turn_translational_force_imbalance[env_ids] = 0.0
        self._turn_translational_force_penalty[env_ids] = 0.0
        self._turn_arm_table_contact_force_n[env_ids] = 0.0
        self._turn_arm_table_contact[env_ids] = False
        self._turn_arm_table_contact_penalty[env_ids] = 0.0
        self._turn_palm_handle_distance[env_ids] = 0.0
        self._turn_min_palm_handle_distance[env_ids] = float("inf")
        self._turn_max_finger_contact_count[env_ids] = 0
        self._turn_ever_palm_contact[env_ids] = False
        self._turn_previous_approach_potential[env_ids] = 0.0
        self._turn_approach_initialized[env_ids] = False
        self._turn_approach_potential[env_ids] = 0.0
        self._turn_approach_progress[env_ids] = 0.0
        self._turn_initial_palm_handle_distance[env_ids] = palm_handle_distance
        self._turn_initial_robot_resample_count[env_ids] = robot_resample_count
        self._turn_qualified_progress[env_ids] = 0.0
        self._turn_unqualified_positive_progress[env_ids] = 0.0
        self._turn_final_hold_valid[env_ids] = False
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
        self._turn_initial_shoulder_to_handle_m[env_ids] = shoulder_reach
        self._turn_state_obs[env_ids] = 0.0
        self._turn_state_obs[env_ids, 1] = 1.0
        self._turn_grasp_obs[env_ids] = 0.0
        self._turn_effort_obs[env_ids] = 0.0
        # Reuse SimToolReal's lifted flag as "grasp acquired" for this engaged
        # tool. This keeps fingertip approach shaping active until a loaded
        # palm grasp is first established, then enables pose progress.
        self._lifted_object[env_ids] = False
        self._closest_keypoint_max_dist[env_ids] = -1.0
        self._write_turn_goal(env_ids, first_target)

    def _sample_clear_initial_pose(
        self,
        env_ids: torch.Tensor,
        *,
        handle_center_x_range: tuple[float, float],
        handle_center_y_range: tuple[float, float],
        easy_yaw_probability: float,
        maximum_shoulder_reach_m: float,
        maximum_palm_handle_distance_m: float,
        z_low: float,
        z_high: float,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Sample a collision-free pose inside the policy's acquisition workspace."""
        count = env_ids.numel()
        xy = torch.zeros(count, 2, device=self.device)
        z = torch.zeros(count, device=self.device)
        yaw = torch.zeros(count, device=self.device)
        shoulder_reach = torch.zeros(count, device=self.device)
        palm_handle_distance = torch.zeros(count, device=self.device)
        pending = torch.arange(count, device=self.device)
        # Joint-state writes invalidate Isaac Lab's body-pose cache. Reading it
        # here runs forward kinematics for the newly randomized reset pose, so
        # collision rejection is based on the actual robot rather than on the
        # construction-time default pose.
        palm_position, _ = palm_center_pose_from_merged_wrist(
            self.robot.data.body_link_pos_w[env_ids, self._palm_body_id],
            self.robot.data.body_link_quat_w[env_ids, self._palm_body_id],
        )
        hand_points = torch.cat((
            palm_position[:, None, :],
            self.robot.data.body_link_pos_w[env_ids][:, self._fingertip_body_ids],
        ), dim=1) - self.scene.env_origins[env_ids, None, :]
        arm_points = (
            self.robot.data.body_link_pos_w[env_ids][:, self._turn_arm_body_ids]
            - self.scene.env_origins[env_ids, None, :]
        )
        shoulder_points = (
            self.robot.data.body_link_pos_w[env_ids, self._turn_shoulder_body_id]
            - self.scene.env_origins[env_ids]
        )
        lengths = self._turn_handle_length[env_ids]
        clearance = (
            0.5 * float(self.cfg.assets.allen_key_handle_across_flats_m)
            + float(self.cfg.allen_turn_initial_hand_clearance_m)
        )
        for _ in range(int(self.cfg.allen_turn_initial_sampling_max_attempts)):
            if pending.numel() == 0:
                break
            sampled_handle_x = torch.empty(
                pending.numel(), device=self.device
            ).uniform_(*handle_center_x_range)
            sampled_handle_y = torch.empty(
                pending.numel(), device=self.device
            ).uniform_(*handle_center_y_range)
            sampled_handle_xy = torch.stack(
                (sampled_handle_x, sampled_handle_y), dim=-1
            )
            sampled_z = torch.empty(pending.numel(), device=self.device).uniform_(
                z_low, z_high
            )
            broad_yaw = torch.empty(pending.numel(), device=self.device).uniform_(
                -math.pi, math.pi
            )
            negative_easy_yaw = torch.empty(
                pending.numel(), device=self.device
            ).uniform_(-math.pi, -2.0 * math.pi / 3.0)
            positive_easy_yaw = torch.empty(
                pending.numel(), device=self.device
            ).uniform_(5.0 * math.pi / 6.0, math.pi)
            easy_yaw = torch.where(
                torch.rand(pending.numel(), device=self.device) < 0.5,
                negative_easy_yaw,
                positive_easy_yaw,
            )
            sampled_yaw = torch.where(
                torch.rand(pending.numel(), device=self.device)
                < easy_yaw_probability,
                easy_yaw,
                broad_yaw,
            )
            direction = torch.stack((
                torch.cos(sampled_yaw), torch.sin(sampled_yaw),
                torch.zeros_like(sampled_yaw),
            ), dim=-1)
            overlap = 0.25 * float(
                self.cfg.assets.allen_key_handle_across_flats_m
            )
            handle_offset = -0.5 * lengths[pending] + overlap
            sampled_xy = sampled_handle_xy - handle_offset[:, None] * direction[:, :2]
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
            sampled_shoulder_reach = minimum_segment_distance(
                shoulder_points[pending, None, :], near, far
            )
            sampled_handle_center = torch.cat(
                (sampled_handle_xy, (sampled_z + 0.03)[:, None]), dim=-1
            )
            sampled_palm_handle_distance = torch.linalg.vector_norm(
                palm_position[pending]
                - self.scene.env_origins[env_ids[pending]]
                - sampled_handle_center,
                dim=-1,
            )
            valid = (hand_distance >= clearance) & (
                arm_distance >= float(self.cfg.allen_turn_initial_arm_clearance_m)
            ) & (sampled_shoulder_reach <= maximum_shoulder_reach_m) & (
                sampled_palm_handle_distance <= maximum_palm_handle_distance_m
            )
            accepted = pending[valid]
            xy[accepted] = sampled_xy[valid]
            z[accepted] = sampled_z[valid]
            yaw[accepted] = sampled_yaw[valid]
            shoulder_reach[accepted] = sampled_shoulder_reach[valid]
            palm_handle_distance[accepted] = sampled_palm_handle_distance[valid]
            pending = pending[~valid]
        valid = torch.ones(count, dtype=torch.bool, device=self.device)
        valid[pending] = False
        return xy, z, yaw, shoulder_reach, palm_handle_distance, valid

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
        torque = torch.zeros(self.num_envs, 3, device=self.device)
        current_yaw = yaw_from_quaternion(self.object.data.root_quat_w)
        angular_velocity = self.object.data.root_ang_vel_w[:, 2]
        angular_displacement = wrap_to_pi(
            current_yaw - self._turn_stiction_yaw
        )
        maximum_torque = (
            float(self.cfg.allen_turn_max_fixture_torque_multiplier)
            * self._turn_coulomb_friction_nm
        )
        resistance, stuck, restuck, clipped = stick_slip_torsional_friction(
            angular_displacement,
            angular_velocity,
            self._turn_friction_stuck,
            kinetic_limit_nm=self._turn_coulomb_friction_nm,
            static_to_kinetic_ratio=float(
                self.cfg.allen_turn_static_to_kinetic_friction_ratio
            ),
            stiction_stiffness_nm_per_rad=float(
                self.cfg.allen_turn_stiction_stiffness_nm_per_rad
            ),
            damping_nm_per_radps=self._turn_damping_nm_per_radps,
            maximum_abs_torque_nm=maximum_torque,
            kinetic_transition_speed_radps=float(
                self.cfg.allen_turn_resistance_transition_speed_radps
            ),
            restick_speed_radps=float(
                self.cfg.allen_turn_friction_restick_speed_radps
            ),
        )
        self._turn_stiction_yaw.copy_(torch.where(
            restuck, current_yaw, self._turn_stiction_yaw
        ))
        self._turn_friction_stuck.copy_(stuck)
        self._turn_fixture_torque_nm.copy_(resistance)
        self._turn_fixture_torque_clipped.copy_(clipped)
        self._turn_fixture_torque_peak_nm.copy_(torch.maximum(
            self._turn_fixture_torque_peak_nm, resistance.abs()
        ))
        self._turn_fixture_torque_clipped_steps.add_(clipped.long())
        torque[:, 2] = resistance
        self.object.set_external_force_and_torque(
            torch.zeros(self.num_envs, 1, 3, device=self.device),
            torque[:, None, :],
            is_global=True,
        )

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._cur_targets)

    def _read_contacts(self) -> None:
        sensors = getattr(self, "_turn_finger_contact_sensors", None)
        if sensors is None or len(sensors) != 5 or any(not group for group in sensors):
            raise RuntimeError(
                "Allen-key turning requires five nonempty finger contact-sensor groups"
            )
        forces = []
        force_vectors = []
        proximal_contacts = []
        resultant_force_w = torch.zeros(self.num_envs, 3, device=self.device)
        summed_contact_load_n = torch.zeros(self.num_envs, device=self.device)
        for finger_id, finger_sensors in enumerate(sensors):
            finger_force = torch.zeros(self.num_envs, device=self.device)
            finger_force_w = torch.zeros(self.num_envs, 3, device=self.device)
            proximal_force = torch.zeros(self.num_envs, device=self.device)
            for link_id, sensor in enumerate(finger_sensors):
                matrix = getattr(
                    getattr(sensor, "data", None), "force_matrix_w", None
                )
                if (
                    matrix is None
                    or matrix.shape[0] != self.num_envs
                    or matrix.shape[-1] != 3
                ):
                    shape = None if matrix is None else tuple(matrix.shape)
                    raise RuntimeError(
                        "Allen-key finger contact sensor has invalid matrix: "
                        f"finger={finger_id}, link={link_id}, shape={shape}"
                    )
                pair_forces = matrix.reshape(self.num_envs, -1, 3)
                friction = getattr(
                    getattr(sensor, "data", None), "friction_forces_w", None
                )
                if friction is None or friction.shape != matrix.shape:
                    shape = None if friction is None else tuple(friction.shape)
                    raise RuntimeError(
                        "Allen-key finger friction sensor has invalid matrix: "
                        f"finger={finger_id}, link={link_id}, shape={shape}"
                    )
                pair_total_forces = pair_forces + friction.reshape(
                    self.num_envs, -1, 3
                )
                link_force = torch.linalg.vector_norm(pair_forces, dim=-1).sum(-1)
                link_force_w = pair_forces.sum(1)
                if not bool(torch.isfinite(link_force).all()) or not bool(
                    torch.isfinite(pair_total_forces).all()
                ):
                    raise RuntimeError(
                        "Allen-key finger contact sensor is non-finite: "
                        f"finger={finger_id}, link={link_id}"
                    )
                finger_force += link_force
                finger_force_w += link_force_w
                resultant_force_w += pair_total_forces.sum(1)
                summed_contact_load_n += torch.linalg.vector_norm(
                    pair_total_forces, dim=-1
                ).sum(-1)
                prim_path = self.cfg.allen_turn_finger_tool_contact_prim_paths[
                    finger_id
                ][link_id]
                link_name = prim_path.rsplit("/", 1)[-1]
                if link_name.endswith(("_MC", "_PP", "_MP")):
                    proximal_force += link_force
            forces.append(finger_force)
            force_vectors.append(finger_force_w)
            proximal_contacts.append(
                proximal_force
                >= float(self.cfg.allen_turn_contact_force_threshold_n)
            )
        self._turn_finger_force_n.copy_(torch.stack(forces, dim=-1))
        self._turn_finger_force_w.copy_(torch.stack(force_vectors, dim=1))
        self._turn_finger_contact.copy_(
            self._turn_finger_force_n
            >= float(self.cfg.allen_turn_contact_force_threshold_n)
        )
        self._turn_proximal_finger_contact.copy_(
            torch.stack(proximal_contacts, dim=-1)
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
        palm_friction = getattr(
            getattr(self._turn_palm_contact_sensor, "data", None),
            "friction_forces_w",
            None,
        )
        if palm_friction is None or palm_friction.shape != matrix.shape:
            shape = None if palm_friction is None else tuple(palm_friction.shape)
            raise RuntimeError(
                f"Allen-key palm friction sensor has invalid matrix {shape}"
            )
        palm_total_forces = matrix.reshape(
            self.num_envs, -1, 3
        ) + palm_friction.reshape(self.num_envs, -1, 3)
        if not bool(torch.isfinite(palm_total_forces).all()):
            raise RuntimeError("Allen-key palm friction sensor is non-finite")
        resultant_force_w += palm_total_forces.sum(1)
        summed_contact_load_n += torch.linalg.vector_norm(
            palm_total_forces, dim=-1
        ).sum(-1)
        force_penalty, imbalance = translational_force_imbalance_penalty(
            resultant_force_w,
            summed_contact_load_n,
            soft_threshold_ratio=float(
                self.cfg.allen_turn_translational_force_soft_threshold_ratio
            ),
        )
        self._turn_contact_resultant_force_w.copy_(resultant_force_w)
        self._turn_contact_resultant_force_n.copy_(
            torch.linalg.vector_norm(resultant_force_w, dim=-1)
        )
        self._turn_summed_contact_load_n.copy_(summed_contact_load_n)
        self._turn_translational_force_imbalance.copy_(imbalance)
        self._turn_translational_force_penalty.copy_(force_penalty)

    def _read_arm_table_contacts(self) -> None:
        sensors = getattr(self, "_turn_arm_table_contact_sensors", None)
        expected = len(self.cfg.allen_turn_arm_table_contact_prim_paths)
        if sensors is None or len(sensors) != expected:
            raise RuntimeError(
                "Allen-key turning arm-table contact sensors are unavailable"
            )
        forces = []
        for link_id, sensor in enumerate(sensors):
            matrix = getattr(getattr(sensor, "data", None), "force_matrix_w", None)
            if (
                matrix is None
                or matrix.shape[0] != self.num_envs
                or matrix.shape[-1] != 3
            ):
                shape = None if matrix is None else tuple(matrix.shape)
                raise RuntimeError(
                    "Allen-key arm-table contact sensor has invalid matrix: "
                    f"link={link_id}, shape={shape}"
                )
            force = torch.linalg.vector_norm(
                matrix.reshape(self.num_envs, -1, 3), dim=-1
            ).sum(-1)
            if not bool(torch.isfinite(force).all()):
                raise RuntimeError(
                    f"Allen-key arm-table contact sensor {link_id} is non-finite"
                )
            forces.append(force)
        force_matrix = torch.stack(forces, dim=-1)
        threshold = float(self.cfg.allen_turn_arm_table_contact_force_threshold_n)
        scale = float(self.cfg.allen_turn_arm_table_contact_force_scale_n)
        normalized_excess = ((force_matrix - threshold) / scale).clamp(0.0, 1.0)
        self._turn_arm_table_contact_force_n.copy_(force_matrix)
        self._turn_arm_table_contact.copy_(force_matrix >= threshold)
        self._turn_arm_table_contact_penalty.copy_(
            normalized_excess.square().amax(-1)
        )

    def _update_metrics(self) -> None:
        compute_intermediate_values(self)
        self._read_contacts()
        self._read_arm_table_contacts()
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

        palm_position, palm_quaternion = self._current_palm_center_pose_w()
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
        finger_contact_count = self._turn_finger_contact.sum(-1)
        contact_support = (
            finger_contact_count
            >= int(self.cfg.allen_turn_minimum_contact_fingers)
        ) | (self._turn_palm_contact & (finger_contact_count >= 1))
        self._turn_contact_support.copy_(contact_support)
        tool_position = self.object.data.root_pos_w
        tool_quaternion = self.object.data.root_quat_w
        handle_near_local = torch.zeros(self.num_envs, 3, device=self.device)
        handle_near_local[:, 0] = float(self.cfg.assets.allen_key_elbow_x_m)
        handle_far_local = handle_near_local.clone()
        handle_far_local[:, 0] -= self._turn_handle_length
        handle_near = tool_position + quat_apply(tool_quaternion, handle_near_local)
        handle_far = tool_position + quat_apply(tool_quaternion, handle_far_local)
        handle_segment = handle_far - handle_near
        segment_sq = handle_segment.square().sum(-1).clamp_min(1.0e-12)
        handle_center, _ = self._current_handle_center_pose_w()
        palm_distance = torch.linalg.vector_norm(
            palm_position - handle_center, dim=-1
        )
        fingertip_position = self.robot.data.body_link_pos_w[:, self._fingertip_body_ids]
        fingertip_fraction = (
            (
                (fingertip_position - handle_near[:, None, :])
                * handle_segment[:, None, :]
            ).sum(-1)
            / segment_sq[:, None]
        ).clamp(0.0, 1.0)
        closest_fingertip_handle_point = (
            handle_near[:, None, :]
            + fingertip_fraction[:, :, None] * handle_segment[:, None, :]
        )
        fingertip_handle_distance = torch.linalg.vector_norm(
            fingertip_position - closest_fingertip_handle_point, dim=-1
        )
        fingertip_handle_distance = (
            fingertip_handle_distance
            - 0.5 * float(self.cfg.assets.allen_key_handle_across_flats_m)
        ).clamp_min(0.0)
        self._turn_palm_handle_distance.copy_(palm_distance)
        self._turn_min_palm_handle_distance.copy_(torch.minimum(
            self._turn_min_palm_handle_distance, palm_distance
        ))
        self._turn_max_finger_contact_count.copy_(torch.maximum(
            self._turn_max_finger_contact_count, finger_contact_count
        ))
        self._turn_ever_palm_contact |= self._turn_palm_contact
        palm_support_distance = float(
            self.cfg.allen_turn_palm_support_distance_stages_m[
                self._turn_curriculum_stage
            ]
        )
        palm_support_margin = float(self.cfg.allen_turn_palm_support_soft_margin_m)
        palm_supported = self._turn_palm_contact | (
            palm_distance <= palm_support_distance
        )
        palm_support_quality = (
            (palm_support_distance + palm_support_margin - palm_distance)
            / palm_support_margin
        ).clamp(0.0, 1.0)
        palm_support_quality = torch.maximum(
            palm_support_quality, self._turn_palm_contact.float()
        )
        grasp_quality, loaded_grasp = loaded_grasp_quality(
            palm_support_quality,
            palm_supported,
            finger_contact_count,
            self._turn_relative_linear_speed,
            self._turn_relative_angular_speed,
            minimum_contact_fingers=int(
                self.cfg.allen_turn_loaded_grasp_minimum_contact_fingers
            ),
            maximum_relative_linear_speed_mps=float(
                self.cfg.allen_turn_max_relative_linear_speed_mps
            ),
            maximum_relative_angular_speed_radps=float(
                self.cfg.allen_turn_max_relative_angular_speed_radps
            ),
        )
        self._turn_palm_supported.copy_(palm_supported)
        self._turn_grasp_quality.copy_(grasp_quality)
        self._turn_max_grasp_quality.copy_(torch.maximum(
            self._turn_max_grasp_quality, grasp_quality
        ))
        self._turn_loaded_grasp.copy_(loaded_grasp)
        self._turn_stable_grasp.copy_(loaded_grasp)
        hold_count, just_confirmed = update_consecutive_grasp_hold(
            loaded_grasp,
            self._turn_grasp_hold_count,
            self._turn_ever_loaded_grasp,
            required_steps=int(self.cfg.allen_turn_acquisition_hold_steps),
        )
        self._turn_grasp_hold_count.copy_(hold_count)
        self._turn_just_loaded_grasp.copy_(just_confirmed)
        self._turn_ever_loaded_grasp |= just_confirmed
        deep_quality, deep_grasp, opposition_cosine = deep_grasp_quality(
            self._turn_finger_contact,
            self._turn_proximal_finger_contact,
            self._turn_palm_contact,
            self._turn_finger_force_w,
            self._turn_relative_linear_speed,
            self._turn_relative_angular_speed,
            minimum_contact_fingers=int(
                self.cfg.allen_turn_deep_grasp_minimum_contact_fingers
            ),
            maximum_opposition_cosine=float(
                self.cfg.allen_turn_deep_grasp_maximum_opposition_cosine
            ),
            maximum_relative_linear_speed_mps=float(
                self.cfg.allen_turn_max_relative_linear_speed_mps
            ),
            maximum_relative_angular_speed_radps=float(
                self.cfg.allen_turn_max_relative_angular_speed_radps
            ),
        )
        self._turn_deep_grasp.copy_(deep_grasp)
        self._turn_deep_grasp_quality.copy_(deep_quality)
        self._turn_deep_grasp_opposition_cosine.copy_(opposition_cosine)
        self._turn_max_deep_grasp_quality.copy_(torch.maximum(
            self._turn_max_deep_grasp_quality, deep_quality
        ))
        self._turn_deep_grasp_hold_count.copy_(torch.where(
            deep_grasp,
            self._turn_deep_grasp_hold_count + 1,
            torch.zeros_like(self._turn_deep_grasp_hold_count),
        ))
        self._turn_deep_grasp_confirmed.copy_(
            self._turn_deep_grasp_hold_count
            >= int(self.cfg.allen_turn_deep_grasp_hold_steps)
        )
        approach_sigma = float(self.cfg.allen_turn_handle_approach_sigma_m)
        palm_approach_potential = torch.exp(-palm_distance / approach_sigma)
        fingertip_approach_potential = torch.exp(
            -fingertip_handle_distance / approach_sigma
        ).mean(-1)
        # Equal palm/fingertip weighting keeps the target a whole-hand grasp;
        # one close fingertip alone cannot maximize the acquisition potential.
        approach_potential = 0.5 * (
            palm_approach_potential + fingertip_approach_potential
        )
        self._turn_approach_potential.copy_(approach_potential)
        approach_progress = approach_potential - self._turn_previous_approach_potential
        approach_progress = torch.where(
            self._turn_approach_initialized & ~self._turn_ever_loaded_grasp,
            approach_progress,
            torch.zeros_like(approach_progress),
        )
        self._turn_approach_progress.copy_(approach_progress)
        self._turn_previous_approach_potential.copy_(approach_potential)
        self._turn_approach_initialized.fill_(True)

        diagnostic_active = self._turn_relative_initialized
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
            contact_code += self._turn_finger_contact[:, finger_id].long() << (finger_id + 1)
        topology_changed = initialized & (
            contact_code != self._turn_previous_contact_code
        )
        self._turn_contact_topology_changes += topology_changed.long()
        contact_lost = (
            initialized & self._turn_previous_stable_grasp
            & ~self._turn_stable_grasp
        )
        contact_reacquired = (
            initialized & ~self._turn_previous_stable_grasp
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

        self._turn_just_subgoal.zero_()
        self._turn_just_succeeded.zero_()
        turning = self._turn_phase == 1
        self._turn_angle_error.copy_(self._turn_target_angle - self._turn_cumulative_angle)
        tolerance = math.radians(float(self.cfg.allen_turn_goal_tolerance_deg))
        at_subgoal = (
            turning
            & (self._turn_angle_error.abs() <= tolerance)
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
                self._closest_keypoint_max_dist[continuing_ids] = -1.0
                self._write_turn_goal(continuing_ids, target)
            if final_ids.numel():
                self._turn_phase[final_ids] = 2
                self._turn_final_hold[final_ids] = 0

        holding = self._turn_phase == 2
        total_rotation = math.radians(
            int(self.cfg.allen_turn_goal_count)
            * float(self.cfg.allen_turn_goal_increment_deg)
        )
        final_target = self._turn_direction * total_rotation
        final_valid = (
            holding
            & ((final_target - self._turn_cumulative_angle).abs() <= tolerance)
            & self._turn_deep_grasp_confirmed
            & self._turn_ever_loaded_grasp
        )
        self._turn_final_hold_valid.copy_(final_valid)
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
        maximum_friction = max(
            high for _, high in self.cfg.allen_turn_friction_ranges_nm
        )
        self._turn_state_obs[:, 7] = (
            self._turn_coulomb_friction_nm / max(maximum_friction, 1.0e-6)
        )
        self._turn_grasp_obs[:, 0] = self._turn_palm_contact.float()
        self._turn_grasp_obs[:, 1:6] = self._turn_finger_contact.float()
        self._turn_grasp_obs[:, 6] = self._turn_stable_grasp.float()
        self._turn_grasp_obs[:, 7] = self._turn_contact_support.float()
        self._turn_effort_obs[:, 0] = self._turn_effort_mean_ratio
        self._turn_effort_obs[:, 1] = self._turn_effort_max_ratio
        self._turn_effort_obs[:, 2] = self._turn_effort_saturation
        self._turn_effort_obs[:, 3] = self._turn_effort_penalty

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
        self._turn_curriculum_successes += int((done & self._turn_success).sum().item())
        minimum = int(self.cfg.allen_turn_curriculum_min_episodes)
        if self._turn_curriculum_completed < minimum:
            return
        success_rate = (
            self._turn_curriculum_successes / self._turn_curriculum_completed
        )
        ready = success_rate >= float(self.cfg.allen_turn_curriculum_success_threshold)
        self._turn_curriculum_conditional_success_rate = success_rate
        last = len(self.cfg.allen_turn_friction_ranges_nm) - 1
        if ready and self._turn_curriculum_stage < last:
            self._turn_curriculum_stage += 1
            self._turn_curriculum_updates += 1
        self._turn_curriculum_completed = 0
        self._turn_curriculum_successes = 0

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_metrics()
        success = self._turn_success
        timeout = self.episode_length_buf >= int(self.cfg.termination.episode_length)
        terminated = success
        truncated = timeout & ~terminated
        done = terminated | truncated
        self._is_success.copy_(success)
        self._termination_reasons = {
            "allen_turn_success": success,
            "timeout": truncated,
        }
        self._episode_final_terms = {
            "allen_turn_subgoals_completed": self._turn_subgoal_index.float(),
            "allen_turn_full_success": self._turn_success.float(),
            "allen_turn_ever_loaded_grasp": self._turn_ever_loaded_grasp.float(),
            "allen_turn_loaded_grasp_at_end": self._turn_loaded_grasp.float(),
            "allen_turn_max_grasp_quality": self._turn_max_grasp_quality,
            "allen_turn_deep_grasp_at_end": self._turn_deep_grasp.float(),
            "allen_turn_max_deep_grasp_quality": (
                self._turn_max_deep_grasp_quality
            ),
            "allen_turn_contact_resultant_force_n": (
                self._turn_contact_resultant_force_n
            ),
            "allen_turn_translational_force_imbalance": (
                self._turn_translational_force_imbalance
            ),
            "allen_turn_ever_palm_contact": self._turn_ever_palm_contact.float(),
            "allen_turn_min_palm_handle_distance_m": (
                self._turn_min_palm_handle_distance
            ),
            "allen_turn_max_finger_contact_count": (
                self._turn_max_finger_contact_count.float()
            ),
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
        self._lifted_object |= self._turn_ever_loaded_grasp
        base_reward = compute_rewards(self)
        base_terms = dict(self._reward_terms)
        base_terms.pop("total_reward", None)
        # Preserve SimToolReal's acquisition -> pose-tracking structure, but
        # replace terms whose geometry is invalid for a socket-engaged key.
        # The key cannot be lifted and fingertip-to-root distance points toward
        # the socket rather than toward the graspable handle.
        for name in ("fingertip_delta_rew", "lifting_rew", "lift_bonus_rew"):
            base_reward = base_reward - base_terms[name]
            base_terms[name] = torch.zeros_like(base_terms[name])
        self._lifted_object.copy_(self._turn_ever_loaded_grasp)

        # Keep the original SimToolReal pose objective. Positive progress and
        # goal bonuses require a currently maintained whole-hand grasp; pose
        # regression remains fully negative.
        for name in ("keypoint_rew", "bonus_rew"):
            raw_term = base_terms[name]
            qualified_term = gate_positive_progress(
                raw_term,
                self._turn_grasp_quality * self._turn_ever_loaded_grasp.float(),
            )
            base_reward = base_reward - raw_term + qualified_term
            base_terms[name] = qualified_term
        self._turn_qualified_progress.copy_(base_terms["keypoint_rew"])
        self._turn_unqualified_positive_progress.copy_(
            torch.relu(self._reward_terms["keypoint_rew"])
            * (1.0 - self._turn_grasp_quality)
        )

        regularization_scale = float(
            self.cfg.allen_turn_regularization_scale_stages[
                self._turn_curriculum_stage
            ]
        )
        for name in ("kuka_actions_penalty", "hand_actions_penalty"):
            raw_term = base_terms[name]
            scaled_term = regularization_scale * raw_term
            base_reward = base_reward - raw_term + scaled_term
            base_terms[name] = scaled_term

        execution_phase = self._turn_phase >= 1
        pregrasp = ~self._turn_ever_loaded_grasp
        handle_approach_reward = (
            float(self.cfg.allen_turn_handle_approach_progress_weight)
            * self._turn_approach_progress
            - float(self.cfg.allen_turn_handle_proximity_penalty_weight)
            * (1.0 - self._turn_approach_potential)
            * pregrasp.float()
        )
        weighted = {
            "handle_approach_rew": handle_approach_reward,
            "first_loaded_grasp_bonus": (
                float(self.cfg.allen_turn_first_loaded_grasp_bonus)
                * self._turn_just_loaded_grasp.float()
            ),
            "grasp_maintenance_rew": (
                float(self.cfg.allen_turn_grasp_maintenance_reward_weight)
                * self._turn_grasp_quality
                * execution_phase.float()
                * self._turn_ever_loaded_grasp.float()
            ),
            "deep_grasp_rew": (
                float(self.cfg.allen_turn_deep_grasp_reward_weight)
                * self._turn_deep_grasp_quality
                * execution_phase.float()
                * self._turn_ever_loaded_grasp.float()
            ),
            "subgoal_bonus": (
                float(self.cfg.allen_turn_subgoal_bonus)
                * self._turn_just_subgoal.float()
                * self._turn_deep_grasp_quality
            ),
            "full_turn_bonus": (
                float(self.cfg.allen_turn_full_turn_bonus)
                * self._turn_just_succeeded.float()
            ),
            "finger_effort_penalty": (
                -regularization_scale
                * float(self.cfg.allen_turn_effort_penalty_weight)
                * self._turn_effort_penalty
            ),
            "translational_force_penalty": (
                -float(self.cfg.allen_turn_translational_force_penalty_weight)
                * self._turn_translational_force_penalty
                * execution_phase.float()
                * self._turn_ever_loaded_grasp.float()
            ),
            "arm_table_contact_penalty": (
                -float(self.cfg.allen_turn_arm_table_contact_penalty_weight)
                * self._turn_arm_table_contact_penalty
            ),
            "action_rate_penalty": (
                -regularization_scale
                * float(self.cfg.allen_turn_action_rate_penalty_weight)
                * self._turn_action_delta_sq
            ),
        }
        reward = base_reward + torch.stack(tuple(weighted.values())).sum(0)
        if not bool(torch.isfinite(reward).all()):
            raise RuntimeError("Allen-key turning reward contains NaN or Inf")
        self._reward_terms = {**base_terms, **weighted, "total_reward": reward}
        self._episode_cumulative_terms = {}
        for name, mask in (
            ("turning", self._turn_phase == 1),
            ("final_hold", self._turn_phase == 2),
        ):
            self._episode_cumulative_terms[f"phase/{name}_reward_sum"] = reward * mask.float()
            self._episode_cumulative_terms[f"phase/{name}_step_count"] = mask.float()
        self.extras.update({
            "allen_turn/stable_grasp_ratio": self._turn_stable_grasp.float().mean(),
            "allen_turn/loaded_grasp_ratio": self._turn_loaded_grasp.float().mean(),
            "allen_turn/grasp_quality_mean": self._turn_grasp_quality.mean(),
            "allen_turn/deep_grasp_ratio": self._turn_deep_grasp.float().mean(),
            "allen_turn/deep_grasp_confirmed_ratio": (
                self._turn_deep_grasp_confirmed.float().mean()
            ),
            "allen_turn/deep_grasp_quality_mean": (
                self._turn_deep_grasp_quality.mean()
            ),
            "allen_turn/deep_grasp_hold_steps_mean": (
                self._turn_deep_grasp_hold_count.float().mean()
            ),
            "allen_turn/deep_grasp_opposition_cosine_mean": (
                self._turn_deep_grasp_opposition_cosine.mean()
            ),
            "allen_turn/proximal_finger_contact_count_mean": (
                self._turn_proximal_finger_contact.float().sum(-1).mean()
            ),
            "allen_turn/contact_resultant_force_mean_n": (
                self._turn_contact_resultant_force_n.mean()
            ),
            "allen_turn/contact_resultant_force_max_n": (
                self._turn_contact_resultant_force_n.max()
            ),
            "allen_turn/summed_contact_load_mean_n": (
                self._turn_summed_contact_load_n.mean()
            ),
            "allen_turn/translational_force_imbalance_mean": (
                self._turn_translational_force_imbalance.mean()
            ),
            "allen_turn/arm_table_contact_ratio": (
                self._turn_arm_table_contact.any(-1).float().mean()
            ),
            "allen_turn/arm_table_contact_link_count_mean": (
                self._turn_arm_table_contact.float().sum(-1).mean()
            ),
            "allen_turn/arm_table_contact_force_mean_n": (
                self._turn_arm_table_contact_force_n.mean()
            ),
            "allen_turn/arm_table_contact_force_max_n": (
                self._turn_arm_table_contact_force_n.max()
            ),
            "allen_turn/ever_loaded_grasp_ratio": (
                self._turn_ever_loaded_grasp.float().mean()
            ),
            "allen_turn/first_loaded_grasp_ratio": (
                self._turn_just_loaded_grasp.float().mean()
            ),
            "allen_turn/grasp_hold_steps_mean": (
                self._turn_grasp_hold_count.float().mean()
            ),
            "allen_turn/contact_support_ratio": self._turn_contact_support.float().mean(),
            "allen_turn/subgoals_completed_mean": self._turn_subgoal_index.float().mean(),
            "allen_turn/full_turn_success_ratio": self._turn_success.float().mean(),
            "allen_turn/signed_progress_mean_deg": torch.rad2deg(
                self._turn_direction * self._turn_cumulative_angle
            ).mean(),
            "allen_turn/angle_error_mean_deg": torch.rad2deg(
                self._turn_angle_error.abs()
            ).mean(),
            "allen_turn/palm_contact_ratio": self._turn_palm_contact.float().mean(),
            "allen_turn/palm_supported_ratio": self._turn_palm_supported.float().mean(),
            "allen_turn/palm_handle_distance_mean_m": (
                self._turn_palm_handle_distance.mean()
            ),
            "allen_turn/initial_palm_handle_distance_mean_m": (
                self._turn_initial_palm_handle_distance.mean()
            ),
            "allen_turn/initial_palm_handle_distance_max_m": (
                self._turn_initial_palm_handle_distance.max()
            ),
            "allen_turn/initial_palm_handle_distance_limit_m": torch.tensor(
                float(self.cfg.allen_turn_max_initial_palm_handle_distance_stages_m[
                    self._turn_curriculum_stage
                ]),
                device=self.device,
            ),
            "allen_turn/initial_robot_resample_count_mean": (
                self._turn_initial_robot_resample_count.float().mean()
            ),
            "allen_turn/approach_potential_mean": self._turn_approach_potential.mean(),
            "allen_turn/qualified_progress_mean": (
                self._turn_qualified_progress.mean()
            ),
            "allen_turn/unqualified_positive_progress_mean": (
                self._turn_unqualified_positive_progress.mean()
            ),
            "allen_turn/finger_contact_count_mean": (
                self._turn_finger_contact.float().sum(-1).mean()
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
            "allen_turn/initial_shoulder_to_handle_mean_m": (
                self._turn_initial_shoulder_to_handle_m.mean()
            ),
            "allen_turn/initial_shoulder_to_handle_max_m": (
                self._turn_initial_shoulder_to_handle_m.max()
            ),
            "allen_turn/fixture_torque_abs_mean_nm": (
                self._turn_fixture_torque_nm.abs().mean()
            ),
            "allen_turn/fixture_torque_abs_max_nm": (
                self._turn_fixture_torque_nm.abs().max()
            ),
            "allen_turn/fixture_torque_episode_peak_mean_nm": (
                self._turn_fixture_torque_peak_nm.mean()
            ),
            "allen_turn/fixture_torque_episode_peak_max_nm": (
                self._turn_fixture_torque_peak_nm.max()
            ),
            "allen_turn/fixture_torque_clip_ratio": (
                self._turn_fixture_torque_clipped.float().mean()
            ),
            "allen_turn/fixture_torque_clipped_steps_mean": (
                self._turn_fixture_torque_clipped_steps.float().mean()
            ),
            "allen_turn/fixture_torque_limit_mean_nm": (
                float(self.cfg.allen_turn_max_fixture_torque_multiplier)
                * self._turn_coulomb_friction_nm.mean()
            ),
            "allen_turn/fixture_stuck_ratio": self._turn_friction_stuck.float().mean(),
            "allen_turn/coulomb_friction_mean_nm": (
                self._turn_coulomb_friction_nm.mean()
            ),
            "allen_turn/coulomb_friction_min_nm": self._turn_coulomb_friction_nm.min(),
            "allen_turn/coulomb_friction_max_nm": self._turn_coulomb_friction_nm.max(),
            "allen_turn/damping_mean_nm_per_radps": (
                self._turn_damping_nm_per_radps.mean()
            ),
            "allen_turn/damping_min_nm_per_radps": (
                self._turn_damping_nm_per_radps.min()
            ),
            "allen_turn/damping_max_nm_per_radps": (
                self._turn_damping_nm_per_radps.max()
            ),
            "curriculum/allen_turn_stage": self._turn_curriculum_stage,
            "curriculum/allen_turn_palm_support_distance_m": (
                self.cfg.allen_turn_palm_support_distance_stages_m[
                    self._turn_curriculum_stage
                ]
            ),
            "curriculum/allen_turn_regularization_scale": (
                self.cfg.allen_turn_regularization_scale_stages[
                    self._turn_curriculum_stage
                ]
            ),
            "curriculum/allen_turn_friction_low_nm": (
                self.cfg.allen_turn_friction_ranges_nm[
                    self._turn_curriculum_stage
                ][0]
            ),
            "curriculum/allen_turn_friction_high_nm": (
                self.cfg.allen_turn_friction_ranges_nm[
                    self._turn_curriculum_stage
                ][1]
            ),
            "curriculum/allen_turn_damping_low_nm_per_radps": (
                self.cfg.allen_turn_damping_ranges_nm_per_radps[
                    self._turn_curriculum_stage
                ][0]
            ),
            "curriculum/allen_turn_damping_high_nm_per_radps": (
                self.cfg.allen_turn_damping_ranges_nm_per_radps[
                    self._turn_curriculum_stage
                ][1]
            ),
            "curriculum/allen_turn_fixture_torque_multiplier": (
                self.cfg.allen_turn_max_fixture_torque_multiplier
            ),
            "curriculum/allen_turn_target_rotation_deg": (
                int(self.cfg.allen_turn_goal_count)
                * float(self.cfg.allen_turn_goal_increment_deg)
            ),
            "curriculum/allen_turn_success_rate": (
                self._turn_curriculum_conditional_success_rate
            ),
            "curriculum/allen_turn_updates": self._turn_curriculum_updates,
        })
        log_step_metrics(self)
        return reward


__all__ = ["SimToolRealAllenKeyTurningEnv"]
