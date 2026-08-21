"""Extrinsic grasp adjustment toward sampled palm-to-tool relationships."""

from __future__ import annotations

import math

import torch
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply,
    quat_from_angle_axis,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
)

from .simtoolreal_inhand_stable_scrape_env import SimToolRealInHandStableScrapeEnv
from .simtoolreal_tacmap_env_cfg import SimToolRealInHandAdjustmentEnvCfg
from .utils.inhand_grasp_bank import (
    collision_box_corners,
    table_root_z_for_lowest_clearance,
)
from .utils.logging_utils import log_step_metrics
from .utils.obs_utils import compute_intermediate_values
from .utils.scrape_pose_utils import TABLE_HALF_HEIGHT, quat_apply_wxyz
from .utils.stable_scrape_utils import consecutive_counter, quaternion_distance_rad

class SimToolRealInHandAdjustmentEnv(SimToolRealInHandStableScrapeEnv):
    """Change the palm-to-tool relationship while returning the tool to its pose."""

    cfg: SimToolRealInHandAdjustmentEnvCfg

    def __init__(self, cfg: SimToolRealInHandAdjustmentEnvCfg, render_mode=None, **kwargs):
        self._validate_adjustment_cfg(cfg)
        self._adjustment_ready = False
        super().__init__(cfg, render_mode, **kwargs)
        n, device = self.num_envs, self.device
        self._adjustment_target_relative_pos = torch.zeros(n, 3, device=device)
        self._adjustment_target_relative_quat = torch.zeros(n, 4, device=device)
        self._adjustment_target_relative_quat[:, 0] = 1.0
        self._adjustment_target_axial_translation = torch.zeros(n, device=device)
        self._adjustment_target_perpendicular_translation = torch.zeros(n, device=device)
        self._adjustment_target_error_obs = torch.zeros(n, 6, device=device)
        self._adjustment_target_palm_error_obs = torch.zeros(n, 6, device=device)
        self._adjustment_target_palm_pos_w = torch.zeros(n, 3, device=device)
        self._adjustment_target_palm_quat_w = torch.zeros(n, 4, device=device)
        self._adjustment_target_palm_quat_w[:, 0] = 1.0
        self._adjustment_target_palm_position_error = torch.zeros(n, device=device)
        self._adjustment_target_palm_rotation_error = torch.zeros(n, device=device)
        self._adjustment_pose_error_obs = torch.zeros(n, 2, device=device)
        self._adjustment_phase_obs = torch.zeros(n, 1, device=device)
        self._adjustment_table_obs = torch.zeros(n, 4, device=device)
        self._adjustment_initial_tool_pos = torch.zeros(n, 3, device=device)
        self._adjustment_initial_tool_quat = torch.zeros(n, 4, device=device)
        self._adjustment_initial_tool_quat[:, 0] = 1.0
        self._adjustment_relative_position_error = torch.zeros(n, device=device)
        self._adjustment_relative_rotation_error = torch.zeros(n, device=device)
        self._adjustment_tool_position_error = torch.zeros(n, device=device)
        self._adjustment_tool_rotation_error = torch.zeros(n, device=device)
        self._adjustment_tool_pose_weight_scale = torch.zeros(n, device=device)
        self._adjustment_success_count = torch.zeros(n, dtype=torch.long, device=device)
        self._adjustment_pose_failure_count = torch.zeros(n, dtype=torch.long, device=device)
        self._adjustment_just_succeeded = torch.zeros(n, dtype=torch.bool, device=device)
        self._adjustment_succeeded = torch.zeros(n, dtype=torch.bool, device=device)
        self._adjustment_relative_position_valid = torch.zeros(n, dtype=torch.bool, device=device)
        self._adjustment_relative_rotation_valid = torch.zeros(n, dtype=torch.bool, device=device)
        self._adjustment_tool_position_valid = torch.zeros(n, dtype=torch.bool, device=device)
        self._adjustment_tool_rotation_valid = torch.zeros(n, dtype=torch.bool, device=device)
        self._adjustment_support_valid = torch.zeros(n, dtype=torch.bool, device=device)
        self._adjustment_combined_valid = torch.zeros(n, dtype=torch.bool, device=device)
        self._adjustment_curriculum_stage = 0
        self._adjustment_curriculum_updates = 0
        self._adjustment_curriculum_eligible = 0
        self._adjustment_curriculum_successes = 0
        self._adjustment_curriculum_success_mean = 0.0
        self._adjustment_curriculum_last_window_size = 0
        self._adjustment_curriculum_evaluations = 0
        self._adjustment_ready = True
        self._reset_idx(torch.arange(n, device=device))

    @staticmethod
    def _validate_adjustment_cfg(cfg: SimToolRealInHandAdjustmentEnvCfg) -> None:
        positive = (
            "adjustment_relative_position_tolerance_m",
            "adjustment_relative_rotation_tolerance_deg",
            "adjustment_tool_position_tolerance_m", "adjustment_tool_rotation_tolerance_deg",
            "adjustment_tool_position_hard_limit_m", "adjustment_tool_rotation_hard_limit_deg",
            "adjustment_relative_position_reward_sigma_m",
            "adjustment_relative_rotation_reward_sigma_deg",
        )
        for name in positive:
            if not math.isfinite(float(getattr(cfg, name))) or float(getattr(cfg, name)) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if int(cfg.adjustment_success_steps) <= 0:
            raise ValueError("adjustment_success_steps must be positive")
        if int(cfg.adjustment_pose_failure_steps) <= 0:
            raise ValueError("adjustment_pose_failure_steps must be positive")
        if int(cfg.adjustment_min_fingertip_support) <= 0:
            raise ValueError("adjustment_min_fingertip_support must be positive")
        clearance_low, clearance_high = cfg.adjustment_table_clearance_range_m
        if float(clearance_low) < 0.0 or float(clearance_high) < float(clearance_low):
            raise ValueError("adjustment table clearance range must be non-negative and ordered")
        stage_fields = (
            cfg.adjustment_finger_perturb_fractions,
            cfg.adjustment_target_axial_translation_m,
            cfg.adjustment_target_perpendicular_translation_m,
            cfg.adjustment_target_rotation_deg,
            cfg.adjustment_relative_position_tolerance_stages_m,
            cfg.adjustment_relative_rotation_tolerance_stages_deg,
            cfg.adjustment_tool_pose_delay_steps,
            cfg.adjustment_tool_pose_ramp_steps,
        )
        if len({len(values) for values in stage_fields}) != 1 or not stage_fields[0]:
            raise ValueError("adjustment curriculum fields must have equal nonzero lengths")
        translation_mode = str(cfg.adjustment_target_translation_mode)
        if translation_mode not in {"anisotropic", "none"}:
            raise ValueError(
                "adjustment_target_translation_mode must be 'anisotropic' or 'none'"
            )
        if translation_mode == "anisotropic":
            if any(float(value) <= 0.0 for value in cfg.adjustment_target_axial_translation_m):
                raise ValueError("adjustment target axial translation stages must be positive")
            if any(
                float(value) <= 0.0
                for value in cfg.adjustment_target_perpendicular_translation_m
            ):
                raise ValueError("adjustment target perpendicular stages must be positive")
            if any(
                float(perpendicular) >= float(axial)
                for axial, perpendicular in zip(
                    cfg.adjustment_target_axial_translation_m,
                    cfg.adjustment_target_perpendicular_translation_m,
                )
            ):
                raise ValueError("perpendicular translation limits must be below axial limits")
        elif any(
            float(value) != 0.0
            for values in (
                cfg.adjustment_target_axial_translation_m,
                cfg.adjustment_target_perpendicular_translation_m,
            )
            for value in values
        ):
            raise ValueError("translation limits must be zero when translation mode is 'none'")
        if str(cfg.adjustment_target_rotation_axis) not in {"random", "tool_x"}:
            raise ValueError(
                "adjustment_target_rotation_axis must be 'random' or 'tool_x'"
            )
        for name in (
            "adjustment_target_axial_fraction_of_tool_length",
            "adjustment_target_perpendicular_fraction_of_tool_thickness",
            "adjustment_target_total_translation_max_m",
        ):
            if not math.isfinite(float(getattr(cfg, name))) or float(getattr(cfg, name)) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if any(float(value) <= 0.0 for value in cfg.adjustment_target_rotation_deg):
            raise ValueError("adjustment target rotation stages must be positive")
        if any(
            float(value) <= 0.0
            for value in cfg.adjustment_relative_position_tolerance_stages_m
        ):
            raise ValueError("adjustment position tolerance stages must be positive")
        if any(
            float(value) <= 0.0
            for value in cfg.adjustment_relative_rotation_tolerance_stages_deg
        ):
            raise ValueError("adjustment rotation tolerance stages must be positive")
        if any(
            later > earlier
            for earlier, later in zip(
                cfg.adjustment_relative_position_tolerance_stages_m,
                cfg.adjustment_relative_position_tolerance_stages_m[1:],
            )
        ):
            raise ValueError("adjustment position tolerances must tighten by stage")
        if any(
            later > earlier
            for earlier, later in zip(
                cfg.adjustment_relative_rotation_tolerance_stages_deg,
                cfg.adjustment_relative_rotation_tolerance_stages_deg[1:],
            )
        ):
            raise ValueError("adjustment rotation tolerances must tighten by stage")
        if any(int(value) < 0 for value in cfg.adjustment_tool_pose_delay_steps):
            raise ValueError("adjustment tool-pose delay stages must be non-negative")
        if any(int(value) <= 0 for value in cfg.adjustment_tool_pose_ramp_steps):
            raise ValueError("adjustment tool-pose ramp stages must be positive")
        if not 0.0 < float(cfg.adjustment_curriculum_success_threshold) <= 1.0:
            raise ValueError("adjustment curriculum success threshold must be in (0, 1]")
        if int(cfg.adjustment_curriculum_min_eligible_count) <= 0:
            raise ValueError("adjustment curriculum eligible count must be positive")

    def _sample_target_relationship(
        self, env_ids: torch.Tensor, relative_pos: torch.Tensor,
        relative_quat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        count = relative_pos.shape[0]
        if str(self.cfg.adjustment_target_translation_mode) == "none":
            delta_tool = torch.zeros(count, 3, device=self.device)
        else:
            bounds = self._scrape_collision_bounds_per_env[env_ids]
            tool_length = bounds[:, 3] - bounds[:, 0]
            tool_thickness = torch.minimum(
                bounds[:, 4] - bounds[:, 1], bounds[:, 5] - bounds[:, 2]
            )
            axial_limit = torch.minimum(
                torch.full((count,), float(
                    self.cfg.adjustment_target_axial_translation_m[
                        self._adjustment_curriculum_stage
                    ]
                ), device=self.device),
                tool_length * float(self.cfg.adjustment_target_axial_fraction_of_tool_length),
            )
            perpendicular_limit = torch.minimum(
                torch.full((count,), float(
                    self.cfg.adjustment_target_perpendicular_translation_m[
                        self._adjustment_curriculum_stage
                    ]
                ), device=self.device),
                tool_thickness * float(
                    self.cfg.adjustment_target_perpendicular_fraction_of_tool_thickness
                ),
            )
            axial = torch.empty(count, device=self.device).uniform_(0.5, 1.0) * axial_limit
            axial *= torch.where(
                torch.rand(count, device=self.device) < 0.5, -1.0, 1.0
            )
            perpendicular_direction = torch.nn.functional.normalize(
                torch.randn(count, 2, device=self.device), dim=-1
            )
            perpendicular = perpendicular_direction * (
                torch.rand(count, 1, device=self.device) * perpendicular_limit.unsqueeze(-1)
            )
            delta_tool = torch.cat((axial.unsqueeze(-1), perpendicular), dim=-1)
            total_limit = float(self.cfg.adjustment_target_total_translation_max_m)
            delta_norm = torch.linalg.vector_norm(delta_tool, dim=-1, keepdim=True)
            delta_tool *= torch.clamp(total_limit / delta_norm.clamp_min(1.0e-8), max=1.0)
        target_pos = relative_pos + quat_apply(relative_quat, delta_tool)

        rotation_limit = math.radians(float(
            self.cfg.adjustment_target_rotation_deg[self._adjustment_curriculum_stage]
        ))
        if str(self.cfg.adjustment_target_rotation_axis) == "tool_x":
            axis = torch.zeros(count, 3, device=self.device)
            axis[:, 0] = 1.0
        else:
            axis = torch.nn.functional.normalize(
                torch.randn(count, 3, device=self.device), dim=-1
            )
        angle = torch.empty(count, device=self.device).uniform_(
            0.5 * rotation_limit, rotation_limit
        )
        angle *= torch.where(
            torch.rand(count, device=self.device) < 0.5, -1.0, 1.0
        )
        # Right composition makes both perturbations tool-frame quantities:
        # T_palm_tool_target = T_palm_tool_initial * T_tool_delta.
        target_quat = quat_mul(relative_quat, quat_from_angle_axis(angle, axis))
        return (
            target_pos,
            target_quat,
            delta_tool[:, 0].abs(),
            torch.linalg.vector_norm(delta_tool[:, 1:], dim=-1),
        )

    def _restore_inhand_state(self, env_ids: torch.Tensor, bank_ids=None) -> None:
        if not getattr(self, "_adjustment_ready", False):
            super()._restore_inhand_state(env_ids, bank_ids)
            return
        count = env_ids.numel()
        asset_ids = self._object_asset_index_per_env[env_ids]
        if bank_ids is None:
            starts = self._inhand_bank_asset_starts[asset_ids]
            counts_per_env = self._inhand_bank_asset_counts[asset_ids]
            source_ids = starts + torch.floor(
                torch.rand(count, device=self.device) * counts_per_env
            ).long()
        else:
            source_ids = torch.as_tensor(bank_ids, device=self.device, dtype=torch.long)
            if source_ids.shape != (count,):
                raise ValueError(
                    f"bank_ids must have shape ({count},), got {tuple(source_ids.shape)}"
                )
            if bool(((source_ids < 0) | (source_ids >= self._inhand_bank_size)).any()):
                raise ValueError("bank_ids contains an out-of-range grasp-bank index")
            if bool((self._inhand_bank_asset_index[source_ids] != asset_ids).any()):
                raise RuntimeError("grasp-bank entries do not match environment tool assets")
        super()._restore_inhand_state(env_ids, source_ids)

        joint_pos = self._inhand_bank_joint_pos[source_ids][:, self._perm_canon_to_lab]
        targets = self._inhand_bank_joint_targets[source_ids][:, self._perm_canon_to_lab]
        fraction = float(
            self.cfg.adjustment_finger_perturb_fractions[self._adjustment_curriculum_stage]
        )
        if fraction > 0.0:
            finger = self._hand_joint_ids
            lower = self._hand_lower[env_ids]
            upper = self._hand_upper[env_ids]
            random_finger = lower + torch.rand_like(lower) * (upper - lower)
            joint_pos[:, finger] += fraction * (random_finger - joint_pos[:, finger])
            targets[:, finger] = joint_pos[:, finger]
        zeros = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, zeros, env_ids=env_ids)
        self._cur_targets[env_ids] = targets
        self._prev_targets[env_ids] = targets
        self.robot.set_joint_position_target(targets, env_ids=env_ids)

        origins = self.scene.env_origins[env_ids]
        object_pos = self._inhand_bank_object_pos[source_ids] + origins
        object_quat = self._inhand_bank_object_quat[source_ids]
        source_relative_pos = self._inhand_bank_relative_pos[source_ids]
        source_relative_quat = self._inhand_bank_relative_quat[source_ids]
        relative_pos = source_relative_pos.clone()
        relative_quat = source_relative_quat.clone()
        (
            target_relative_pos,
            target_relative_quat,
            target_axial_translation,
            target_perpendicular_translation,
        ) = self._sample_target_relationship(env_ids, relative_pos, relative_quat)
        self.object.write_root_pose_to_sim(
            torch.cat((object_pos, object_quat), dim=-1), env_ids=env_ids
        )
        self.object.write_root_velocity_to_sim(
            torch.zeros(count, 6, device=self.device), env_ids=env_ids
        )

        table_quat = torch.zeros(count, 4, device=self.device)
        table_quat[:, 0] = 1.0
        normal = torch.zeros(count, 3, device=self.device)
        normal[:, 2] = 1.0
        corners = collision_box_corners(self._scrape_collision_bounds_per_env[env_ids])
        corners_w = object_pos.unsqueeze(1) + quat_apply_wxyz(
            object_quat.unsqueeze(1).expand(-1, 8, -1).reshape(-1, 4),
            corners.reshape(-1, 3),
        ).reshape(count, 8, 3)
        clearance_low, clearance_high = self.cfg.adjustment_table_clearance_range_m
        clearance = torch.empty(count, device=self.device).uniform_(
            float(clearance_low), float(clearance_high)
        )
        table_z = table_root_z_for_lowest_clearance(
            corners_w, normal, origins, clearance, table_half_height_m=TABLE_HALF_HEIGHT
        )
        table_pos = origins.clone()
        table_pos[:, 2] += table_z
        self.table.write_root_pose_to_sim(
            torch.cat((table_pos, table_quat), dim=-1), env_ids=env_ids
        )
        self._table_z_per_env[env_ids] = table_z
        self._table_quat_wxyz_per_env[env_ids] = table_quat

        self._write_goal(env_ids, object_pos, object_quat, object_pos)
        self._stable_relative_pos[env_ids] = relative_pos
        self._stable_relative_quat[env_ids] = relative_quat
        self._stable_grasp_reference_pos[env_ids] = target_relative_pos
        self._stable_grasp_reference_quat[env_ids] = target_relative_quat
        self._stable_prev_relative_pos[env_ids] = relative_pos
        self._stable_prev_relative_quat[env_ids] = relative_quat
        self._adjustment_target_relative_pos[env_ids] = target_relative_pos
        self._adjustment_target_relative_quat[env_ids] = target_relative_quat
        tool_to_target_palm_quat = quat_inv(target_relative_quat)
        tool_to_target_palm_pos = quat_apply(
            tool_to_target_palm_quat, -target_relative_pos
        )
        target_palm_pos, target_palm_quat = combine_frame_transforms(
            object_pos,
            object_quat,
            tool_to_target_palm_pos,
            tool_to_target_palm_quat,
        )
        self._adjustment_target_palm_pos_w[env_ids] = target_palm_pos
        self._adjustment_target_palm_quat_w[env_ids] = target_palm_quat
        self._adjustment_target_axial_translation[env_ids] = target_axial_translation
        self._adjustment_target_perpendicular_translation[env_ids] = (
            target_perpendicular_translation
        )
        self._adjustment_initial_tool_pos[env_ids] = object_pos
        self._adjustment_initial_tool_quat[env_ids] = object_quat
        translation_scale = max(
            float(self.cfg.adjustment_target_total_translation_max_m), 1.0e-6
        )
        relative_delta, relative_delta_quat = subtract_frame_transforms(
            relative_pos,
            relative_quat,
            target_relative_pos,
            target_relative_quat,
        )
        relative_rotation_delta = self._quaternion_log_vector(relative_delta_quat)
        self._adjustment_target_error_obs[env_ids, :3] = (
            relative_delta / translation_scale
        )
        self._adjustment_target_error_obs[env_ids, 3:] = (
            relative_rotation_delta / math.pi
        )
        tool_to_initial_palm_quat = quat_inv(relative_quat)
        tool_to_initial_palm_pos = quat_apply(
            tool_to_initial_palm_quat, -relative_pos
        )
        initial_palm_pos, initial_palm_quat = combine_frame_transforms(
            object_pos,
            object_quat,
            tool_to_initial_palm_pos,
            tool_to_initial_palm_quat,
        )
        palm_delta, palm_delta_quat = subtract_frame_transforms(
            initial_palm_pos,
            initial_palm_quat,
            target_palm_pos,
            target_palm_quat,
        )
        palm_rotation_delta = self._quaternion_log_vector(palm_delta_quat)
        self._adjustment_target_palm_error_obs[env_ids, :3] = (
            palm_delta / translation_scale
        )
        self._adjustment_target_palm_error_obs[env_ids, 3:] = (
            palm_rotation_delta / math.pi
        )
        self._adjustment_pose_error_obs[env_ids] = 0.0
        self._adjustment_phase_obs[env_ids] = 0.0
        self._adjustment_table_obs[env_ids, 0] = table_z
        self._adjustment_table_obs[env_ids, 1:] = normal
        self._adjustment_success_count[env_ids] = 0
        self._adjustment_pose_failure_count[env_ids] = 0
        self._adjustment_just_succeeded[env_ids] = False
        self._adjustment_succeeded[env_ids] = False
        self._stable_grasp_loss_count[env_ids] = 0
        self._stable_previous_action[env_ids] = self._inhand_bank_last_action[source_ids]
        self._stable_prev_tool_velocity[env_ids] = 0.0

    @staticmethod
    def _quaternion_log_vector(delta: torch.Tensor) -> torch.Tensor:
        delta = delta * torch.where(
            delta[:, :1] < 0.0, -torch.ones_like(delta[:, :1]),
            torch.ones_like(delta[:, :1]),
        )
        vector = delta[:, 1:]
        sin_half = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
        angle = 2.0 * torch.atan2(sin_half, delta[:, :1].clamp_min(1.0e-8))
        axis = vector / sin_half.clamp_min(1.0e-8)
        return axis * angle

    def _update_adjustment_metrics(self) -> None:
        relative_delta, relative_delta_quat = subtract_frame_transforms(
            self._stable_relative_pos,
            self._stable_relative_quat,
            self._adjustment_target_relative_pos,
            self._adjustment_target_relative_quat,
        )
        rotation_delta = self._quaternion_log_vector(relative_delta_quat)
        self._adjustment_relative_position_error.copy_(
            torch.linalg.vector_norm(relative_delta, dim=-1)
        )
        self._adjustment_relative_rotation_error.copy_(
            torch.linalg.vector_norm(rotation_delta, dim=-1)
        )
        translation_scale = max(
            float(self.cfg.adjustment_target_total_translation_max_m), 1.0e-6
        )
        self._adjustment_target_error_obs[:, :3] = relative_delta / translation_scale
        self._adjustment_target_error_obs[:, 3:] = rotation_delta / math.pi
        palm_pos = self.robot.data.body_link_pos_w[:, self._palm_body_id]
        palm_quat = self.robot.data.body_link_quat_w[:, self._palm_body_id]
        palm_delta, palm_delta_quat = subtract_frame_transforms(
            palm_pos,
            palm_quat,
            self._adjustment_target_palm_pos_w,
            self._adjustment_target_palm_quat_w,
        )
        palm_rotation_delta = self._quaternion_log_vector(palm_delta_quat)
        self._adjustment_target_palm_error_obs[:, :3] = palm_delta / translation_scale
        self._adjustment_target_palm_error_obs[:, 3:] = palm_rotation_delta / math.pi
        self._adjustment_target_palm_position_error.copy_(
            torch.linalg.vector_norm(palm_delta, dim=-1)
        )
        self._adjustment_target_palm_rotation_error.copy_(
            torch.linalg.vector_norm(palm_rotation_delta, dim=-1)
        )
        self._adjustment_tool_position_error.copy_(torch.linalg.vector_norm(
            self.object.data.root_pos_w - self._adjustment_initial_tool_pos, dim=-1
        ))
        self._adjustment_tool_rotation_error.copy_(quaternion_distance_rad(
            self.object.data.root_quat_w, self._adjustment_initial_tool_quat
        ))
        self._adjustment_pose_error_obs[:, 0] = (
            self._adjustment_tool_position_error
            / float(self.cfg.adjustment_tool_position_tolerance_m)
        )
        self._adjustment_pose_error_obs[:, 1] = (
            self._adjustment_tool_rotation_error
            / math.radians(float(self.cfg.adjustment_tool_rotation_tolerance_deg))
        )
        delay = int(
            self.cfg.adjustment_tool_pose_delay_steps[self._adjustment_curriculum_stage]
        )
        ramp = int(
            self.cfg.adjustment_tool_pose_ramp_steps[self._adjustment_curriculum_stage]
        )
        scale = ((self.episode_length_buf.float() - delay) / float(ramp)).clamp(0.0, 1.0)
        self._adjustment_tool_pose_weight_scale.copy_(scale)
        self._adjustment_phase_obs[:, 0] = scale

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._frame_counter += 1
        compute_intermediate_values(self)
        self._stable_support_count.copy_((
            self._curr_fingertip_distances
            < float(self.cfg.acquisition_max_fingertip_distance_m)
        ).sum(dim=-1))
        self._update_relative_motion()
        self._update_adjustment_metrics()
        position_tolerance = float(
            self.cfg.adjustment_relative_position_tolerance_stages_m[
                self._adjustment_curriculum_stage
            ]
        )
        rotation_tolerance = math.radians(float(
            self.cfg.adjustment_relative_rotation_tolerance_stages_deg[
                self._adjustment_curriculum_stage
            ]
        ))
        self._adjustment_support_valid.copy_(
            self._stable_support_count >= int(self.cfg.adjustment_min_fingertip_support)
        )
        self._adjustment_relative_position_valid.copy_(
            self._adjustment_relative_position_error <= position_tolerance
        )
        self._adjustment_relative_rotation_valid.copy_(
            self._adjustment_relative_rotation_error <= rotation_tolerance
        )
        self._adjustment_tool_position_valid.copy_(
            self._adjustment_tool_position_error
            <= float(self.cfg.adjustment_tool_position_tolerance_m)
        )
        self._adjustment_tool_rotation_valid.copy_(
            self._adjustment_tool_rotation_error
            <= math.radians(float(self.cfg.adjustment_tool_rotation_tolerance_deg))
        )
        valid = (
            self._adjustment_support_valid
            & self._adjustment_relative_position_valid
            & self._adjustment_relative_rotation_valid
            & self._adjustment_tool_position_valid
            & self._adjustment_tool_rotation_valid
        )
        self._adjustment_combined_valid.copy_(valid)
        self._adjustment_success_count.copy_(consecutive_counter(
            valid, self._adjustment_success_count
        ))
        success = self._adjustment_success_count >= int(self.cfg.adjustment_success_steps)
        self._adjustment_just_succeeded.copy_(success & ~self._adjustment_succeeded)
        self._adjustment_succeeded.copy_(self._adjustment_succeeded | success)
        # compute_intermediate_values() also computes the inherited pose-goal
        # success flag. For this task, success exclusively means that the grasp
        # adjustment criteria have been stable for the configured window.
        self._is_success.copy_(self._adjustment_succeeded)
        local_z = self.object.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        fall = local_z < 0.1
        raw_pose_failure = (
            (self._adjustment_tool_position_error > float(
                self.cfg.adjustment_tool_position_hard_limit_m
            ))
            | (self._adjustment_tool_rotation_error > math.radians(float(
                self.cfg.adjustment_tool_rotation_hard_limit_deg
            )))
        )
        self._adjustment_pose_failure_count.copy_(consecutive_counter(
            raw_pose_failure & (self._adjustment_tool_pose_weight_scale >= 1.0),
            self._adjustment_pose_failure_count,
        ))
        pose_failure = (
            self._adjustment_pose_failure_count >= int(
                self.cfg.adjustment_pose_failure_steps
            )
        )
        terminated = fall | pose_failure
        if self.cfg.adjustment_terminate_on_success:
            terminated |= success
        truncated = self.episode_length_buf >= self.max_episode_length
        completed = terminated | truncated
        completed_count = int(completed.sum().item())
        if completed_count:
            self._adjustment_curriculum_eligible += completed_count
            self._adjustment_curriculum_successes += int((success & completed).sum().item())
            last_stage = len(self.cfg.adjustment_finger_perturb_fractions) - 1
            if (
                self._adjustment_curriculum_eligible
                >= int(self.cfg.adjustment_curriculum_min_eligible_count)
            ):
                self._adjustment_curriculum_success_mean = (
                    self._adjustment_curriculum_successes
                    / self._adjustment_curriculum_eligible
                )
                self._adjustment_curriculum_last_window_size = (
                    self._adjustment_curriculum_eligible
                )
                self._adjustment_curriculum_evaluations += 1
                if (
                    self._adjustment_curriculum_stage < last_stage
                    and self._adjustment_curriculum_success_mean
                    >= float(self.cfg.adjustment_curriculum_success_threshold)
                ):
                    self._adjustment_curriculum_stage += 1
                    self._adjustment_curriculum_updates += 1
                # Evaluate independent recent windows. Keeping all failures from
                # the start of training makes a recovered policy unable to ever
                # reach the curriculum threshold.
                self._adjustment_curriculum_eligible = 0
                self._adjustment_curriculum_successes = 0
        self._termination_reasons = {
            "fall": fall, "tool_pose_failure": pose_failure,
            "adjustment_success": success, "timeout": truncated,
        }
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        relative_position_reward = torch.exp(
            -self._adjustment_relative_position_error
            / float(self.cfg.adjustment_relative_position_reward_sigma_m)
        )
        relative_rotation_reward = torch.exp(
            -self._adjustment_relative_rotation_error
            / math.radians(float(self.cfg.adjustment_relative_rotation_reward_sigma_deg))
        )
        pose_scale = self._adjustment_tool_pose_weight_scale
        weighted = {
            "target_relative_position_rew": relative_position_reward * float(
                self.cfg.adjustment_relative_position_reward_weight
            ),
            "target_relative_rotation_rew": relative_rotation_reward * float(
                self.cfg.adjustment_relative_rotation_reward_weight
            ),
            "tool_position_penalty": -self._adjustment_tool_position_error * pose_scale * float(
                self.cfg.adjustment_tool_position_penalty_weight
            ),
            "tool_rotation_penalty": -self._adjustment_tool_rotation_error * pose_scale * float(
                self.cfg.adjustment_tool_rotation_penalty_weight
            ),
            "fingertip_support_rew": (
                self._stable_support_count.float() / 5.0
            ).clamp(0.0, 1.0) * float(
                self.cfg.adjustment_fingertip_support_reward_weight
            ),
            "action_rate_penalty": -self._stable_action_delta_sq_mean * float(
                self.cfg.adjustment_action_rate_penalty_weight
            ),
            "success_bonus": self._adjustment_just_succeeded.float() * float(self.cfg.adjustment_success_bonus),
        }
        reward = torch.stack(tuple(weighted.values())).sum(0)
        self._reward_terms = {**weighted, "total_reward": reward}
        self.extras.update({f"reward/{name}": value.mean() for name, value in weighted.items()})
        self.extras.update({
            "adjustment/target_relative_position_error_mean_m": (
                self._adjustment_relative_position_error.mean()
            ),
            "adjustment/target_relative_rotation_error_mean_deg": torch.rad2deg(
                self._adjustment_relative_rotation_error
            ).mean(),
            "adjustment/target_palm_position_error_mean_m": (
                self._adjustment_target_palm_position_error.mean()
            ),
            "adjustment/target_palm_rotation_error_mean_deg": torch.rad2deg(
                self._adjustment_target_palm_rotation_error
            ).mean(),
            "adjustment/tool_position_error_mean_m": self._adjustment_tool_position_error.mean(),
            "adjustment/tool_rotation_error_mean_deg": torch.rad2deg(
                self._adjustment_tool_rotation_error
            ).mean(),
            "adjustment/success_ratio": self._adjustment_succeeded.float().mean(),
            "adjustment/fingertip_support_mean": self._stable_support_count.float().mean(),
            "adjustment/tool_pose_weight_scale_mean": pose_scale.mean(),
            "adjustment/target_axial_translation_mean_m": (
                self._adjustment_target_axial_translation.mean()
            ),
            "adjustment/target_perpendicular_translation_mean_m": (
                self._adjustment_target_perpendicular_translation.mean()
            ),
            "adjustment/raw_pose_failure_ratio": (
                self._adjustment_pose_failure_count > 0
            ).float().mean(),
            "adjustment/criterion_relative_position_ratio": (
                self._adjustment_relative_position_valid.float().mean()
            ),
            "adjustment/criterion_relative_rotation_ratio": (
                self._adjustment_relative_rotation_valid.float().mean()
            ),
            "adjustment/criterion_tool_position_ratio": (
                self._adjustment_tool_position_valid.float().mean()
            ),
            "adjustment/criterion_tool_rotation_ratio": (
                self._adjustment_tool_rotation_valid.float().mean()
            ),
            "adjustment/criterion_support_ratio": (
                self._adjustment_support_valid.float().mean()
            ),
            "adjustment/criterion_combined_ratio": (
                self._adjustment_combined_valid.float().mean()
            ),
            "curriculum/adjustment_stage": self._adjustment_curriculum_stage,
            "curriculum/adjustment_success_mean": self._adjustment_curriculum_success_mean,
            "curriculum/adjustment_eligible_count": self._adjustment_curriculum_eligible,
            "curriculum/adjustment_last_window_size": (
                self._adjustment_curriculum_last_window_size
            ),
            "curriculum/adjustment_evaluations": self._adjustment_curriculum_evaluations,
            "curriculum/finger_perturb_fraction": self.cfg.adjustment_finger_perturb_fractions[
                self._adjustment_curriculum_stage
            ],
            "curriculum/target_axial_translation_m": (
                self.cfg.adjustment_target_axial_translation_m[
                    self._adjustment_curriculum_stage
                ]
            ),
            "curriculum/target_perpendicular_translation_m": (
                self.cfg.adjustment_target_perpendicular_translation_m[
                    self._adjustment_curriculum_stage
                ]
            ),
            "curriculum/target_rotation_deg": self.cfg.adjustment_target_rotation_deg[
                self._adjustment_curriculum_stage
            ],
            "curriculum/relative_position_tolerance_m": (
                self.cfg.adjustment_relative_position_tolerance_stages_m[
                    self._adjustment_curriculum_stage
                ]
            ),
            "curriculum/relative_rotation_tolerance_deg": (
                self.cfg.adjustment_relative_rotation_tolerance_stages_deg[
                    self._adjustment_curriculum_stage
                ]
            ),
            "curriculum/tool_pose_delay_steps": self.cfg.adjustment_tool_pose_delay_steps[
                self._adjustment_curriculum_stage
            ],
            "curriculum/tool_pose_ramp_steps": self.cfg.adjustment_tool_pose_ramp_steps[
                self._adjustment_curriculum_stage
            ],
        })
        log_step_metrics(self)
        return reward


__all__ = ["SimToolRealInHandAdjustmentEnv", "SimToolRealInHandAdjustmentEnvCfg"]
