"""Stable-contact scraping with frozen-policy acquisition handoff."""

from __future__ import annotations

import math

import torch
from isaaclab.utils.math import subtract_frame_transforms

from .simtoolreal_scrape_pose_env import SimToolRealTacMapScrapePoseEnv
from .simtoolreal_tacmap_env_cfg import SimToolRealStableScrapeEnvCfg
from .utils.logging_utils import log_step_metrics
from .utils.obs_utils import compute_intermediate_values
from .utils.scrape_pose_utils import (
    edge_contact_points_w, table_top_state, tangent_basis_from_yaw,
)
from .utils.stable_scrape_utils import (
    ACQUISITION_PHASE,
    CONTACT_APPROACH_PHASE,
    SCRAPE_PHASE,
    advance_reflected_path,
    consecutive_counter,
    quaternion_distance_rad,
    stable_scrape_reward_terms,
)


class SimToolRealStableScrapeEnv(SimToolRealTacMapScrapePoseEnv):
    """Acquire with a frozen actor, then learn smooth contact-rich scraping."""

    cfg: SimToolRealStableScrapeEnvCfg

    def __init__(self, cfg: SimToolRealStableScrapeEnvCfg, render_mode=None, **kwargs):
        self._validate_stable_cfg(cfg)
        super().__init__(cfg, render_mode, **kwargs)
        n, device = self.num_envs, self.device
        self._stable_phase = torch.zeros(n, dtype=torch.long, device=device)
        self._stable_phase_steps = torch.zeros(n, dtype=torch.long, device=device)
        self._stable_acquisition_steps = torch.zeros(n, dtype=torch.long, device=device)
        self._stable_candidate_pos = torch.zeros(n, 3, device=device)
        self._stable_candidate_quat = torch.zeros(n, 4, device=device)
        self._stable_candidate_quat[:, 0] = 1.0
        self._stable_acquisition_count = torch.zeros(n, dtype=torch.long, device=device)
        self._stable_contact_count = torch.zeros(n, dtype=torch.long, device=device)
        self._stable_grasp_loss_count = torch.zeros(n, dtype=torch.long, device=device)
        self._stable_support_count = torch.zeros(n, dtype=torch.long, device=device)
        self._stable_relative_pos = torch.zeros(n, 3, device=device)
        self._stable_relative_quat = torch.zeros(n, 4, device=device)
        self._stable_relative_quat[:, 0] = 1.0
        self._stable_prev_relative_pos = self._stable_relative_pos.clone()
        self._stable_prev_relative_quat = self._stable_relative_quat.clone()
        self._stable_relative_linear_speed = torch.zeros(n, device=device)
        self._stable_relative_angular_speed = torch.zeros(n, device=device)
        self._stable_prev_tool_velocity = torch.zeros(n, 3, device=device)
        self._stable_tool_acceleration = torch.zeros(n, device=device)
        self._stable_previous_action = torch.zeros(n, self.cfg.action_space, device=device)
        self._stable_action_delta_sq_mean = torch.zeros(n, device=device)
        self._stable_contact_goal_pos_w = torch.zeros(n, 3, device=device)
        self._stable_contact_goal_quat_w = torch.zeros(n, 4, device=device)
        self._stable_contact_goal_quat_w[:, 0] = 1.0
        self._stable_contact_anchor_w = torch.zeros(n, 3, device=device)
        self._stable_path_direction_w = torch.zeros(n, 3, device=device)
        self._stable_path_offset = torch.zeros(n, device=device)
        self._stable_path_sign = torch.ones(n, device=device)
        self._stable_target_velocity_w = torch.zeros(n, 3, device=device)
        self._stable_over_force = torch.zeros(n, device=device)
        self._stable_previous_contact = torch.zeros(n, dtype=torch.bool, device=device)
        self._stable_contact_churn = torch.zeros(n, device=device)
        self._stable_pose_sigma = float(cfg.pose_tracking_sigma_start_m)
        self._stable_curriculum_success = 0.0
        self._stable_curriculum_updates = 0
        self._stable_last_curriculum_update = 0
        self._stable_handoff_total = 0
        ids = torch.arange(n, device=device)
        self._reset_stable_episode(ids)

    @staticmethod
    def _validate_stable_cfg(cfg: SimToolRealStableScrapeEnvCfg) -> None:
        if cfg.frozen_acquisition_obs_dim != 140:
            raise ValueError("V1 requires the exact 140-value vanilla actor prefix")
        if cfg.acquisition_stability_steps <= 0 or cfg.approach_contact_steps <= 0:
            raise ValueError("phase persistence requirements must be positive")
        if cfg.scrape_path_speed_mps <= 0.0 or cfg.scrape_path_half_length_m <= 0.0:
            raise ValueError("scrape path speed and half-length must be positive")
        if cfg.soft_contact_normal_force_limit >= cfg.hard_contact_normal_force_limit:
            raise ValueError("soft force limit must be below hard force limit")

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)
        if hasattr(self, "_stable_phase"):
            self._reset_stable_episode(env_ids)

    def _write_goal(self, env_ids: torch.Tensor, pos: torch.Tensor, quat: torch.Tensor, anchor: torch.Tensor) -> None:
        pose = torch.cat((pos, quat), dim=-1)
        self.goal_viz.write_root_pose_to_sim(pose, env_ids=env_ids)
        self.goal_viz.write_root_velocity_to_sim(
            torch.zeros(env_ids.numel(), 6, device=self.device), env_ids=env_ids
        )
        self._scrape_edge_anchor_w[env_ids] = anchor
        self._clear_goal_trackers(env_ids)

    def _palm_tool_relative(self) -> tuple[torch.Tensor, torch.Tensor]:
        return subtract_frame_transforms(
            self.robot.data.body_link_pos_w[:, self._palm_body_id],
            self.robot.data.body_link_quat_w[:, self._palm_body_id],
            self.object.data.root_pos_w,
            self.object.data.root_quat_w,
        )

    def _reset_stable_episode(self, env_ids: torch.Tensor) -> None:
        self._stable_phase[env_ids] = ACQUISITION_PHASE
        self._stable_phase_steps[env_ids] = 0
        self._stable_acquisition_steps[env_ids] = 0
        self._stable_acquisition_count[env_ids] = 0
        self._stable_contact_count[env_ids] = 0
        self._stable_grasp_loss_count[env_ids] = 0
        self._stable_path_offset[env_ids] = 0.0
        self._stable_path_sign[env_ids] = torch.where(
            torch.rand(env_ids.numel(), device=self.device) < 0.5, -1.0, 1.0
        )
        self._stable_target_velocity_w[env_ids] = 0.0
        self._stable_over_force[env_ids] = 0.0
        self._stable_previous_action[env_ids] = 0.0
        self._stable_action_delta_sq_mean[env_ids] = 0.0
        self._stable_previous_contact[env_ids] = False
        self._stable_contact_churn[env_ids] = 0.0

        self._stable_contact_goal_pos_w[env_ids] = self.goal_viz.data.root_pos_w[env_ids]
        self._stable_contact_goal_quat_w[env_ids] = self.goal_viz.data.root_quat_w[env_ids]
        self._stable_contact_anchor_w[env_ids] = self._scrape_edge_anchor_w[env_ids]
        table_quat = getattr(self, "_table_quat_wxyz_per_env", self.table.data.root_quat_w)
        _, normal = table_top_state(self.table.data.root_pos_w, table_quat)
        _, forward = tangent_basis_from_yaw(self._scrape_edge_yaw, normal)
        self._stable_path_direction_w[env_ids] = forward[env_ids]

        relative_pos, relative_quat = self._palm_tool_relative()
        self._stable_relative_pos[env_ids] = relative_pos[env_ids]
        self._stable_relative_quat[env_ids] = relative_quat[env_ids]
        self._stable_prev_relative_pos[env_ids] = relative_pos[env_ids]
        self._stable_prev_relative_quat[env_ids] = relative_quat[env_ids]
        self._stable_prev_tool_velocity[env_ids] = self.object.data.root_lin_vel_w[env_ids]

        hover_pos = self._stable_contact_goal_pos_w[env_ids] + normal[env_ids] * float(
            self.cfg.acquisition_hover_height_m
        )
        hover_anchor = self._stable_contact_anchor_w[env_ids] + normal[env_ids] * float(
            self.cfg.acquisition_hover_height_m
        )
        self._write_goal(
            env_ids, hover_pos, self._stable_contact_goal_quat_w[env_ids], hover_anchor
        )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        canonical = actions.clone().to(self.device)
        delta = canonical - self._stable_previous_action
        self._stable_action_delta_sq_mean.copy_(delta.square().mean(dim=-1))
        self._stable_previous_action.copy_(canonical)
        super()._pre_physics_step(actions)

    def _advance_scrape_reference(self) -> None:
        active = self._stable_phase == SCRAPE_PHASE
        if not bool(active.any()):
            self._stable_target_velocity_w.zero_()
            return
        ids = active.nonzero(as_tuple=False).squeeze(-1)
        offset, sign = advance_reflected_path(
            self._stable_path_offset[ids], self._stable_path_sign[ids],
            distance=float(self.cfg.scrape_path_speed_mps) * float(self.step_dt),
            half_length=float(self.cfg.scrape_path_half_length_m),
        )
        self._stable_path_offset[ids] = offset
        self._stable_path_sign[ids] = sign
        delta = self._stable_path_direction_w[ids] * offset.unsqueeze(-1)
        pos = self._stable_contact_goal_pos_w[ids] + delta
        anchor = self._stable_contact_anchor_w[ids] + delta
        self._write_goal(ids, pos, self._stable_contact_goal_quat_w[ids], anchor)
        self._stable_target_velocity_w.zero_()
        self._stable_target_velocity_w[ids] = (
            self._stable_path_direction_w[ids]
            * sign.unsqueeze(-1)
            * float(self.cfg.scrape_path_speed_mps)
        )

    def _update_relative_motion(self) -> None:
        pos, quat = self._palm_tool_relative()
        dt = max(float(self.step_dt), 1.0e-6)
        self._stable_relative_linear_speed.copy_(
            torch.linalg.vector_norm(pos - self._stable_prev_relative_pos, dim=-1) / dt
        )
        self._stable_relative_angular_speed.copy_(
            quaternion_distance_rad(quat, self._stable_prev_relative_quat) / dt
        )
        velocity = self.object.data.root_lin_vel_w
        self._stable_tool_acceleration.copy_(
            torch.linalg.vector_norm(velocity - self._stable_prev_tool_velocity, dim=-1) / dt
        )
        self._stable_relative_pos.copy_(pos)
        self._stable_relative_quat.copy_(quat)
        self._stable_prev_relative_pos.copy_(pos)
        self._stable_prev_relative_quat.copy_(quat)
        self._stable_prev_tool_velocity.copy_(velocity)

    def _update_acquisition(self, force: torch.Tensor) -> torch.Tensor:
        acquisition = self._stable_phase == ACQUISITION_PHASE
        self._stable_acquisition_steps += acquisition.long()
        _, table_normal = table_top_state(
            self.table.data.root_pos_w,
            getattr(self, "_table_quat_wxyz_per_env", self.table.data.root_quat_w),
        )
        edge_points = edge_contact_points_w(
            self.object.data.root_pos_w, self.object.data.root_quat_w,
            self._scrape_x_tip_per_env, self._scrape_y_min_per_env,
            self._scrape_y_max_per_env, self._scrape_z_contact_per_env,
        )
        table_top, _ = table_top_state(
            self.table.data.root_pos_w,
            getattr(self, "_table_quat_wxyz_per_env", self.table.data.root_quat_w),
        )
        edge_clearance = (
            (edge_points[:, 1] - table_top) * table_normal
        ).sum(dim=-1)
        base = (
            acquisition
            & (edge_clearance > float(self.cfg.acquisition_min_edge_clearance_m))
            & (self._stable_support_count >= int(self.cfg.acquisition_min_fingertips))
            & (force < float(self.cfg.acquisition_max_table_force_n))
        )
        starting = base & (self._stable_acquisition_count == 0)
        self._stable_candidate_pos[starting] = self._stable_relative_pos[starting]
        self._stable_candidate_quat[starting] = self._stable_relative_quat[starting]
        drift = torch.linalg.vector_norm(
            self._stable_relative_pos - self._stable_candidate_pos, dim=-1
        )
        rotation = torch.rad2deg(quaternion_distance_rad(
            self._stable_relative_quat, self._stable_candidate_quat
        ))
        stable = (
            base
            & (drift <= float(self.cfg.acquisition_position_drift_m))
            & (rotation <= float(self.cfg.acquisition_rotation_drift_deg))
        )
        self._stable_acquisition_count.copy_(
            consecutive_counter(stable, self._stable_acquisition_count)
        )
        restart = base & ~stable
        self._stable_candidate_pos[restart] = self._stable_relative_pos[restart]
        self._stable_candidate_quat[restart] = self._stable_relative_quat[restart]
        handoff = acquisition & (
            self._stable_acquisition_count >= int(self.cfg.acquisition_stability_steps)
        )
        if bool(handoff.any()):
            ids = handoff.nonzero(as_tuple=False).squeeze(-1)
            self._stable_phase[ids] = CONTACT_APPROACH_PHASE
            self._stable_phase_steps[ids] = 0
            self._stable_handoff_total += int(ids.numel())
            self._write_goal(
                ids, self._stable_contact_goal_pos_w[ids],
                self._stable_contact_goal_quat_w[ids], self._stable_contact_anchor_w[ids]
            )
        return handoff

    def _update_approach(self, edge_error: torch.Tensor) -> torch.Tensor:
        approach = self._stable_phase == CONTACT_APPROACH_PHASE
        ready = (
            approach
            & (self._stable_support_count >= 2)
            & (edge_error <= float(self.cfg.approach_edge_tolerance_m))
            & (self._contact_force_reward_ramp >= 1.0)
        )
        self._stable_contact_count.copy_(
            consecutive_counter(ready, self._stable_contact_count)
        )
        entered = approach & (
            self._stable_contact_count >= int(self.cfg.approach_contact_steps)
        )
        if bool(entered.any()):
            ids = entered.nonzero(as_tuple=False).squeeze(-1)
            self._stable_phase[ids] = SCRAPE_PHASE
            self._stable_phase_steps[ids] = 0
        return entered

    def _update_curriculum(self, eligible: torch.Tensor, success: torch.Tensor) -> None:
        count = int(eligible.sum().item())
        self._stable_curriculum_success = (
            float(success[eligible].float().mean().item()) if count else 0.0
        )
        interval = int(self.cfg.termination.tolerance_curriculum_interval)
        if self._frame_counter - self._stable_last_curriculum_update < interval:
            return
        if count < int(self.cfg.stable_curriculum_min_eligible_count):
            return
        if self._stable_curriculum_success < float(self.cfg.stable_curriculum_success_threshold):
            return
        new_sigma = max(
            self._stable_pose_sigma * float(self.cfg.stable_curriculum_increment),
            float(self.cfg.pose_tracking_sigma_target_m),
        )
        if new_sigma != self._stable_pose_sigma:
            self._stable_pose_sigma = new_sigma
            self._stable_curriculum_updates += 1
        self._stable_last_curriculum_update = self._frame_counter

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._frame_counter += 1
        self._stable_phase_steps += 1
        self._advance_scrape_reference()
        compute_intermediate_values(self)
        force = self._sensor_normal_force()
        _, edge_error, _ = self._compute_edge_contact_reward()
        self._scrape_edge_contact_error.copy_(edge_error)
        self._stable_support_count.copy_(
            (self._curr_fingertip_distances < float(
                self.cfg.acquisition_max_fingertip_distance_m
            )).sum(dim=-1)
        )
        self._update_relative_motion()
        interval_force = self._scrape_table_normal_force_interval
        self._stable_over_force.copy_(
            torch.clamp(
                interval_force - float(self.cfg.soft_contact_normal_force_limit), min=0.0
            )
        )
        contact_now = self._contact_force_reward_ramp >= 1.0
        self._stable_contact_churn.copy_((contact_now ^ self._stable_previous_contact).float())
        self._stable_previous_contact.copy_(contact_now)
        self._update_acquisition(force)
        self._update_approach(edge_error)

        post_grasp = self._stable_phase != ACQUISITION_PHASE
        lost_support = post_grasp & (self._stable_support_count < 2)
        self._stable_grasp_loss_count.copy_(
            consecutive_counter(lost_support, self._stable_grasp_loss_count)
        )
        grasp_loss = self._stable_grasp_loss_count >= int(self.cfg.grasp_loss_grace_steps)
        local_z = self.object.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        fall = local_z < 0.1
        hard_force = interval_force > float(self.cfg.hard_contact_normal_force_limit)
        acquisition_timeout = (
            (self._stable_phase == ACQUISITION_PHASE)
            & (self._stable_acquisition_steps >= int(self.cfg.acquisition_timeout_steps))
        )
        terminated = fall | grasp_loss | hard_force | acquisition_timeout
        truncated = self.episode_length_buf >= self.max_episode_length

        eligible = self._stable_phase == SCRAPE_PHASE
        success = (
            eligible & contact_now & (self._stable_support_count >= 2)
            & (self._keypoints_max_dist <= self._stable_pose_sigma)
            & (edge_error <= self._edge_contact_sigma())
        )
        self._is_success.copy_(success)
        self._update_curriculum(eligible, success)
        self._termination_reasons = {
            "fall": fall, "grasp_loss": grasp_loss, "hard_force": hard_force,
            "acquisition_timeout": acquisition_timeout, "timeout": truncated,
        }
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        edge_rew, edge_error, edge_weight = self._compute_edge_contact_reward()
        edge_score = edge_rew / max(float(edge_weight), 1.0e-6)
        terms = stable_scrape_reward_terms(
            phase=self._stable_phase,
            pose_error_m=self._keypoints_max_dist,
            pose_sigma_m=self._stable_pose_sigma,
            edge_score=edge_score,
            persistent_contact=self._contact_force_reward_ramp >= 1.0,
            support_count=self._stable_support_count,
            relative_linear_speed=self._stable_relative_linear_speed,
            relative_angular_speed=self._stable_relative_angular_speed,
            action_delta_sq_mean=self._stable_action_delta_sq_mean,
            tool_acceleration=self._stable_tool_acceleration,
            normal_force_n=self._scrape_table_normal_force_interval,
            soft_force_limit_n=float(self.cfg.soft_contact_normal_force_limit),
        )
        weighted = {
            "pose_tracking_rew": terms["tracking"] * float(self.cfg.pose_reward_weight),
            "edge_contact_rew": terms["edge"] * float(self.cfg.edge_contact_reward_max_weight),
            "contact_presence_rew": terms["contact"] * float(self.cfg.contact_presence_reward_weight),
            "support_rew": terms["support"] * float(self.cfg.support_reward_weight),
            "slip_penalty": terms["slip"] * float(self.cfg.slip_penalty_weight),
            "spin_penalty": terms["spin"] * float(self.cfg.spin_penalty_weight),
            "action_rate_penalty": terms["action_rate"] * float(self.cfg.action_rate_penalty_weight),
            "tool_acceleration_penalty": terms["acceleration"] * float(self.cfg.tool_acceleration_penalty_weight),
            "over_force_penalty": terms["over_force"] * float(self.cfg.over_force_penalty_weight),
            "contact_churn_penalty": -0.1 * self._stable_contact_churn * (self._stable_phase != ACQUISITION_PHASE),
        }
        reward = torch.stack(tuple(weighted.values()), dim=0).sum(dim=0)
        self._reward_terms = {**weighted, "total_reward": reward}
        self.extras.update({f"reward/{name}": value.mean() for name, value in weighted.items()})
        self.extras.update({
            "phase/acquisition_ratio": (self._stable_phase == ACQUISITION_PHASE).float().mean(),
            "phase/contact_approach_ratio": (self._stable_phase == CONTACT_APPROACH_PHASE).float().mean(),
            "phase/scrape_ratio": (self._stable_phase == SCRAPE_PHASE).float().mean(),
            "phase/handoff_total": self._stable_handoff_total,
            "stability/support_count_mean": self._stable_support_count.float().mean(),
            "stability/relative_linear_speed_mean": self._stable_relative_linear_speed.mean(),
            "stability/relative_angular_speed_mean": self._stable_relative_angular_speed.mean(),
            "stability/contact_persistence_mean": self._contact_force_reward_ramp.mean(),
            "safety/contact_force_interval_mean": self._scrape_table_normal_force_interval.mean(),
            "safety/over_force_mean": self._stable_over_force.mean(),
            "curriculum/pose_tracking_tolerance": self._stable_pose_sigma,
            "curriculum/pose_tracking_success_mean": self._stable_curriculum_success,
            "curriculum/pose_tracking_updates": self._stable_curriculum_updates,
            "task/pose_error_mean": self._keypoints_max_dist.mean(),
            "task/edge_error_mean": edge_error.mean(),
            "task/target_speed_mean": torch.linalg.vector_norm(
                self._stable_target_velocity_w, dim=-1
            ).mean(),
        })
        log_step_metrics(self)
        return reward


__all__ = ["SimToolRealStableScrapeEnv", "SimToolRealStableScrapeEnvCfg"]
