"""Pose-only table scraping task built on TacMap SimToolReal."""

from __future__ import annotations

import math

import torch

from .simtoolreal_tacmap_env import SimToolRealTacMapEnv
from .simtoolreal_tacmap_env_cfg import SimToolRealTacMapScrapePoseEnvCfg
from .utils.logging_utils import log_step_metrics
from .utils.obs_utils import compute_intermediate_values
from .utils.reward_utils import compute_rewards
from .utils.scrape_pose_utils import (
    contact_force_reward,
    edge_contact_points_w,
    edge_contact_reward,
    load_urdf_collision_bounds,
    sample_edge_contact_goal_pose,
    table_top_state,
)
from .utils.termination_utils import update_tolerance_curriculum


class SimToolRealTacMapScrapePoseEnv(SimToolRealTacMapEnv):
    """TacMap task with original pose reward and edge-contact goal sampling."""

    cfg: SimToolRealTacMapScrapePoseEnvCfg

    def __init__(
        self,
        cfg: SimToolRealTacMapScrapePoseEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        cfg.reset.table_reset_pitch_roll_range_deg = float(
            cfg.table_pitch_roll_range_deg
        )
        super().__init__(cfg, render_mode, **kwargs)
        self._scrape_edge_anchor_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._scrape_edge_yaw = torch.zeros(self.num_envs, device=self.device)
        self._scrape_edge_contact_error = torch.zeros(self.num_envs, device=self.device)
        self._scrape_table_normal_force_raw = torch.zeros(
            self.num_envs, device=self.device
        )
        self._scrape_table_normal_force = torch.zeros(self.num_envs, device=self.device)
        self._contact_force_filter_initialized = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._scrape_target_contact_normal_force = torch.zeros(
            self.num_envs, device=self.device
        )
        self._validate_contact_reward_curricula_cfg()
        self._current_edge_contact_sigma = float(
            self.cfg.edge_contact_reward_sigma_start_m
        )
        self._current_contact_force_sigma = float(self.cfg.contact_force_sigma_start)
        self._edge_contact_curriculum_success_mean = 0.0
        self._contact_force_curriculum_success_mean = 0.0
        self._last_edge_contact_curriculum_update = 0
        self._last_contact_force_curriculum_update = 0
        self._edge_contact_curriculum_update_count = 0
        self._contact_force_curriculum_update_count = 0
        self._cache_scrape_tool_bounds()
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._reset_scrape_goal_pose(env_ids, resample_edge=True)
        self._clear_goal_trackers(env_ids)

    def _setup_scene(self) -> None:
        super()._setup_scene()
        self._tool_table_contact_sensor = None
        self._tool_table_contact_sensor_failed = False
        self._tool_table_contact_sensor_error = ""
        if not bool(
            getattr(self.cfg, "enable_tool_table_contact_force_reward", False)
        ):
            return
        try:
            from isaaclab.sensors import ContactSensor, ContactSensorCfg

            sensor_cfg = ContactSensorCfg(
                prim_path=self.cfg.tool_table_contact_sensor_prim_path,
                update_period=float(self.cfg.tool_table_contact_sensor_update_period),
                history_length=int(self.cfg.tool_table_contact_sensor_history_len),
                debug_vis=False,
                track_pose=False,
                track_contact_points=False,
                track_friction_forces=False,
                track_air_time=False,
                force_threshold=float(
                    self.cfg.tool_table_contact_sensor_force_threshold
                ),
                filter_prim_paths_expr=list(
                    self.cfg.tool_table_contact_sensor_filter_paths
                ),
            )
            self._tool_table_contact_sensor = ContactSensor(sensor_cfg)
            self.scene.sensors["tool_table_contact_sensor"] = (
                self._tool_table_contact_sensor
            )
        except Exception as exc:  # pragma: no cover - runtime/Kit dependent.
            self._tool_table_contact_sensor_failed = True
            self._tool_table_contact_sensor_error = repr(exc)
            raise RuntimeError(
                "enable_tool_table_contact_force_reward=true, but the "
                f"tool-table ContactSensor could not be created: {exc!r}"
            ) from exc

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)
        if hasattr(self, "_scrape_edge_anchor_w"):
            self._reset_contact_force_filter(env_ids)
            self._reset_scrape_goal_pose(env_ids, resample_edge=True)
            self._clear_goal_trackers(env_ids)

    def _validate_tightening_curriculum_cfg(
        self,
        *,
        name: str,
        start: float,
        target: float,
        increment: float,
        threshold: float,
    ) -> None:
        if start <= 0.0 or target <= 0.0:
            raise ValueError(f"{name} curriculum thresholds must be positive.")
        if start < target:
            raise ValueError(
                f"{name} start threshold ({start}) must be >= target threshold ({target}) "
                "because this curriculum tightens the reward tolerance."
            )
        if not (0.0 < increment <= 1.0):
            raise ValueError(
                f"{name} increment must be in (0, 1], got {increment}."
            )
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(
                f"{name} success threshold must be in [0, 1], got {threshold}."
            )

    def _validate_contact_reward_curricula_cfg(self) -> None:
        self._validate_tightening_curriculum_cfg(
            name="edge_contact_reward_sigma",
            start=float(self.cfg.edge_contact_reward_sigma_start_m),
            target=float(self.cfg.edge_contact_reward_sigma_target_m),
            increment=float(self.cfg.edge_contact_reward_sigma_increment),
            threshold=float(self.cfg.edge_contact_curriculum_success_threshold),
        )
        self._validate_tightening_curriculum_cfg(
            name="contact_force_sigma",
            start=float(self.cfg.contact_force_sigma_start),
            target=float(self.cfg.contact_force_sigma_target),
            increment=float(self.cfg.contact_force_sigma_increment),
            threshold=float(self.cfg.contact_force_curriculum_success_threshold),
        )
        filter_alpha = float(self.cfg.contact_force_filter_alpha)
        if not (0.0 < filter_alpha <= 1.0):
            raise ValueError(
                "contact_force_filter_alpha must be in (0, 1], got "
                f"{filter_alpha}."
            )

    def _contact_force_target_range(self) -> tuple[float, float]:
        fixed_target = getattr(self.cfg, "target_contact_normal_force", None)
        if fixed_target is not None:
            lo = hi = float(fixed_target)
        else:
            target_range = tuple(
                float(v) for v in self.cfg.target_contact_normal_force_range
            )
            if len(target_range) != 2:
                raise ValueError(
                    "target_contact_normal_force_range must contain exactly two values."
                )
            lo, hi = target_range
        if lo <= 0.0 or hi <= 0.0:
            raise ValueError(
                "target contact normal force values must be positive, got "
                f"({lo}, {hi})."
            )
        if lo > hi:
            raise ValueError(
                "target_contact_normal_force_range lower bound must be <= upper bound, "
                f"got ({lo}, {hi})."
            )
        max_force = float(self.cfg.max_contact_normal_force)
        if hi > max_force:
            raise ValueError(
                "target_contact_normal_force_range upper bound must be <= "
                f"max_contact_normal_force ({max_force}), got {hi}."
            )
        return lo, hi

    def _reset_contact_force_targets(self, env_ids: torch.Tensor) -> None:
        lo, hi = self._contact_force_target_range()
        if lo == hi:
            self._scrape_target_contact_normal_force[env_ids] = lo
            return
        self._scrape_target_contact_normal_force[env_ids] = torch.empty(
            env_ids.numel(), device=self.device
        ).uniform_(lo, hi)

    def _reset_contact_force_filter(self, env_ids: torch.Tensor) -> None:
        self._scrape_table_normal_force_raw[env_ids] = 0.0
        self._scrape_table_normal_force[env_ids] = 0.0
        self._contact_force_filter_initialized[env_ids] = False

    def _cache_scrape_tool_bounds(self) -> None:
        bounds = [load_urdf_collision_bounds(path) for path in self._object_urdf_paths]
        bounds_t = torch.tensor(bounds, device=self.device, dtype=torch.float32)
        per_env_bounds = bounds_t[self._object_asset_index_per_env]
        self._scrape_x_tip_per_env = per_env_bounds[:, 3]
        self._scrape_y_min_per_env = per_env_bounds[:, 1]
        self._scrape_y_max_per_env = per_env_bounds[:, 4]
        self._scrape_y_center_per_env = 0.5 * (
            per_env_bounds[:, 1] + per_env_bounds[:, 4]
        )
        self._scrape_z_contact_per_env = per_env_bounds[:, 2]

    def _clear_goal_trackers(self, env_ids: torch.Tensor) -> None:
        self._closest_keypoint_max_dist[env_ids] = -1.0
        self._closest_fingertip_dist[env_ids] = -1.0
        self._near_goal_steps[env_ids] = 0

    def _reset_scrape_goal_pose(
        self, env_ids: torch.Tensor, *, resample_edge: bool = False
    ) -> None:
        n = env_ids.numel()
        tilt_range_rad = (
            math.radians(float(self.cfg.edge_tilt_range_deg[0])),
            math.radians(float(self.cfg.edge_tilt_range_deg[1])),
        )
        edge_yaw = None
        if not resample_edge:
            edge_yaw = self._scrape_edge_yaw[env_ids]
        goal_pos_w, goal_quat_wxyz, edge_anchor_w, edge_yaw = sample_edge_contact_goal_pose(
            table_pos_w=self.table.data.root_pos_w[env_ids],
            table_quat_wxyz=getattr(
                self, "_table_quat_wxyz_per_env", self.table.data.root_quat_w
            )[env_ids],
            x_tip=self._scrape_x_tip_per_env[env_ids],
            y_center=self._scrape_y_center_per_env[env_ids],
            z_contact=self._scrape_z_contact_per_env[env_ids],
            xy_half_range=tuple(float(v) for v in self.cfg.edge_contact_xy_range_m),
            edge_yaw_range_rad=math.radians(float(self.cfg.edge_contact_yaw_range_deg)),
            tilt_range_rad=tilt_range_rad,
            device=self.device,
            edge_yaw=edge_yaw,
        )
        self._reset_contact_force_targets(env_ids)
        pose = torch.cat((goal_pos_w, goal_quat_wxyz), dim=-1)
        self.goal_viz.write_root_pose_to_sim(pose, env_ids=env_ids)
        self.goal_viz.write_root_velocity_to_sim(
            torch.zeros(n, 6, device=self.device), env_ids=env_ids
        )
        self._scrape_edge_anchor_w[env_ids] = edge_anchor_w
        self._scrape_edge_yaw[env_ids] = edge_yaw

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        update_tolerance_curriculum(self)
        self.cfg.reset.table_reset_pitch_roll_range_deg = float(
            self.cfg.table_pitch_roll_range_deg
        )
        compute_intermediate_values(self)

        is_success = self._is_success
        self._successes = self._successes + is_success.long()
        goal_reset_ids = is_success.nonzero(as_tuple=False).squeeze(-1)
        if goal_reset_ids.numel() > 0:
            self._clear_goal_trackers(goal_reset_ids)
            self._reset_scrape_goal_pose(goal_reset_ids, resample_edge=False)
            self.episode_length_buf[goal_reset_ids] = 0

        object_z_local = (
            self.object.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        )
        fall = object_z_local < 0.1
        if self.cfg.termination.max_consecutive_successes > 0:
            max_successes = self._successes >= self.cfg.termination.max_consecutive_successes
        else:
            max_successes = torch.zeros_like(fall)
        hand_far = self._curr_fingertip_distances.max(dim=-1).values > 1.5
        terminated = fall | max_successes | hand_far
        truncated = self.episode_length_buf >= self.max_episode_length
        self._termination_reasons = {
            "fall": fall,
            "max_successes": max_successes,
            "hand_far": hand_far,
            "timeout": truncated,
        }
        return terminated, truncated

    def _sensor_normal_force(self) -> torch.Tensor:
        if not bool(
            getattr(self.cfg, "enable_tool_table_contact_force_reward", False)
        ):
            return torch.zeros(self.num_envs, device=self.device)
        sensor = getattr(self, "_tool_table_contact_sensor", None)
        if sensor is None:
            raise RuntimeError(
                "enable_tool_table_contact_force_reward=true, but "
                "_tool_table_contact_sensor is not available."
            )
        data = getattr(sensor, "data", None)
        if data is None:
            raise RuntimeError("tool-table ContactSensor has no data object.")
        force = getattr(data, "force_matrix_w", None)
        if force is None:
            raise RuntimeError(
                "tool-table ContactSensor has no force_matrix_w. Pair-filtered force "
                "data is required; refusing to fall back to unfiltered net_forces_w."
            )
        if force.ndim < 3 or force.shape[-1] != 3:
            raise RuntimeError(
                "tool-table ContactSensor force_matrix_w has invalid shape "
                f"{tuple(force.shape)}; expected (..., 3)."
            )
        if force.shape[0] != self.num_envs or any(
            dim == 0 for dim in force.shape[1:-1]
        ):
            raise RuntimeError(
                "tool-table ContactSensor force_matrix_w is not populated for all envs: "
                f"shape={tuple(force.shape)}, num_envs={self.num_envs}."
            )
        vec = force.sum(dim=tuple(range(1, force.ndim - 1)))
        if vec.shape != (self.num_envs, 3):
            raise RuntimeError(
                "tool-table ContactSensor force reduction produced invalid shape "
                f"{tuple(vec.shape)}; expected ({self.num_envs}, 3)."
            )
        if not torch.isfinite(vec).all():
            raise RuntimeError(
                "tool-table ContactSensor force tensor contains NaN or Inf."
            )
        table_quat = getattr(
            self, "_table_quat_wxyz_per_env", self.table.data.root_quat_w
        )
        _, table_normal = table_top_state(self.table.data.root_pos_w, table_quat)
        raw_force = torch.abs((vec * table_normal).sum(dim=-1))
        alpha = float(self.cfg.contact_force_filter_alpha)
        filtered_force = torch.lerp(
            self._scrape_table_normal_force, raw_force, alpha
        )
        filtered_force = torch.where(
            self._contact_force_filter_initialized, filtered_force, raw_force
        )
        if not torch.isfinite(filtered_force).all():
            raise RuntimeError("Filtered tool-table normal force contains NaN or Inf.")
        self._scrape_table_normal_force_raw.copy_(raw_force)
        self._scrape_table_normal_force.copy_(filtered_force)
        self._contact_force_filter_initialized.fill_(True)
        return filtered_force

    def _edge_contact_sigma(self) -> float:
        return float(
            getattr(
                self,
                "_current_edge_contact_sigma",
                float(self.cfg.edge_contact_reward_sigma_start_m),
            )
        )

    def _contact_force_sigma(self) -> float:
        return float(
            getattr(
                self,
                "_current_contact_force_sigma",
                float(self.cfg.contact_force_sigma_start),
            )
        )

    def _maybe_tighten_contact_curriculum(
        self,
        *,
        current_attr: str,
        last_update_attr: str,
        update_count_attr: str,
        success_mean: float,
        success_threshold: float,
        target: float,
        increment: float,
    ) -> bool:
        interval = int(self.cfg.termination.tolerance_curriculum_interval)
        if self._frame_counter - int(getattr(self, last_update_attr)) < interval:
            return False
        if success_mean < float(success_threshold):
            return False

        current = float(getattr(self, current_attr))
        new_value = max(current * float(increment), float(target))
        if new_value != current:
            setattr(self, current_attr, new_value)
            setattr(
                self, update_count_attr, int(getattr(self, update_count_attr)) + 1
            )
        setattr(self, last_update_attr, int(self._frame_counter))
        return new_value != current

    def _update_contact_reward_curricula(
        self,
        *,
        edge_success_mean: float,
        force_success_mean: float,
        force_enabled: bool,
    ) -> None:
        self._edge_contact_curriculum_success_mean = float(edge_success_mean)
        self._contact_force_curriculum_success_mean = float(force_success_mean)
        self._maybe_tighten_contact_curriculum(
            current_attr="_current_edge_contact_sigma",
            last_update_attr="_last_edge_contact_curriculum_update",
            update_count_attr="_edge_contact_curriculum_update_count",
            success_mean=float(edge_success_mean),
            success_threshold=float(self.cfg.edge_contact_curriculum_success_threshold),
            target=float(self.cfg.edge_contact_reward_sigma_target_m),
            increment=float(self.cfg.edge_contact_reward_sigma_increment),
        )
        if force_enabled:
            self._maybe_tighten_contact_curriculum(
                current_attr="_current_contact_force_sigma",
                last_update_attr="_last_contact_force_curriculum_update",
                update_count_attr="_contact_force_curriculum_update_count",
                success_mean=float(force_success_mean),
                success_threshold=float(self.cfg.contact_force_curriculum_success_threshold),
                target=float(self.cfg.contact_force_sigma_target),
                increment=float(self.cfg.contact_force_sigma_increment),
            )

    def _force_reward_weight(self) -> float:
        return float(self.cfg.edge_contact_reward_max_weight) * float(
            self.cfg.contact_force_reward_relative_weight
        )

    def _compute_contact_force_reward(
        self, edge_score: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        normal_force = self._sensor_normal_force()
        force_score, over_force = contact_force_reward(
            normal_force,
            self._scrape_target_contact_normal_force,
            self._contact_force_sigma(),
            float(self.cfg.max_contact_normal_force),
        )
        weight = self._force_reward_weight() if bool(
            getattr(self.cfg, "enable_tool_table_contact_force_reward", False)
        ) else 0.0
        reward = force_score * edge_score.detach() * weight
        return reward, normal_force, over_force, weight

    def _edge_contact_reward_weight(self) -> float:
        return float(self.cfg.edge_contact_reward_max_weight)

    def _compute_edge_contact_reward(self) -> tuple[torch.Tensor, torch.Tensor, float]:
        edge_points = edge_contact_points_w(
            self.object.data.root_pos_w,
            self.object.data.root_quat_w,
            self._scrape_x_tip_per_env,
            self._scrape_y_min_per_env,
            self._scrape_y_max_per_env,
            self._scrape_z_contact_per_env,
        )
        table_quat = getattr(
            self, "_table_quat_wxyz_per_env", self.table.data.root_quat_w
        )
        score, error = edge_contact_reward(
            edge_points,
            self.table.data.root_pos_w,
            table_quat,
            self._edge_contact_sigma(),
        )
        weight = self._edge_contact_reward_weight()
        return score * weight, error, weight

    def _get_rewards(self) -> torch.Tensor:
        reward = compute_rewards(self)
        edge_rew, edge_error, edge_weight = self._compute_edge_contact_reward()
        edge_score = (
            edge_rew / max(float(edge_weight), 1.0e-6)
            if edge_weight > 0.0
            else torch.zeros_like(edge_rew)
        )
        (
            force_rew,
            normal_force,
            over_force,
            force_weight,
        ) = self._compute_contact_force_reward(edge_score)
        reward = reward + edge_rew + force_rew
        self._reward_terms["edge_contact_rew"] = edge_rew
        self._reward_terms["contact_force_rew"] = force_rew
        self._reward_terms["total_reward"] = reward
        self._scrape_edge_contact_error = edge_error
        self._scrape_table_normal_force = normal_force

        edge_sigma = float(self._edge_contact_sigma())
        force_sigma = float(self._contact_force_sigma())
        target_force = self._scrape_target_contact_normal_force
        force_error = torch.abs(normal_force - target_force)
        force_enabled = bool(
            getattr(self.cfg, "enable_tool_table_contact_force_reward", False)
        )
        edge_success = edge_error <= edge_sigma
        force_success = (force_error <= force_sigma) & (over_force <= 0.0)
        if not force_enabled:
            force_success = torch.zeros_like(force_success)
        edge_success_mean = float(edge_success.float().mean().item())
        force_success_mean = float(force_success.float().mean().item())
        self._update_contact_reward_curricula(
            edge_success_mean=edge_success_mean,
            force_success_mean=force_success_mean,
            force_enabled=force_enabled,
        )

        self.extras["scrape_pose/edge_contact_rew_mean"] = edge_rew.mean()
        self.extras["scrape_pose/edge_contact_error_mean"] = edge_error.mean()
        self.extras["scrape_pose/edge_contact_reward_weight"] = float(edge_weight)
        self.extras["scrape_pose/edge_contact_sigma"] = edge_sigma
        self.extras["scrape_pose/contact_force_rew_mean"] = force_rew.mean()
        self.extras["scrape_pose/contact_force_weight"] = float(force_weight)
        self.extras["scrape_pose/contact_force_sigma"] = force_sigma
        self.extras["scrape_pose/contact_force_raw_mean"] = (
            self._scrape_table_normal_force_raw.mean()
        )
        self.extras["scrape_pose/contact_force_mean"] = normal_force.mean()
        self.extras["scrape_pose/contact_force_filter_abs_delta_mean"] = torch.abs(
            self._scrape_table_normal_force_raw - normal_force
        ).mean()
        self.extras["scrape_pose/contact_force_filter_alpha"] = float(
            self.cfg.contact_force_filter_alpha
        )
        self.extras["scrape_pose/contact_force_target_mean"] = target_force.mean()
        self.extras["scrape_pose/contact_force_target_min"] = target_force.min()
        self.extras["scrape_pose/contact_force_target_max"] = target_force.max()
        self.extras["scrape_pose/contact_force_error_mean"] = force_error.mean()
        self.extras["scrape_pose/contact_over_force_mean"] = over_force.mean()
        self.extras["curriculum/edge_contact_threshold"] = float(
            self._edge_contact_sigma()
        )
        self.extras["curriculum/edge_contact_success_mean"] = float(
            self._edge_contact_curriculum_success_mean
        )
        self.extras["curriculum/contact_force_threshold"] = float(
            self._contact_force_sigma()
        )
        self.extras["curriculum/contact_force_success_mean"] = float(
            self._contact_force_curriculum_success_mean
        )
        if self._tool_table_contact_sensor_failed:
            raise RuntimeError(
                "tool-table ContactSensor previously failed: "
                f"{self._tool_table_contact_sensor_error}"
            )
        self.extras["scrape_pose/edge_anchor_z_mean"] = self._scrape_edge_anchor_w[
            :, 2
        ].mean()
        self.extras["scrape_pose/table_pitch_roll_range_deg"] = float(
            self.cfg.table_pitch_roll_range_deg
        )
        log_step_metrics(self)
        return reward


__all__ = ["SimToolRealTacMapScrapePoseEnv", "SimToolRealTacMapScrapePoseEnvCfg"]
