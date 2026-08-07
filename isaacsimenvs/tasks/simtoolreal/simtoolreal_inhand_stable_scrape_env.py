"""Post-grasp stable scraping initialized from verified compliant grasps."""

from __future__ import annotations

import math

import torch
from isaaclab.utils.math import quat_from_angle_axis, quat_mul

from .simtoolreal_stable_scrape_env import SimToolRealStableScrapeEnv
from .simtoolreal_tacmap_env_cfg import SimToolRealInHandStableScrapeEnvCfg
from .utils.inhand_grasp_bank import (
    collision_box_corners,
    load_grasp_bank,
    sha256_file,
    table_root_z_for_lowest_clearance,
)
from .utils.scrape_pose_utils import (
    TABLE_HALF_HEIGHT,
    edge_contact_points_w,
    quat_apply_wxyz,
    sample_edge_contact_goal_pose,
    table_top_state,
)
from .utils.stable_scrape_utils import CONTACT_APPROACH_PHASE


class SimToolRealInHandStableScrapeEnv(SimToolRealStableScrapeEnv):
    """Train only the compliant post-grasp approach and scrape behavior."""

    cfg: SimToolRealInHandStableScrapeEnvCfg

    def __init__(self, cfg: SimToolRealInHandStableScrapeEnvCfg, render_mode=None, **kwargs):
        self._validate_inhand_cfg(cfg)
        self._inhand_bank_payload = load_grasp_bank(
            cfg.grasp_bank_path, minimum_entries=int(cfg.grasp_bank_min_entries)
        )
        checkpoint_hash = sha256_file(cfg.grasp_bank_source_checkpoint_path)
        expected_checkpoint_hash = str(
            self._inhand_bank_payload["source_checkpoint_sha256"]
        )
        if checkpoint_hash != expected_checkpoint_hash:
            raise RuntimeError(
                "in-hand grasp bank checkpoint hash does not match the configured source: "
                f"bank={expected_checkpoint_hash}, configured={checkpoint_hash}"
            )
        self._inhand_ready = False
        super().__init__(cfg, render_mode, **kwargs)
        actual_asset_hash = sha256_file(self._object_urdf_paths[0])
        expected_asset_hash = str(self._inhand_bank_payload["asset_sha256"])
        if actual_asset_hash != expected_asset_hash:
            raise RuntimeError(
                "in-hand grasp bank asset hash does not match the spawned eraser: "
                f"bank={expected_asset_hash}, spawned={actual_asset_hash}"
            )
        self._materialize_grasp_bank()
        self._inhand_curriculum_stage = 0
        self._inhand_curriculum_updates = 0
        self._inhand_reset_bank_index = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._inhand_initial_clearance = torch.zeros(self.num_envs, device=self.device)
        self._inhand_initial_min_box_clearance = torch.zeros(
            self.num_envs, device=self.device
        )
        self._inhand_target_min_box_clearance = torch.zeros(
            self.num_envs, device=self.device
        )
        self._inhand_target_rotation_error = torch.zeros(
            self.num_envs, device=self.device
        )
        self._inhand_expected_support = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._inhand_ready = True
        self._reset_idx(torch.arange(self.num_envs, device=self.device))

    @staticmethod
    def _validate_inhand_cfg(cfg: SimToolRealInHandStableScrapeEnvCfg) -> None:
        clearances = tuple(cfg.inhand_clearance_stages_m)
        angles = tuple(cfg.inhand_table_angle_stages_deg)
        if not clearances or len(clearances) != len(angles):
            raise ValueError("in-hand clearance and angle curricula must have equal nonzero length")
        for low, high in clearances:
            if float(low) <= 0.0 or float(high) < float(low):
                raise ValueError("in-hand clearance ranges must be positive and ordered")
        if any(float(angle) < 0.0 or float(angle) >= 45.0 for angle in angles):
            raise ValueError("in-hand table angles must be in [0, 45) degrees")
        if int(cfg.grasp_bank_min_entries) <= 0:
            raise ValueError("grasp_bank_min_entries must be positive")
        for name in (
            "inhand_target_yaw_delta_deg", "inhand_target_tilt_delta_deg",
            "inhand_target_max_rotation_deg",
        ):
            if float(getattr(cfg, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.0 < float(cfg.inhand_target_max_rotation_deg) < 180.0:
            raise ValueError("inhand_target_max_rotation_deg must be in (0, 180)")
        if int(cfg.inhand_target_sampling_attempts) <= 0:
            raise ValueError("inhand_target_sampling_attempts must be positive")

    def _materialize_grasp_bank(self) -> None:
        entries = self._inhand_bank_payload["entries"]
        tensor_fields = {
            "joint_pos_canonical": "_inhand_bank_joint_pos",
            "joint_targets_canonical": "_inhand_bank_joint_targets",
            "last_action_canonical": "_inhand_bank_last_action",
            "object_pos_local": "_inhand_bank_object_pos",
            "object_quat_wxyz": "_inhand_bank_object_quat",
            "palm_to_tool_pos": "_inhand_bank_relative_pos",
            "palm_to_tool_quat_wxyz": "_inhand_bank_relative_quat",
        }
        for source, destination in tensor_fields.items():
            setattr(
                self,
                destination,
                torch.tensor(
                    [entry[source] for entry in entries],
                    dtype=torch.float32,
                    device=self.device,
                ),
            )
        self._inhand_bank_support = torch.tensor(
            [entry["verification"]["support_count"] for entry in entries],
            dtype=torch.long,
            device=self.device,
        )
        self._inhand_bank_reference_yaw = torch.tensor(
            [entry["reference_edge_yaw_rad"] for entry in entries],
            dtype=torch.float32,
            device=self.device,
        )
        self._inhand_bank_reference_tilt = torch.tensor(
            [entry["reference_edge_tilt_rad"] for entry in entries],
            dtype=torch.float32,
            device=self.device,
        )
        self._inhand_bank_tactile_fingers = torch.tensor(
            [entry["verification"]["tactile_finger_count_min"] for entry in entries],
            dtype=torch.long,
            device=self.device,
        )
        self._inhand_bank_size = len(entries)

        lower = self._joint_lower_canon.unsqueeze(0)
        upper = self._joint_upper_canon.unsqueeze(0)
        tolerance = float(self.cfg.grasp_bank_joint_limit_tolerance_rad)
        below = (lower - self._inhand_bank_joint_pos).clamp_min(0.0)
        above = (self._inhand_bank_joint_pos - upper).clamp_min(0.0)
        maximum_violation = torch.maximum(below, above).max()
        if float(maximum_violation.item()) > tolerance:
            raise RuntimeError(
                "grasp bank joint-limit violation exceeds configured tolerance: "
                f"maximum={float(maximum_violation.item()):.7f} rad, "
                f"tolerance={tolerance:.7f} rad"
            )

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)
        if getattr(self, "_inhand_ready", False):
            self._restore_inhand_state(env_ids)

    def _sample_table_quaternion(self, count: int) -> torch.Tensor:
        angle_deg = float(
            self.cfg.inhand_table_angle_stages_deg[self._inhand_curriculum_stage]
        )
        quat = torch.zeros(count, 4, device=self.device)
        quat[:, 0] = 1.0
        if angle_deg == 0.0:
            return quat
        angles = torch.empty(count, 2, device=self.device).uniform_(
            -math.radians(angle_deg), math.radians(angle_deg)
        )
        x_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(count, -1)
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(count, -1)
        return quat_mul(
            quat_from_angle_axis(angles[:, 1], y_axis),
            quat_from_angle_axis(angles[:, 0], x_axis),
        )

    def _sample_clearance(self, count: int) -> torch.Tensor:
        low, high = self.cfg.inhand_clearance_stages_m[
            self._inhand_curriculum_stage
        ]
        if float(low) == float(high):
            return torch.full((count,), float(low), device=self.device)
        return torch.empty(count, device=self.device).uniform_(float(low), float(high))

    def _box_signed_clearance(
        self,
        env_ids: torch.Tensor,
        object_pos: torch.Tensor,
        object_quat: torch.Tensor,
        table_pos: torch.Tensor,
        table_quat: torch.Tensor,
    ) -> torch.Tensor:
        corners = collision_box_corners(self._scrape_collision_bounds_per_env[env_ids])
        count = env_ids.numel()
        corners_w = object_pos.unsqueeze(1) + quat_apply_wxyz(
            object_quat.unsqueeze(1).expand(-1, 8, -1).reshape(-1, 4),
            corners.reshape(-1, 3),
        ).reshape(count, 8, 3)
        table_top, normal = table_top_state(table_pos, table_quat)
        return ((corners_w - table_top.unsqueeze(1)) * normal.unsqueeze(1)).sum(-1).min(-1).values

    @staticmethod
    def _quaternion_error_deg(actual: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        alignment = torch.abs((actual * target).sum(dim=-1)).clamp(0.0, 1.0)
        return torch.rad2deg(2.0 * torch.acos(alignment))

    def _sample_nearby_contact_target(
        self,
        env_ids: torch.Tensor,
        bank_ids: torch.Tensor,
        object_quat: torch.Tensor,
        table_pos: torch.Tensor,
        table_quat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        count = env_ids.numel()
        target_pos = torch.zeros(count, 3, device=self.device)
        target_quat = torch.zeros(count, 4, device=self.device)
        target_anchor = torch.zeros(count, 3, device=self.device)
        edge_yaw = torch.zeros(count, device=self.device)
        rotation_error = torch.full((count,), float("inf"), device=self.device)
        pending = torch.ones(count, dtype=torch.bool, device=self.device)
        yaw_delta = math.radians(float(self.cfg.inhand_target_yaw_delta_deg))
        tilt_delta = math.radians(float(self.cfg.inhand_target_tilt_delta_deg))
        tilt_low = math.radians(float(self.cfg.edge_tilt_range_deg[0]))
        tilt_high = math.radians(float(self.cfg.edge_tilt_range_deg[1]))

        for _ in range(int(self.cfg.inhand_target_sampling_attempts)):
            local_ids = pending.nonzero(as_tuple=False).squeeze(-1)
            if local_ids.numel() == 0:
                break
            reference_yaw = self._inhand_bank_reference_yaw[bank_ids[local_ids]]
            reference_tilt = self._inhand_bank_reference_tilt[bank_ids[local_ids]]
            sampled_yaw = reference_yaw + torch.empty_like(reference_yaw).uniform_(
                -yaw_delta, yaw_delta
            )
            sampled_tilt = (
                reference_tilt
                + torch.empty_like(reference_tilt).uniform_(-tilt_delta, tilt_delta)
            ).clamp(tilt_low, tilt_high)
            pos, quat, anchor, yaw = sample_edge_contact_goal_pose(
                table_pos_w=table_pos[local_ids],
                table_quat_wxyz=table_quat[local_ids],
                x_tip=self._scrape_x_tip_per_env[env_ids[local_ids]],
                y_center=self._scrape_y_center_per_env[env_ids[local_ids]],
                z_contact=self._scrape_z_contact_per_env[env_ids[local_ids]],
                xy_half_range=tuple(float(v) for v in self.cfg.edge_contact_xy_range_m),
                edge_yaw_range_rad=0.0,
                tilt_range_rad=(tilt_low, tilt_high),
                device=self.device,
                edge_yaw=sampled_yaw,
                edge_tilt=sampled_tilt,
            )
            error = self._quaternion_error_deg(object_quat[local_ids], quat)
            accepted = error <= float(self.cfg.inhand_target_max_rotation_deg)
            accepted_ids = local_ids[accepted]
            target_pos[accepted_ids] = pos[accepted]
            target_quat[accepted_ids] = quat[accepted]
            target_anchor[accepted_ids] = anchor[accepted]
            edge_yaw[accepted_ids] = yaw[accepted]
            rotation_error[accepted_ids] = error[accepted]
            pending[accepted_ids] = False

        if bool(pending.any()):
            failed_count = int(pending.sum().item())
            raise RuntimeError(
                "failed to sample an edge-contact target near the initial grasp "
                f"for {failed_count}/{count} environments after "
                f"{self.cfg.inhand_target_sampling_attempts} attempts"
            )
        return target_pos, target_quat, target_anchor, edge_yaw, rotation_error

    def _restore_inhand_state(
        self, env_ids: torch.Tensor, bank_ids: torch.Tensor | None = None
    ) -> None:
        count = env_ids.numel()
        if bank_ids is None:
            bank_ids = torch.randint(
                0, self._inhand_bank_size, (count,), device=self.device
            )
        else:
            bank_ids = torch.as_tensor(bank_ids, device=self.device, dtype=torch.long)
            if bank_ids.shape != (count,):
                raise ValueError(
                    f"bank_ids must have shape ({count},), got {tuple(bank_ids.shape)}"
                )
            if bool(((bank_ids < 0) | (bank_ids >= self._inhand_bank_size)).any()):
                raise ValueError("bank_ids contains an out-of-range grasp-bank index")
        self._inhand_reset_bank_index[env_ids] = bank_ids
        joint_pos_canonical = self._inhand_bank_joint_pos[bank_ids]
        target_canonical = self._inhand_bank_joint_targets[bank_ids]
        joint_pos = joint_pos_canonical[:, self._perm_canon_to_lab]
        # This task starts from a static in-hand state. The bank records source
        # velocities for provenance, but independently replaying coupled robot
        # and free-tool velocities introduces an artificial relative impulse.
        joint_vel = torch.zeros_like(joint_pos)
        targets = target_canonical[:, self._perm_canon_to_lab]
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._cur_targets[env_ids] = targets
        self._prev_targets[env_ids] = targets
        self.robot.set_joint_position_target(targets, env_ids=env_ids)

        origins = self.scene.env_origins[env_ids]
        object_pos = self._inhand_bank_object_pos[bank_ids] + origins
        object_quat = self._inhand_bank_object_quat[bank_ids]
        object_velocity = torch.zeros(count, 6, device=self.device)
        self.object.write_root_pose_to_sim(
            torch.cat((object_pos, object_quat), dim=-1), env_ids=env_ids
        )
        self.object.write_root_velocity_to_sim(object_velocity, env_ids=env_ids)

        table_quat = self._sample_table_quaternion(count)
        local_normal = torch.zeros(count, 3, device=self.device)
        local_normal[:, 2] = 1.0
        normal = quat_apply_wxyz(table_quat, local_normal)
        clearance = self._sample_clearance(count)

        # Acquisition grasps have arbitrary tool orientation, so place the
        # table below the lowest collision point rather than below the scrape edge.
        corners = collision_box_corners(self._scrape_collision_bounds_per_env[env_ids])
        corners_w = object_pos.unsqueeze(1) + quat_apply_wxyz(
            object_quat.unsqueeze(1).expand(-1, 8, -1).reshape(-1, 4),
            corners.reshape(-1, 3),
        ).reshape(count, 8, 3)
        table_local_z = table_root_z_for_lowest_clearance(
            corners_w,
            normal,
            origins,
            clearance,
            table_half_height_m=TABLE_HALF_HEIGHT,
        )
        table_pos = origins.clone()
        table_pos[:, 2] += table_local_z
        self.table.write_root_pose_to_sim(
            torch.cat((table_pos, table_quat), dim=-1), env_ids=env_ids
        )
        self._table_z_per_env[env_ids] = table_local_z
        self._table_quat_wxyz_per_env[env_ids] = table_quat

        (
            target_pos,
            target_quat,
            target_anchor,
            edge_yaw,
            target_rotation_error,
        ) = self._sample_nearby_contact_target(
            env_ids, bank_ids, object_quat, table_pos, table_quat
        )
        target_edge_points = edge_contact_points_w(
            target_pos,
            target_quat,
            self._scrape_x_tip_per_env[env_ids],
            self._scrape_y_min_per_env[env_ids],
            self._scrape_y_max_per_env[env_ids],
            self._scrape_z_contact_per_env[env_ids],
        )
        initial_min = self._box_signed_clearance(
            env_ids, object_pos, object_quat, table_pos, table_quat
        )
        target_min = self._box_signed_clearance(
            env_ids, target_pos, target_quat, table_pos, table_quat
        )
        tolerance = float(self.cfg.inhand_reset_penetration_tolerance_m)
        if bool((initial_min < -tolerance).any()):
            bad = float(initial_min.min().item())
            raise RuntimeError(f"in-hand reset penetrates table by {-bad:.6f} m")
        if bool((target_min < -tolerance).any()):
            bad = float(target_min.min().item())
            raise RuntimeError(f"in-hand contact target penetrates table by {-bad:.6f} m")

        self._stable_phase[env_ids] = CONTACT_APPROACH_PHASE
        self._stable_phase_steps[env_ids] = 0
        self._stable_contact_goal_pos_w[env_ids] = target_pos
        self._stable_contact_goal_quat_w[env_ids] = target_quat
        self._stable_contact_anchor_w[env_ids] = target_anchor
        edge_direction = torch.nn.functional.normalize(
            target_edge_points[:, 2] - target_edge_points[:, 0], dim=-1
        )
        self._stable_path_direction_w[env_ids] = torch.nn.functional.normalize(
            torch.cross(edge_direction, normal, dim=-1), dim=-1
        )
        self._stable_path_offset[env_ids] = 0.0
        self._stable_target_velocity_w[env_ids] = 0.0
        self._scrape_edge_yaw[env_ids] = edge_yaw
        self._write_goal(env_ids, target_pos, target_quat, target_anchor)

        relative_pos = self._inhand_bank_relative_pos[bank_ids]
        relative_quat = self._inhand_bank_relative_quat[bank_ids]
        self._stable_relative_pos[env_ids] = relative_pos
        self._stable_relative_quat[env_ids] = relative_quat
        self._stable_grasp_reference_pos[env_ids] = relative_pos
        self._stable_grasp_reference_quat[env_ids] = relative_quat
        self._stable_grasp_position_error[env_ids] = 0.0
        self._stable_grasp_rotation_error_deg[env_ids] = 0.0
        self._stable_grasp_retained[env_ids] = True
        self._stable_prev_relative_pos[env_ids] = relative_pos
        self._stable_prev_relative_quat[env_ids] = relative_quat
        self._stable_prev_tool_velocity[env_ids] = object_velocity[:, :3]
        self._stable_previous_action[env_ids] = self._inhand_bank_last_action[bank_ids]
        self._stable_support_count[env_ids] = self._inhand_bank_support[bank_ids]
        self._inhand_expected_support[env_ids] = self._inhand_bank_support[bank_ids]
        self._inhand_initial_clearance[env_ids] = clearance
        self._inhand_initial_min_box_clearance[env_ids] = initial_min
        self._inhand_target_min_box_clearance[env_ids] = target_min
        self._inhand_target_rotation_error[env_ids] = target_rotation_error
        self._object_init_z[env_ids] = self._inhand_bank_object_pos[bank_ids, 2]
        self._lifted_object[env_ids] = True
        self._reset_contact_force_filter(env_ids)

    def _update_curriculum(self, eligible: torch.Tensor, success: torch.Tensor) -> None:
        count = int(eligible.sum().item())
        self._stable_curriculum_success = (
            float(success[eligible].float().mean().item()) if count else 0.0
        )
        interval = int(self.cfg.termination.tolerance_curriculum_interval)
        if self._frame_counter - self._stable_last_curriculum_update < interval:
            return
        if count < int(self.cfg.inhand_curriculum_min_eligible_count):
            return
        if self._stable_curriculum_success < float(
            self.cfg.inhand_curriculum_success_threshold
        ):
            return
        new_sigma = max(
            self._stable_pose_sigma * float(self.cfg.stable_curriculum_increment),
            float(self.cfg.pose_tracking_sigma_target_m),
        )
        if new_sigma != self._stable_pose_sigma:
            self._stable_pose_sigma = new_sigma
            self._stable_curriculum_updates += 1
        last_stage = len(self.cfg.inhand_table_angle_stages_deg) - 1
        if self._inhand_curriculum_stage < last_stage:
            self._inhand_curriculum_stage += 1
            self._inhand_curriculum_updates += 1
        self._stable_last_curriculum_update = self._frame_counter

    def _get_rewards(self) -> torch.Tensor:
        reward = super()._get_rewards()
        low, high = self.cfg.inhand_clearance_stages_m[
            self._inhand_curriculum_stage
        ]
        self.extras.update({
            "inhand/curriculum_stage": self._inhand_curriculum_stage,
            "inhand/curriculum_updates": self._inhand_curriculum_updates,
            "inhand/table_angle_limit_deg": self.cfg.inhand_table_angle_stages_deg[
                self._inhand_curriculum_stage
            ],
            "inhand/clearance_min_m": low,
            "inhand/clearance_max_m": high,
            "inhand/reset_bank_index_mean": self._inhand_reset_bank_index.float().mean(),
            "inhand/expected_support_mean": self._inhand_expected_support.float().mean(),
            "inhand/bank_tactile_fingers_mean": self._inhand_bank_tactile_fingers[
                self._inhand_reset_bank_index
            ].float().mean(),
            "inhand/target_rotation_error_mean_deg": self._inhand_target_rotation_error.mean(),
            "inhand/target_rotation_error_max_deg": self._inhand_target_rotation_error.max(),
            "inhand/initial_box_clearance_min": self._inhand_initial_min_box_clearance.min(),
            "inhand/target_box_clearance_min": self._inhand_target_min_box_clearance.min(),
        })
        return reward


__all__ = ["SimToolRealInHandStableScrapeEnv", "SimToolRealInHandStableScrapeEnvCfg"]
