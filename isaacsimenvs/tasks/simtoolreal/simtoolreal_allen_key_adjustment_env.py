"""Palm-supported Allen-key adjustment around an engaged screw axis."""

from __future__ import annotations

import math

import torch
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils.math import quat_apply, quat_from_angle_axis, quat_inv, quat_mul

from .simtoolreal_inhand_adjustment_env import SimToolRealInHandAdjustmentEnv
from .simtoolreal_tacmap_env_cfg import SimToolRealAllenKeyAdjustmentEnvCfg
from .utils.adjustment_utils import (
    ALLEN_WORKSPACE_TIERS,
    allen_pair_curriculum_mask,
    allen_workspace_sampling_weights,
    palm_keypoint_error,
    screw_axis_orbit_errors,
)
from .utils.logging_utils import log_step_metrics
from .utils.obs_utils import compute_intermediate_values
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
            (
                0.264, 0.06, float(cfg.allen_handle_across_flats_m),
                *cfg.allen_screw_axis_tool, *cfg.allen_screw_pivot_tool_m,
            ),
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
        self._allen_fixture_workpiece_pos = torch.zeros(n, 3, device=device)
        self._allen_fixture_workpiece_quat = torch.zeros(n, 4, device=device)
        self._allen_fixture_workpiece_quat[:, 0] = 1.0
        self._allen_reset_yaw_rad = torch.zeros(n, device=device)
        self._allen_target_bank_index = torch.zeros(n, dtype=torch.long, device=device)
        self._allen_source_workspace_tier = torch.zeros(n, dtype=torch.long, device=device)
        self._allen_target_quality_improvement = torch.zeros(n, device=device)
        self._allen_sampled_pair_translation_m = torch.zeros(n, device=device)
        self._allen_sampled_pair_rotation_deg = torch.zeros(n, device=device)
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
        if any(not math.isfinite(value) or not 0.0 <= value <= 90.0 for value in yaw_ranges):
            raise ValueError("Allen-key reset yaw ranges must be finite and in [0, 90]")
        translation_range = tuple(
            float(value) for value in cfg.allen_target_pair_translation_range_m
        )
        rotation_range = tuple(
            float(value) for value in cfg.allen_target_pair_rotation_range_deg
        )
        for name, values, allow_zero in (
            ("translation", translation_range, True),
            ("rotation", rotation_range, False),
        ):
            if (
                len(values) != 2 or not all(math.isfinite(value) for value in values)
                or values[0] < 0.0 or (not allow_zero and values[0] == 0.0)
                or values[1] <= values[0]
            ):
                raise ValueError(f"Allen-key target pair {name} range is invalid")
        sigma = tuple(float(value) for value in cfg.allen_pose_sigma_stages_m)
        if len(sigma) != len(cfg.adjustment_target_rotation_deg):
            raise ValueError("Allen-key pose sigma curriculum length is inconsistent")
        if any(value <= 0.0 or not math.isfinite(value) for value in sigma):
            raise ValueError("Allen-key pose sigmas must be finite and positive")
        if any(later >= earlier for earlier, later in zip(sigma, sigma[1:])):
            raise ValueError("Allen-key pose sigma curriculum must strictly tighten")
        stage_count = len(cfg.adjustment_target_rotation_deg)
        for name in (
            "allen_workspace_tier_probabilities_stages",
            "allen_pair_max_translation_stages_m",
            "allen_pair_max_rotation_stages_deg",
        ):
            if len(getattr(cfg, name)) != stage_count:
                raise ValueError(f"{name} must have one value per curriculum stage")
        for probabilities in cfg.allen_workspace_tier_probabilities_stages:
            if (
                len(probabilities) != len(ALLEN_WORKSPACE_TIERS)
                or any(not math.isfinite(float(value)) or float(value) < 0.0 for value in probabilities)
                or not math.isclose(sum(float(value) for value in probabilities), 1.0, abs_tol=1.0e-6)
            ):
                raise ValueError(
                    "Allen-key workspace tier probabilities must be non-negative and sum to one"
                )
        for name in (
            "allen_pair_max_translation_stages_m",
            "allen_pair_max_rotation_stages_deg",
        ):
            values = tuple(float(value) for value in getattr(cfg, name))
            if any(not math.isfinite(value) or value <= 0.0 for value in values):
                raise ValueError(f"{name} values must be finite and positive")
            if any(later < earlier for earlier, later in zip(values, values[1:])):
                raise ValueError(f"{name} must be non-decreasing")
        if (
            not math.isfinite(float(cfg.allen_target_min_quality_improvement))
            or float(cfg.allen_target_min_quality_improvement) < 0.0
        ):
            raise ValueError("Allen-key target quality improvement must be finite and non-negative")
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
            "allen_hidden_table_offset_m",
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

    def _materialize_grasp_bank(self) -> None:
        """Materialize bank tensors and require a useful target for every start."""
        super()._materialize_grasp_bank()
        relative_pos = self._inhand_bank_relative_pos
        relative_quat = self._inhand_bank_relative_quat
        tool_to_palm_quat = quat_inv(relative_quat)
        tool_to_palm_pos = quat_apply(tool_to_palm_quat, -relative_pos)
        translation = torch.cdist(tool_to_palm_pos, tool_to_palm_pos)
        alignment = torch.abs(relative_quat @ relative_quat.T).clamp(0.0, 1.0)
        rotation_deg = torch.rad2deg(2.0 * torch.acos(alignment))
        translation_low, translation_high = (
            float(value) for value in self.cfg.allen_target_pair_translation_range_m
        )
        rotation_low, rotation_high = (
            float(value) for value in self.cfg.allen_target_pair_rotation_range_deg
        )
        distinct = ~torch.eye(
            self._inhand_bank_size, dtype=torch.bool, device=self.device
        )
        meaningful = (translation >= translation_low) | (rotation_deg >= rotation_low)
        bounded = (translation <= translation_high) & (rotation_deg <= rotation_high)
        self._allen_target_pair_valid = distinct & meaningful & bounded
        entries = self._inhand_bank_payload.get("entries", [])
        if len(entries) != self._inhand_bank_size:
            raise RuntimeError("Allen-key bank entry count changed during materialization")
        declared = torch.zeros_like(self._allen_target_pair_valid)
        self._allen_target_arm_joint_0 = torch.full(
            (self._inhand_bank_size, self._inhand_bank_size),
            float("nan"),
            device=self.device,
        )
        graph_present = all(
            entry.get("verification", {}).get("valid_target_ids") is not None
            and entry.get("verification", {}).get("target_arm_joint_0_rad") is not None
            for entry in entries
        )
        if not graph_present and not bool(self.cfg.allen_require_valid_target_pairs):
            # Provisional banks have no screened graph yet. Use each target
            # entry's actual first arm joint for the reset-yaw intersection;
            # zero is not generally inside the feasible interval.
            self._allen_target_arm_joint_0.copy_(
                self._inhand_bank_joint_pos[:, 0].unsqueeze(0).expand(
                    self._inhand_bank_size, -1
                )
            )
            declared.fill_(True)
        for source_id, entry in enumerate(entries):
            verification = entry.get("verification", {})
            target_ids = verification.get("valid_target_ids")
            target_joint_0 = verification.get("target_arm_joint_0_rad")
            if target_ids is None or target_joint_0 is None:
                if not graph_present and not bool(self.cfg.allen_require_valid_target_pairs):
                    continue
                raise RuntimeError(
                    "Allen-key grasp bank is missing its IK-valid target graph; "
                    "regenerate it with adapt_allen_key_grasp_bank.py"
                )
            if len(target_ids) != len(target_joint_0):
                raise RuntimeError(
                    f"Allen-key target graph row {source_id} has mismatched fields"
                )
            for target_id, joint_0 in zip(target_ids, target_joint_0, strict=True):
                target_id = int(target_id)
                joint_0 = float(joint_0)
                if not 0 <= target_id < self._inhand_bank_size or not math.isfinite(joint_0):
                    raise RuntimeError(
                        f"Allen-key target graph row {source_id} is invalid"
                    )
                declared[source_id, target_id] = True
                self._allen_target_arm_joint_0[source_id, target_id] = joint_0
        self._allen_target_pair_valid &= declared
        candidates_per_start = self._allen_target_pair_valid.sum(dim=-1)
        target_only = torch.tensor(
            [
                bool(entry.get("verification", {}).get("target_only", False))
                for entry in entries
            ],
            dtype=torch.bool,
            device=self.device,
        )
        invalid_start = (candidates_per_start == 0) & ~target_only
        if bool(self.cfg.allen_require_valid_target_pairs) and bool(invalid_start.any()):
            invalid = torch.nonzero(
                invalid_start, as_tuple=False
            ).squeeze(-1).tolist()
            raise RuntimeError(
                "Allen-key grasp bank has no valid manipulation target for start entries "
                f"{invalid}; regenerate the bank instead of using a synthetic fallback"
            )
        self._allen_pair_translation_m = translation
        self._allen_pair_rotation_deg = rotation_deg
        workspace_tiers = []
        functional_quality = []
        for index, entry in enumerate(entries):
            verification = entry.get("verification", {})
            tier = verification.get("rollout_workspace_tier_id")
            quality = verification.get("rollout_functional_quality")
            if bool(self.cfg.allen_workspace_conditioned_sampling) and (
                tier is None or quality is None
            ):
                raise RuntimeError(
                    f"Allen-key rollout bank entry {index} lacks workspace/quality metadata"
                )
            workspace_tiers.append(0 if tier is None else int(tier))
            functional_quality.append(0.0 if quality is None else float(quality))
        self._allen_bank_workspace_tier = torch.tensor(
            workspace_tiers, dtype=torch.long, device=self.device
        )
        self._allen_bank_functional_quality = torch.tensor(
            functional_quality, dtype=torch.float32, device=self.device
        )
        if bool(((self._allen_bank_workspace_tier < 0) | (
            self._allen_bank_workspace_tier >= len(ALLEN_WORKSPACE_TIERS)
        )).any()):
            raise RuntimeError("Allen-key bank contains an invalid workspace tier")
        if not bool(torch.isfinite(self._allen_bank_functional_quality).all()):
            raise RuntimeError("Allen-key bank contains non-finite functional quality")

    def _active_target_graph(self) -> torch.Tensor:
        graph = self._allen_target_pair_valid
        if not (
            bool(self.cfg.allen_workspace_conditioned_sampling)
            and getattr(self, "_allen_ready", False)
        ):
            return graph
        stage = self._adjustment_curriculum_stage
        graph = allen_pair_curriculum_mask(
            graph,
            self._allen_pair_translation_m,
            self._allen_pair_rotation_deg,
            maximum_translation_m=float(
                self.cfg.allen_pair_max_translation_stages_m[stage]
            ),
            maximum_rotation_deg=float(
                self.cfg.allen_pair_max_rotation_stages_deg[stage]
            ),
        )
        improvement = (
            self._allen_bank_functional_quality.unsqueeze(0)
            - self._allen_bank_functional_quality.unsqueeze(1)
        )
        return graph & (
            improvement >= float(self.cfg.allen_target_min_quality_improvement)
        )

    def _sample_workspace_source_ids(self, count: int) -> torch.Tensor:
        graph = self._active_target_graph()
        available = graph.any(dim=-1)
        available_ids = torch.nonzero(available, as_tuple=False).squeeze(-1)
        if available_ids.numel() == 0:
            raise RuntimeError(
                "Allen-key curriculum has no start with a valid improved target"
            )
        stage = int(getattr(self, "_adjustment_curriculum_stage", 0))
        probabilities = tuple(float(value) for value in (
            self.cfg.allen_workspace_tier_probabilities_stages[
                stage
            ]
        ))
        try:
            available_weights = allen_workspace_sampling_weights(
                self._allen_bank_workspace_tier[available_ids], probabilities
            )
        except ValueError as exc:
            raise RuntimeError(
                "Allen-key curriculum requests an unavailable workspace tier"
            ) from exc
        sampled = torch.multinomial(available_weights, count, replacement=True)
        return available_ids[sampled]

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
        source_ids = self._inhand_reset_bank_index[env_ids]
        graph = self._active_target_graph()
        weights = graph[source_ids].float()
        if bool(self.cfg.allen_workspace_conditioned_sampling) and getattr(
            self, "_allen_ready", False
        ):
            quality = self._allen_bank_functional_quality
            quality = quality - quality.min() + 1.0
            weights *= quality.unsqueeze(0)
        missing = weights.sum(dim=-1) == 0
        if bool(self.cfg.allen_require_valid_target_pairs) and bool(missing.any()):
            raise RuntimeError("Allen-key reset selected a start without a valid target")
        if bool(missing.any()):
            weights[missing, source_ids[missing]] = 1.0
        target_ids = torch.multinomial(weights, 1).squeeze(-1)
        if hasattr(self, "_allen_target_bank_index"):
            self._allen_target_bank_index[env_ids] = target_ids
        target_pos = self._inhand_bank_relative_pos[target_ids]
        target_quat = self._inhand_bank_relative_quat[target_ids]
        rotation = torch.deg2rad(self._allen_pair_rotation_deg[source_ids, target_ids])
        translation = self._allen_pair_translation_m[source_ids, target_ids]
        if hasattr(self, "_allen_source_workspace_tier"):
            self._allen_source_workspace_tier[env_ids] = (
                self._allen_bank_workspace_tier[source_ids]
            )
            self._allen_target_quality_improvement[env_ids] = (
                self._allen_bank_functional_quality[target_ids]
                - self._allen_bank_functional_quality[source_ids]
            )
            self._allen_sampled_pair_translation_m[env_ids] = translation
            self._allen_sampled_pair_rotation_deg[env_ids] = torch.rad2deg(rotation)
        return target_pos, target_quat, rotation, translation

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
        if (
            bank_ids is None
            and bool(self.cfg.allen_workspace_conditioned_sampling)
        ):
            bank_ids = self._sample_workspace_source_ids(env_ids.numel())
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
        table_pos = self.scene.env_origins[env_ids].clone()
        table_pos[:, 2] -= float(self.cfg.allen_hidden_table_offset_m)
        table_quat = torch.zeros(count, 4, device=self.device)
        table_quat[:, 0] = 1.0
        self.table.write_root_pose_to_sim(
            torch.cat((table_pos, table_quat), dim=-1), env_ids=env_ids
        )
        self._table_z_per_env[env_ids] = table_pos[:, 2] - self.scene.env_origins[env_ids, 2]
        self._table_quat_wxyz_per_env[env_ids] = table_quat
        if getattr(self, "_allen_ready", False):
            self._allen_hold_count[env_ids] = 0
            self._allen_succeeded[env_ids] = False
            self._allen_just_succeeded[env_ids] = False
            self._allen_previous_pose_potential[env_ids] = 0.0
            self._allen_fixture_tool_pos[env_ids] = self.object.data.root_pos_w[env_ids]
            self._allen_fixture_tool_quat[env_ids] = self.object.data.root_quat_w[env_ids]
            self._allen_fixture_workpiece_pos[env_ids] = workpiece_pos
            self._allen_fixture_workpiece_quat[env_ids] = workpiece_quat

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
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self.workpiece.write_root_pose_to_sim(
            torch.cat((
                self._allen_fixture_workpiece_pos,
                self._allen_fixture_workpiece_quat,
            ), dim=-1),
            env_ids=all_env_ids,
        )
        self.workpiece.write_root_velocity_to_sim(
            torch.zeros(self.num_envs, 6, device=self.device), env_ids=all_env_ids
        )
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
        source_ids = self._inhand_reset_bank_index[env_ids]
        target_ids = self._allen_target_bank_index[env_ids]
        target_joint_0 = self._allen_target_arm_joint_0[source_ids, target_ids]
        if not bool(torch.isfinite(target_joint_0).all()):
            raise RuntimeError("sampled Allen-key target has no finite arm-yaw solution")
        low = torch.maximum(
            torch.full_like(current_targets, -limit),
            self._arm_lower[env_ids, 0] - current_targets,
        )
        high = torch.minimum(
            torch.full_like(current_targets, limit),
            self._arm_upper[env_ids, 0] - current_targets,
        )
        low = torch.maximum(low, self._arm_lower[env_ids, 0] - target_joint_0)
        high = torch.minimum(high, self._arm_upper[env_ids, 0] - target_joint_0)
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
        geometric_support_quality = (
            self._stable_support_count.float() / 3.0
        ).clamp(0.0, 1.0)
        flexion_ids = self._allen_flexion_joint_ids
        limits = self.robot.data.joint_pos_limits[:, flexion_ids]
        closure = (
            (self.robot.data.joint_pos[:, flexion_ids] - limits[:, :, 0])
            / (limits[:, :, 1] - limits[:, :, 0]).clamp_min(1.0e-6)
        ).mean(-1).clamp(0.0, 1.0)
        self._allen_flexion_closure.copy_(closure)
        self._allen_contact_quality.copy_(
            0.35 * closure + 0.20 * fingertip_force_quality
            + 0.25 * self._allen_palm_contact.float()
            + 0.20 * geometric_support_quality
        )
        multi_contact_support = (
            self._allen_palm_contact
            | (self._allen_fingertip_contact_count >= 2)
            | (self._stable_support_count >= 3)
        )
        self._allen_final_grasp_valid.copy_(
            (closure >= float(self.cfg.allen_min_flexion_closure_fraction))
            & multi_contact_support
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
        # Pose writes in _apply_action happen before the final physics substep.
        # Reassert the kinematic fixture before reading state so contact impulses
        # cannot leak a one-substep socket/tool displacement into RL metrics.
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self.workpiece.write_root_pose_to_sim(
            torch.cat((
                self._allen_fixture_workpiece_pos,
                self._allen_fixture_workpiece_quat,
            ), dim=-1),
            env_ids=all_env_ids,
        )
        self.workpiece.write_root_velocity_to_sim(
            torch.zeros(self.num_envs, 6, device=self.device), env_ids=all_env_ids
        )
        release_start = int(self.cfg.allen_adjustment_steps) + int(
            self.cfg.allen_closure_steps
        )
        fixture_ids = torch.nonzero(
            self.episode_length_buf < release_start, as_tuple=False
        ).squeeze(-1)
        if fixture_ids.numel():
            self.object.write_root_pose_to_sim(
                torch.cat((
                    self._allen_fixture_tool_pos[fixture_ids],
                    self._allen_fixture_tool_quat[fixture_ids],
                ), dim=-1),
                env_ids=fixture_ids,
            )
            self.object.write_root_velocity_to_sim(
                torch.zeros(fixture_ids.numel(), 6, device=self.device),
                env_ids=fixture_ids,
            )
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
        adjustment_phase = self.episode_length_buf < closure_start
        closure_phase = closure_or_release & ~release_phase
        self._episode_cumulative_terms = {}
        for phase_name, phase_mask in (
            ("adjustment", adjustment_phase),
            ("closure", closure_phase),
            ("release", release_phase),
        ):
            self._episode_cumulative_terms[f"phase/{phase_name}_reward_sum"] = (
                reward * phase_mask.float()
            )
            self._episode_cumulative_terms[f"phase/{phase_name}_step_count"] = (
                phase_mask.float()
            )
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
            "allen/source_easy_ratio": (
                self._allen_source_workspace_tier == 0
            ).float().mean(),
            "allen/source_support_ratio": (
                self._allen_source_workspace_tier == 1
            ).float().mean(),
            "allen/source_broad_ratio": (
                self._allen_source_workspace_tier == 2
            ).float().mean(),
            "allen/target_quality_improvement_mean": (
                self._allen_target_quality_improvement.mean()
            ),
            "allen/sampled_pair_translation_mean_m": (
                self._allen_sampled_pair_translation_m.mean()
            ),
            "allen/sampled_pair_rotation_mean_deg": (
                self._allen_sampled_pair_rotation_deg.mean()
            ),
            "curriculum/allen_stage": self._adjustment_curriculum_stage,
            "curriculum/allen_success_mean": self._allen_curriculum_success_mean,
            "curriculum/allen_palm_pose_sigma_m": pose_sigma,
            "curriculum/allen_pair_max_translation_m": (
                self.cfg.allen_pair_max_translation_stages_m[
                    self._adjustment_curriculum_stage
                ]
            ),
            "curriculum/allen_pair_max_rotation_deg": (
                self.cfg.allen_pair_max_rotation_stages_deg[
                    self._adjustment_curriculum_stage
                ]
            ),
            "curriculum/allen_target_translation_min_m": (
                self.cfg.allen_target_pair_translation_range_m[0]
            ),
            "curriculum/allen_target_translation_max_m": (
                self.cfg.allen_target_pair_translation_range_m[1]
            ),
            "curriculum/allen_target_rotation_min_deg": (
                self.cfg.allen_target_pair_rotation_range_deg[0]
            ),
            "curriculum/allen_target_rotation_max_deg": (
                self.cfg.allen_target_pair_rotation_range_deg[1]
            ),
            "curriculum/allen_reset_yaw_range_deg": (
                self.cfg.allen_reset_yaw_range_stages_deg[
                    self._adjustment_curriculum_stage
                ]
            ),
        })
        log_step_metrics(self)
        return reward


__all__ = ["SimToolRealAllenKeyAdjustmentEnv"]
