"""Fixed-grasp, one-dimensional normal-force control task."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, Sdf, UsdPhysics

from .simtoolreal_scrape_pose_env import SimToolRealTacMapScrapePoseEnv
from .simtoolreal_tacmap_env_cfg import SimToolRealFixedGraspNormalForceEnvCfg
from .utils.contact_force_controllability import (
    damped_least_squares,
    quaternion_error_vector,
)
from .utils.scrape_pose_utils import contact_force_reward, table_top_state


_FORCE_OBS = (
    "fixed_force_target",
    "fixed_force_measured",
    "fixed_force_error_signed",
    "fixed_normal_offset",
    "fixed_normal_velocity",
    "fixed_prev_action",
)
_BLIND_OBS = (
    "fixed_force_target",
    "fixed_normal_offset",
    "fixed_normal_velocity",
    "fixed_prev_action",
)
_CRITIC_OBS = _FORCE_OBS


def load_fixed_grasp_spec(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Fixed-grasp specification does not exist: {path}. "
            "Run validate_contact_force_controllability.py with --fixed-grasp-export."
        )
    with path.open() as stream:
        spec = json.load(stream)
    required = {"tool_type", "joint_positions", "palm_to_tool_pos", "palm_to_tool_quat_wxyz"}
    missing = required - set(spec)
    if missing:
        raise ValueError(f"Fixed-grasp specification is missing fields: {sorted(missing)}")
    if not isinstance(spec["joint_positions"], dict) or len(spec["joint_positions"]) != 29:
        raise ValueError("fixed-grasp joint_positions must contain exactly 29 named joints")
    for name, size in (("palm_to_tool_pos", 3), ("palm_to_tool_quat_wxyz", 4)):
        values = spec[name]
        if not isinstance(values, list) or len(values) != size:
            raise ValueError(f"fixed-grasp {name} must contain {size} values")
        if not bool(torch.isfinite(torch.tensor(values, dtype=torch.float32)).all()):
            raise ValueError(f"fixed-grasp {name} contains NaN or Inf")
    return spec


def fixed_force_observation_lists(feedback_mode: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    mode = str(feedback_mode).lower()
    if mode == "force":
        policy = _FORCE_OBS
    elif mode == "blind":
        policy = _BLIND_OBS
    elif mode == "tactile":
        policy = _BLIND_OBS + ("tacmap",)
    else:
        raise ValueError(
            f"feedback_mode must be one of force/tactile/blind, got {feedback_mode!r}"
        )
    return policy, _CRITIC_OBS


def fixed_force_reward_terms(
    measured_force: torch.Tensor,
    target_force: torch.Tensor,
    action_delta: torch.Tensor,
    *,
    sigma: float,
    soft_force_limit: float,
    max_force: float,
    force_weight: float,
    quadratic_error_weight: float,
    quadratic_error_scale: float,
    over_force_weight: float,
    action_rate_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if float(quadratic_error_scale) <= 0.0:
        raise ValueError("quadratic_error_scale must be positive")
    score, _ = contact_force_reward(measured_force, target_force, sigma, max_force)
    soft_over_force = torch.clamp(measured_force - float(soft_force_limit), min=0.0)
    force_span = max(float(max_force) - float(soft_force_limit), 1.0e-6)
    force_rew = float(force_weight) * score
    quadratic_error_penalty = -float(quadratic_error_weight) * (
        (measured_force - target_force) / float(quadratic_error_scale)
    ).square()
    over_force_penalty = -float(over_force_weight) * (soft_over_force / force_span).square()
    action_rate_penalty = -float(action_rate_weight) * action_delta.square()
    total = (
        force_rew
        + quadratic_error_penalty
        + over_force_penalty
        + action_rate_penalty
    )
    return total, {
        "force_tracking_rew": force_rew,
        "quadratic_force_error_penalty": quadratic_error_penalty,
        "over_force_penalty": over_force_penalty,
        "action_rate_penalty": action_rate_penalty,
        "total_reward": total,
    }


def integrate_normal_action(
    offset: torch.Tensor,
    action: torch.Tensor,
    *,
    velocity_limit: float,
    step_dt: float,
    offset_min: float,
    offset_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Integrate normalized normal velocity without modifying either input."""
    velocity = action.clamp(-1.0, 1.0) * float(velocity_limit)
    next_offset = torch.clamp(
        offset + velocity * float(step_dt),
        float(offset_min),
        float(offset_max),
    )
    return next_offset, velocity


def pi_force_control(
    force_error: torch.Tensor,
    integral: torch.Tensor,
    *,
    step_dt: float,
    kp: float,
    ki: float,
    integral_limit: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute normalized PI action with conditional anti-windup."""
    if float(integral_limit) <= 0.0:
        raise ValueError("integral_limit must be positive")
    candidate = torch.clamp(
        integral + force_error * float(step_dt),
        -float(integral_limit),
        float(integral_limit),
    )
    candidate_action = float(kp) * force_error + float(ki) * candidate
    drives_further_into_saturation = (
        ((candidate_action > 1.0) & (force_error > 0.0))
        | ((candidate_action < -1.0) & (force_error < 0.0))
    )
    next_integral = torch.where(drives_further_into_saturation, integral, candidate)
    action = torch.clamp(
        float(kp) * force_error + float(ki) * next_integral, -1.0, 1.0
    )
    return action, next_integral


class SimToolRealFixedGraspNormalForceEnv(SimToolRealTacMapScrapePoseEnv):
    """Track randomized normal-force targets with one normal-velocity action."""

    cfg: SimToolRealFixedGraspNormalForceEnvCfg

    def __init__(
        self,
        cfg: SimToolRealFixedGraspNormalForceEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        self._fixed_grasp_spec = load_fixed_grasp_spec(cfg.fixed_grasp_spec_path)
        cfg.assets.handle_head_types = (str(self._fixed_grasp_spec["tool_type"]),)
        cfg.assets.num_assets_per_type = 1
        cfg.assets.shuffle_assets = False
        cfg.action_space = 1
        cfg.reset.table_reset_pitch_roll_range_deg = float(cfg.table_pitch_roll_range_deg)
        policy_fields, critic_fields = fixed_force_observation_lists(cfg.feedback_mode)
        cfg.obs.obs_list = policy_fields
        cfg.obs.state_list = critic_fields
        cfg.enable_vbts = str(cfg.feedback_mode).lower() == "tactile"
        self._fixed_joint_pos_by_name = {
            str(name): float(value)
            for name, value in self._fixed_grasp_spec["joint_positions"].items()
        }
        super().__init__(cfg, render_mode, **kwargs)

        joint_names = list(self.robot.data.joint_names)
        missing = set(joint_names) - set(self._fixed_joint_pos_by_name)
        extra = set(self._fixed_joint_pos_by_name) - set(joint_names)
        if missing or extra:
            raise RuntimeError(
                f"Fixed-grasp joint-name mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
            )
        fixed_joint_pos = torch.tensor(
            [self._fixed_joint_pos_by_name[name] for name in joint_names],
            device=self.device,
            dtype=torch.float32,
        )
        limits = self.robot.data.joint_pos_limits[0]
        fixed_joint_pos = torch.clamp(
            fixed_joint_pos,
            limits[:, 0] + 1.0e-5,
            limits[:, 1] - 1.0e-5,
        )
        self._fixed_robot_joint_pos = fixed_joint_pos.unsqueeze(0).expand(self.num_envs, -1).clone()
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self.robot.write_joint_state_to_sim(
            self._fixed_robot_joint_pos,
            torch.zeros_like(self._fixed_robot_joint_pos),
            env_ids=all_env_ids,
        )
        self._cur_targets.copy_(self._fixed_robot_joint_pos)
        self._prev_targets.copy_(self._fixed_robot_joint_pos)
        self.robot.set_joint_position_target(self._fixed_robot_joint_pos)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)
        self._fixed_palm_pos_w = self.robot.data.body_link_pos_w[
            :, self._palm_body_id
        ].clone()
        self._fixed_palm_quat_w = self.robot.data.body_link_quat_w[
            :, self._palm_body_id
        ].clone()
        self._write_fixed_tool_pose(all_env_ids)
        self.scene.write_data_to_sim()
        self._create_fixed_grasp_joints()
        self._normal_offset = torch.zeros(self.num_envs, device=self.device)
        self._normal_velocity = torch.zeros(self.num_envs, device=self.device)
        self._previous_normal_action = torch.zeros(self.num_envs, device=self.device)
        self._action_delta = torch.zeros(self.num_envs, device=self.device)
        self._episode_within_steps = torch.zeros(self.num_envs, device=self.device)
        self._episode_force_steps = torch.zeros(self.num_envs, device=self.device)
        self._fixed_grasp_drift = torch.zeros(self.num_envs, device=self.device)
        self._previous_filtered_force = torch.zeros(self.num_envs, device=self.device)
        self._force_derivative = torch.zeros(self.num_envs, device=self.device)
        self._reset_idx(all_env_ids)

    def _write_fixed_tool_pose(self, env_ids: torch.Tensor) -> None:
        n = env_ids.numel()
        rel_pos = torch.tensor(
            self._fixed_grasp_spec["palm_to_tool_pos"], device=self.device
        ).expand(n, -1)
        rel_quat = torch.tensor(
            self._fixed_grasp_spec["palm_to_tool_quat_wxyz"], device=self.device
        ).expand(n, -1)
        object_pos, object_quat = combine_frame_transforms(
            self.robot.data.body_link_pos_w[env_ids, self._palm_body_id],
            self.robot.data.body_link_quat_w[env_ids, self._palm_body_id],
            rel_pos,
            rel_quat,
        )
        self.object.write_root_pose_to_sim(
            torch.cat((object_pos, object_quat), dim=-1), env_ids=env_ids
        )
        self.object.write_root_velocity_to_sim(
            torch.zeros(n, 6, device=self.device), env_ids=env_ids
        )

    def _create_fixed_grasp_joints(self) -> None:
        stage = get_current_stage()
        pos = [float(v) for v in self._fixed_grasp_spec["palm_to_tool_pos"]]
        quat = [float(v) for v in self._fixed_grasp_spec["palm_to_tool_quat_wxyz"]]
        for env_id in range(self.cfg.scene.num_envs):
            root = f"/World/envs/env_{env_id}"
            palm_path = f"{root}/Robot/iiwa14_link_7"
            tool_path = f"{root}/Object/object_root"
            for body_path in (palm_path, tool_path):
                if not stage.GetPrimAtPath(body_path).IsValid():
                    raise RuntimeError(f"Fixed-grasp rigid body prim is missing: {body_path}")
            joint = UsdPhysics.FixedJoint.Define(stage, f"{root}/FixedGraspJoint")
            joint.CreateExcludeFromArticulationAttr().Set(True)
            joint.CreateBody0Rel().SetTargets([Sdf.Path(palm_path)])
            joint.CreateBody1Rel().SetTargets([Sdf.Path(tool_path)])
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*pos))
            joint.CreateLocalRot0Attr().Set(Gf.Quatf(quat[0], Gf.Vec3f(*quat[1:])))
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
            joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0)))

    def _reset_idx(self, env_ids) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        if not hasattr(self, "_fixed_robot_joint_pos"):
            return
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        joint_pos = self._fixed_robot_joint_pos[env_ids]
        self.robot.write_joint_state_to_sim(
            joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids
        )
        self._cur_targets[env_ids] = joint_pos
        self._prev_targets[env_ids] = joint_pos

        self._write_fixed_tool_pose(env_ids)
        self._normal_offset[env_ids] = 0.0
        self._normal_velocity[env_ids] = 0.0
        self._previous_normal_action[env_ids] = 0.0
        self._action_delta[env_ids] = 0.0
        self._episode_within_steps[env_ids] = 0.0
        self._episode_force_steps[env_ids] = 0.0
        self._previous_filtered_force[env_ids] = 0.0
        self._force_derivative[env_ids] = 0.0
        self._reset_contact_force_targets(env_ids)
        self._reset_contact_force_filter(env_ids)

    def _table_normal(self) -> torch.Tensor:
        table_quat = getattr(
            self, "_table_quat_wxyz_per_env", self.table.data.root_quat_w
        )
        _, normal = table_top_state(self.table.data.root_pos_w, table_quat)
        if not torch.isfinite(normal).all():
            raise RuntimeError("Table normal contains NaN or Inf.")
        return normal

    def _palm_jacobian(self) -> torch.Tensor:
        jacobians = self.robot.root_physx_view.get_jacobians()
        body_index = int(self._palm_body_id) - (1 if self.robot.is_fixed_base else 0)
        jacobian = jacobians[:, body_index, :, self._arm_joint_ids]
        expected = (self.num_envs, 6, len(self._arm_joint_ids))
        if jacobian.shape != expected or not torch.isfinite(jacobian).all():
            raise RuntimeError(
                f"Expected finite palm Jacobian {expected}, got {tuple(jacobian.shape)}"
            )
        return jacobian

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions.shape != (self.num_envs, 1):
            raise RuntimeError(
                f"Fixed-force action must have shape ({self.num_envs}, 1), got {tuple(actions.shape)}"
            )
        actions = actions.to(self.device).squeeze(-1).clamp(-1.0, 1.0)
        if not torch.isfinite(actions).all():
            raise RuntimeError("Fixed-force action contains NaN or Inf.")
        self._action_delta.copy_(actions - self._previous_normal_action)
        self._previous_normal_action.copy_(actions)
        next_offset, velocity = integrate_normal_action(
            self._normal_offset,
            actions,
            velocity_limit=float(self.cfg.normal_velocity_limit_mps),
            step_dt=float(self.step_dt),
            offset_min=float(self.cfg.normal_offset_min_m),
            offset_max=float(self.cfg.normal_offset_max_m),
        )
        self._normal_velocity.copy_(velocity)
        self._normal_offset.copy_(next_offset)

        palm_pos = self.robot.data.body_link_pos_w[:, self._palm_body_id]
        palm_quat = self.robot.data.body_link_quat_w[:, self._palm_body_id]
        target_pos = self._fixed_palm_pos_w - self._normal_offset.unsqueeze(-1) * self._table_normal()
        linear = float(self.cfg.dls_linear_gain) * (target_pos - palm_pos)
        angular = float(self.cfg.dls_angular_gain) * quaternion_error_vector(
            palm_quat, self._fixed_palm_quat_w
        )
        qdot = damped_least_squares(
            self._palm_jacobian(),
            torch.cat((linear, angular), dim=-1),
            float(self.cfg.dls_damping),
        ).clamp(
            -float(self.cfg.dls_joint_velocity_limit),
            float(self.cfg.dls_joint_velocity_limit),
        )
        if not torch.isfinite(qdot).all():
            raise RuntimeError("DLS joint velocity contains NaN or Inf.")
        arm_targets = self._prev_targets[:, self._arm_joint_ids] + qdot * float(self.step_dt)
        arm_targets = torch.clamp(arm_targets, self._arm_lower, self._arm_upper)
        freeze_mask = getattr(self, "_diagnostic_freeze_arm_targets", None)
        if freeze_mask is not None:
            if freeze_mask.shape != (self.num_envs,) or freeze_mask.dtype != torch.bool:
                raise RuntimeError(
                    "_diagnostic_freeze_arm_targets must be a boolean "
                    f"({self.num_envs},) tensor"
                )
            arm_targets = torch.where(
                freeze_mask.unsqueeze(-1),
                self._prev_targets[:, self._arm_joint_ids],
                arm_targets,
            )
        self._cur_targets.copy_(self._fixed_robot_joint_pos)
        self._cur_targets[:, self._arm_joint_ids] = arm_targets
        self._prev_targets.copy_(self._cur_targets)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        palm_pos = self.robot.data.body_link_pos_w[:, self._palm_body_id]
        palm_quat = self.robot.data.body_link_quat_w[:, self._palm_body_id]
        rel_pos, _ = subtract_frame_transforms(
            palm_pos,
            palm_quat,
            self.object.data.root_link_pos_w,
            self.object.data.root_link_quat_w,
        )
        expected_rel = torch.tensor(
            self._fixed_grasp_spec["palm_to_tool_pos"], device=self.device
        ).expand(self.num_envs, -1)
        self._fixed_grasp_drift.copy_((rel_pos - expected_rel).norm(dim=-1))
        drift = self._fixed_grasp_drift > float(self.cfg.fixed_grasp_max_drift_m)
        over_force = self._scrape_table_normal_force > float(
            self.cfg.max_contact_normal_force
        )
        truncated = self.episode_length_buf >= self.max_episode_length
        self._termination_reasons = {
            "fixed_grasp_drift": drift,
            "over_force": over_force,
            "timeout": truncated,
        }
        return drift | over_force, truncated

    def _get_rewards(self) -> torch.Tensor:
        force = self._sensor_normal_force()
        self._force_derivative.copy_(
            (force - self._previous_filtered_force) / float(self.step_dt)
        )
        self._previous_filtered_force.copy_(force)
        target = self._scrape_target_contact_normal_force
        reward, terms = fixed_force_reward_terms(
            force,
            target,
            self._action_delta,
            sigma=float(self.cfg.contact_force_sigma_start),
            soft_force_limit=float(self.cfg.soft_contact_normal_force_limit),
            max_force=float(self.cfg.max_contact_normal_force),
            force_weight=float(self.cfg.force_reward_weight),
            quadratic_error_weight=float(
                self.cfg.quadratic_force_error_penalty_weight
            ),
            quadratic_error_scale=float(self.cfg.quadratic_force_error_scale_n),
            over_force_weight=float(self.cfg.over_force_penalty_weight),
            action_rate_weight=float(self.cfg.action_rate_penalty_weight),
        )
        error = torch.abs(force - target)
        within = error <= float(self.cfg.force_success_tolerance_n)
        self._episode_within_steps.add_(within.float())
        self._episode_force_steps.add_(1.0)
        self._reward_terms = terms
        self.extras["episode_cumulative"] = terms
        self.extras["episode_final"] = {
            "within_1n_ratio": self._episode_within_steps
            / self._episode_force_steps.clamp_min(1.0),
            **{
                f"done_{name}": value.float()
                for name, value in self._termination_reasons.items()
            },
        }
        metrics = {
            "force_measured_mean": force.mean(),
            "force_target_mean": target.mean(),
            "force_error_mae": error.mean(),
            "within_1n_ratio": within.float().mean(),
            "normal_offset_mean": self._normal_offset.mean(),
            "normal_action_mean": self._previous_normal_action.mean(),
            "force_derivative_mean": self._force_derivative.mean(),
            "force_derivative_abs_mean": self._force_derivative.abs().mean(),
            "fixed_grasp_drift_max": self._fixed_grasp_drift.max(),
            "raw_force_mean": self._scrape_table_normal_force_raw.mean(),
        }
        for lower in range(2, 6):
            upper = lower + 1
            mask = (target >= float(lower)) & (target < float(upper))
            count = int(mask.sum())
            metrics[f"target_bin_{lower}_{upper}/sample_fraction"] = mask.float().mean()
            if count == 0:
                continue
            bin_force = force[mask]
            bin_target = target[mask]
            metrics.update(
                {
                    f"target_bin_{lower}_{upper}/target_mean": bin_target.mean(),
                    f"target_bin_{lower}_{upper}/measured_force_mean": bin_force.mean(),
                    f"target_bin_{lower}_{upper}/force_error_mae": (
                        bin_force - bin_target
                    ).abs().mean(),
                    f"target_bin_{lower}_{upper}/force_derivative_mean": (
                        self._force_derivative[mask].mean()
                    ),
                    f"target_bin_{lower}_{upper}/force_derivative_abs_mean": (
                        self._force_derivative[mask].abs().mean()
                    ),
                    f"target_bin_{lower}_{upper}/action_mean": (
                        self._previous_normal_action[mask].mean()
                    ),
                    f"target_bin_{lower}_{upper}/offset_mean": (
                        self._normal_offset[mask].mean()
                    ),
                }
            )
        for name, value in metrics.items():
            self.extras[f"fixed_force/{name}"] = value
        for name, value in terms.items():
            self.extras[f"reward/{name}"] = value.mean()
        if self._tool_table_contact_sensor_failed:
            raise RuntimeError(
                f"tool-table ContactSensor failed: {self._tool_table_contact_sensor_error}"
            )
        if not torch.isfinite(reward).all():
            raise RuntimeError("Fixed-force reward contains NaN or Inf.")
        return reward

    def _get_observations(self) -> dict[str, torch.Tensor]:
        observations = super()._get_observations()
        for name, value in observations.items():
            if not torch.isfinite(value).all():
                raise RuntimeError(f"Fixed-force {name} observations contain NaN or Inf.")
        return observations


__all__ = [
    "SimToolRealFixedGraspNormalForceEnv",
    "SimToolRealFixedGraspNormalForceEnvCfg",
    "fixed_force_observation_lists",
    "fixed_force_reward_terms",
    "integrate_normal_action",
    "load_fixed_grasp_spec",
    "pi_force_control",
]
