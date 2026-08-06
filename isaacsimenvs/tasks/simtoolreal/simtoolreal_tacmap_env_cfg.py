"""TacMap-enabled SimToolReal config.

This config extends the official SimToolReal IsaacSim task. The base TacMap
variant keeps the policy and critic observation contract unchanged; the tactile
contact variant appends compact per-finger contact features to the actor
observation.
"""

from __future__ import annotations

from pathlib import Path

from isaaclab.sensors.ray_caster import patterns
from isaaclab.utils import configclass

from isaacsimenvs.sensors.tacmap import SharpaTacmapCfg

from .simtoolreal_env_cfg import AssetsCfg, ObsCfg, ResetCfg, SimToolRealEnvCfg


_TACMAP_ROOT = Path(__file__).resolve().parents[3] / "assets" / "tacmap"
_ELASTOMER_OFFSET_ROT_WXYZ = (0.5, 0.5, -0.5, 0.5)
_BASE_OBS = ObsCfg()


@configclass
class SimToolRealTacMapEnvCfg(SimToolRealEnvCfg):
    use_tacmap: bool = True
    enable_vbts: bool = True
    resolution_step: int = 20
    vbts_update_period: float = 1.0 / 60.0
    vbts_target_prim_expr: str = "/World/envs/env_.*/Object/.*/visuals"
    # Track the actual rigid body only. A broad `/Object/.*` view also asks
    # PhysX to resolve non-physics scopes such as `/Object/Looks`.
    vbts_target_rigid_expr: str = "/World/envs/env_.*/Object/object_root"

    points_npy_4f: str = str(_TACMAP_ROOT / "tactileSensor_map_4F_point_origin.npy")
    normals_npy_4f: str = str(_TACMAP_ROOT / "tactileSensor_map_4F_normal_origin.npy")
    points_npy_th: str = str(_TACMAP_ROOT / "tactileSensor_map_TH_point.npy")
    normals_npy_th: str = str(_TACMAP_ROOT / "tactileSensor_map_TH_normal.npy")

    vbts_sensor: list[SharpaTacmapCfg] = [
        SharpaTacmapCfg(
            prim_path="/World/envs/env_.*/Robot/left_thumb_DP",
            mesh_prim_paths=[
                SharpaTacmapCfg.RaycastTargetCfg(
                    prim_expr=vbts_target_prim_expr,
                    track_mesh_transforms=True,
                )
            ],
            update_period=vbts_update_period,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=(0.5, 0.5)),
            offset=SharpaTacmapCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=_ELASTOMER_OFFSET_ROT_WXYZ,
                convention="world",
            ),
            data_types=["distance_along_normal"],
            target_rigid_expr=vbts_target_rigid_expr,
            points_npy=points_npy_th,
            normals_npy=normals_npy_th,
            resolution_step=resolution_step,
            max_distance=0.015,
            correction_scale=1e-3,
        ),
        SharpaTacmapCfg(
            prim_path="/World/envs/env_.*/Robot/left_index_DP",
            mesh_prim_paths=[
                SharpaTacmapCfg.RaycastTargetCfg(
                    prim_expr=vbts_target_prim_expr,
                    track_mesh_transforms=True,
                )
            ],
            update_period=vbts_update_period,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=(0.5, 0.5)),
            offset=SharpaTacmapCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=_ELASTOMER_OFFSET_ROT_WXYZ,
                convention="world",
            ),
            data_types=["distance_along_normal"],
            target_rigid_expr=vbts_target_rigid_expr,
            points_npy=points_npy_4f,
            normals_npy=normals_npy_4f,
            resolution_step=resolution_step,
            max_distance=0.015,
            correction_scale=1e-3,
        ),
        SharpaTacmapCfg(
            prim_path="/World/envs/env_.*/Robot/left_middle_DP",
            mesh_prim_paths=[
                SharpaTacmapCfg.RaycastTargetCfg(
                    prim_expr=vbts_target_prim_expr,
                    track_mesh_transforms=True,
                )
            ],
            update_period=vbts_update_period,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=(0.5, 0.5)),
            offset=SharpaTacmapCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=_ELASTOMER_OFFSET_ROT_WXYZ,
                convention="world",
            ),
            data_types=["distance_along_normal"],
            target_rigid_expr=vbts_target_rigid_expr,
            points_npy=points_npy_4f,
            normals_npy=normals_npy_4f,
            resolution_step=resolution_step,
            max_distance=0.015,
            correction_scale=1e-3,
        ),
        SharpaTacmapCfg(
            prim_path="/World/envs/env_.*/Robot/left_ring_DP",
            mesh_prim_paths=[
                SharpaTacmapCfg.RaycastTargetCfg(
                    prim_expr=vbts_target_prim_expr,
                    track_mesh_transforms=True,
                )
            ],
            update_period=vbts_update_period,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=(0.5, 0.5)),
            offset=SharpaTacmapCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=_ELASTOMER_OFFSET_ROT_WXYZ,
                convention="world",
            ),
            data_types=["distance_along_normal"],
            target_rigid_expr=vbts_target_rigid_expr,
            points_npy=points_npy_4f,
            normals_npy=normals_npy_4f,
            resolution_step=resolution_step,
            max_distance=0.015,
            correction_scale=1e-3,
        ),
        SharpaTacmapCfg(
            prim_path="/World/envs/env_.*/Robot/left_pinky_DP",
            mesh_prim_paths=[
                SharpaTacmapCfg.RaycastTargetCfg(
                    prim_expr=vbts_target_prim_expr,
                    track_mesh_transforms=True,
                )
            ],
            update_period=vbts_update_period,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.01, size=(0.5, 0.5)),
            offset=SharpaTacmapCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=_ELASTOMER_OFFSET_ROT_WXYZ,
                convention="world",
            ),
            data_types=["distance_along_normal"],
            target_rigid_expr=vbts_target_rigid_expr,
            points_npy=points_npy_4f,
            normals_npy=normals_npy_4f,
            resolution_step=resolution_step,
            max_distance=0.015,
            correction_scale=1e-3,
        ),
    ]

    def compute_tacmap_obs_size(self) -> int:
        side_h = 240 // int(self.resolution_step)
        side_w = 240 // int(self.resolution_step)
        return len(self.vbts_sensor) * side_h * side_w


@configclass
class SimToolRealTacMapContactEnvCfg(SimToolRealTacMapEnvCfg):
    """TacMap task variant with Sharpa rotation-style tactile contact features."""

    include_tacmap_in_policy: bool = True
    tacmap_obs_normalization: float = 255.0
    enable_tactile: bool = True
    binary_contact: bool = False
    contact_smooth: float = 0.5
    contact_latency: float = 0.005
    contact_threshold: float = 0.05
    contact_sensor_noise: float = 0.01
    tacmap_history_len: int = 5
    disable_tactile_ids: list[int] = []
    # Current checkpoints use contact/depth-mean/depth-max/centroid-x/centroid-y.
    # Disable depth only when restoring legacy three-feature tactile policies.
    tacmap_policy_include_depth: bool = True

    obs: ObsCfg = ObsCfg(
        obs_list=_BASE_OBS.obs_list + ("tacmap",),
        state_list=_BASE_OBS.state_list,
        clamp_abs_observations=_BASE_OBS.clamp_abs_observations,
    )

    def compute_tacmap_obs_size(self) -> int:
        if self.tacmap_history_len <= 0:
            raise ValueError("tacmap_history_len must be positive.")
        features_per_sensor = 5 if self.tacmap_policy_include_depth else 3
        return (
            len(self.vbts_sensor)
            * features_per_sensor
            * int(self.tacmap_history_len)
        )


@configclass
class SimToolRealTacMapScrapePoseEnvCfg(SimToolRealTacMapContactEnvCfg):
    """Pose-only scrape finetuning task with edge-contact goal sampling."""

    assets: AssetsCfg = AssetsCfg(
        handle_head_types=("spatula", "eraser", "brush", "marker"),
    )

    obs: ObsCfg = ObsCfg(
        obs_list=_BASE_OBS.obs_list
        + ("tacmap", "scrape_target_contact_normal_force"),
        state_list=_BASE_OBS.state_list,
        clamp_abs_observations=_BASE_OBS.clamp_abs_observations,
    )

    # Keep the table pitch/roll domain randomization and the original
    # SimToolReal pose reward as the main objective.
    table_pitch_roll_range_deg: float = 8.0

    # The target pose anchors the lower leading tip edge on the tabletop. The
    # edge may translate along the table and rotate around itself.
    edge_contact_xy_range_m: tuple[float, float] = (0.08, 0.08)
    edge_contact_yaw_range_deg: float = 90.0
    edge_tilt_range_deg: tuple[float, float] = (5.0, 45.0)

    # Geometry-only auxiliary reward for keeping that same selected edge on the
    # table. The reward scale is constant; the distance threshold tightens only
    # when the edge-contact pass rate exceeds the configured threshold.
    edge_contact_reward_max_weight: float = 1.0
    edge_contact_reward_sigma_start_m: float = 0.03
    edge_contact_reward_sigma_target_m: float = 0.005
    edge_contact_reward_sigma_increment: float = 0.9
    edge_contact_curriculum_success_threshold: float = 0.8

    # Optional force term. It is multiplied by the edge-contact geometry score.
    # The reward scale is constant; the force tolerance tightens only when the
    # force-tracking pass rate among eligible environments exceeds the
    # configured threshold with enough samples to make the estimate reliable.
    enable_tool_table_contact_force_reward: bool = True
    enable_tool_table_contact_sensor: bool = False
    contact_force_reward_relative_weight: float = 0.5
    # If target_contact_normal_force is set, it pins a fixed target for backward
    # compatibility. Otherwise each env samples from target_contact_normal_force_range.
    target_contact_normal_force: float | None = None
    target_contact_normal_force_range: tuple[float, float] = (2.0, 6.0)
    max_contact_normal_force: float = 20.0
    contact_force_sigma_start: float = 8.0
    contact_force_sigma_target: float = 2.0
    contact_force_sigma_increment: float = 0.9
    contact_force_curriculum_success_threshold: float = 0.8
    contact_force_curriculum_min_eligible_count: int = 64
    # Scrape force is averaged over all physics samples in one policy interval.
    # These settings are validated against decimation at environment startup.
    contact_force_use_control_interval_average: bool = True
    tool_table_contact_sensor_update_period: float = 0.0
    tool_table_contact_sensor_history_len: int = 2
    tool_table_contact_sensor_force_threshold: float = 0.1
    # EMA is applied after control-interval averaging, and initialized only
    # after persistent contact clears the onset grace period.
    contact_force_filter_alpha: float = 0.2
    contact_force_onset_threshold_n: float = 0.1
    contact_force_onset_grace_steps: int = 3
    contact_force_reward_ramp_steps: int = 6
    contact_force_huber_delta_n: float = 1.0
    contact_force_grasp_min_fingertips: int = 2
    contact_force_grasp_max_fingertip_distance_m: float = 0.12
    tool_table_contact_sensor_prim_path: str = "/World/envs/env_.*/Object/object_root"
    tool_table_contact_sensor_filter_paths: list[str] = ["/World/envs/env_.*/Table/box"]


@configclass
class SimToolRealStableScrapeEnvCfg(SimToolRealTacMapScrapePoseEnvCfg):
    """Frozen-acquisition, moving-reference stable scraping task."""

    use_tacmap: bool = False
    enable_vbts: bool = False
    include_tacmap_in_policy: bool = False
    enable_tool_table_contact_force_reward: bool = False
    enable_tool_table_contact_sensor: bool = True

    # The vanilla observation tuple is an exact prefix for the frozen actor.
    frozen_acquisition_obs_dim: int = 140
    obs: ObsCfg = ObsCfg(
        obs_list=_BASE_OBS.obs_list
        + ("stable_target_tangent_velocity", "stable_phase"),
        state_list=_BASE_OBS.state_list
        + (
            "scrape_tool_table_normal_force", "scrape_edge_contact_error",
            "scrape_table_height", "scrape_table_normal", "stable_pose_error",
            "stable_support_count", "stable_contact_persistence",
            "stable_relative_linear_speed", "stable_relative_angular_speed",
            "stable_over_force", "stable_target_tangent_velocity", "stable_phase",
        ),
        clamp_abs_observations=_BASE_OBS.clamp_abs_observations,
    )

    acquisition_hover_height_m: float = 0.12
    acquisition_timeout_steps: int = 600
    acquisition_min_fingertips: int = 2
    acquisition_max_fingertip_distance_m: float = 0.12
    acquisition_min_edge_clearance_m: float = 0.03
    acquisition_max_table_force_n: float = 0.1
    acquisition_position_drift_m: float = 0.005
    acquisition_rotation_drift_deg: float = 2.0
    acquisition_stability_steps: int = 15
    approach_edge_tolerance_m: float = 0.01
    approach_contact_steps: int = 15
    grasp_loss_grace_steps: int = 5
    scrape_path_half_length_m: float = 0.04
    scrape_path_speed_mps: float = 0.02
    edge_contact_xy_range_m: tuple[float, float] = (0.04, 0.04)

    pose_reward_weight: float = 5.0
    edge_contact_reward_max_weight: float = 5.0
    contact_presence_reward_weight: float = 1.0
    support_reward_weight: float = 1.0
    slip_penalty_weight: float = 1.0
    spin_penalty_weight: float = 0.1
    action_rate_penalty_weight: float = 0.01
    tool_acceleration_penalty_weight: float = 0.01
    over_force_penalty_weight: float = 0.1
    soft_contact_normal_force_limit: float = 12.0
    hard_contact_normal_force_limit: float = 20.0
    pose_tracking_sigma_start_m: float = 0.03
    pose_tracking_sigma_target_m: float = 0.005
    stable_curriculum_increment: float = 0.9
    stable_curriculum_success_threshold: float = 0.8
    stable_curriculum_min_eligible_count: int = 64


@configclass
class SimToolRealFixedGraspNormalForceEnvCfg(SimToolRealTacMapScrapePoseEnvCfg):
    """One-action diagnostic for normal-force RL controllability."""

    action_space: int = 1
    episode_length_s: float = 5.0
    feedback_mode: str = "force"  # force | tactile | blind
    fixed_grasp_spec_path: str = str(
        Path(__file__).resolve().parents[3]
        / "assets"
        / "fixed_grasp"
        / "spatula_fixed_grasp.json"
    )

    target_contact_normal_force: float | None = None
    target_contact_normal_force_range: tuple[float, float] = (2.0, 6.0)
    contact_force_sigma_start: float = 2.0
    contact_force_sigma_target: float = 2.0
    max_contact_normal_force: float = 20.0
    soft_contact_normal_force_limit: float = 12.0
    force_success_tolerance_n: float = 1.0
    contact_force_filter_alpha: float = 0.2
    contact_force_use_control_interval_average: bool = False
    contact_force_onset_grace_steps: int = 0
    contact_force_reward_ramp_steps: int = 1

    normal_velocity_limit_mps: float = 0.005
    normal_offset_min_m: float = -0.01
    normal_offset_max_m: float = 0.05
    normal_offset_limit_m: float = 0.05
    dls_damping: float = 0.05
    dls_linear_gain: float = 5.0
    dls_angular_gain: float = 3.0
    dls_joint_velocity_limit: float = 0.5
    fixed_grasp_max_drift_m: float = 0.005
    arm_drive_damping_scale: float = 1.0
    contact_max_depenetration_velocity_mps: float = 1000.0

    force_reward_weight: float = 1.0
    quadratic_force_error_penalty_weight: float = 0.5
    quadratic_force_error_scale_n: float = 4.0
    over_force_penalty_weight: float = 0.1
    action_rate_penalty_weight: float = 0.01

    table_pitch_roll_range_deg: float = 8.0
    reset: ResetCfg = ResetCfg(
        reset_position_noise_x=0.0,
        reset_position_noise_y=0.0,
        reset_position_noise_z=0.0,
        reset_dof_pos_random_interval_arm=0.0,
        reset_dof_pos_random_interval_fingers=0.0,
        reset_dof_vel_random_interval=0.0,
        table_reset_z=0.37,
        table_reset_z_range=0.01,
        table_reset_pitch_roll_range_deg=8.0,
    )
