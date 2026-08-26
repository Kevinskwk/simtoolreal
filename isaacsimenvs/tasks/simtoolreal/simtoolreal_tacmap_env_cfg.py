"""TacMap-enabled SimToolReal config.

This config extends the official SimToolReal IsaacSim task. The base TacMap
variant keeps the policy and critic observation contract unchanged; the tactile
contact variant appends compact per-finger contact features to the actor
observation.
"""

from __future__ import annotations

from pathlib import Path

from isaaclab.sim import SimulationCfg
from isaaclab.sensors.ray_caster import patterns
from isaaclab.utils import configclass

from isaacsimenvs.sensors.tacmap import SharpaTacmapCfg

from .simtoolreal_env_cfg import (
    AssetsCfg,
    DomainRandomizationCfg,
    ObsCfg,
    ResetCfg,
    SimToolRealEnvCfg,
    _default_sim_cfg,
)


_TACMAP_ROOT = Path(__file__).resolve().parents[3] / "assets" / "tacmap"
_ELASTOMER_OFFSET_ROT_WXYZ = (0.5, 0.5, -0.5, 0.5)
_BASE_OBS = ObsCfg()


def _allen_turn_sim_cfg() -> SimulationCfg:
    cfg = _default_sim_cfg()
    cfg.physx.min_velocity_iteration_count = 2
    cfg.physx.max_velocity_iteration_count = 2
    cfg.physx.enable_external_forces_every_iteration = True
    return cfg


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

    # Optional pair-filtered PhysX measurements used by the wrench
    # observability gate. A separate sensor is required for every finger:
    # Isaac Lab does not guarantee correct pair filtering when one sensor
    # expression resolves to multiple bodies in an environment.
    enable_fingertip_tool_contact_sensors: bool = False
    fingertip_tool_contact_sensor_update_period: float = 0.0
    fingertip_tool_contact_max_data_count: int = 16
    fingertip_tool_contact_prim_paths: tuple[str, ...] = (
        "/World/envs/env_.*/Robot/left_thumb_DP",
        "/World/envs/env_.*/Robot/left_index_DP",
        "/World/envs/env_.*/Robot/left_middle_DP",
        "/World/envs/env_.*/Robot/left_ring_DP",
        "/World/envs/env_.*/Robot/left_pinky_DP",
    )
    fingertip_tool_contact_filter_paths: tuple[str, ...] = (
        "/World/envs/env_.*/Object/object_root",
    )
    enable_palm_tool_contact_sensor: bool = False
    palm_tool_contact_sensor_prim_path: str = (
        "/World/envs/env_.*/Robot/iiwa14_link_7"
    )
    palm_tool_contact_sensor_filter_paths: tuple[str, ...] = (
        "/World/envs/env_.*/Object/object_root",
    )

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
    frozen_acquisition_coefficient_id: float = 0.0
    frozen_acquisition_phase_field: str = "stable_phase"
    frozen_acquisition_phase_source: str = "policy"
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
    grasp_retention_position_tolerance_m: float = 0.015
    grasp_retention_rotation_tolerance_deg: float = 15.0
    scrape_path_half_length_m: float = 0.04
    scrape_path_speed_mps: float = 0.02
    scrape_velocity_activation_pose_error_m: float = 0.02
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
class SimToolRealInHandStableScrapeEnvCfg(SimToolRealStableScrapeEnvCfg):
    """Post-grasp stabilization gate initialized from compliant snapshots."""

    assets: AssetsCfg = AssetsCfg(
        handle_head_types=("eraser",),
        object_urdf=str(
            Path(__file__).resolve().parents[3]
            / "assets"
            / "urdf"
            / "objects"
            / "eraser_tactile_canonical.urdf"
        ),
        object_scale=(
            2.9373215824170767,
            0.5126639800346792,
            1.2951200580119278,
        ),
    )
    domain_randomization: DomainRandomizationCfg = DomainRandomizationCfg(
        use_obs_delay=False,
        use_action_delay=False,
        use_object_state_delay_noise=False,
        joint_velocity_obs_noise_std=0.0,
        force_scale=0.0,
        torque_scale=0.0,
        force_prob_range=(1.0e-12, 1.0e-12),
        torque_prob_range=(1.0e-12, 1.0e-12),
    )
    grasp_bank_path: str = str(
        Path(__file__).resolve().parents[3]
        / "assets"
        / "grasp_banks"
        / "eraser_canonical_v2.json"
    )
    grasp_bank_source_checkpoint_path: str = str(
        Path(__file__).resolve().parents[3] / "pretrained_policy" / "model.pth"
    )
    grasp_bank_min_entries: int = 64
    grasp_bank_joint_limit_tolerance_rad: float = 5.0e-4
    inhand_clearance_stages_m: tuple[tuple[float, float], ...] = (
        (0.05, 0.05),
        (0.045, 0.055),
        (0.04, 0.06),
        (0.04, 0.06),
    )
    inhand_table_angle_stages_deg: tuple[float, ...] = (0.0, 2.0, 5.0, 8.0)
    inhand_curriculum_success_threshold: float = 0.8
    inhand_curriculum_min_eligible_count: int = 64
    inhand_reset_penetration_tolerance_m: float = 1.0e-4
    inhand_target_yaw_delta_deg: float = 5.0
    inhand_target_tilt_delta_deg: float = 5.0
    inhand_target_max_rotation_deg: float = 30.0
    inhand_target_sampling_attempts: int = 32
    grasp_loss_grace_steps: int = 15


@configclass
class SimToolRealInHandAdjustmentEnvCfg(SimToolRealInHandStableScrapeEnvCfg):
    """Extrinsic in-hand grasp adjustment toward a palm-to-tool target."""

    episode_length_s: float = 8.0
    obs: ObsCfg = ObsCfg(
        obs_list=_BASE_OBS.obs_list + (
            "adjustment_target_error", "adjustment_target_palm_error",
            "adjustment_pose_error",
            "adjustment_phase", "adjustment_table",
        ),
        state_list=_BASE_OBS.state_list + (
            "stable_support_count", "stable_relative_linear_speed",
            "stable_relative_angular_speed", "adjustment_target_error",
            "adjustment_target_palm_error", "adjustment_pose_error",
            "adjustment_phase", "adjustment_table",
        ),
        clamp_abs_observations=_BASE_OBS.clamp_abs_observations,
    )

    adjustment_success_steps: int = 20
    adjustment_relative_position_tolerance_m: float = 0.002
    adjustment_relative_rotation_tolerance_deg: float = 2.0
    adjustment_relative_position_tolerance_stages_m: tuple[float, ...] = (
        0.005, 0.004, 0.003, 0.0025, 0.002
    )
    adjustment_relative_rotation_tolerance_stages_deg: tuple[float, ...] = (
        5.0, 4.0, 3.0, 2.5, 2.0
    )
    adjustment_tool_position_tolerance_m: float = 0.02
    adjustment_tool_rotation_tolerance_deg: float = 10.0
    adjustment_tool_position_hard_limit_m: float = 0.15
    adjustment_tool_rotation_hard_limit_deg: float = 75.0
    adjustment_pose_failure_steps: int = 15
    adjustment_terminate_on_success: bool = True
    adjustment_min_fingertip_support: int = 2
    adjustment_table_clearance_range_m: tuple[float, float] = (0.002, 0.03)
    adjustment_curriculum_success_threshold: float = 0.80
    adjustment_curriculum_min_eligible_count: int = 256
    adjustment_finger_perturb_fractions: tuple[float, ...] = (0.0, 0.01, 0.02, 0.03, 0.05)
    adjustment_target_translation_mode: str = "anisotropic"  # anisotropic | none
    adjustment_target_rotation_axis: str = "random"  # random | tool_x
    adjustment_target_axial_translation_m: tuple[float, ...] = (0.008, 0.015, 0.025, 0.035, 0.045)
    adjustment_target_perpendicular_translation_m: tuple[float, ...] = (0.002, 0.003, 0.005, 0.0075, 0.010)
    adjustment_target_rotation_deg: tuple[float, ...] = (10.0, 20.0, 30.0, 45.0, 60.0)
    adjustment_target_axial_fraction_of_tool_length: float = 0.25
    adjustment_target_perpendicular_fraction_of_tool_thickness: float = 0.30
    adjustment_target_total_translation_max_m: float = 0.05
    adjustment_tool_pose_delay_steps: tuple[int, ...] = (180, 150, 120, 90, 60)
    adjustment_tool_pose_ramp_steps: tuple[int, ...] = (180, 150, 120, 90, 60)
    adjustment_relative_position_reward_weight: float = 2.0
    adjustment_relative_rotation_reward_weight: float = 1.0
    adjustment_relative_position_reward_sigma_m: float = 0.015
    adjustment_relative_rotation_reward_sigma_deg: float = 10.0
    adjustment_tool_position_penalty_weight: float = 20.0
    adjustment_tool_rotation_penalty_weight: float = 0.5
    adjustment_fingertip_support_reward_weight: float = 0.25
    adjustment_action_rate_penalty_weight: float = 0.01
    adjustment_success_bonus: float = 10.0


@configclass
class SimToolRealScrewdriverAxialAdjustmentEnvCfg(SimToolRealInHandAdjustmentEnvCfg):
    """Screwdriver regrasp with rotation only about the tool-local handle axis."""

    assets: AssetsCfg = AssetsCfg(
        handle_head_types=("screwdriver",),
        object_urdf="",
        object_scale=None,
        num_assets_per_type=20,
        procedural_asset_seed=42,
        shuffle_assets=True,
        object_pool_limit=0,
    )
    grasp_bank_path: str = str(
        Path(__file__).resolve().parents[3]
        / "outputs"
        / "inhand_adjustment_cache"
        / "screwdriver_seed42_n20"
        / "grasps.json"
    )
    grasp_bank_min_entries: int = 4
    adjustment_finger_perturb_fractions: tuple[float, ...] = (0.0, 0.01, 0.02, 0.03)
    adjustment_target_translation_mode: str = "none"
    # The Allen override samples about local -Z through allen_screw_pivot_tool_m.
    adjustment_target_rotation_axis: str = "tool_x"
    adjustment_target_axial_translation_m: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    adjustment_target_perpendicular_translation_m: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    adjustment_target_rotation_deg: tuple[float, ...] = (10.0, 20.0, 30.0, 45.0)
    adjustment_relative_position_tolerance_stages_m: tuple[float, ...] = (
        0.005, 0.004, 0.003, 0.002
    )
    adjustment_relative_rotation_tolerance_stages_deg: tuple[float, ...] = (
        5.0, 4.0, 3.0, 2.0
    )
    adjustment_tool_pose_delay_steps: tuple[int, ...] = (180, 150, 120, 90)
    adjustment_tool_pose_ramp_steps: tuple[int, ...] = (180, 150, 120, 90)


@configclass
class SimToolRealAllenKeyAdjustmentEnvCfg(SimToolRealInHandAdjustmentEnvCfg):
    """Palm-supported Allen-key regrasp around the engaged screw axis."""

    assets: AssetsCfg = AssetsCfg(
        table_urdf=str(
            Path(__file__).resolve().parents[3]
            / "assets" / "urdf" / "table_allen_disabled.urdf"
        ),
        handle_head_types=("screwdriver",),
        object_urdf=str(
            Path(__file__).resolve().parents[3]
            / "assets" / "urdf" / "objects" / "allen_key_canonical.urdf"
        ),
        object_scale=(3.5, 0.5, 1.5),
        workpiece_urdf=str(
            Path(__file__).resolve().parents[3]
            / "assets" / "urdf" / "workpieces" / "allen_key_hex_socket.urdf"
        ),
    )
    grasp_bank_path: str = str(
        Path(__file__).resolve().parents[3]
        / "assets" / "grasp_banks" / "allen_key_manipulation_v3.json"
    )
    grasp_bank_min_entries: int = 6
    episode_length_s: float = 8.0
    enable_palm_tool_contact_sensor: bool = True
    enable_fingertip_tool_contact_sensors: bool = True
    adjustment_terminate_on_success: bool = False
    adjustment_finger_perturb_fractions: tuple[float, ...] = (0.0,) * 5
    adjustment_target_translation_mode: str = "none"
    adjustment_target_rotation_axis: str = "tool_x"
    adjustment_target_axial_translation_m: tuple[float, ...] = (0.0,) * 5
    adjustment_target_perpendicular_translation_m: tuple[float, ...] = (0.0,) * 5
    # Target sampling is fixed across the curriculum. Only the palm-keypoint
    # accuracy sigma below tightens as endpoint success improves.
    adjustment_target_rotation_deg: tuple[float, ...] = (60.0,) * 5
    adjustment_relative_position_tolerance_stages_m: tuple[float, ...] = (
        0.008, 0.007, 0.006, 0.005, 0.004
    )
    adjustment_relative_rotation_tolerance_stages_deg: tuple[float, ...] = (
        6.0, 5.0, 4.0, 3.0, 2.5
    )
    adjustment_tool_pose_delay_steps: tuple[int, ...] = (0,) * 5
    adjustment_tool_pose_ramp_steps: tuple[int, ...] = (1,) * 5
    allen_adjustment_steps: int = 280
    allen_closure_steps: int = 120
    allen_release_steps: int = 80
    allen_success_hold_steps: int = 45
    # Start and target relationships are sampled from distinct physically
    # validated bank entries. A pair must clear at least one lower bound and
    # both upper bounds, preventing trivial and implausibly distant targets.
    allen_target_pair_translation_range_m: tuple[float, float] = (0.012, 0.090)
    allen_target_pair_rotation_range_deg: tuple[float, float] = (18.0, 100.0)
    allen_require_valid_target_pairs: bool = True
    allen_pose_sigma_stages_m: tuple[float, ...] = (0.060, 0.040, 0.025, 0.015, 0.010)
    # The bank itself covers a physically validated 100-degree world-yaw span.
    # Do not rotate those grasps into unscreened arm configurations at reset.
    allen_reset_yaw_range_stages_deg: tuple[float, ...] = (0.0,) * 5
    # Disabled for the legacy bank. The workspace task below enables strict
    # rollout-derived source and target curricula.
    allen_workspace_conditioned_sampling: bool = False
    allen_workspace_tier_probabilities_stages: tuple[
        tuple[float, float, float], ...
    ] = ((1.0, 0.0, 0.0),) * 5
    allen_pair_max_translation_stages_m: tuple[float, ...] = (0.09,) * 5
    allen_pair_max_rotation_stages_deg: tuple[float, ...] = (100.0,) * 5
    allen_target_min_quality_improvement: float = 0.0
    allen_palm_keypoints_m: tuple[tuple[float, float, float], ...] = (
        (0.0, 0.0, 0.0), (0.05, 0.0, 0.0), (-0.05, 0.0, 0.0),
        (0.0, 0.035, 0.0), (0.0, -0.035, 0.0), (0.0, 0.0, 0.03),
    )
    allen_screw_axis_tool: tuple[float, float, float] = (0.0, 0.0, -1.0)
    allen_screw_pivot_tool_m: tuple[float, float, float] = (0.192, 0.0, -0.03)
    allen_handle_across_flats_m: float = 0.010
    allen_workpiece_from_tool_m: tuple[float, float, float] = (0.192, 0.0, -0.095)
    allen_hidden_table_offset_m: float = 1.0
    allen_socket_lateral_tolerance_m: float = 0.006
    allen_socket_insertion_tolerance_m: float = 0.008
    allen_socket_tilt_tolerance_deg: float = 5.0
    allen_palm_contact_threshold_n: float = 0.01
    allen_fingertip_contact_threshold_n: float = 0.05
    allen_fingertip_contact_quality_saturation_count: int = 2
    allen_closure_flexion_fraction: float = 0.15
    allen_min_flexion_closure_fraction: float = 0.35
    allen_release_linear_speed_tolerance_mps: float = 0.04
    allen_release_angular_speed_tolerance_radps: float = 1.0
    allen_release_challenge_force_n: float = 1.0
    allen_release_challenge_torque_nm: float = 0.02
    allen_tool_position_tolerance_m: float = 0.010
    allen_tool_rotation_tolerance_deg: float = 5.0
    obs: ObsCfg = ObsCfg(
        obs_list=_BASE_OBS.obs_list + (
            "allen_current_palm_tool", "allen_target_palm_tool",
            "allen_target_error", "allen_geometry", "allen_socket_state",
            "allen_phase",
        ),
        state_list=_BASE_OBS.state_list + (
            "stable_support_count", "stable_relative_linear_speed",
            "stable_relative_angular_speed", "allen_current_palm_tool",
            "allen_target_palm_tool", "allen_target_error", "allen_geometry",
            "allen_socket_state", "allen_phase", "allen_validity",
        ),
        clamp_abs_observations=_BASE_OBS.clamp_abs_observations,
    )


@configclass
class SimToolRealAllenKeyWorkspaceAdjustmentEnvCfg(
    SimToolRealAllenKeyAdjustmentEnvCfg
):
    """Rollout-conditioned Allen-key adjustment from weaker to functional grasps."""

    grasp_bank_path: str = str(
        Path(__file__).resolve().parents[3]
        / "assets" / "grasp_banks" / "allen_key_rollout_adjustment_v1.json"
    )
    grasp_bank_min_entries: int = 18
    allen_workspace_conditioned_sampling: bool = True
    # Stage zero follows the measured 69% acquisition region. Broader support
    # and hard poses enter only after endpoint hold success advances curriculum.
    allen_workspace_tier_probabilities_stages: tuple[
        tuple[float, float, float], ...
    ] = (
        (0.80, 0.20, 0.00),
        (0.65, 0.30, 0.05),
        (0.50, 0.40, 0.10),
        (0.40, 0.40, 0.20),
        (0.30, 0.40, 0.30),
    )
    allen_pair_max_translation_stages_m: tuple[float, ...] = (
        0.060, 0.060, 0.075, 0.075, 0.090
    )
    allen_pair_max_rotation_stages_deg: tuple[float, ...] = (
        60.0, 60.0, 80.0, 80.0, 100.0
    )
    allen_target_min_quality_improvement: float = 0.25


@configclass
class SimToolRealAllenKeyPalmDownAdjustmentEnvCfg(
    SimToolRealAllenKeyWorkspaceAdjustmentEnvCfg
):
    """Palm-down side-changing regrasp around a thick Allen-key handle."""

    assets: AssetsCfg = AssetsCfg(
        table_urdf=str(
            Path(__file__).resolve().parents[3]
            / "assets" / "urdf" / "table_allen_disabled.urdf"
        ),
        handle_head_types=("screwdriver",),
        object_urdf=str(
            Path(__file__).resolve().parents[3]
            / "assets" / "urdf" / "objects" / "allen_key_thick_handle.urdf"
        ),
        object_scale=(1.0, 1.0, 1.0),
        workpiece_urdf=str(
            Path(__file__).resolve().parents[3]
            / "assets" / "urdf" / "workpieces" / "allen_key_hex_socket.urdf"
        ),
    )
    grasp_bank_path: str = str(
        Path(__file__).resolve().parents[3]
        / "assets" / "grasp_banks" / "allen_key_palm_down_v1.json"
    )
    grasp_bank_min_entries: int = 8
    allen_handle_across_flats_m: float = 0.030
    allen_target_pair_translation_range_m: tuple[float, float] = (0.0, 0.200)
    allen_target_pair_rotation_range_deg: tuple[float, float] = (40.0, 100.0)
    allen_workspace_tier_probabilities_stages: tuple[
        tuple[float, float, float], ...
    ] = ((1.0, 0.0, 0.0),) * 5
    allen_pair_max_translation_stages_m: tuple[float, ...] = (
        0.20, 0.20, 0.20, 0.20, 0.20
    )
    # Pair difficulty is fixed by the screened rollout graph. Curriculum
    # progression tightens endpoint accuracy instead of hiding valid goals.
    allen_pair_max_rotation_stages_deg: tuple[float, ...] = (100.0,) * 5
    allen_target_min_quality_improvement: float = 0.0
    allen_pose_sigma_stages_m: tuple[float, ...] = (
        0.080, 0.060, 0.040, 0.025, 0.015
    )


@configclass
class SimToolRealAllenKeyTurningEnvCfg(SimToolRealTacMapEnvCfg):
    """Single-policy 360-degree Allen-key pose tracking under resistance."""

    sim: SimulationCfg = _allen_turn_sim_cfg()

    assets: AssetsCfg = AssetsCfg(
        table_urdf=str(
            Path(__file__).resolve().parents[3]
            / "assets" / "urdf" / "table_allen_disabled.urdf"
        ),
        workpiece_urdf=str(
            Path(__file__).resolve().parents[3]
            / "assets" / "urdf" / "workpieces" / "allen_key_hex_socket.urdf"
        ),
        # The ideal screw-axis fixture supplies the physical constraint and
        # calibrated load. Keeping these synchronized meshes collidable would
        # add timestep-dependent socket impulses to the resistance target.
        workpiece_collision_enabled=False,
        handle_head_types=("screwdriver",),
        allen_key_lengths_m=(0.20, 0.22, 0.24, 0.264, 0.28, 0.30, 0.32),
        allen_key_handle_across_flats_m=0.020,
        allen_key_short_leg_length_m=0.060,
        allen_key_elbow_x_m=0.192,
    )
    use_tacmap: bool = False
    enable_vbts: bool = False
    include_tacmap_in_policy: bool = False
    enable_fingertip_tool_contact_sensors: bool = False
    enable_palm_tool_contact_sensor: bool = True
    # One ContactSensor is created per surviving rigid link, then contacts are
    # reduced to five per-finger values. A contact on any link of a finger is
    # sufficient; multiple contacting links still count as one finger.
    allen_turn_finger_tool_contact_prim_paths: tuple[tuple[str, ...], ...] = (
        (
            "/World/envs/env_.*/Robot/left_thumb_CMC_VL",
            "/World/envs/env_.*/Robot/left_thumb_MC",
            "/World/envs/env_.*/Robot/left_thumb_MCP_VL",
            "/World/envs/env_.*/Robot/left_thumb_PP",
            "/World/envs/env_.*/Robot/left_thumb_DP",
        ),
        (
            "/World/envs/env_.*/Robot/left_index_MCP_VL",
            "/World/envs/env_.*/Robot/left_index_PP",
            "/World/envs/env_.*/Robot/left_index_MP",
            "/World/envs/env_.*/Robot/left_index_DP",
        ),
        (
            "/World/envs/env_.*/Robot/left_middle_MCP_VL",
            "/World/envs/env_.*/Robot/left_middle_PP",
            "/World/envs/env_.*/Robot/left_middle_MP",
            "/World/envs/env_.*/Robot/left_middle_DP",
        ),
        (
            "/World/envs/env_.*/Robot/left_ring_MCP_VL",
            "/World/envs/env_.*/Robot/left_ring_PP",
            "/World/envs/env_.*/Robot/left_ring_MP",
            "/World/envs/env_.*/Robot/left_ring_DP",
        ),
        (
            "/World/envs/env_.*/Robot/left_pinky_MC",
            "/World/envs/env_.*/Robot/left_pinky_MCP_VL",
            "/World/envs/env_.*/Robot/left_pinky_PP",
            "/World/envs/env_.*/Robot/left_pinky_MP",
            "/World/envs/env_.*/Robot/left_pinky_DP",
        ),
    )
    episode_length_s: float = 45.0

    allen_turn_screw_axis_tool: tuple[float, float, float] = (0.0, 0.0, -1.0)
    allen_turn_screw_pivot_tool_m: tuple[float, float, float] = (0.192, 0.0, -0.030)
    allen_turn_socket_root_from_pivot_m: tuple[float, float, float] = (0.0, 0.0, -0.065)
    allen_turn_goal_increment_deg: float = 30.0
    allen_turn_goal_count: int = 12
    allen_turn_goal_tolerance_deg: float = 6.0
    allen_turn_goal_hold_steps: int = 10
    allen_turn_final_hold_steps: int = 30
    allen_turn_minimum_contact_fingers: int = 2
    allen_turn_loaded_grasp_minimum_contact_fingers: int = 2
    allen_turn_acquisition_hold_steps: int = 20
    # Center-to-center distance from the physical palm mesh to the tool handle.
    # The stage curriculum tightens the palm-supported grasp envelope.
    allen_turn_palm_support_distance_stages_m: tuple[float, ...] = (
        0.105, 0.100, 0.095, 0.090, 0.085, 0.080, 0.075
    )
    allen_turn_palm_support_soft_margin_m: float = 0.040
    allen_turn_contact_force_threshold_n: float = 0.05
    allen_turn_palm_contact_threshold_n: float = 0.05
    allen_turn_max_relative_linear_speed_mps: float = 0.08
    allen_turn_max_relative_angular_speed_radps: float = 2.0
    # A deep grasp excludes unilateral finger pushing. It requires thumb and
    # opposing-finger force closure plus actual palm/proximal-link support.
    allen_turn_deep_grasp_minimum_contact_fingers: int = 3
    allen_turn_deep_grasp_maximum_opposition_cosine: float = -0.25
    allen_turn_deep_grasp_hold_steps: int = 10

    # Keep the graspable long-handle center in the original SimToolReal XY
    # reset range. The Z ranges below refer to the screw pivot; the horizontal
    # handle lies 3 cm above it. Workspace sampling remains fixed while the
    # curriculum increases fixture resistance and regularization.
    allen_turn_handle_center_x_range_stages_m: tuple[tuple[float, float], ...] = (
        (-0.10, 0.10), (-0.10, 0.10), (-0.10, 0.10), (-0.10, 0.10),
        (-0.10, 0.10), (-0.10, 0.10), (-0.10, 0.10),
    )
    allen_turn_handle_center_y_range_stages_m: tuple[tuple[float, float], ...] = (
        (-0.10, 0.10), (-0.10, 0.10), (-0.10, 0.10), (-0.10, 0.10),
        (-0.10, 0.10), (-0.10, 0.10), (-0.10, 0.10),
    )
    allen_turn_easy_yaw_probability_stages: tuple[float, ...] = (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    )
    allen_turn_max_shoulder_to_handle_stages_m: tuple[float, ...] = (
        0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.84
    )
    allen_turn_max_initial_palm_handle_distance_stages_m: tuple[float, ...] = (
        0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35
    )
    allen_turn_z_range_stages_m: tuple[tuple[float, float], ...] = (
        (0.50, 0.60), (0.50, 0.60), (0.50, 0.60), (0.50, 0.60),
        (0.50, 0.60), (0.50, 0.60), (0.50, 0.60),
    )
    # Per-episode Coulomb friction and viscous damping are randomized. The
    # curriculum expands these ranges; no constant load is applied.
    allen_turn_friction_ranges_nm: tuple[tuple[float, float], ...] = (
        (0.040, 0.080), (0.060, 0.120), (0.100, 0.200), (0.160, 0.320),
        (0.250, 0.500), (0.350, 0.700), (0.500, 1.000),
    )
    allen_turn_damping_ranges_nm_per_radps: tuple[tuple[float, float], ...] = (
        (0.025, 0.070), (0.040, 0.100), (0.060, 0.150), (0.090, 0.220),
        (0.130, 0.320), (0.180, 0.450), (0.250, 0.600),
    )
    allen_turn_stiction_stiffness_nm_per_rad: float = 1.0
    allen_turn_static_to_kinetic_friction_ratio: float = 1.5
    # Bound the combined Coulomb and viscous reaction. Without this cap, a
    # contact transient can turn damping into an arbitrarily large impulse.
    allen_turn_max_fixture_torque_multiplier: float = 2.0
    allen_turn_resistance_transition_speed_radps: float = 0.05
    allen_turn_friction_restick_speed_radps: float = 0.03
    allen_turn_curriculum_min_episodes: int = 4096
    allen_turn_curriculum_success_threshold: float = 0.60

    # Keep the original SimToolReal acquisition -> pose-tracking structure.
    # The socket-engaged key cannot be lifted, so handle approach and loaded
    # grasp replace the original fingertip-to-root and lift terms.
    allen_turn_handle_approach_progress_weight: float = 20.0
    allen_turn_handle_approach_sigma_m: float = 0.10
    allen_turn_handle_proximity_penalty_weight: float = 0.05
    allen_turn_first_loaded_grasp_bonus: float = 10.0
    allen_turn_grasp_maintenance_reward_weight: float = 0.25
    allen_turn_deep_grasp_reward_weight: float = 1.0
    allen_turn_subgoal_bonus: float = 8.0
    allen_turn_full_turn_bonus: float = 100.0

    # Regularization is weak while acquisition is being learned, then returns
    # to its configured strength as the task curriculum advances.
    allen_turn_regularization_scale_stages: tuple[float, ...] = (
        0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 1.00
    )
    allen_turn_effort_soft_threshold_fraction: float = 0.45
    allen_turn_effort_penalty_weight: float = 2.0
    allen_turn_action_rate_penalty_weight: float = 0.01
    allen_turn_constraint_position_tolerance_m: float = 0.005
    allen_turn_constraint_tilt_tolerance_deg: float = 5.0
    allen_turn_hidden_table_offset_m: float = 1.0
    allen_turn_initial_hand_clearance_m: float = 0.015
    allen_turn_initial_arm_clearance_m: float = 0.055
    allen_turn_initial_sampling_max_attempts: int = 128
    allen_turn_initial_robot_resampling_max_attempts: int = 16
    # Bound overlap recovery so a bad finger contact cannot launch the key in
    # one physics frame. The revolute constraint handles the fixture geometry.
    contact_max_depenetration_velocity_mps: float = 2.0

    reset: ResetCfg = ResetCfg(
        reset_dof_pos_random_interval_arm=0.1,
        reset_dof_pos_random_interval_fingers=0.1,
        reset_dof_vel_random_interval=0.5,
    )
    domain_randomization: DomainRandomizationCfg = DomainRandomizationCfg(
        use_obs_delay=False,
        use_action_delay=False,
        use_object_state_delay_noise=False,
        joint_velocity_obs_noise_std=0.0,
        force_scale=0.0,
        force_prob_range=(1.0e-12, 1.0e-12),
        torque_scale=0.0,
        torque_prob_range=(1.0e-12, 1.0e-12),
    )
    obs: ObsCfg = ObsCfg(
        obs_list=_BASE_OBS.obs_list,
        state_list=_BASE_OBS.state_list,
        clamp_abs_observations=10.0,
    )

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
