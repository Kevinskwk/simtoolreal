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
    vbts_target_rigid_expr: str = "/World/envs/env_.*/Object/.*"

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

    obs: ObsCfg = ObsCfg(
        obs_list=_BASE_OBS.obs_list + ("tacmap",),
        state_list=_BASE_OBS.state_list,
        clamp_abs_observations=_BASE_OBS.clamp_abs_observations,
    )

    def compute_tacmap_obs_size(self) -> int:
        if self.tacmap_history_len <= 0:
            raise ValueError("tacmap_history_len must be positive.")
        return len(self.vbts_sensor) * 5 * int(self.tacmap_history_len)


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
    # force-tracking pass rate exceeds the configured threshold.
    enable_tool_table_contact_force_reward: bool = True
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
    tool_table_contact_sensor_update_period: float = 1.0 / 60.0
    tool_table_contact_sensor_history_len: int = 1
    tool_table_contact_sensor_force_threshold: float = 0.1
    # EMA coefficient for the current tool-table force sample. Set to 1.0 to
    # disable temporal filtering.
    contact_force_filter_alpha: float = 0.2
    tool_table_contact_sensor_prim_path: str = "/World/envs/env_.*/Object/object_root"
    tool_table_contact_sensor_filter_paths: list[str] = ["/World/envs/env_.*/Table/box"]


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
