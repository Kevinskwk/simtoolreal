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

from .simtoolreal_env_cfg import ObsCfg, SimToolRealEnvCfg


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
        return len(self.vbts_sensor) * 3 * int(self.tacmap_history_len)
