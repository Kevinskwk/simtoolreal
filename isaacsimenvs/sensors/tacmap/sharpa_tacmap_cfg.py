from dataclasses import MISSING
from typing import Literal

from isaaclab.sensors.ray_caster.multi_mesh_ray_caster_cfg import MultiMeshRayCasterCfg
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.utils import configclass


@configclass
class SharpaTacmapCfg(MultiMeshRayCasterCfg):
    points_npy: str = MISSING
    normals_npy: str = MISSING
    resolution_step: int = MISSING

    @configclass
    class OffsetCfg:
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        convention: Literal["opengl", "ros", "world"] = "ros"

    offset: OffsetCfg = OffsetCfg()

    data_types: list[str] = ["distance_along_normal"]
    target_rigid_expr: str | None = None
    correction_scale: float = 1e-3
    pts_offsets: float = 0.0
    max_distance: float = 0.015
    cpd_max_dist: float = 0.5
    debug_viz: bool = False
    pattern_cfg: PinholeCameraPatternCfg = MISSING
