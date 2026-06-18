from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch
import torchvision.transforms as transforms
from pxr import UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.sensors.ray_caster.multi_mesh_ray_caster import MultiMeshRayCaster
from isaaclab.sensors.ray_caster.multi_mesh_ray_caster_data import MultiMeshRayCasterData
from isaaclab.sensors.ray_caster.ray_cast_utils import obtain_world_pose_from_view
from isaaclab.sim.views import XformPrimView
from isaaclab.utils.warp import raycast_dynamic_meshes

from .tacmap_utils import deform_quantize

if TYPE_CHECKING:
    from .sharpa_tacmap_cfg import SharpaTacmapCfg


logger = logging.getLogger(__name__)


class SharpaTacmap(MultiMeshRayCaster):
    cfg: "SharpaTacmapCfg"
    UNSUPPORTED_TYPES: ClassVar[set[str]] = set()

    def __init__(self, cfg: "SharpaTacmapCfg"):
        for name in cfg.data_types:
            if name not in ["distance_along_normal"]:
                raise ValueError(f"Unsupported data type: {name}")
        super().__init__(cfg)
        self._data = MultiMeshRayCasterData()

    @property
    def data(self) -> MultiMeshRayCasterData:
        self._update_outdated_buffers()
        return self._data

    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._frame[env_ids] = 0

    def _initialize_rays_impl(self):
        self._ALL_INDICES = torch.arange(self._view.count, device=self._device, dtype=torch.long)
        self._frame = torch.zeros(self._view.count, device=self._device, dtype=torch.long)

        resolution_step = self.cfg.resolution_step
        pts_np = np.load(self.cfg.points_npy)[::resolution_step, ::resolution_step, :]
        nrm_np = np.load(self.cfg.normals_npy)[::resolution_step, ::resolution_step, :]
        self.image_shape = pts_np.shape[:2]

        pts_np = np.asarray(pts_np).reshape(-1, 3)
        nrm_np = np.asarray(nrm_np).reshape(-1, 3)

        nrm_norm = np.linalg.norm(nrm_np, axis=-1, keepdims=True) + 1e-12
        nrm_np = nrm_np / nrm_norm

        pts = torch.tensor(pts_np, dtype=torch.float32, device=self._device)
        nrms = torch.tensor(nrm_np, dtype=torch.float32, device=self._device)

        blur_kernel = max(1, 9 // self.cfg.resolution_step)
        self.blur = transforms.GaussianBlur(kernel_size=(blur_kernel, blur_kernel), sigma=(1.5, 1.5))

        pts = pts * self.cfg.correction_scale
        nrms = -nrms
        pts = pts + self.cfg.pts_offsets * nrms

        self.num_rays = pts.shape[0]
        self._create_buffers()

        self.ray_starts_att = pts.unsqueeze(0).repeat(self._view.count, 1, 1)
        self.ray_directions_att = nrms.unsqueeze(0).repeat(self._view.count, 1, 1)
        self.ray_hits_w = torch.zeros(self._view.count, self.num_rays, 3, device=self._device)

        quat_w = math_utils.convert_camera_frame_orientation_convention(
            torch.tensor([self.cfg.offset.rot], device=self._device),
            origin=self.cfg.offset.convention,
            target="world",
        )
        self._offset_quat = quat_w.repeat(self._view.count, 1)
        self._offset_pos = torch.tensor(list(self.cfg.offset.pos), device=self._device).repeat(self._view.count, 1)

        self._ray_starts_w = torch.zeros(self._view.count, self.num_rays, 3, device=self.device)
        self._ray_directions_w = torch.zeros(self._view.count, self.num_rays, 3, device=self.device)

    def _create_buffers(self):
        self._data.pos_w = torch.zeros((self._view.count, 3), device=self._device)
        self._data.quat_w = torch.zeros((self._view.count, 4), device=self._device)
        self._data.output = {
            "distance_along_normal": torch.zeros(
                (self._view.count, self.num_rays, 1), device=self._device, dtype=torch.uint8
            )
        }
        self._data.image_mesh_ids = torch.zeros((self._num_envs, self.num_rays, 1), device=self.device, dtype=torch.int16)
        self.drift = torch.zeros((self._view.count, 3), device=self._device, dtype=torch.float32)
        self.bias = torch.zeros((self._view.count, 3), device=self._device, dtype=torch.float32)
        self.ray_cast_drift = torch.zeros((self._view.count, 3), device=self._device, dtype=torch.float32)
        self.ray_cast_bias = torch.zeros((self._view.count, 3), device=self._device, dtype=torch.float32)

    def _update_ray_infos(self, env_ids: Sequence[int]):
        pos_w, quat_w = self._obtain_world_pose_from_view_safe(self._view, env_ids)
        pos_w, quat_w = math_utils.combine_frame_transforms(
            pos_w, quat_w, self._offset_pos[env_ids], self._offset_quat[env_ids]
        )
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w[env_ids] = quat_w

        ray_starts_w = math_utils.quat_apply(quat_w.repeat(1, self.num_rays), self.ray_starts_att[env_ids])
        ray_starts_w += pos_w.unsqueeze(1)
        ray_directions_w = math_utils.quat_apply(quat_w.repeat(1, self.num_rays), self.ray_directions_att[env_ids])

        self._ray_starts_w[env_ids] = ray_starts_w
        self._ray_directions_w[env_ids] = ray_directions_w

    def _obtain_world_pose_from_view_safe(self, view, env_ids):
        try:
            return obtain_world_pose_from_view(view, env_ids)
        except AttributeError as exc:
            if not isinstance(view, XformPrimView) or "hierarchy" not in str(exc):
                raise
            # Isaac Lab 2.3 on some installs lacks usdrt.hierarchy, so Fabric world-pose reads fail.
            # Fall back to the USD path for XformPrimView-based tracked meshes.
            if hasattr(view, "_use_fabric"):
                view._use_fabric = False
            return view.get_world_poses(indices=env_ids)

    def _obtain_trackable_prim_view(self, target_prim_path: str):
        target_rigid_expr = getattr(self.cfg, "target_rigid_expr", None)
        if not target_rigid_expr:
            return super()._obtain_trackable_prim_view(target_prim_path)

        mesh_prims = sim_utils.find_matching_prims(target_prim_path)
        view_prims = [
            prim
            for prim in sim_utils.find_matching_prims(target_rigid_expr)
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI) or prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        if len(mesh_prims) != len(view_prims):
            logger.warning(
                "TacMap target_rigid_expr '%s' resolved %d physics prims for %d mesh prims under '%s'. "
                "Falling back to Isaac Lab default tracking.",
                target_rigid_expr,
                len(view_prims),
                len(mesh_prims),
                target_prim_path,
            )
            return super()._obtain_trackable_prim_view(target_prim_path)

        first_view_prim = view_prims[0] if view_prims else None
        if first_view_prim is None:
            return super()._obtain_trackable_prim_view(target_prim_path)

        if first_view_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim_view = self._physics_sim_view.create_articulation_view(target_rigid_expr.replace(".*", "*"))
        elif first_view_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim_view = self._physics_sim_view.create_rigid_body_view(target_rigid_expr.replace(".*", "*"))
        else:
            return super()._obtain_trackable_prim_view(target_prim_path)

        positions = []
        quaternions = []
        for mesh_prim, view_prim in zip(mesh_prims, view_prims):
            pos, orientation = sim_utils.resolve_prim_pose(mesh_prim, view_prim)
            positions.append(torch.tensor(pos, dtype=torch.float32, device=self.device))
            quaternions.append(torch.tensor(orientation, dtype=torch.float32, device=self.device))

        return prim_view, (torch.stack(positions), torch.stack(quaternions))

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        self._update_ray_infos(env_ids)

        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self.device)

        self._frame[env_ids] += 1

        mesh_idx = 0
        for view, target_cfg in zip(self._mesh_views, self._raycast_targets_cfg):
            if not target_cfg.track_mesh_transforms:
                mesh_idx += self._num_meshes_per_env[target_cfg.prim_expr]
                continue

            pos_w, ori_w = self._obtain_world_pose_from_view_safe(view, None)
            pos_w = pos_w.squeeze(0) if len(pos_w.shape) == 3 else pos_w
            ori_w = ori_w.squeeze(0) if len(ori_w.shape) == 3 else ori_w

            if target_cfg.prim_expr in MultiMeshRayCaster.mesh_offsets:
                pos_offset, ori_offset = MultiMeshRayCaster.mesh_offsets[target_cfg.prim_expr]
                pos_w -= pos_offset
                ori_w = math_utils.quat_mul(ori_offset.expand(ori_w.shape[0], -1), ori_w)

            count = view.count
            if count != 1:
                count = count // self._num_envs
                pos_w = pos_w.view(self._num_envs, count, 3)
                ori_w = ori_w.view(self._num_envs, count, 4)

            self._mesh_positions_w[:, mesh_idx : mesh_idx + count] = pos_w
            self._mesh_orientations_w[:, mesh_idx : mesh_idx + count] = ori_w
            mesh_idx += count

        self.ray_hits_w[env_ids], ray_depth, _, _, ray_mesh_ids = raycast_dynamic_meshes(
            self._ray_starts_w[env_ids],
            self._ray_directions_w[env_ids],
            mesh_ids_wp=self._mesh_ids_wp,
            max_dist=self.cfg.max_distance,
            mesh_positions_w=self._mesh_positions_w[env_ids],
            mesh_orientations_w=self._mesh_orientations_w[env_ids],
            return_distance=True,
            return_normal=False,
        )

        cpd_distance = ray_depth
        cpd_eps = -1e-4
        cpd_advance = (cpd_distance.clamp_min(0.0) + cpd_eps).unsqueeze(-1)
        cpd_ray_starts = self._ray_starts_w[env_ids] + self._ray_directions_w[env_ids] * cpd_advance

        _, cpd_ray_depth, _, _, _ = raycast_dynamic_meshes(
            cpd_ray_starts,
            -self._ray_directions_w[env_ids],
            mesh_ids_wp=self._mesh_ids_wp,
            max_dist=self.cfg.cpd_max_dist,
            mesh_positions_w=self._mesh_positions_w[env_ids],
            mesh_orientations_w=self._mesh_orientations_w[env_ids],
            return_distance=True,
            return_normal=False,
            return_mesh_id=False,
        )

        cpd_keep = torch.isfinite(cpd_distance) & torch.isfinite(cpd_ray_depth)
        ray_depth_after_cpd = torch.where(cpd_keep, cpd_distance, torch.zeros_like(cpd_distance))

        if "distance_along_normal" in self._data.output:
            export_ray_depth = torch.where(
                torch.isfinite(ray_depth_after_cpd), ray_depth_after_cpd, torch.zeros_like(ray_depth_after_cpd)
            )
            export_ray_depth_quantize = deform_quantize(export_ray_depth.view(-1, self.num_rays, 1))
            export_ray_depth_quantize = export_ray_depth_quantize.reshape(
                len(env_ids), self.image_shape[0], self.image_shape[1]
            )
            export_ray_depth_quantize = self.blur(export_ray_depth_quantize)
            self._data.output["distance_along_normal"][env_ids] = export_ray_depth_quantize.view(
                -1, self.num_rays, 1
            )

        if self.cfg.update_mesh_ids:
            self._data.image_mesh_ids[env_ids] = ray_mesh_ids.view(-1, self.num_rays, 1)
