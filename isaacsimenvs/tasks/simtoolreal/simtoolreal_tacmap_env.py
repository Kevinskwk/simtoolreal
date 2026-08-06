"""TacMap wrapper for the official SimToolReal IsaacSim env."""

from __future__ import annotations

import torch

from isaacsimenvs.sensors.tacmap import SharpaTacmap

from .simtoolreal_env import SimToolRealEnv
from .simtoolreal_tacmap_env_cfg import (
    SimToolRealTacMapContactEnvCfg,
    SimToolRealTacMapEnvCfg,
)
from .utils.obs_utils import register_obs_field_size


class SimToolRealTacMapEnv(SimToolRealEnv):
    cfg: SimToolRealTacMapEnvCfg

    def __init__(
        self, cfg: SimToolRealTacMapEnvCfg, render_mode: str | None = None, **kwargs
    ) -> None:
        self._sync_vbts_sensor_cfgs(cfg)
        self._vbts_side_h = 240 // int(cfg.resolution_step)
        self._vbts_side_w = 240 // int(cfg.resolution_step)
        register_obs_field_size("tacmap", cfg.compute_tacmap_obs_size())
        super().__init__(cfg, render_mode, **kwargs)
        num_sensors = len(getattr(self, "_vbts_sensor", []))
        self.vbts_deform = torch.zeros(
            (self.num_envs, num_sensors, self._vbts_side_h, self._vbts_side_w),
            dtype=torch.uint8,
            device=self.device,
        )
        self.last_contacts = torch.zeros(
            (self.num_envs, num_sensors),
            dtype=torch.float32,
            device=self.device,
        )
        self._prev_raw_contacts = torch.zeros_like(self.last_contacts)
        history_len = int(getattr(self.cfg, "tacmap_history_len", 1))
        features_per_sensor = (
            5 if bool(getattr(self.cfg, "tacmap_policy_include_depth", True)) else 3
        )
        self._tacmap_policy_obs_history = torch.zeros(
            (self.num_envs, history_len, num_sensors * features_per_sensor),
            dtype=torch.float32,
            device=self.device,
        )
        self._last_tacmap_policy_obs_frame = torch.zeros(
            (self.num_envs, num_sensors * features_per_sensor),
            dtype=torch.float32,
            device=self.device,
        )
        self._tacmap_grid_y, self._tacmap_grid_x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, self._vbts_side_h, device=self.device),
            torch.linspace(-1.0, 1.0, self._vbts_side_w, device=self.device),
            indexing="ij",
        )

    @staticmethod
    def _sync_vbts_sensor_cfgs(cfg: SimToolRealTacMapEnvCfg) -> None:
        """Apply top-level TacMap YAML overrides to the nested sensor configs."""
        resolution_step = int(cfg.resolution_step)
        for sensor_cfg in getattr(cfg, "vbts_sensor", []):
            sensor_cfg.resolution_step = resolution_step
            sensor_cfg.update_period = float(cfg.vbts_update_period)
            sensor_cfg.target_rigid_expr = cfg.vbts_target_rigid_expr
            for target_cfg in getattr(sensor_cfg, "mesh_prim_paths", []):
                target_cfg.prim_expr = cfg.vbts_target_prim_expr

    def _setup_scene(self) -> None:
        super()._setup_scene()
        self._vbts_sensor: list[SharpaTacmap] = []
        if not getattr(self.cfg, "enable_vbts", False):
            return

        for sensor_id, sensor_cfg in enumerate(self.cfg.vbts_sensor):
            sensor = SharpaTacmap(sensor_cfg)
            self._vbts_sensor.append(sensor)
            self.scene.sensors[f"vbts_sensor_{sensor_id}"] = sensor

    def _reset_idx(self, env_ids) -> None:
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if hasattr(self, "vbts_deform") and env_ids.numel() > 0:
            self.vbts_deform[env_ids] = 0
        if hasattr(self, "last_contacts") and env_ids.numel() > 0:
            self.last_contacts[env_ids] = 0
        if hasattr(self, "_prev_raw_contacts") and env_ids.numel() > 0:
            self._prev_raw_contacts[env_ids] = 0
        if hasattr(self, "_tacmap_policy_obs_history") and env_ids.numel() > 0:
            self._tacmap_policy_obs_history[env_ids] = 0
        if hasattr(self, "_last_tacmap_policy_obs_frame") and env_ids.numel() > 0:
            self._last_tacmap_policy_obs_frame[env_ids] = 0

    def _update_tacmap_buffers(self) -> None:
        if not getattr(self.cfg, "enable_vbts", False) or not self._vbts_sensor:
            return

        vbts_deform = torch.cat(
            [
                self._reshape_vbts_output(sensor).unsqueeze(1)
                for sensor in self._vbts_sensor
            ],
            dim=1,
        )
        self.vbts_deform = vbts_deform.clone()

    def _reshape_vbts_output(self, sensor: SharpaTacmap) -> torch.Tensor:
        output = sensor.data.output["distance_along_normal"]
        expected_numel = self.num_envs * self._vbts_side_h * self._vbts_side_w
        if output.numel() != expected_numel:
            per_env = (
                output.numel() // self.num_envs
                if output.numel() % self.num_envs == 0
                else None
            )
            raise RuntimeError(
                "TacMap output shape mismatch: "
                f"env resolution_step={self.cfg.resolution_step}, "
                f"sensor resolution_step={sensor.cfg.resolution_step}, "
                f"expected per-env map={self._vbts_side_h}x{self._vbts_side_w}, "
                f"actual output shape={tuple(output.shape)}, actual per-env elements={per_env}."
            )
        return output.reshape(self.num_envs, self._vbts_side_h, self._vbts_side_w)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._update_tacmap_buffers()
        return super()._get_observations()

    def get_tacmap_obs(self) -> torch.Tensor:
        self._update_tacmap_buffers()
        return self.vbts_deform

    def _get_tacmap_policy_obs_frame(self) -> torch.Tensor:
        scale = float(getattr(self.cfg, "tacmap_obs_normalization", 255.0))
        if scale <= 0.0:
            raise ValueError("cfg.tacmap_obs_normalization must be positive.")
        tactile = self.vbts_deform.to(torch.float32) / scale
        threshold = float(getattr(self.cfg, "contact_threshold", 0.05))
        contact_mask = tactile > threshold
        contact_weights = torch.where(contact_mask, tactile, torch.zeros_like(tactile))

        contact_area = contact_mask.to(torch.float32).mean(dim=(-1, -2))
        contact_count = contact_mask.to(torch.float32).sum(dim=(-1, -2))
        contact_mass = contact_weights.sum(dim=(-1, -2)).clamp_min(1.0e-6)
        contact_depth_mean = torch.where(
            contact_count > 0.0,
            contact_weights.sum(dim=(-1, -2)) / contact_count.clamp_min(1.0),
            torch.zeros_like(contact_area),
        )
        contact_depth_max = torch.where(
            contact_mask, tactile, torch.zeros_like(tactile)
        ).amax(dim=(-1, -2))
        contact_cx = (
            (contact_weights * self._tacmap_grid_x).sum(dim=(-1, -2))
            / contact_mass
        )
        contact_cy = (
            (contact_weights * self._tacmap_grid_y).sum(dim=(-1, -2))
            / contact_mass
        )
        active_contact = contact_area > 0.0
        contact_cx = torch.where(active_contact, contact_cx, torch.zeros_like(contact_cx))
        contact_cy = torch.where(active_contact, contact_cy, torch.zeros_like(contact_cy))

        disabled = getattr(self.cfg, "disable_tactile_ids", [])
        if disabled:
            disabled_ids = torch.as_tensor(disabled, device=self.device, dtype=torch.long)
            contact_area[:, disabled_ids] = 0.0
            contact_depth_mean[:, disabled_ids] = 0.0
            contact_depth_max[:, disabled_ids] = 0.0
            contact_cx[:, disabled_ids] = 0.0
            contact_cy[:, disabled_ids] = 0.0

        if bool(getattr(self.cfg, "binary_contact", False)):
            contacts = torch.where(contact_area > 0.0, 1.0, 0.0)
            latency = torch.where(
                torch.rand_like(self.last_contacts)
                < float(getattr(self.cfg, "contact_latency", 0.005)),
                1.0,
                0.0,
            )
            self.last_contacts = self.last_contacts * latency + contacts * (1.0 - latency)
            noise_mask = torch.where(
                torch.rand_like(self.last_contacts)
                < float(getattr(self.cfg, "contact_sensor_noise", 0.01)),
                0.0,
                1.0,
            )
            sensed_contacts = torch.where(
                self.last_contacts > 0.1,
                noise_mask * self.last_contacts,
                self.last_contacts,
            )
        else:
            smooth = float(getattr(self.cfg, "contact_smooth", 0.5))
            smooth_contacts = (
                contact_area * smooth + self._prev_raw_contacts * (1.0 - smooth)
            )
            self._prev_raw_contacts = contact_area.clone()
            latency = torch.where(
                torch.rand_like(self.last_contacts)
                < float(getattr(self.cfg, "contact_latency", 0.005)),
                1.0,
                0.0,
            )
            self.last_contacts = (
                self.last_contacts * latency + smooth_contacts * (1.0 - latency)
            )
            sensed_contacts = self.last_contacts.clone()

        if disabled:
            sensed_contacts[:, disabled_ids] = 0.0
        contact_present = sensed_contacts > 0.0
        contact_depth_mean = torch.where(
            contact_present, contact_depth_mean, torch.zeros_like(contact_depth_mean)
        )
        contact_depth_max = torch.where(
            contact_present, contact_depth_max, torch.zeros_like(contact_depth_max)
        )
        contact_cx = torch.where(contact_present, contact_cx, torch.zeros_like(contact_cx))
        contact_cy = torch.where(contact_present, contact_cy, torch.zeros_like(contact_cy))
        if not bool(getattr(self.cfg, "enable_tactile", True)):
            sensed_contacts[:] = 0.0
            contact_depth_mean[:] = 0.0
            contact_depth_max[:] = 0.0
            contact_cx[:] = 0.0
            contact_cy[:] = 0.0
        features = [sensed_contacts]
        if bool(getattr(self.cfg, "tacmap_policy_include_depth", True)):
            features.extend((contact_depth_mean, contact_depth_max))
        features.extend((contact_cx, contact_cy))
        frame = torch.stack(features, dim=-1).reshape(self.num_envs, -1)
        if not hasattr(self, "_last_tacmap_policy_obs_frame"):
            self._last_tacmap_policy_obs_frame = frame.clone()
        else:
            self._last_tacmap_policy_obs_frame.copy_(frame)
        return frame

    def get_last_tacmap_policy_frame(self) -> torch.Tensor:
        """Return the compact frame most recently assembled for observations."""
        if not hasattr(self, "_last_tacmap_policy_obs_frame"):
            raise RuntimeError("TacMap compact observation frame is unavailable")
        return self._last_tacmap_policy_obs_frame

    def get_tacmap_policy_obs(self) -> torch.Tensor:
        frame = self._get_tacmap_policy_obs_frame()
        history = self._tacmap_policy_obs_history
        episode_start = self.episode_length_buf == 0
        if episode_start.any():
            history[episode_start] = frame[episode_start].unsqueeze(1)
        history = torch.roll(history, shifts=1, dims=1)
        history[:, 0, :] = frame
        self._tacmap_policy_obs_history = history
        return history.reshape(self.num_envs, -1)


__all__ = [
    "SimToolRealTacMapEnv",
    "SimToolRealTacMapContactEnvCfg",
    "SimToolRealTacMapEnvCfg",
]
