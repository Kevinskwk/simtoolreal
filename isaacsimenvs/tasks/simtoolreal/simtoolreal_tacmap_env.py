"""TacMap wrapper for the official SimToolReal IsaacSim env."""

from __future__ import annotations

import torch

from isaacsimenvs.sensors.tacmap import SharpaTacmap

from .simtoolreal_env import SimToolRealEnv
from .simtoolreal_tacmap_env_cfg import SimToolRealTacMapEnvCfg


class SimToolRealTacMapEnv(SimToolRealEnv):
    cfg: SimToolRealTacMapEnvCfg

    def __init__(
        self, cfg: SimToolRealTacMapEnvCfg, render_mode: str | None = None, **kwargs
    ) -> None:
        self._vbts_side_h = 240 // int(cfg.resolution_step)
        self._vbts_side_w = 240 // int(cfg.resolution_step)
        super().__init__(cfg, render_mode, **kwargs)
        num_sensors = len(getattr(self, "_vbts_sensor", []))
        self.vbts_deform = torch.zeros(
            (self.num_envs, num_sensors, self._vbts_side_h, self._vbts_side_w),
            dtype=torch.uint8,
            device=self.device,
        )

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

    def _update_tacmap_buffers(self) -> None:
        if not getattr(self.cfg, "enable_vbts", False) or not self._vbts_sensor:
            return

        vbts_deform = torch.cat(
            [
                sensor.data.output["distance_along_normal"]
                .reshape(self.num_envs, self._vbts_side_h, self._vbts_side_w)
                .unsqueeze(1)
                for sensor in self._vbts_sensor
            ],
            dim=1,
        )
        self.vbts_deform = vbts_deform.clone()

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._update_tacmap_buffers()
        return super()._get_observations()

    def get_tacmap_obs(self) -> torch.Tensor:
        self._update_tacmap_buffers()
        return self.vbts_deform


__all__ = [
    "SimToolRealTacMapEnv",
    "SimToolRealTacMapEnvCfg",
]
