"""Dataset validation and models for controlled tool-wrench observability probes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import Dataset


WRENCH_MODE_NAMES = (
    "zero",
    "force_x_pos", "force_x_neg",
    "force_y_pos", "force_y_neg",
    "force_z_pos", "force_z_neg",
    "torque_x_pos", "torque_x_neg",
    "torque_y_pos", "torque_y_neg",
    "torque_z_pos", "torque_z_neg",
)

REQUIRED_KEYS = {
    "proprio_state",
    "tactile_compact",
    "commanded_force_palm_n",
    "commanded_torque_palm_nm",
    "wrench_mode",
    "valid",
    "metadata",
}

IMPULSE_KEYS = {
    "finger_normal_impulse_palm_ns",
    "finger_tangential_impulse_palm_ns",
}


def validate_episode(episode: dict) -> int:
    missing = REQUIRED_KEYS - set(episode)
    if missing:
        raise ValueError(f"wrench episode is missing keys: {sorted(missing)}")
    state = episode["proprio_state"]
    if not isinstance(state, torch.Tensor) or state.ndim != 2 or state.shape[1] <= 0:
        raise ValueError("proprio_state must have shape (T, D) with D > 0")
    length = int(state.shape[0])
    shapes = {
        "tactile_compact": (length, 5, 5),
        "commanded_force_palm_n": (length, 3),
        "commanded_torque_palm_nm": (length, 3),
        "wrench_mode": (length,),
        "valid": (length,),
    }
    for key, shape in shapes.items():
        value = episode[key]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"{key} has shape {getattr(value, 'shape', None)}; expected {shape}")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"{key} contains NaN or Inf")
    if not torch.isfinite(state).all():
        raise ValueError("proprio_state contains NaN or Inf")
    mode = episode["wrench_mode"].long()
    if bool(((mode < 0) | (mode >= len(WRENCH_MODE_NAMES))).any()):
        raise ValueError("wrench_mode contains an invalid class")
    if not isinstance(episode["metadata"], dict):
        raise ValueError("metadata must be a dictionary")
    return length


def validate_impulse_episode(episode: dict) -> int:
    """Validate the strict v2 finger-tool impulse extension."""
    length = validate_episode(episode)
    missing = IMPULSE_KEYS - set(episode)
    if missing:
        raise ValueError(f"impulse episode is missing keys: {sorted(missing)}")
    for key in sorted(IMPULSE_KEYS):
        value = episode[key]
        expected = (length, 5, 3)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
            raise ValueError(
                f"{key} has shape {getattr(value, 'shape', None)}; expected {expected}"
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"{key} contains NaN or Inf")
    return length


def stable_split(group: str, seed: int = 0) -> str:
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "little") % 10_000
    if bucket < 7000:
        return "train"
    if bucket < 8500:
        return "validation"
    return "test"


@dataclass(frozen=True)
class EpisodeRef:
    path: Path
    episode_index: int
    split: str
    length: int


def catalog(shards: Sequence[Path], split_seed: int = 0) -> list[EpisodeRef]:
    refs: list[EpisodeRef] = []
    for path in shards:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        episodes = payload.get("episodes") if isinstance(payload, dict) else None
        if not isinstance(episodes, list):
            raise ValueError(f"invalid wrench-observability shard: {path}")
        for index, episode in enumerate(episodes):
            length = validate_episode(episode)
            group = str(episode["metadata"].get("split_group", ""))
            if not group:
                raise ValueError(f"episode {index} in {path} has no split_group")
            refs.append(EpisodeRef(path, index, stable_split(group, split_seed), length))
    return refs


class WrenchWindowDataset(Dataset):
    """In-memory windows; collection defaults keep this comfortably below RAM limits."""

    def __init__(
        self,
        refs: Sequence[EpisodeRef],
        split: str,
        history: int = 5,
        stride: int = 1,
    ) -> None:
        if history <= 0 or stride <= 0:
            raise ValueError("history and stride must be positive")
        self.history = history
        self.episodes: list[dict] = []
        self.indices: list[tuple[int, int]] = []
        by_path: dict[Path, dict] = {}
        for ref in refs:
            if ref.split != split:
                continue
            payload = by_path.setdefault(
                ref.path, torch.load(ref.path, map_location="cpu", weights_only=False)
            )
            episode = payload["episodes"][ref.episode_index]
            local = len(self.episodes)
            self.episodes.append(episode)
            valid = episode["valid"].bool()
            for end in range(history - 1, ref.length, stride):
                if bool(valid[end - history + 1 : end + 1].all()):
                    self.indices.append((local, end))
        if not self.indices:
            raise ValueError(f"no valid {split} windows were found")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_index, end = self.indices[index]
        episode = self.episodes[episode_index]
        start = end - self.history + 1
        force = episode["commanded_force_palm_n"][end].float()
        torque = episode["commanded_torque_palm_nm"][end].float()
        sample = {
            "state": episode["proprio_state"][start : end + 1].float(),
            "compact": episode["tactile_compact"][start : end + 1].float(),
            "force": force,
            "torque": torque,
            "mode": episode["wrench_mode"][end].long(),
        }
        if IMPULSE_KEYS.issubset(episode):
            sample["impulse"] = torch.cat(
                (
                    episode["finger_normal_impulse_palm_ns"][start : end + 1],
                    episode["finger_tangential_impulse_palm_ns"][start : end + 1],
                ),
                dim=-1,
            ).float()
        return sample


class WrenchProbe(nn.Module):
    def __init__(self, state_dim: int, variant: str, hidden_dim: int = 128) -> None:
        super().__init__()
        valid = {
            "state", "compact", "state_compact", "normal_impulse",
            "tangential_impulse", "impulse", "impulse_compact",
        }
        if variant not in valid:
            raise ValueError(f"unknown probe variant {variant!r}")
        self.variant = variant
        input_dim = 0
        if "state" in variant:
            input_dim += state_dim
        if "compact" in variant:
            input_dim += 25
        if variant in {"normal_impulse", "tangential_impulse"}:
            input_dim += 15
        elif "impulse" in variant:
            input_dim += 30
        self.temporal = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.trunk = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.mode_head = nn.Linear(hidden_dim, len(WRENCH_MODE_NAMES))
        self.wrench_head = nn.Linear(hidden_dim, 6)

    def forward(
        self, state: torch.Tensor, compact: torch.Tensor,
        impulse: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = []
        if "state" in self.variant:
            features.append(state)
        if "compact" in self.variant:
            features.append(compact.flatten(2))
        if "impulse" in self.variant:
            if impulse is None or impulse.ndim != 4 or impulse.shape[-2:] != (5, 6):
                raise ValueError("impulse input must have shape (B, T, 5, 6)")
            if self.variant == "normal_impulse":
                features.append(impulse[..., :3].flatten(2))
            elif self.variant == "tangential_impulse":
                features.append(impulse[..., 3:].flatten(2))
            else:
                features.append(impulse.flatten(2))
        encoded, _ = self.temporal(torch.cat(features, dim=-1))
        hidden = self.trunk(encoded[:, -1])
        return self.mode_head(hidden), self.wrench_head(hidden)


__all__ = [
    "WRENCH_MODE_NAMES",
    "IMPULSE_KEYS",
    "WrenchProbe",
    "WrenchWindowDataset",
    "catalog",
    "stable_split",
    "validate_episode",
    "validate_impulse_episode",
]
