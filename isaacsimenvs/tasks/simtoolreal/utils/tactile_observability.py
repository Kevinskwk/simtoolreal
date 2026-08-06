"""Offline labels, datasets, and probe models for tactile observability."""

from __future__ import annotations

import hashlib
import math
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.utils.data import Dataset


REQUIRED_EPISODE_KEYS = {
    "actor_state",
    "deployable_geometry",
    "oracle_geometry",
    "tactile_compact",
    "tactile_raw",
    "force_interval_n",
    "edge_error_m",
    "palm_tool_pos",
    "palm_tool_quat",
    "fingertip_count",
    "object_fallen",
}


@dataclass(frozen=True)
class LabelConfig:
    control_dt_s: float = 1.0 / 60.0
    contact_threshold_n: float = 0.1
    edge_threshold_m: float = 0.005
    event_horizon_steps: int = 5
    event_persistence_steps: int = 3
    slip_translation_speed_mps: float = 0.03
    slip_rotation_speed_radps: float = 1.0
    slip_persistence_steps: int = 2
    instability_horizon_steps: int = 15
    instability_translation_m: float = 0.03
    loss_translation_m: float = 0.06


def _require_tensor(episode: dict, key: str) -> torch.Tensor:
    value = episode.get(key)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"episode[{key!r}] must be a tensor")
    return value


def validate_episode(episode: dict) -> int:
    """Validate one saved post-grasp episode and return its length."""
    missing = REQUIRED_EPISODE_KEYS - set(episode)
    if missing:
        raise ValueError(f"episode is missing required keys: {sorted(missing)}")
    length = int(_require_tensor(episode, "actor_state").shape[0])
    if length <= 0:
        raise ValueError("episode must contain at least one frame")
    expected_shapes = {
        "actor_state": (length, 140),
        "deployable_geometry": (length, 13),
        "oracle_geometry": (length, 3),
        "tactile_compact": (length, 5, 5),
        "tactile_raw": (length, 5, 12, 12),
        "force_interval_n": (length,),
        "edge_error_m": (length,),
        "palm_tool_pos": (length, 3),
        "palm_tool_quat": (length, 4),
        "fingertip_count": (length,),
        "object_fallen": (length,),
    }
    for key, shape in expected_shapes.items():
        value = _require_tensor(episode, key)
        if tuple(value.shape) != shape:
            raise ValueError(
                f"episode[{key!r}] has shape {tuple(value.shape)}; expected {shape}"
            )
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"episode[{key!r}] contains NaN or Inf")
    if _require_tensor(episode, "tactile_raw").dtype != torch.uint8:
        raise ValueError("tactile_raw must use uint8 storage")
    reference_pos = _require_tensor(episode, "grasp_reference_pos")
    reference_quat = _require_tensor(episode, "grasp_reference_quat")
    if tuple(reference_pos.shape) != (3,) or tuple(reference_quat.shape) != (4,):
        raise ValueError("grasp reference pose must have shapes (3,) and (4,)")
    if not torch.isfinite(reference_pos).all() or not torch.isfinite(reference_quat).all():
        raise ValueError("grasp reference pose contains NaN or Inf")
    return length


def quaternion_angle_rad(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Shortest sign-invariant angle between batches of wxyz quaternions."""
    if a.shape != b.shape or a.shape[-1] != 4:
        raise ValueError(f"quaternion shapes must match and end in 4: {a.shape}, {b.shape}")
    a = torch.nn.functional.normalize(a.float(), dim=-1)
    b = torch.nn.functional.normalize(b.float(), dim=-1)
    dot = torch.abs((a * b).sum(dim=-1)).clamp(0.0, 1.0)
    return 2.0 * torch.acos(dot)


def _future_any(values: torch.Tensor, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return whether a true value occurs in the next horizon and an eligibility mask."""
    if values.ndim != 1 or values.dtype != torch.bool:
        raise ValueError("_future_any expects a one-dimensional bool tensor")
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    length = values.shape[0]
    result = torch.zeros(length, dtype=torch.bool)
    eligible = torch.zeros(length, dtype=torch.bool)
    if length > horizon:
        windows = values[1:].unfold(0, horizon, 1)
        count = windows.shape[0]
        eligible[:count] = True
        result[:count] = windows.any(dim=-1)
    return result, eligible


def _persistent_start(values: torch.Tensor, persistence: int) -> torch.Tensor:
    if persistence <= 0:
        raise ValueError("persistence must be positive")
    result = torch.zeros_like(values)
    if values.shape[0] >= persistence:
        windows = values.unfold(0, persistence, 1)
        result[: windows.shape[0]] = windows.all(dim=-1)
    return result


def _persistent_end(values: torch.Tensor, persistence: int) -> torch.Tensor:
    if persistence <= 0:
        raise ValueError("persistence must be positive")
    result = torch.zeros_like(values)
    if values.shape[0] >= persistence:
        windows = values.unfold(0, persistence, 1)
        result[persistence - 1 :] = windows.all(dim=-1)
    return result


def derive_labels(episode: dict, cfg: LabelConfig = LabelConfig()) -> dict[str, torch.Tensor]:
    """Create non-leaking current and future labels from a complete episode."""
    length = validate_episode(episode)
    force = episode["force_interval_n"].float()
    edge_error = episode["edge_error_m"].float()
    any_contact = force >= cfg.contact_threshold_n
    intended_contact = any_contact & (edge_error <= cfg.edge_threshold_m)
    contact_mode = torch.zeros(length, dtype=torch.long)
    contact_mode[intended_contact] = 1
    contact_mode[any_contact & ~intended_contact] = 2

    stable_contact_start = _persistent_start(
        intended_contact, cfg.event_persistence_steps
    )
    stable_absence_start = _persistent_start(
        ~intended_contact, cfg.event_persistence_steps
    )
    onset, onset_future_ok = _future_any(stable_contact_start, cfg.event_horizon_steps)
    loss, loss_future_ok = _future_any(stable_absence_start, cfg.event_horizon_steps)
    onset_eligible = (~intended_contact) & onset_future_ok
    loss_eligible = intended_contact & loss_future_ok

    position = episode["palm_tool_pos"].float()
    quaternion = episode["palm_tool_quat"].float()
    reference_pos = episode["grasp_reference_pos"].float().view(1, 3)
    reference_quat = episode["grasp_reference_quat"].float().view(1, 4)
    translation_drift = torch.linalg.vector_norm(position - reference_pos, dim=-1)
    rotation_drift_deg = torch.rad2deg(
        quaternion_angle_rad(quaternion, reference_quat.expand_as(quaternion))
    )
    translation_speed = torch.zeros(length)
    rotation_speed = torch.zeros(length)
    if length > 1:
        translation_speed[1:] = torch.linalg.vector_norm(
            position[1:] - position[:-1], dim=-1
        ) / cfg.control_dt_s
        rotation_speed[1:] = quaternion_angle_rad(
            quaternion[1:], quaternion[:-1]
        ) / cfg.control_dt_s
    slip_candidate = (
        (translation_speed > cfg.slip_translation_speed_mps)
        | (rotation_speed > cfg.slip_rotation_speed_radps)
    ) & (translation_drift < cfg.instability_translation_m)
    # Mark the current frame once at least N of the latest N frames indicate motion.
    slip = _persistent_end(slip_candidate, cfg.slip_persistence_steps)

    unstable_now = translation_drift >= cfg.instability_translation_m
    instability, instability_eligible = _future_any(
        unstable_now, cfg.instability_horizon_steps
    )
    hard_loss = (
        (translation_drift >= cfg.loss_translation_m)
        | episode["object_fallen"].bool()
        | _persistent_end(
            episode["fingertip_count"].long() < 2,
            cfg.instability_horizon_steps,
        )
    )

    return {
        "contact_mode": contact_mode,
        "onset": onset.float(),
        "onset_eligible": onset_eligible,
        "loss": loss.float(),
        "loss_eligible": loss_eligible,
        "slip": slip.float(),
        "slip_eligible": torch.ones(length, dtype=torch.bool),
        "instability": instability.float(),
        "instability_eligible": instability_eligible,
        "hard_loss": hard_loss.float(),
        "hard_loss_eligible": torch.ones(length, dtype=torch.bool),
        "relative_speed": torch.stack(
            (torch.log1p(translation_speed), torch.log1p(rotation_speed)), dim=-1
        ),
        "translation_drift_m": translation_drift,
        "rotation_drift_deg": rotation_drift_deg,
    }


def stable_bucket(value: str, seed: int, buckets: int = 10_000) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % buckets


def assign_split(metadata: dict, seed: int = 0, holdout_tools: bool = True) -> str:
    """Assign a stable episode split without frame-level leakage."""
    episode_id = str(metadata["episode_id"])
    tool_id = str(metadata.get("tool_id", "unknown"))
    key = tool_id if holdout_tools and tool_id != "unknown" else episode_id
    bucket = stable_bucket(key, seed)
    if bucket < 7000:
        return "train"
    if bucket < 8500:
        return "validation"
    return "test"


class EpisodeShardCache:
    def __init__(self, capacity: int = 4) -> None:
        self.capacity = capacity
        self.cache: OrderedDict[Path, dict] = OrderedDict()

    def load(self, path: Path) -> dict:
        if path in self.cache:
            value = self.cache.pop(path)
            self.cache[path] = value
            return value
        value = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(value, dict) or not isinstance(value.get("episodes"), list):
            raise ValueError(f"invalid observability shard: {path}")
        self.cache[path] = value
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return value


CATALOG_CACHE_VERSION = 1


def _catalog_fingerprint(
    paths: Sequence[Path], split_seed: int, holdout_tools: bool
) -> str:
    digest = hashlib.sha256()
    digest.update(
        f"{CATALOG_CACHE_VERSION}:{split_seed}:{int(holdout_tools)}".encode("utf-8")
    )
    for path in paths:
        stat = path.stat()
        digest.update(
            f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
        )
    return digest.hexdigest()[:20]


class EpisodeCatalog:
    """Validated episodes and labels shared by every split and history view."""

    def __init__(
        self,
        shard_paths: Sequence[Path],
        split_seed: int = 0,
        holdout_tools: bool = True,
        preload: bool = True,
        cache_dir: Path | None = None,
        rebuild_cache: bool = False,
        verbose: bool = False,
    ) -> None:
        paths = sorted(Path(item) for item in shard_paths)
        if not paths:
            raise ValueError("at least one shard path is required")
        self.cache = EpisodeShardCache()
        fingerprint = _catalog_fingerprint(paths, split_seed, holdout_tools)
        cache_path = None
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"catalog_{fingerprint}.pt"

        self.records: list[dict] = []
        cache_hit = cache_path is not None and cache_path.exists() and not rebuild_cache
        if cache_hit:
            if verbose:
                print(f"[dataset] loading derived-label cache {cache_path}", flush=True)
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            if (
                not isinstance(payload, dict)
                or payload.get("version") != CATALOG_CACHE_VERSION
                or payload.get("fingerprint") != fingerprint
                or not isinstance(payload.get("records"), list)
            ):
                raise ValueError(
                    f"invalid tactile observability catalog cache: {cache_path}"
                )
            self.records = payload["records"]
        else:
            progress_interval = max(1, len(paths) // 20)
            if verbose:
                print(
                    f"[dataset] deriving labels for {len(paths)} shards once",
                    flush=True,
                )
            for shard_number, path in enumerate(paths, start=1):
                shard = torch.load(path, map_location="cpu", weights_only=False)
                episodes = shard.get("episodes") if isinstance(shard, dict) else None
                if not isinstance(episodes, list):
                    raise ValueError(f"invalid observability shard: {path}")
                for episode_index, episode in enumerate(episodes):
                    length = validate_episode(episode)
                    metadata = dict(episode.get("metadata", {}))
                    if "episode_id" not in metadata:
                        raise ValueError(
                            f"episode {episode_index} in {path} has no episode_id"
                        )
                    self.records.append(
                        {
                            "path": path,
                            "episode_index": episode_index,
                            "episode": episode if preload else None,
                            "length": length,
                            "metadata": metadata,
                            "split": assign_split(
                                metadata, split_seed, holdout_tools
                            ),
                            "labels": derive_labels(episode),
                        }
                    )
                if verbose and (
                    shard_number % progress_interval == 0
                    or shard_number == len(paths)
                ):
                    print(
                        f"[dataset] processed {shard_number}/{len(paths)} shards, "
                        f"{len(self.records)} episodes",
                        flush=True,
                    )
            if cache_path is not None:
                cached_records = [
                    {key: value for key, value in record.items() if key != "episode"}
                    for record in self.records
                ]
                temporary = cache_path.with_suffix(".tmp")
                torch.save(
                    {
                        "version": CATALOG_CACHE_VERSION,
                        "fingerprint": fingerprint,
                        "records": cached_records,
                    },
                    temporary,
                )
                temporary.replace(cache_path)
                if verbose:
                    print(f"[dataset] wrote derived-label cache {cache_path}", flush=True)

        if cache_hit and preload:
            records_by_path: dict[Path, list[dict]] = {}
            for record in self.records:
                record["path"] = Path(record["path"])
                records_by_path.setdefault(record["path"], []).append(record)
            if verbose:
                print(
                    f"[dataset] preloading {len(records_by_path)} source shards",
                    flush=True,
                )
            progress_interval = max(1, len(records_by_path) // 10)
            for shard_number, (path, records) in enumerate(
                records_by_path.items(), start=1
            ):
                shard = torch.load(path, map_location="cpu", weights_only=False)
                episodes = shard.get("episodes") if isinstance(shard, dict) else None
                if not isinstance(episodes, list):
                    raise ValueError(f"invalid observability shard: {path}")
                for record in records:
                    record["episode"] = episodes[record["episode_index"]]
                if verbose and (
                    shard_number % progress_interval == 0
                    or shard_number == len(records_by_path)
                ):
                    print(
                        f"[dataset] preloaded {shard_number}/{len(records_by_path)} shards",
                        flush=True,
                    )

    def episode(self, record: dict) -> dict:
        episode = record.get("episode")
        if episode is not None:
            return episode
        return self.cache.load(Path(record["path"]))["episodes"][
            record["episode_index"]
        ]


class EpisodeWindowDataset(Dataset):
    """Causal windows over complete episodes stored in PyTorch shards."""

    def __init__(
        self,
        shard_paths: Sequence[Path],
        split: str,
        history: int = 5,
        split_seed: int = 0,
        holdout_tools: bool = True,
        stride: int = 1,
        catalog: EpisodeCatalog | None = None,
    ) -> None:
        if history <= 0 or stride <= 0:
            raise ValueError("history and stride must be positive")
        self.history = history
        self.stride = stride
        self.catalog = catalog or EpisodeCatalog(
            shard_paths,
            split_seed=split_seed,
            holdout_tools=holdout_tools,
        )
        self.records = [
            record for record in self.catalog.records if record["split"] == split
        ]
        self.episode_metadata = [record["metadata"] for record in self.records]
        self.cumulative_windows: list[int] = []
        total = 0
        for record in self.records:
            count = max(0, (record["length"] - history) // stride + 1)
            total += count
            self.cumulative_windows.append(total)
        if total == 0:
            raise ValueError(f"no windows found for split {split!r}")

    def __len__(self) -> int:
        return self.cumulative_windows[-1]

    def iter_episode_views(self):
        """Yield each episode, its labels, and sampled current-frame indices."""
        for record in self.records:
            steps = torch.arange(
                self.history - 1, record["length"], self.stride
            )
            yield self.catalog.episode(record), record["labels"], steps

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        record_index = bisect_right(self.cumulative_windows, index)
        previous = (
            self.cumulative_windows[record_index - 1] if record_index else 0
        )
        record = self.records[record_index]
        step = self.history - 1 + (index - previous) * self.stride
        episode = self.catalog.episode(record)
        labels = record["labels"]
        start = step - self.history + 1
        actor_state = episode["actor_state"][start : step + 1].float()
        state_geometry = torch.cat(
            (
                actor_state,
                episode["deployable_geometry"][start : step + 1].float(),
            ),
            dim=-1,
        )
        result: dict[str, torch.Tensor | str] = {
            "state": actor_state,
            "state_geometry": state_geometry,
            "compact": episode["tactile_compact"][start : step + 1].float(),
            "raw": episode["tactile_raw"][start : step + 1],
            "oracle": episode["oracle_geometry"][start : step + 1].float(),
            "episode_id": str(episode["metadata"]["episode_id"]),
            "tool_id": str(episode["metadata"].get("tool_id", "unknown")),
            "policy_source": str(
                episode["metadata"].get("policy_source", "unknown")
            ),
        }
        for key, value in labels.items():
            result[key] = value[step]
        return result


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class RawTacmapEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.project = MLP(5 * 32, 128, 64)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 5 or tuple(value.shape[-3:]) != (5, 12, 12):
            raise ValueError(f"raw tactile input must be [B,H,5,12,12], got {value.shape}")
        batch, history = value.shape[:2]
        fingers = value.float().reshape(batch * history * 5, 1, 12, 12) / 255.0
        encoded = self.cnn(fingers).reshape(batch, history, 5 * 32)
        return self.project(encoded)


class ObservabilityProbe(nn.Module):
    """Matched causal probe for state, compact tactile, or raw tactile inputs."""

    def __init__(self, state_dim: int = 140, variant: str = "state_compact") -> None:
        super().__init__()
        valid = {"state", "compact", "state_compact", "state_raw", "oracle"}
        if variant not in valid:
            raise ValueError(f"unknown probe variant {variant!r}; expected {sorted(valid)}")
        self.variant = variant
        self.state_encoder = MLP(state_dim, 128, 128)
        self.compact_encoder = MLP(25, 64, 64)
        self.raw_encoder = RawTacmapEncoder()
        self.fusion = MLP(192, 192, 128)
        self.temporal = nn.GRU(128, 128, batch_first=True)
        self.contact_mode_head = nn.Linear(128, 3)
        self.binary_heads = nn.ModuleDict(
            {name: nn.Linear(128, 1) for name in ("onset", "loss", "slip", "instability", "hard_loss")}
        )
        self.speed_head = nn.Linear(128, 2)

    def forward(
        self,
        state: torch.Tensor,
        compact: torch.Tensor,
        raw: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if state.ndim != 3 or compact.ndim != 4 or raw.ndim != 5:
            raise ValueError("probe inputs must include batch and history dimensions")
        state_feature = self.state_encoder(state)
        compact_feature = self.compact_encoder(compact.flatten(-2))
        if self.variant == "compact":
            state_feature = torch.zeros_like(state_feature)
            tactile_feature = compact_feature
        elif self.variant == "state_raw":
            tactile_feature = self.raw_encoder(raw)
        elif self.variant in {"state", "oracle"}:
            tactile_feature = torch.zeros_like(compact_feature)
        else:
            tactile_feature = compact_feature
        fused = self.fusion(torch.cat((state_feature, tactile_feature), dim=-1))
        sequence, _ = self.temporal(fused)
        feature = sequence[:, -1]
        output = {"contact_mode": self.contact_mode_head(feature)}
        output.update({name: head(feature).squeeze(-1) for name, head in self.binary_heads.items()})
        output["relative_speed"] = self.speed_head(feature)
        return output


def label_config_dict(cfg: LabelConfig) -> dict:
    return asdict(cfg)


def discover_shards(roots: Iterable[Path]) -> list[Path]:
    paths = sorted({path for root in roots for path in Path(root).glob("shard_*.pt")})
    if not paths:
        raise FileNotFoundError(f"no shard_*.pt files found under {[str(root) for root in roots]}")
    return paths


__all__ = [
    "EpisodeCatalog",
    "EpisodeWindowDataset",
    "LabelConfig",
    "ObservabilityProbe",
    "assign_split",
    "derive_labels",
    "discover_shards",
    "label_config_dict",
    "quaternion_angle_rad",
    "validate_episode",
]
