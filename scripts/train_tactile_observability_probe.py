#!/usr/bin/env python3
"""Train and evaluate tactile-observability probes from collected shards."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

_UTILS_PATH = (
    REPO_ROOT
    / "isaacsimenvs/tasks/simtoolreal/utils/tactile_observability.py"
)
_UTILS_SPEC = importlib.util.spec_from_file_location(
    "tactile_observability_offline", _UTILS_PATH
)
if _UTILS_SPEC is None or _UTILS_SPEC.loader is None:
    raise RuntimeError(f"could not load observability utilities from {_UTILS_PATH}")
_UTILS = importlib.util.module_from_spec(_UTILS_SPEC)
sys.modules[_UTILS_SPEC.name] = _UTILS
_UTILS_SPEC.loader.exec_module(_UTILS)
EpisodeCatalog = _UTILS.EpisodeCatalog
EpisodeWindowDataset = _UTILS.EpisodeWindowDataset
ObservabilityProbe = _UTILS.ObservabilityProbe
discover_shards = _UTILS.discover_shards


BINARY_HEADS = ("onset", "loss", "slip", "instability", "hard_loss")
PRIMARY_HEADS = ("onset", "loss", "slip", "instability")
VARIANTS = (
    "state",
    "state_geometry",
    "compact",
    "state_compact",
    "state_geometry_compact",
    "state_raw",
    "state_compact_shuffled",
    "state_compact_h1",
    "oracle",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_roots", nargs="+", type=Path)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "outputs/tactile_observability/probes")
    parser.add_argument(
        "--resume-dir",
        type=Path,
        default=None,
        help="Reuse completed variant/seed results in an interrupted output directory.",
    )
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "outputs/tactile_observability/cache")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--build-cache-only", action="store_true")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--history", type=int, default=5)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--episode-split",
        action="store_true",
        help="Split by episode instead of holding out collection split groups.",
    )
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--log-every-batches", type=int, default=100)
    parser.add_argument("--bootstrap-repetitions", type=int, default=100)
    parser.add_argument("--bootstrap-score-bins", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def binary_average_precision(target: np.ndarray, score: np.ndarray) -> float:
    target = target.astype(bool)
    positives = int(target.sum())
    if positives == 0 or positives == target.size:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    ranked = target[order]
    precision = np.cumsum(ranked) / np.arange(1, ranked.size + 1)
    return float(precision[ranked].sum() / positives)


def binary_auroc(target: np.ndarray, score: np.ndarray) -> float:
    target = target.astype(bool)
    positives = int(target.sum())
    negatives = int((~target).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(score, kind="stable")
    ranks = np.empty(order.size, dtype=np.float64)
    sorted_score = score[order]
    start = 0
    while start < order.size:
        end = start + 1
        while end < order.size and sorted_score[end] == sorted_score[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    return float((ranks[target].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def threshold_metrics(target: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    target = target.astype(bool)
    pred = score >= threshold
    tp = int((pred & target).sum())
    fp = int((pred & ~target).sum())
    tn = int((~pred & ~target).sum())
    fn = int((~pred & target).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return {
        "f1": 2.0 * precision * recall / max(precision + recall, 1.0e-12),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
    }


def best_f1_threshold(target: np.ndarray, score: np.ndarray) -> float:
    if target.sum() == 0 or target.sum() == target.size:
        raise RuntimeError("validation event class lacks positive or negative support")
    candidates = np.unique(np.quantile(score, np.linspace(0.0, 1.0, 201)))
    return float(max(candidates, key=lambda value: threshold_metrics(target, score, float(value))["f1"]))


def macro_f1(target: np.ndarray, prediction: np.ndarray, classes: int = 3) -> float:
    present = [label for label in range(classes) if np.any(target == label)]
    if not present:
        return float("nan")
    scores = []
    for label in present:
        scores.append(threshold_metrics(target == label, (prediction == label).astype(float), 0.5)["f1"])
    return float(np.mean(scores))


def compute_normalization(
    dataset: EpisodeWindowDataset,
) -> dict[str, torch.Tensor]:
    """Compute frame statistics without materializing overlapping windows."""
    sums: dict[str, torch.Tensor] = {}
    square_sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = defaultdict(int)
    for episode, _, steps in dataset.iter_episode_views():
        actor_state = episode["actor_state"][steps].float()
        values = {
            "state": actor_state,
            "state_geometry": torch.cat(
                (actor_state, episode["deployable_geometry"][steps].float()),
                dim=-1,
            ),
            "compact": episode["tactile_compact"][steps].float().reshape(-1, 5),
            "oracle": episode["oracle_geometry"][steps].float(),
        }
        for key, value in values.items():
            value = value.reshape(-1, value.shape[-1])
            sums[key] = sums.get(key, torch.zeros(value.shape[-1])) + value.sum(dim=0)
            square_sums[key] = square_sums.get(
                key, torch.zeros(value.shape[-1])
            ) + value.square().sum(dim=0)
            counts[key] += value.shape[0]
    result = {}
    for key in sums:
        mean = sums[key] / counts[key]
        variance = square_sums[key] / counts[key] - mean.square()
        result[f"{key}_mean"] = mean
        result[f"{key}_std"] = variance.clamp_min(1.0e-8).sqrt()
    return result



def normalize_batch(
    batch: dict,
    stats: dict[str, torch.Tensor],
    device: torch.device,
    variant: str,
) -> dict:
    result = {}
    for key in ("state", "state_geometry", "compact", "oracle"):
        value = batch[key].to(device=device, dtype=torch.float32)
        result[key] = (
            value - stats[f"{key}_mean"].to(device)
        ) / stats[f"{key}_std"].to(device)
    result["raw"] = batch["raw"].to(device)
    if variant == "oracle":
        result["state"] = result["oracle"]
    elif variant in {"state_geometry", "state_geometry_compact"}:
        result["state"] = result["state_geometry"]
    for key in ("contact_mode", "relative_speed", *BINARY_HEADS):
        result[key] = batch[key].to(device)
    for key in (
        "onset_eligible",
        "loss_eligible",
        "slip_eligible",
        "instability_eligible",
        "hard_loss_eligible",
    ):
        result[key] = batch[key].to(device).bool()
    return result

def collect_support(
    dataset: EpisodeWindowDataset,
) -> dict[str, tuple[int, int]]:
    """Count labels directly from cached episodes instead of DataLoader samples."""
    counts = {name: [0, 0] for name in BINARY_HEADS}
    contact_counts = [0, 0, 0]
    for _, labels, steps in dataset.iter_episode_views():
        contact_mode = labels["contact_mode"][steps]
        mode_counts = torch.bincount(contact_mode, minlength=3)
        for label in range(3):
            contact_counts[label] += int(mode_counts[label])
        for name in BINARY_HEADS:
            mask = labels[f"{name}_eligible"][steps].bool()
            target = labels[name][steps][mask].bool()
            counts[name][0] += int((~target).sum())
            counts[name][1] += int(target.sum())
    if sum(count > 0 for count in contact_counts) < 2:
        raise RuntimeError(
            f"contact mode needs at least two observed classes: {contact_counts}"
        )
    for name, (negative, positive) in counts.items():
        if negative == 0 or positive == 0:
            raise RuntimeError(
                f"{name} lacks class support: negative={negative}, positive={positive}"
            )
    return {name: tuple(value) for name, value in counts.items()}


def loss_for_batch(output: dict, batch: dict, support: dict[str, tuple[int, int]]) -> torch.Tensor:
    contact_counts = torch.bincount(batch["contact_mode"].long(), minlength=3).float()
    contact_weight = contact_counts.sum() / contact_counts.clamp_min(1.0)
    contact_weight = (contact_weight / contact_weight.mean()).clamp(max=10.0)
    loss = F.cross_entropy(output["contact_mode"], batch["contact_mode"].long(), weight=contact_weight)
    for name in BINARY_HEADS:
        mask = batch[f"{name}_eligible"]
        negative, positive = support[name]
        pos_weight = torch.tensor(min(negative / positive, 10.0), device=loss.device)
        if mask.any():
            loss = loss + F.binary_cross_entropy_with_logits(
                output[name][mask], batch[name].float()[mask], pos_weight=pos_weight
            )
    loss = loss + F.smooth_l1_loss(output["relative_speed"], batch["relative_speed"].float())
    return loss


@torch.no_grad()
def predict(model, loader, stats, device, variant) -> dict[str, np.ndarray]:
    model.eval()
    values: dict[str, list[np.ndarray]] = defaultdict(list)
    for raw_batch in loader:
        batch = normalize_batch(raw_batch, stats, device, variant)
        if variant == "state_compact_shuffled" and batch["compact"].shape[0] > 1:
            batch["compact"] = torch.roll(batch["compact"], 1, dims=0)
        output = model(batch["state"], batch["compact"], batch["raw"])
        episode_ids = np.asarray(raw_batch["episode_id"], dtype=object)
        policy_sources = np.asarray(raw_batch["policy_source"], dtype=object)
        values["contact_mode_target"].append(batch["contact_mode"].cpu().numpy())
        values["contact_mode_score"].append(output["contact_mode"].softmax(-1).cpu().numpy())
        values["contact_mode_episode_id"].append(episode_ids)
        values["contact_mode_policy_source"].append(policy_sources)
        values["relative_speed_target"].append(batch["relative_speed"].cpu().numpy())
        values["relative_speed_score"].append(output["relative_speed"].cpu().numpy())
        for name in BINARY_HEADS:
            mask = batch[f"{name}_eligible"]
            values[f"{name}_target"].append(batch[name][mask].cpu().numpy())
            values[f"{name}_score"].append(output[name][mask].sigmoid().cpu().numpy())
            mask_cpu = mask.cpu().numpy()
            values[f"{name}_episode_id"].append(episode_ids[mask_cpu])
            values[f"{name}_policy_source"].append(policy_sources[mask_cpu])
    return {key: np.concatenate(items) for key, items in values.items()}


def bootstrap_primary_auprc(
    values: dict[str, np.ndarray],
    repetitions: int = 100,
    score_bins: int = 256,
) -> list[float]:
    """Episode-cluster bootstrap using per-episode score histograms.

    Sorting every resampled test window made the previous implementation scale
    quadratically in episodes and windows. Histograms preserve clustered
    resampling while bounding each replicate by ``score_bins`` ranked groups.
    """
    if repetitions <= 0:
        return [float("nan"), float("nan")]
    if score_bins < 16:
        raise ValueError("bootstrap score bins must be at least 16")
    episode_ids = np.unique(values["contact_mode_episode_id"])
    if episode_ids.size < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(0)
    episode_lookup = {episode_id: index for index, episode_id in enumerate(episode_ids)}
    sampled_counts = rng.multinomial(
        episode_ids.size,
        np.full(episode_ids.size, 1.0 / episode_ids.size),
        size=repetitions,
    ).astype(np.float32)
    replicate_heads = []
    for name in PRIMARY_HEADS:
        target = values[f"{name}_target"].astype(bool)
        score = np.clip(values[f"{name}_score"], 0.0, 1.0)
        ids = values[f"{name}_episode_id"]
        episode_index = np.fromiter(
            (episode_lookup[item] for item in ids), dtype=np.int64, count=ids.size
        )
        bin_index = np.minimum(
            (score * score_bins).astype(np.int64), score_bins - 1
        )
        positive = np.zeros((episode_ids.size, score_bins), dtype=np.float32)
        negative = np.zeros_like(positive)
        np.add.at(positive, (episode_index[target], bin_index[target]), 1.0)
        np.add.at(negative, (episode_index[~target], bin_index[~target]), 1.0)

        positive = (sampled_counts @ positive)[:, ::-1]
        negative = (sampled_counts @ negative)[:, ::-1]
        cumulative_positive = np.cumsum(positive, axis=1)
        cumulative_total = np.cumsum(positive + negative, axis=1)
        precision = cumulative_positive / np.maximum(cumulative_total, 1.0)
        total_positive = positive.sum(axis=1)
        average_precision = (
            (precision * positive).sum(axis=1) / np.maximum(total_positive, 1.0)
        )
        average_precision[total_positive == 0] = np.nan
        replicate_heads.append(average_precision)
    scores = np.nanmean(np.stack(replicate_heads, axis=1), axis=1)
    return np.nanquantile(scores, [0.025, 0.975]).tolist()


def subset_predictions(values: dict[str, np.ndarray], source: str) -> dict[str, np.ndarray]:
    result = {}
    contact_mask = values["contact_mode_policy_source"] == source
    common_keys = (
        "contact_mode_target", "contact_mode_score", "contact_mode_episode_id",
        "contact_mode_policy_source", "relative_speed_target",
        "relative_speed_score",
    )
    for key in common_keys:
        result[key] = values[key][contact_mask]
    for name in BINARY_HEADS:
        mask = values[f"{name}_policy_source"] == source
        for suffix in ("target", "score", "episode_id", "policy_source"):
            key = f"{name}_{suffix}"
            result[key] = values[key][mask]
    return result


def evaluate_predictions(
    values: dict[str, np.ndarray],
    thresholds: dict[str, float] | None = None,
    include_groups: bool = True,
    include_bootstrap: bool = True,
    bootstrap_repetitions: int = 100,
    bootstrap_score_bins: int = 256,
) -> tuple[dict, dict]:
    thresholds = dict(thresholds or {})
    result = {}
    contact_target = values["contact_mode_target"]
    contact_prediction = values["contact_mode_score"].argmax(axis=-1)
    result["contact_mode_macro_f1"] = macro_f1(contact_target, contact_prediction)
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (contact_target, contact_prediction), 1)
    result["contact_mode_confusion"] = confusion.tolist()
    for name in BINARY_HEADS:
        target = values[f"{name}_target"]
        score = values[f"{name}_score"]
        threshold = thresholds.get(name)
        if threshold is None:
            threshold = best_f1_threshold(target, score)
            thresholds[name] = threshold
        result[name] = {
            "support": int(target.size),
            "positive": int(target.sum()),
            "auprc": binary_average_precision(target, score),
            "auroc": binary_auroc(target, score),
            "threshold": threshold,
            **threshold_metrics(target, score, threshold),
        }
    target = values["relative_speed_target"]
    score = values["relative_speed_score"]
    residual = target - score
    result["relative_speed"] = {
        "mae": np.abs(residual).mean(axis=0).tolist(),
        "r2": (
            1.0
            - (residual * residual).sum(axis=0)
            / np.maximum(
                ((target - target.mean(axis=0)) ** 2).sum(axis=0), 1.0e-12
            )
        ).tolist(),
    }
    result["primary_macro_auprc"] = float(np.nanmean([result[name]["auprc"] for name in PRIMARY_HEADS]))
    result["primary_macro_auprc_ci95"] = (
        bootstrap_primary_auprc(
            values,
            repetitions=bootstrap_repetitions,
            score_bins=bootstrap_score_bins,
        )
        if include_bootstrap
        else None
    )
    if include_groups:
        result["by_policy_source"] = {}
        for source in sorted(set(values["contact_mode_policy_source"].tolist())):
            subset = subset_predictions(values, source)
            try:
                grouped, _ = evaluate_predictions(
                    subset,
                    thresholds,
                    include_groups=False,
                    include_bootstrap=False,
                )
            except (RuntimeError, ValueError):
                continue
            result["by_policy_source"][source] = grouped
    return result, thresholds


def gate_decision(results: list[dict]) -> dict:
    by_variant = defaultdict(dict)
    for row in results:
        by_variant[row["variant"]][row["seed"]] = row["test"]
    required = {"state", "state_compact", "state_compact_shuffled"}
    if not required.issubset(by_variant):
        return {"evaluated": False, "reason": f"gate requires variants {sorted(required)}"}
    common_seeds = sorted(set.intersection(*(set(by_variant[name]) for name in required)))
    if not common_seeds:
        return {"evaluated": False, "reason": "gate variants have no common seeds"}
    heads = {}
    for head in PRIMARY_HEADS:
        deltas = [
            by_variant["state_compact"][seed][head]["auprc"]
            - by_variant["state"][seed][head]["auprc"]
            for seed in common_seeds
        ]
        shuffled = [
            by_variant["state_compact_shuffled"][seed][head]["auprc"]
            - by_variant["state"][seed][head]["auprc"]
            for seed in common_seeds
        ]
        heads[head] = {
            "delta_by_seed": deltas,
            "mean_delta": float(np.mean(deltas)),
            "same_positive_sign": all(value > 0.0 for value in deltas),
            "shuffled_mean_delta": float(np.mean(shuffled)),
            "passes": bool(
                np.mean(deltas) >= 0.05
                and all(value > 0.0 for value in deltas)
                and np.mean(shuffled) <= 0.02
            ),
        }
    return {"evaluated": True, "passed": any(value["passes"] for value in heads.values()), "heads": heads}


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def plot_summary(all_results: list[dict], output_path: Path) -> None:
    names = [name for name in ("contact_mode_macro_f1", *PRIMARY_HEADS)]
    variants = sorted({item["variant"] for item in all_results})
    means = []
    errors = []
    for variant in variants:
        rows = [item for item in all_results if item["variant"] == variant]
        matrix = []
        for row in rows:
            matrix.append(
                [
                    row["test"][name]
                    if name == "contact_mode_macro_f1"
                    else row["test"][name]["auprc"]
                    for name in names
                ]
            )
        matrix = np.asarray(matrix)
        means.append(matrix.mean(axis=0))
        errors.append(matrix.std(axis=0))
    x = np.arange(len(variants))
    width = 0.16
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for index, name in enumerate(names):
        ax.bar(
            x + (index - 2) * width,
            np.asarray(means)[:, index],
            width,
            yerr=np.asarray(errors)[:, index],
            label=name.replace("_", " "),
        )
    labels = [
        item.replace("state_", "S+").replace("compact", "Tac")
        for item in variants
    ]
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Held-out score")
    ax.legend(ncol=3, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.history <= 0 or args.batch_size <= 0 or args.epochs <= 0:
        raise ValueError("history, batch size, and epochs must be positive")
    if args.bootstrap_repetitions <= 0 or args.bootstrap_score_bins < 16:
        raise ValueError(
            "bootstrap repetitions must be positive and score bins at least 16"
        )
    shards = discover_shards(args.data_roots)
    if args.resume_dir is None:
        run_dir = args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = args.resume_dir.resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"resume directory does not exist: {run_dir}")
    device = torch.device(args.device)
    print(
        f"[setup] {len(shards)} shards, device={device}, output={run_dir}",
        flush=True,
    )
    catalog = EpisodeCatalog(
        shards,
        split_seed=args.split_seed,
        holdout_tools=not args.episode_split,
        preload=True,
        cache_dir=args.cache_dir,
        rebuild_cache=args.rebuild_cache,
        verbose=True,
    )
    if args.build_cache_only:
        print("[output] preprocessing cache is ready", flush=True)
        return
    datasets = {}
    prepared = {}
    for history in sorted({args.history, 1}):
        datasets[history] = {
            split: EpisodeWindowDataset(
                shards,
                split,
                history=history,
                stride=args.stride,
                catalog=catalog,
            )
            for split in ("train", "validation", "test")
        }
        split_sizes = {
            split: len(dataset) for split, dataset in datasets[history].items()
        }
        print(
            f"[dataset] history={history} windows={split_sizes}",
            flush=True,
        )
        print(f"[dataset] computing history={history} normalization", flush=True)
        prepared[history] = {
            "stats": compute_normalization(datasets[history]["train"]),
            "support": collect_support(datasets[history]["train"]),
        }
    all_results = []
    for variant in args.variants:
        history = 1 if variant == "state_compact_h1" else args.history
        loaders = {
            split: DataLoader(
                dataset, batch_size=args.batch_size, shuffle=split == "train",
                num_workers=args.num_workers, pin_memory=device.type == "cuda",
            )
            for split, dataset in datasets[history].items()
        }
        stats = prepared[history]["stats"]
        support = prepared[history]["support"]
        actor_state_dim = datasets[history]["train"].actor_state_dim
        split_dims = {
            dataset.actor_state_dim for dataset in datasets[history].values()
        }
        if split_dims != {actor_state_dim}:
            raise ValueError(
                f"actor-state dimensions differ across splits: {sorted(split_dims)}"
            )
        print(
            f"[train] variant={variant} history={history} "
            f"train_windows={len(datasets[history]['train'])}",
            flush=True,
        )
        model_variant = {
            "state_compact_shuffled": "state_compact",
            "state_compact_h1": "state_compact",
            "state_geometry": "state",
            "state_geometry_compact": "state_compact",
        }.get(variant, variant)
        for seed in args.seeds:
            result_path = run_dir / f"{variant}_seed{seed}.json"
            checkpoint_path = run_dir / f"{variant}_seed{seed}.pt"
            if result_path.exists() and not checkpoint_path.exists():
                raise RuntimeError(
                    f"saved metrics have no model checkpoint: {result_path}"
                )
            if result_path.exists():
                result = json.loads(result_path.read_text())
                if result.get("variant") != variant or int(result.get("seed", -1)) != seed:
                    raise RuntimeError(f"saved result identity mismatch: {result_path}")
                all_results.append(result)
                print(f"[resume] reusing {variant} seed={seed}", flush=True)
                continue
            seed_everything(seed)
            if variant == "oracle":
                state_dim = 3
            elif variant in {"state_geometry", "state_geometry_compact"}:
                state_dim = actor_state_dim + 13
            else:
                state_dim = actor_state_dim
            model = ObservabilityProbe(state_dim=state_dim, variant=model_variant).to(device)
            if checkpoint_path.exists():
                saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                if saved.get("variant") != variant or int(saved.get("state_dim", -1)) != state_dim:
                    raise RuntimeError(f"saved checkpoint identity mismatch: {checkpoint_path}")
                best_state = saved.get("model")
                history_rows = saved.get("history", [])
                if not isinstance(best_state, dict):
                    raise RuntimeError(f"saved checkpoint has no model: {checkpoint_path}")
                print(
                    f"[resume] evaluating trained checkpoint {variant} seed={seed}",
                    flush=True,
                )
            else:
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
                )
                best_state = None
                best_score = -float("inf")
                stale = 0
                history_rows = []
                for epoch in range(args.epochs):
                    model.train()
                    total_loss = 0.0
                    batches = 0
                    for batch_index, raw_batch in enumerate(loaders["train"]):
                        batch = normalize_batch(raw_batch, stats, device, variant)
                        if variant == "state_compact_shuffled" and batch["compact"].shape[0] > 1:
                            batch["compact"] = batch["compact"][torch.randperm(batch["compact"].shape[0], device=device)]
                        output = model(batch["state"], batch["compact"], batch["raw"])
                        loss = loss_for_batch(output, batch, support)
                        if not torch.isfinite(loss):
                            raise RuntimeError(f"non-finite loss for {variant}, seed {seed}")
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        optimizer.step()
                        total_loss += float(loss.item())
                        batches += 1
                        if (
                            args.log_every_batches > 0
                            and batches % args.log_every_batches == 0
                        ):
                            print(
                                f"[{variant} seed={seed}] epoch={epoch + 1} "
                                f"batch={batches}/{len(loaders['train'])} "
                                f"loss={total_loss / batches:.4f}",
                                flush=True,
                            )
                        if args.max_train_batches and batch_index + 1 >= args.max_train_batches:
                            break
                    validation_values = predict(model, loaders["validation"], stats, device, variant)
                    validation, _ = evaluate_predictions(
                        validation_values,
                        include_groups=False,
                        include_bootstrap=False,
                    )
                    score = validation["primary_macro_auprc"]
                    history_rows.append(
                        {
                            "epoch": epoch + 1,
                            "train_loss": total_loss / max(batches, 1),
                            "validation_macro_auprc": score,
                        }
                    )
                    print(
                        f"[{variant} seed={seed}] epoch={epoch + 1} "
                        f"loss={history_rows[-1]['train_loss']:.4f} "
                        f"val_AUPRC={score:.4f}",
                        flush=True,
                    )
                    if score > best_score:
                        best_score = score
                        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                        stale = 0
                    else:
                        stale += 1
                        if stale >= args.patience:
                            break
                if best_state is None:
                    raise RuntimeError("training produced no valid checkpoint")
                torch.save(
                    {
                        "model": best_state,
                        "normalization": stats,
                        "thresholds": None,
                        "variant": variant,
                        "state_dim": state_dim,
                        "history": history_rows,
                        "status": "trained",
                    },
                    checkpoint_path,
                )
            model.load_state_dict(best_state)
            validation_values = predict(model, loaders["validation"], stats, device, variant)
            validation, thresholds = evaluate_predictions(
                validation_values,
                bootstrap_repetitions=args.bootstrap_repetitions,
                bootstrap_score_bins=args.bootstrap_score_bins,
            )
            test_values = predict(model, loaders["test"], stats, device, variant)
            test, _ = evaluate_predictions(
                test_values,
                thresholds,
                bootstrap_repetitions=args.bootstrap_repetitions,
                bootstrap_score_bins=args.bootstrap_score_bins,
            )
            result = {
                "variant": variant,
                "seed": seed,
                "validation": validation,
                "test": test,
                "history": history_rows,
            }
            all_results.append(result)
            torch.save(
                {
                    "model": best_state,
                    "normalization": stats,
                    "thresholds": thresholds,
                    "variant": variant,
                    "state_dim": state_dim,
                    "history": history_rows,
                    "status": "complete",
                },
                checkpoint_path,
            )
            result_path.write_text(json.dumps(finite_json(result), indent=2) + "\n")
    summary = {
        "args": vars(args)
        | {
            "data_roots": [str(item) for item in args.data_roots],
            "output_root": str(args.output_root),
        },
        "shards": [str(path) for path in shards],
        "gate": gate_decision(all_results),
        "results": all_results,
    }
    (run_dir / "summary.json").write_text(json.dumps(finite_json(summary), indent=2, default=str) + "\n")
    plot_summary(all_results, run_dir / "probe_comparison.png")
    print(f"[output] {run_dir}", flush=True)


if __name__ == "__main__":
    main()
