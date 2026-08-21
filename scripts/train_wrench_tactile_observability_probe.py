#!/usr/bin/env python3
"""Train matched probes on controlled held-tool wrench/TacMap sequences."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import importlib.util
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


_UTILS_PATH = (
    Path(__file__).resolve().parents[1]
    / "isaacsimenvs/tasks/simtoolreal/utils/wrench_tactile_observability.py"
)
_SPEC = importlib.util.spec_from_file_location("wrench_tactile_observability_offline", _UTILS_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load probe utilities from {_UTILS_PATH}")
_UTILS = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _UTILS
_SPEC.loader.exec_module(_UTILS)
WRENCH_MODE_NAMES = _UTILS.WRENCH_MODE_NAMES
WrenchProbe = _UTILS.WrenchProbe
WrenchWindowDataset = _UTILS.WrenchWindowDataset
catalog = _UTILS.catalog


REPO_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (
    "state",
    "compact",
    "state_compact",
    "compact_shuffled",
    "state_compact_shuffled",
    "normal_impulse",
    "normal_impulse_shuffled",
    "tangential_impulse",
    "tangential_impulse_shuffled",
    "impulse",
    "impulse_shuffled",
    "impulse_compact",
    "impulse_compact_shuffled",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path,
        default=REPO_ROOT / "outputs/wrench_tactile_observability/probes",
    )
    parser.add_argument(
        "--resume-dir", type=Path, default=None,
        help="Continue an interrupted run in-place, skipping validated variant/seed outputs.",
    )
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--history", type=int, default=5)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="DataLoader workers. Keep at 0: this dataset is already memory-resident.",
    )
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_shards(data_dir: Path) -> list[Path]:
    paths = sorted(data_dir.resolve().glob("shard_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no shard_*.pt files found under {data_dir}")
    return paths


def state_statistics(dataset: WrenchWindowDataset) -> tuple[torch.Tensor, torch.Tensor]:
    count = 0
    total = None
    total_sq = None
    for episode in dataset.episodes:
        values = episode["proprio_state"].double()
        total = values.sum(0) if total is None else total + values.sum(0)
        total_sq = values.square().sum(0) if total_sq is None else total_sq + values.square().sum(0)
        count += values.shape[0]
    if count == 0 or total is None or total_sq is None:
        raise RuntimeError("training split has no state frames")
    mean = total / count
    variance = (total_sq / count - mean.square()).clamp_min(1.0e-8)
    return mean.float(), variance.sqrt().float()


def impulse_statistics(dataset: WrenchWindowDataset) -> tuple[torch.Tensor, torch.Tensor]:
    count = 0
    total = torch.zeros(5, 6, dtype=torch.float64)
    total_sq = torch.zeros_like(total)
    for episode in dataset.episodes:
        if "finger_normal_impulse_palm_ns" not in episode:
            raise ValueError("dataset does not contain v2 finger impulse measurements")
        values = torch.cat(
            (
                episode["finger_normal_impulse_palm_ns"],
                episode["finger_tangential_impulse_palm_ns"],
            ),
            dim=-1,
        ).double()
        total += values.sum(0)
        total_sq += values.square().sum(0)
        count += values.shape[0]
    if count == 0:
        raise RuntimeError("training split has no impulse frames")
    mean = total / count
    variance = (total_sq / count - mean.square()).clamp_min(1.0e-12)
    return mean.float(), variance.sqrt().float()


def prepare_batch(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    impulse_mean: torch.Tensor,
    impulse_std: torch.Tensor,
    variant: str,
    max_force: float,
    max_torque: float,
) -> dict[str, torch.Tensor]:
    result = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    result["state"] = (result["state"] - mean) / std
    if "impulse" in result:
        result["impulse"] = (
            result["impulse"] - impulse_mean
        ) / impulse_std
    if variant.endswith("_shuffled") and result["compact"].shape[0] > 1:
        # A fixed cyclic mismatch is deterministic in evaluation and never leaks the label.
        result["compact"] = torch.roll(
            result["compact"], shifts=max(1, result["compact"].shape[0] // 2), dims=0
        )
        if "impulse" in variant:
            result["impulse"] = torch.roll(
                result["impulse"],
                shifts=max(1, result["impulse"].shape[0] // 2),
                dims=0,
            )
    result["wrench_target"] = torch.cat(
        (result["force"] / max_force, result["torque"] / max_torque), dim=-1
    )
    return result


def macro_f1(target: np.ndarray, prediction: np.ndarray) -> float:
    scores = []
    for class_id in range(len(WRENCH_MODE_NAMES)):
        true_positive = np.sum((target == class_id) & (prediction == class_id))
        false_positive = np.sum((target != class_id) & (prediction == class_id))
        false_negative = np.sum((target == class_id) & (prediction != class_id))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


@torch.no_grad()
def evaluate(
    model: WrenchProbe,
    loader: DataLoader,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    impulse_mean: torch.Tensor,
    impulse_std: torch.Tensor,
    variant: str,
    max_force: float,
    max_torque: float,
) -> dict[str, float]:
    model.eval()
    targets, predictions, true_wrench, predicted_wrench = [], [], [], []
    for raw_batch in loader:
        batch = prepare_batch(
            raw_batch, device, mean, std, impulse_mean, impulse_std,
            variant, max_force, max_torque,
        )
        logits, normalized = model(
            batch["state"], batch["compact"], batch.get("impulse")
        )
        targets.append(batch["mode"].cpu())
        predictions.append(logits.argmax(-1).cpu())
        true_wrench.append(torch.cat((batch["force"], batch["torque"]), -1).cpu())
        scale = torch.tensor(
            [max_force] * 3 + [max_torque] * 3, device=device
        )
        predicted_wrench.append((normalized * scale).cpu())
    target = torch.cat(targets).numpy()
    prediction = torch.cat(predictions).numpy()
    truth = torch.cat(true_wrench).numpy()
    estimate = torch.cat(predicted_wrench).numpy()
    force_active = np.linalg.norm(truth[:, :3], axis=1) > 0.0
    torque_active = np.linalg.norm(truth[:, 3:], axis=1) > 0.0
    force_mae = np.linalg.norm(estimate[force_active, :3] - truth[force_active, :3], axis=1)
    torque_mae = np.linalg.norm(estimate[torque_active, 3:] - truth[torque_active, 3:], axis=1)
    return {
        "mode_accuracy": float(np.mean(target == prediction)),
        "mode_macro_f1": macro_f1(target, prediction),
        "force_vector_mae_n": float(np.mean(force_mae)),
        "torque_vector_mae_nm": float(np.mean(torque_mae)),
        "samples": int(target.size),
    }


def plot_results(results: list[dict], output_dir: Path) -> None:
    variants = [variant for variant in VARIANTS if any(r["variant"] == variant for r in results)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    specifications = (
        ("mode_macro_f1", "Wrench mode macro-F1", (0.0, 1.0)),
        ("force_vector_mae_n", "Force vector MAE (N)", None),
        ("torque_vector_mae_nm", "Torque vector MAE (N m)", None),
    )
    labels = [value.replace("state", "proprio").replace("_", "\n") for value in variants]
    for axis, (key, title, limits) in zip(axes, specifications):
        means, errors = [], []
        for variant in variants:
            values = [r["test"][key] for r in results if r["variant"] == variant]
            means.append(np.mean(values))
            errors.append(np.std(values))
        axis.bar(np.arange(len(variants)), means, yerr=errors, color="#3B6EA8", capsize=3)
        axis.set_xticks(np.arange(len(variants)), labels, fontsize=8)
        axis.set_title(title, fontsize=11)
        if limits:
            axis.set_ylim(*limits)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Controlled tool-wrench observability (mean +/- SD across seeds)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "matched_probe_results.png", dpi=200)
    fig.savefig(output_dir / "matched_probe_results.pdf")
    plt.close(fig)


def load_completed_results(
    output_dir: Path, variants: list[str], seeds: list[int]
) -> list[dict]:
    """Load only complete JSON/checkpoint pairs from an interrupted run."""
    requested = {(variant, seed) for variant in variants for seed in seeds}
    completed: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for json_path in sorted(output_dir.glob("*_seed*.json")):
        result = json.loads(json_path.read_text())
        key = (str(result.get("variant")), int(result.get("seed", -1)))
        if key not in requested:
            continue
        if key in seen:
            raise RuntimeError(f"duplicate completed result for {key} in {output_dir}")
        checkpoint_path = output_dir / f"{key[0]}_seed{key[1]}.pt"
        if not checkpoint_path.is_file():
            raise RuntimeError(f"result has no matching checkpoint: {json_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("variant") != key[0] or "model" not in checkpoint:
            raise RuntimeError(f"invalid checkpoint paired with {json_path}")
        if not isinstance(result.get("test"), dict) or "mode_macro_f1" not in result["test"]:
            raise RuntimeError(f"incomplete result file: {json_path}")
        completed.append(result)
        seen.add(key)
    return completed


def main() -> None:
    args = parse_args()
    if args.history <= 0 or args.stride <= 0 or args.batch_size <= 0:
        raise ValueError("history, stride, and batch size must be positive")
    if args.epochs <= 0 or args.patience <= 0:
        raise ValueError("epochs and patience must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    refs = catalog(find_shards(args.data_dir), args.split_seed)
    counts = {split: sum(ref.split == split for ref in refs) for split in ("train", "validation", "test")}
    if min(counts.values()) == 0:
        raise RuntimeError(f"grouped data split is empty: {counts}")
    datasets = {
        split: WrenchWindowDataset(refs, split, args.history, args.stride)
        for split in ("train", "validation", "test")
    }
    mean, std = state_statistics(datasets["train"])
    needs_impulse = any("impulse" in variant for variant in args.variants)
    if needs_impulse:
        impulse_mean, impulse_std = impulse_statistics(datasets["train"])
    else:
        # Preserve compatibility with schema-v1 datasets when only the
        # original proprio/TacMap variants are requested.
        impulse_mean = torch.zeros(5, 6)
        impulse_std = torch.ones(5, 6)
    state_dim = int(mean.numel())
    max_force = max(
        float(episode["commanded_force_palm_n"].abs().max())
        for episode in datasets["train"].episodes
    )
    max_torque = max(
        float(episode["commanded_torque_palm_nm"].abs().max())
        for episode in datasets["train"].episodes
    )
    if max_force <= 0.0 or max_torque <= 0.0:
        raise RuntimeError("training data does not contain both force and torque commands")
    if args.resume_dir is None:
        output_dir = args.output_root.resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=False)
        results: list[dict] = []
    else:
        output_dir = args.resume_dir.resolve()
        if not output_dir.is_dir():
            raise FileNotFoundError(f"resume directory does not exist: {output_dir}")
        results = load_completed_results(output_dir, args.variants, args.seeds)
        print(
            f"[resume] loaded {len(results)}/{len(args.variants) * len(args.seeds)} "
            f"completed fits from {output_dir}",
            flush=True,
        )
    completed_keys = {(item["variant"], int(item["seed"])) for item in results}
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
        for split, dataset in datasets.items()
    }
    run_config = {
        "data_dir": str(args.data_dir.resolve()),
        "variants": list(args.variants),
        "seeds": list(args.seeds),
        "history": args.history,
        "stride": args.stride,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "split_seed": args.split_seed,
    }
    execution = {
        "started_at": datetime.now().astimezone().isoformat(),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "resume": args.resume_dir is not None,
    }
    config_path = output_dir / "run_config.json"
    if config_path.exists():
        existing_config = json.loads(config_path.read_text())
        comparable_keys = ("data_dir", "history", "stride", "split_seed")
        mismatches = {
            key: (existing_config.get(key), run_config.get(key))
            for key in comparable_keys
            if existing_config.get(key) != run_config.get(key)
        }
        if mismatches:
            raise RuntimeError(f"resume configuration mismatch: {mismatches}")
        history = existing_config.setdefault("execution_history", [])
        if not history:
            history.append({
                "started_at": None,
                "batch_size": existing_config.get("batch_size"),
                "num_workers": existing_config.get("num_workers"),
                "resume": True,
            })
        history.append(execution)
        config_path.write_text(json.dumps(existing_config, indent=2) + "\n")
    else:
        run_config["execution_history"] = [execution]
        config_path.write_text(json.dumps(run_config, indent=2) + "\n")
    for variant in args.variants:
        model_variant = variant.removesuffix("_shuffled")
        for seed in args.seeds:
            if (variant, seed) in completed_keys:
                print(f"[resume] skip completed {variant} seed={seed}", flush=True)
                continue
            seed_all(seed)
            model = WrenchProbe(state_dim, model_variant).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
            )
            mean_device, std_device = mean.to(device), std.to(device)
            impulse_mean_device = impulse_mean.to(device)
            impulse_std_device = impulse_std.to(device)
            best_score = -float("inf")
            best_state = None
            stale = 0
            history = []
            for epoch in range(1, args.epochs + 1):
                model.train()
                loss_sum = 0.0
                batches = 0
                for raw_batch in loaders["train"]:
                    batch = prepare_batch(
                        raw_batch, device, mean_device, std_device,
                        impulse_mean_device, impulse_std_device, variant,
                        max_force, max_torque,
                    )
                    logits, wrench = model(
                        batch["state"], batch["compact"], batch.get("impulse")
                    )
                    loss = F.cross_entropy(logits, batch["mode"]) + F.smooth_l1_loss(
                        wrench, batch["wrench_target"], beta=0.2
                    )
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"non-finite loss for {variant}, seed {seed}")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    loss_sum += float(loss)
                    batches += 1
                validation = evaluate(
                    model, loaders["validation"], device, mean_device, std_device,
                    impulse_mean_device, impulse_std_device,
                    variant, max_force, max_torque,
                )
                history.append({"epoch": epoch, "loss": loss_sum / batches, **validation})
                print(
                    f"[{variant} seed={seed}] epoch={epoch} loss={loss_sum / batches:.4f} "
                    f"val_F1={validation['mode_macro_f1']:.3f} "
                    f"force_MAE={validation['force_vector_mae_n']:.3f}N "
                    f"torque_MAE={validation['torque_vector_mae_nm']:.4f}Nm",
                    flush=True,
                )
                score = validation["mode_macro_f1"]
                if score > best_score:
                    best_score = score
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                    if stale >= args.patience:
                        break
            if best_state is None:
                raise RuntimeError("training did not produce a checkpoint")
            model.load_state_dict(best_state)
            test = evaluate(
                model, loaders["test"], device, mean_device, std_device,
                impulse_mean_device, impulse_std_device,
                variant, max_force, max_torque,
            )
            result = {
                "variant": variant,
                "seed": seed,
                "validation_best_macro_f1": best_score,
                "test": test,
                "history": history,
            }
            results.append(result)
            torch.save(
                {
                    "model": best_state,
                    "variant": variant,
                    "state_mean": mean,
                    "state_std": std,
                    "impulse_mean": impulse_mean,
                    "impulse_std": impulse_std,
                    "max_force_n": max_force,
                    "max_torque_nm": max_torque,
                    "history": history,
                },
                output_dir / f"{variant}_seed{seed}.pt",
            )
            (output_dir / f"{variant}_seed{seed}.json").write_text(
                json.dumps(result, indent=2, allow_nan=False) + "\n"
            )
    summary = {
        "created_at": datetime.now().astimezone().isoformat(),
        "data_dir": str(args.data_dir.resolve()),
        "split_episode_counts": counts,
        "history": args.history,
        "stride": args.stride,
        "wrench_modes": list(WRENCH_MODE_NAMES),
        "results": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    plot_results(results, output_dir)
    print(f"[output] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
