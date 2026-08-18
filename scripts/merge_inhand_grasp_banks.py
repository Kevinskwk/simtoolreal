#!/usr/bin/env python3
"""Merge compatible partial in-hand grasp banks into one validated bank."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
UTILS_PATH = (
    REPO_ROOT / "isaacsimenvs/tasks/simtoolreal/utils/inhand_grasp_bank.py"
)


def load_utils():
    spec = importlib.util.spec_from_file_location("inhand_grasp_bank_merge_utils", UTILS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load grasp-bank utilities from {UTILS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", type=Path, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing = [path for path in args.inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"partial grasp banks are missing: {missing}")
    utils = load_utils()
    payloads = [json.loads(path.read_text()) for path in args.inputs]
    merged = utils.merge_grasp_banks(payloads)
    merged["source_banks"] = [str(path.resolve()) for path in args.inputs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2, allow_nan=False) + "\n")
    print(
        f"[output] {args.output.resolve()} entries={len(merged['entries'])} "
        f"partials={len(payloads)} duplicates_removed={merged['duplicates_removed']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
