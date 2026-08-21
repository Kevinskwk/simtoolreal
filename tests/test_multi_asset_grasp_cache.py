from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "isaacsimenvs/tasks/simtoolreal/utils/inhand_grasp_bank.py"
spec = importlib.util.spec_from_file_location("multi_asset_grasp_bank_test", PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def make_multi_cache() -> dict:
    source = json.loads(
        (ROOT / "assets/grasp_banks/eraser_canonical_v2.json").read_text()
    )
    entry = copy.deepcopy(source["entries"][0])
    common = {
        "schema_version": 3,
        "kind": "simtoolreal_multi_asset_grasp_cache",
        "source_checkpoint_sha256": source["source_checkpoint_sha256"],
        "policy_coefficient_id": 0.0,
        "tactile_rich_fraction_min": 0.0,
        "tactile_min_fingers": 1,
        "control_dt_s": source["control_dt_s"],
    }
    common["assets"] = [
        {
            "asset_index": 0,
            "tool_type": "eraser",
            "object_name": "procedural_0000_eraser",
            "asset_sha256": "1" * 64,
            "object_scale": [1.0, 1.0, 1.0],
            "entries": [copy.deepcopy(entry)],
        },
        {
            "asset_index": 1,
            "tool_type": "hammer",
            "object_name": "procedural_0001_hammer",
            "asset_sha256": "2" * 64,
            "object_scale": [1.0, 1.0, 1.0],
            "entries": [copy.deepcopy(entry)],
        },
    ]
    return common


def test_multi_asset_cache_validates_and_flattens_in_asset_order():
    payload = make_multi_cache()
    module.validate_grasp_bank(payload)
    entries, asset_indices = module.flatten_multi_asset_grasp_bank(payload)
    assert len(entries) == 2
    assert asset_indices == [0, 1]


def test_multi_asset_cache_rejects_missing_per_asset_coverage():
    payload = make_multi_cache()
    payload["assets"][1]["entries"] = []
    with pytest.raises(ValueError, match="at least 1 entries"):
        module.validate_grasp_bank(payload)


def test_multi_asset_cache_rejects_noncontiguous_indices():
    payload = make_multi_cache()
    payload["assets"][1]["asset_index"] = 3
    with pytest.raises(ValueError, match="contiguous"):
        module.validate_grasp_bank(payload)
