#!/usr/bin/env bash
# Build and strictly validate the rollout-conditioned Allen-key grasp bank.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

ROLLOUT_DIR="${ROLLOUT_DIR:-${ROOT}/outputs/allen_key_pretrained_turning/20260822_stable_init_headless_12288}"
PROVISIONAL="${PROVISIONAL:-${ROOT}/outputs/allen_key_rollout_bank/provisional_rollout_grasps.json}"
OUTPUT="${OUTPUT:-${ROOT}/assets/grasp_banks/allen_key_rollout_adjustment_v1.json}"
SOURCE_ENTRIES="${SOURCE_ENTRIES:-96}"
FINAL_ENTRIES="${FINAL_ENTRIES:-24}"

python scripts/prepare_allen_key_rollout_grasp_bank.py \
  --rollout-dir "${ROLLOUT_DIR}" \
  --output "${PROVISIONAL}" \
  --desired-entries "${SOURCE_ENTRIES}"

python scripts/adapt_allen_key_grasp_bank.py \
  --source-bank "${PROVISIONAL}" \
  --direct-source-entries \
  --max-source-entries "${SOURCE_ENTRIES}" \
  --desired-entries "${FINAL_ENTRIES}" \
  --output "${OUTPUT}" \
  --headless

python scripts/validate_allen_key_workspace_bank.py --grasp-bank "${OUTPUT}"
python scripts/validate_allen_key_target_sampling.py \
  --grasp-bank "${OUTPUT}" \
  --maximum-reset-yaw-deg 0

