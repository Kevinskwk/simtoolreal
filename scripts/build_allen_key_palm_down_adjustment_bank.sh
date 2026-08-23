#!/usr/bin/env bash
# Rebuild and physically validate the palm-down Allen-key adjustment bank.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

ROLLOUT_DIR="${ROLLOUT_DIR:-${ROOT}/outputs/allen_key_pretrained_turning/thick_handle_90deg_bank_20260823}"
OUTPUT="${OUTPUT:-${ROOT}/assets/grasp_banks/allen_key_palm_down_v1.json}"
VIEW_DIR="${VIEW_DIR:-${ROOT}/outputs/allen_key_grasp_bank_visualization/palm_down_thick_v2}"

# These rollout environments failed either the 120-step source replay gate or
# physical target screening during the deterministic bank construction pass.
REJECTED_SOURCE_ENVS=(
  2477 1979 2316 3403 2308 45 320 1914 3466 912 3281 1316 3678 2972
  2212 2871 2494 3552 540 655 833 3419 2931 414 3441 3978 1141 700
)

python scripts/build_allen_key_palm_down_adjustment_bank.py \
  --rollout-dir "${ROLLOUT_DIR}" \
  --output "${OUTPUT}" \
  --pairs 8 \
  --exclude-source-env-ids "${REJECTED_SOURCE_ENVS[@]}"

python scripts/validate_allen_key_target_sampling.py \
  --grasp-bank "${OUTPUT}" \
  --minimum-translation-m 0 \
  --maximum-translation-m 0.20 \
  --minimum-rotation-deg 40 \
  --maximum-reset-yaw-deg 0

python scripts/visualize_allen_key_grasp_bank.py \
  --grasp-bank "${OUTPUT}" \
  --output-dir "${VIEW_DIR}" \
  --settle-steps 120 \
  --headless

python scripts/validate_allen_key_adjustment_task.py \
  --palm-down \
  --grasp-bank "${OUTPUT}" \
  --all-target-pairs \
  --settle-steps 120 \
  --headless
