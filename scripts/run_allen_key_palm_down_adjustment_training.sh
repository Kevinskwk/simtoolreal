#!/usr/bin/env bash
# Finetune the palm-down Allen-key side-changing adjustment gate.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CHECKPOINT="${CHECKPOINT:-${ROOT}/pretrained_policy/model.pth}"
GRASP_BANK="${GRASP_BANK:-${ROOT}/assets/grasp_banks/allen_key_palm_down_v1.json}"
NUM_ENVS="${NUM_ENVS:-12288}"
MAX_EPOCHS="${MAX_EPOCHS:-12000}"
STAMP="$(date +%Y%m%d_%H%M%S)"

python - "${GRASP_BANK}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
entries = payload.get("entries", [])
sources = [entry for entry in entries if not entry["verification"].get("target_only", False)]
if payload.get("kind") != "allen_key_palm_down_adjustment_bank":
    raise SystemExit(f"ERROR: {path} is not a palm-down Allen-key adjustment bank")
if len(sources) < 8:
    raise SystemExit(f"ERROR: expected at least eight resettable grasps, found {len(sources)}")
for index, entry in enumerate(sources):
    verification = entry["verification"]
    if verification.get("hold_steps", 0) < 120:
        raise SystemExit(f"ERROR: source {index} has not passed the 120-step replay gate")
    if verification.get("palm_down_cosine", 0.0) < 0.65:
        raise SystemExit(f"ERROR: source {index} is not palm-down")
    if len(verification.get("valid_target_ids", [])) != 1:
        raise SystemExit(f"ERROR: source {index} does not have exactly one validated goal")
print(f"[allen-key] using {len(sources)} validated palm-down start/goal pairs")
PY

python scripts/validate_allen_key_target_sampling.py \
  --grasp-bank "${GRASP_BANK}" \
  --minimum-translation-m 0 \
  --maximum-translation-m 0.20 \
  --minimum-rotation-deg 40 \
  --maximum-reset-yaw-deg 0

python scripts/visualize_allen_key_grasp_bank.py \
  --grasp-bank "${GRASP_BANK}" \
  --output-dir "${ROOT}/outputs/allen_key_grasp_bank_visualization/preflight_${STAMP}" \
  --settle-steps 120 \
  --headless

python scripts/validate_allen_key_adjustment_task.py \
  --palm-down \
  --grasp-bank "${GRASP_BANK}" \
  --all-target-pairs \
  --settle-steps 120 \
  --headless

python isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-AllenKey-PalmDown-Adjustment-Direct-v0 \
  --agent rl_games_sapg_cfg_entry_point \
  --headless \
  --capture_viewer \
  --checkpoint "${CHECKPOINT}" \
  --checkpoint_load_mode expand_obs \
  --wandb_activate \
  --wandb_project simtoolreal \
  --wandb_name "allen_key_palm_down_adjustment_${STAMP}" \
  "env.grasp_bank_path=${GRASP_BANK}" \
  "env.grasp_bank_source_checkpoint_path=${ROOT}/pretrained_policy/model.pth" \
  env.scene.num_envs="${NUM_ENVS}" \
  agent.params.config.max_epochs="${MAX_EPOCHS}" \
  agent.params.config.minibatch_size=32768 \
  agent.params.config.central_value_config.minibatch_size=32768 \
  agent.params.config.expl_coef_block_size=4096 \
  agent.params.config.learning_rate=5e-5 \
  agent.params.config.central_value_config.learning_rate=5e-5
