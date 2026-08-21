#!/usr/bin/env bash
# Finetune palm-supported Allen-key adjustment about the engaged screw axis.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

VANILLA_CHECKPOINT="${ROOT}/pretrained_policy/model.pth"
CHECKPOINT="${CHECKPOINT:-${VANILLA_CHECKPOINT}}"
GRASP_BANK_SOURCE_CHECKPOINT="${GRASP_BANK_SOURCE_CHECKPOINT:-${VANILLA_CHECKPOINT}}"
CHECKPOINT_LOAD_MODE="${CHECKPOINT_LOAD_MODE:-expand_obs}"
GRASP_BANK="${GRASP_BANK:-${ROOT}/assets/grasp_banks/allen_key_canonical_v1.json}"
NUM_ENVS="${NUM_ENVS:-12288}"
MAX_EPOCHS="${MAX_EPOCHS:-12000}"
STAMP="$(date +%Y%m%d_%H%M%S)"

python - "${GRASP_BANK}" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"ERROR: Allen-key grasp bank does not exist: {path}")
payload = json.loads(path.read_text())
entries = payload.get("entries", [])
if payload.get("tool_type") != "allen_key" or not entries:
    raise SystemExit("ERROR: grasp bank is not a non-empty Allen-key bank")
for index, entry in enumerate(entries):
    metrics = entry.get("verification", {})
    if metrics.get("palm_contact_ratio", 0.0) < 0.95:
        raise SystemExit(f"ERROR: grasp {index} lacks validated palm contact")
    if metrics.get("socket_valid_ratio", 0.0) < 0.95:
        raise SystemExit(f"ERROR: grasp {index} lacks validated socket engagement")
print(f"[allen-key] using {len(entries)} physically validated grasp(s) from {path}")
PY

python isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-AllenKey-Adjustment-Direct-v0 \
  --agent rl_games_sapg_cfg_entry_point \
  --headless \
  --capture_viewer \
  --checkpoint "${CHECKPOINT}" \
  --checkpoint_load_mode "${CHECKPOINT_LOAD_MODE}" \
  --wandb_activate \
  --wandb_project simtoolreal \
  --wandb_name "allen_key_palm_supported_adjustment_${STAMP}" \
  "env.grasp_bank_path=${GRASP_BANK}" \
  "env.grasp_bank_source_checkpoint_path=${GRASP_BANK_SOURCE_CHECKPOINT}" \
  env.scene.num_envs="${NUM_ENVS}" \
  agent.params.config.max_epochs="${MAX_EPOCHS}" \
  agent.params.config.minibatch_size=32768 \
  agent.params.config.central_value_config.minibatch_size=32768 \
  agent.params.config.expl_coef_block_size=4096 \
  agent.params.config.learning_rate=5e-5 \
  agent.params.config.central_value_config.learning_rate=5e-5
