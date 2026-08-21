#!/usr/bin/env bash
# Finetune sampled palm-to-tool adjustment from a procedural grasp bank.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

VANILLA_CHECKPOINT="${ROOT}/pretrained_policy/model.pth"
CHECKPOINT="${CHECKPOINT:-${VANILLA_CHECKPOINT}}"
GRASP_BANK_SOURCE_CHECKPOINT="${GRASP_BANK_SOURCE_CHECKPOINT:-${VANILLA_CHECKPOINT}}"
CHECKPOINT_LOAD_MODE="${CHECKPOINT_LOAD_MODE:-expand_obs}"
ASSETS_PER_DISTRIBUTION="${ASSETS_PER_DISTRIBUTION:-20}"
BANK_GRASPS_PER_ASSET="${BANK_GRASPS_PER_ASSET:-4}"
NUM_ENVS="${NUM_ENVS:-12288}"
MAX_EPOCHS="${MAX_EPOCHS:-12000}"
STAMP="$(date +%Y%m%d_%H%M%S)"
CACHE_ROOT="${CACHE_ROOT:-${ROOT}/outputs/inhand_adjustment_cache/procedural_seed42_n${ASSETS_PER_DISTRIBUTION}}"
GRASP_BANK="${GRASP_BANK:-${CACHE_ROOT}/grasps.json}"
mkdir -p "${CACHE_ROOT}"

if ! python - "${GRASP_BANK}" "${BANK_GRASPS_PER_ASSET}" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
quota = int(sys.argv[2])
if not path.is_file():
    raise SystemExit(1)
payload = json.loads(path.read_text())
assets = payload.get("assets", [])
raise SystemExit(0 if assets and all(len(asset.get("entries", [])) >= quota for asset in assets) else 1)
PY
then
  echo "ERROR: procedural grasp bank is missing or incomplete: ${GRASP_BANK}" >&2
  echo "Run: GRASPS_PER_ASSET=${BANK_GRASPS_PER_ASSET} bash scripts/collect_procedural_adjustment_grasp_bank.sh" >&2
  exit 1
fi

python isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-InHand-Adjustment-Direct-v0 \
  --agent rl_games_sapg_cfg_entry_point \
  --headless \
  --capture_viewer \
  --checkpoint "${CHECKPOINT}" \
  --checkpoint_load_mode "${CHECKPOINT_LOAD_MODE}" \
  --wandb_activate \
  --wandb_project simtoolreal \
  --wandb_name "extrinsic_grasp_adjustment_procedural240_${STAMP}" \
  "env.grasp_bank_path=${GRASP_BANK}" \
  "env.grasp_bank_source_checkpoint_path=${GRASP_BANK_SOURCE_CHECKPOINT}" \
  "env.assets.object_urdf=" \
  'env.assets.handle_head_types=[hammer,screwdriver,marker,spatula,eraser,brush]' \
  env.assets.num_assets_per_type="${ASSETS_PER_DISTRIBUTION}" \
  env.assets.procedural_asset_seed=42 \
  env.assets.shuffle_assets=true \
  env.assets.object_pool_limit=0 \
  env.grasp_bank_min_entries="${BANK_GRASPS_PER_ASSET}" \
  env.scene.num_envs="${NUM_ENVS}" \
  agent.params.config.max_epochs="${MAX_EPOCHS}" \
  agent.params.config.minibatch_size=32768 \
  agent.params.config.central_value_config.minibatch_size=32768 \
  agent.params.config.expl_coef_block_size=4096 \
  agent.params.config.learning_rate=5e-5 \
  agent.params.config.central_value_config.learning_rate=5e-5
