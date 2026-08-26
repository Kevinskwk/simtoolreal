#!/usr/bin/env bash
# Finetune SimToolReal with event-gated Allen-key turning and regrasping.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CHECKPOINT="${CHECKPOINT:-${ROOT}/pretrained_policy/model.pth}"
NUM_ENVS="${NUM_ENVS:-12288}"
MAX_EPOCHS="${MAX_EPOCHS:-12000}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
WANDB_NAME="${WANDB_NAME:-allen_key_clean_regrasp_${STAMP}}"
PRETRAINED_SAPG_GROUPS=6

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "ERROR: checkpoint does not exist: ${CHECKPOINT}" >&2
  exit 2
fi

if (( NUM_ENVS % PRETRAINED_SAPG_GROUPS != 0 )); then
  echo "ERROR: NUM_ENVS=${NUM_ENVS} must be divisible by ${PRETRAINED_SAPG_GROUPS} to preserve the pretrained SAPG heads" >&2
  exit 2
fi
EXPL_BLOCK_SIZE=$((NUM_ENVS / PRETRAINED_SAPG_GROUPS))
# One reset frame plus up to 4500 policy transitions. A normal timeout is
# finalized at its reset boundary and therefore contains exactly 4500 frames.
CAPTURE_VIEWER_LEN="${CAPTURE_VIEWER_LEN:-4501}"
CAPTURE_VIEWER_INTERVAL="${CAPTURE_VIEWER_INTERVAL:-6000}"

if [[ "${SKIP_PREFLIGHT}" != "1" ]]; then
  python scripts/validate_allen_key_turning_task.py \
    --output "${ROOT}/outputs/allen_key_turning_validation/preflight_${STAMP}.html" \
    --headless
fi

python isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-AllenKey-Turning-Direct-v0 \
  --agent rl_games_sapg_cfg_entry_point \
  --headless \
  --capture_viewer \
  --capture_viewer_len "${CAPTURE_VIEWER_LEN}" \
  --capture_viewer_interval "${CAPTURE_VIEWER_INTERVAL}" \
  --capture_viewer_episode_aligned \
  --checkpoint "${CHECKPOINT}" \
  --checkpoint_load_mode weights \
  --wandb_activate \
  --wandb_project simtoolreal \
  --wandb_name "${WANDB_NAME}" \
  env.scene.num_envs="${NUM_ENVS}" \
  agent.params.config.max_epochs="${MAX_EPOCHS}" \
  agent.params.config.minibatch_size=98304 \
  agent.params.config.central_value_config.minibatch_size=98304 \
  agent.params.config.expl_coef_block_size="${EXPL_BLOCK_SIZE}" \
  agent.params.config.learning_rate=5e-5 \
  agent.params.config.central_value_config.learning_rate=5e-5
