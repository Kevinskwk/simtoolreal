#!/usr/bin/env bash
# Finetune rollout-conditioned Allen-key adjustment from the vanilla policy.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CHECKPOINT="${CHECKPOINT:-${ROOT}/pretrained_policy/model.pth}"
GRASP_BANK="${GRASP_BANK:-${ROOT}/assets/grasp_banks/allen_key_rollout_adjustment_v1.json}"
NUM_ENVS="${NUM_ENVS:-12288}"
MAX_EPOCHS="${MAX_EPOCHS:-12000}"
VALIDATION_ENVS="${VALIDATION_ENVS:-8}"
STAMP="$(date +%Y%m%d_%H%M%S)"

python scripts/validate_allen_key_workspace_bank.py --grasp-bank "${GRASP_BANK}"
python scripts/validate_allen_key_target_sampling.py \
  --grasp-bank "${GRASP_BANK}" \
  --maximum-reset-yaw-deg 0
python scripts/validate_allen_key_adjustment_task.py \
  --workspace-conditioned \
  --grasp-bank "${GRASP_BANK}" \
  --num-envs "${VALIDATION_ENVS}" \
  --settle-steps 30 \
  --headless \
  --output "${ROOT}/outputs/allen_key_workspace_adjustment_validation/preflight_${STAMP}.html"

python isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-AllenKey-Workspace-Adjustment-Direct-v0 \
  --agent rl_games_sapg_cfg_entry_point \
  --headless \
  --capture_viewer \
  --checkpoint "${CHECKPOINT}" \
  --checkpoint_load_mode expand_obs \
  --wandb_activate \
  --wandb_project simtoolreal \
  --wandb_name "allen_key_workspace_adjustment_${STAMP}" \
  "env.grasp_bank_path=${GRASP_BANK}" \
  "env.grasp_bank_source_checkpoint_path=${ROOT}/pretrained_policy/model.pth" \
  env.scene.num_envs="${NUM_ENVS}" \
  agent.params.config.max_epochs="${MAX_EPOCHS}" \
  agent.params.config.minibatch_size=32768 \
  agent.params.config.central_value_config.minibatch_size=32768 \
  agent.params.config.expl_coef_block_size=4096 \
  agent.params.config.learning_rate=5e-5 \
  agent.params.config.central_value_config.learning_rate=5e-5
