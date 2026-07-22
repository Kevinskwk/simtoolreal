#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
WANDB_PROJECT="${WANDB_PROJECT:-simtoolreal}"
RUN_TAG="${RUN_TAG:-tactile_depth_twostage_$(date +%Y%m%d_%H%M%S)}"
VANILLA_CHECKPOINT="${VANILLA_CHECKPOINT:-$REPO_ROOT/pretrained_policy/model.pth}"
STAGE1_MAX_EPOCHS="${STAGE1_MAX_EPOCHS:-12000}"
STAGE2_MAX_EPOCHS="${STAGE2_MAX_EPOCHS:-}"
CAPTURE_VIEWER="${CAPTURE_VIEWER:-1}"

if [[ ! -f "$VANILLA_CHECKPOINT" ]]; then
  echo "[two-stage] ERROR: vanilla checkpoint not found: $VANILLA_CHECKPOINT" >&2
  exit 1
fi

COMMON_CLI=(
  isaacsimenvs/train.py
  --task Isaacsimenvs-SimToolReal-TacMap-Scrape-Direct-v0
  --agent rl_games_sapg_cfg_entry_point
  --headless
)

if [[ "$CAPTURE_VIEWER" != "0" ]]; then
  COMMON_CLI+=(--capture_viewer)
fi

WANDB_CLI=(
  --wandb_activate
  --wandb_project "$WANDB_PROJECT"
)
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  WANDB_CLI+=(--wandb_entity "$WANDB_ENTITY")
fi

COMMON_OVERRIDES=(
  'env.use_tacmap=true'
  'env.enable_vbts=true'
  'env.include_tacmap_in_policy=true'
  'env.tacmap_history_len=5'
  'env.obs.obs_list=[joint_pos,joint_vel,prev_action_targets,palm_pos,palm_rot,object_rot,fingertip_pos_rel_palm,keypoints_rel_palm,keypoints_rel_goal,object_scales,tacmap]'
  'env.enable_tool_table_contact_force_reward=true'
  'env.scene.num_envs=12288'
  'agent.params.config.minibatch_size=98304'
  'agent.params.config.central_value_config.minibatch_size=98304'
  'agent.params.config.expl_coef_block_size=4096'
  'agent.params.config.learning_rate=5e-5'
  'agent.params.config.central_value_config.learning_rate=5e-5'
)

STAGE1_NAME="${STAGE1_WANDB_NAME:-${RUN_TAG}_stage1_pretrain_expandobs_12k}"
STAGE2_NAME="${STAGE2_WANDB_NAME:-${RUN_TAG}_stage2_contact_force}"
MARKER="$(mktemp /tmp/simtoolreal_stage1_start.XXXXXX)"
touch "$MARKER"
cleanup() {
  rm -f "$MARKER"
}
trap cleanup EXIT

echo "[two-stage] Stage 1: vanilla -> tactile-depth expand_obs for ${STAGE1_MAX_EPOCHS} epochs"
echo "[two-stage] Stage 1 wandb_name: $STAGE1_NAME"
"$PYTHON_BIN" "${COMMON_CLI[@]}" \
  --checkpoint "$VANILLA_CHECKPOINT" \
  --checkpoint_load_mode expand_obs \
  "${WANDB_CLI[@]}" \
  --wandb_name "$STAGE1_NAME" \
  "${COMMON_OVERRIDES[@]}" \
  "agent.params.config.max_epochs=${STAGE1_MAX_EPOCHS}"

STAGE1_CKPT="$({
  find outputs -path "*/0_simtoolreal_sapg/nn/last_0_simtoolreal_sapg_ep_${STAGE1_MAX_EPOCHS}_*.pth" -newer "$MARKER" -printf '%T@ %p\n' 2>/dev/null || true
} | sort -nr | awk 'NR==1 { $1=""; sub(/^ /, ""); print }')"

if [[ -z "$STAGE1_CKPT" || ! -f "$STAGE1_CKPT" ]]; then
  echo "[two-stage] ERROR: could not find a new stage-1 ep_${STAGE1_MAX_EPOCHS} checkpoint." >&2
  echo "[two-stage] Expected pattern: outputs/*/*/0_simtoolreal_sapg/nn/last_0_simtoolreal_sapg_ep_${STAGE1_MAX_EPOCHS}_*.pth" >&2
  exit 1
fi

echo "[two-stage] Stage 1 checkpoint: $STAGE1_CKPT"
echo "[two-stage] Stage 2: load stage-1 weights with contact-force reward config"
echo "[two-stage] Stage 2 wandb_name: $STAGE2_NAME"

STAGE2_OVERRIDES=("${COMMON_OVERRIDES[@]}")
if [[ -n "$STAGE2_MAX_EPOCHS" ]]; then
  STAGE2_OVERRIDES+=("agent.params.config.max_epochs=${STAGE2_MAX_EPOCHS}")
fi

"$PYTHON_BIN" "${COMMON_CLI[@]}" \
  --checkpoint "$STAGE1_CKPT" \
  --checkpoint_load_mode weights \
  "${WANDB_CLI[@]}" \
  --wandb_name "$STAGE2_NAME" \
  "${STAGE2_OVERRIDES[@]}"
