#!/usr/bin/env bash
# Finetune end-to-end Allen-key acquisition and loaded 360-degree turning.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CALIBRATION_JSON="${CALIBRATION_JSON:-}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/pretrained_policy/model.pth}"
NUM_ENVS="${NUM_ENVS:-12288}"
MAX_EPOCHS="${MAX_EPOCHS:-12000}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"

if [[ -z "${CALIBRATION_JSON}" || ! -f "${CALIBRATION_JSON}" ]]; then
  echo "ERROR: set CALIBRATION_JSON to a completed calibration.json" >&2
  exit 2
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "ERROR: checkpoint does not exist: ${CHECKPOINT}" >&2
  exit 2
fi

CALIBRATED_TORQUE="$(python - "${CALIBRATION_JSON}" <<'PY'
import json
import math
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
if payload.get("schema_version") != 1:
    raise SystemExit(f"ERROR: unsupported calibration schema in {path}")
value = payload.get("recommended_training_torque_nm")
if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
    raise SystemExit(f"ERROR: calibration has invalid recommended torque: {value!r}")
if int(payload.get("num_grasps", 0)) < 8:
    raise SystemExit("ERROR: calibration used fewer than eight grasps")
print(f"{value:.9g}")
PY
)"

echo "[allen-turn] calibrated maximum=${CALIBRATED_TORQUE} N m"
if [[ "${SKIP_PREFLIGHT}" != "1" ]]; then
  python scripts/validate_allen_key_turning_task.py \
    --test-torque-nm "$(python - "${CALIBRATED_TORQUE}" <<'PY'
import sys
print(max(0.02, 0.2 * float(sys.argv[1])))
PY
)" \
    --output "${ROOT}/outputs/allen_key_turning_validation/preflight_${STAMP}.html" \
    --headless
fi

python isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-AllenKey-Turning-Direct-v0 \
  --agent rl_games_sapg_cfg_entry_point \
  --headless \
  --capture_viewer \
  --checkpoint "${CHECKPOINT}" \
  --checkpoint_load_mode expand_obs \
  --wandb_activate \
  --wandb_project simtoolreal \
  --wandb_name "allen_key_end_to_end_turning_${STAMP}" \
  env.allen_turn_calibrated_torque_nm="${CALIBRATED_TORQUE}" \
  env.allen_turn_require_calibrated_load=true \
  env.scene.num_envs="${NUM_ENVS}" \
  agent.params.config.max_epochs="${MAX_EPOCHS}" \
  agent.params.config.minibatch_size=98304 \
  agent.params.config.central_value_config.minibatch_size=98304 \
  agent.params.config.expl_coef_block_size=4096 \
  agent.params.config.learning_rate=5e-5 \
  agent.params.config.central_value_config.learning_rate=5e-5
