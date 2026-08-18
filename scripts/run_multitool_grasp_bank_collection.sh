#!/usr/bin/env bash
# Adaptively collect and replay-filter grasp banks for all DexToolBench tools.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "sharpa" && "${SHARPA_CONDA_REEXEC:-0}" != "1" ]]; then
    CONDA_BIN="${CONDA_EXE:-/home/showlab/miniforge3/bin/conda}"
    if [[ ! -x "${CONDA_BIN}" ]]; then
        echo "ERROR: activate sharpa; conda was not found at ${CONDA_BIN}" >&2
        exit 2
    fi
    export SHARPA_CONDA_REEXEC=1
    exec "${CONDA_BIN}" run --no-capture-output -n sharpa bash "$0" "$@"
fi

CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/pretrained_policy/model.pth}"
POLICY_CONFIG="${POLICY_CONFIG:-${REPO_ROOT}/pretrained_policy/config.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/multitool_grasp_banks/$(date +%Y%m%d_%H%M%S)}"
SEEDS_STRING="${SEEDS:-0 1 2 3}"
TRIAL_REQUESTED_ENTRIES="${TRIAL_REQUESTED_ENTRIES:-512}"
NUM_ENVS="${NUM_ENVS:-1024}"
ACQUISITION_STEPS="${ACQUISITION_STEPS:-2400}"
TRIAL_STEP_INCREMENT="${TRIAL_STEP_INCREMENT:-600}"
MAX_ACQUISITION_STEPS="${MAX_ACQUISITION_STEPS:-4800}"
MAX_TRIALS="${MAX_TRIALS:-20}"
TRIAL_SEED_STRIDE="${TRIAL_SEED_STRIDE:-10000}"
REPLAY_RESETS_MIN="${REPLAY_RESETS_MIN:-1024}"
REPLAY_PASSES_PER_ENTRY="${REPLAY_PASSES_PER_ENTRY:-2}"
REPLAY_NUM_ENVS="${REPLAY_NUM_ENVS:-256}"
REPLAY_HOLD_STEPS="${REPLAY_HOLD_STEPS:-120}"
FILTERED_ENTRIES="${FILTERED_ENTRIES:-256}"
CURRICULUM_STAGE="${CURRICULUM_STAGE:-3}"
TACTILE_ENTRY_FRACTION="${TACTILE_ENTRY_FRACTION:-0.0}"
TOOL_NAMES="${TOOL_NAMES:-mallet_hammer claw_hammer long_screwdriver short_screwdriver handle_eraser flat_eraser flat_spatula spoon_spatula sharpie_marker staples_marker red_brush blue_brush}"

read -r -a TOOLS <<< "${TOOL_NAMES}"
read -r -a SEEDS_ARRAY <<< "${SEEDS_STRING}"

for value_name in TRIAL_REQUESTED_ENTRIES NUM_ENVS ACQUISITION_STEPS \
    MAX_ACQUISITION_STEPS MAX_TRIALS REPLAY_RESETS_MIN \
    TRIAL_SEED_STRIDE REPLAY_PASSES_PER_ENTRY REPLAY_NUM_ENVS \
    REPLAY_HOLD_STEPS FILTERED_ENTRIES; do
    value="${!value_name}"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: ${value_name} must be a positive integer, got ${value}" >&2
        exit 2
    fi
done
if [[ ! "${TRIAL_STEP_INCREMENT}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: TRIAL_STEP_INCREMENT must be a non-negative integer" >&2
    exit 2
fi
if (( ${#TOOLS[@]} == 0 || ${#SEEDS_ARRAY[@]} == 0 )); then
    echo "ERROR: TOOL_NAMES and SEEDS must both be non-empty" >&2
    exit 2
fi
if [[ ! -f "${CHECKPOINT}" || ! -f "${POLICY_CONFIG}" ]]; then
    echo "ERROR: checkpoint or policy config is missing" >&2
    exit 2
fi
if (( TRIAL_REQUESTED_ENTRIES > NUM_ENVS )); then
    echo "ERROR: TRIAL_REQUESTED_ENTRIES cannot exceed NUM_ENVS" >&2
    exit 2
fi
if (( ACQUISITION_STEPS > MAX_ACQUISITION_STEPS )); then
    echo "ERROR: ACQUISITION_STEPS cannot exceed MAX_ACQUISITION_STEPS" >&2
    exit 2
fi

bank_count() {
    python -c '
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(2)
payload = json.loads(path.read_text())
entries = payload.get("entries")
if not isinstance(entries, list) or not entries:
    raise SystemExit(3)
print(len(entries))
' "$1"
}

mkdir -p "${OUTPUT_ROOT}"
STATUS_FILE="${OUTPUT_ROOT}/status.tsv"
printf 'object\tseed\tresult\ttrials\traw_entries\trobust_entries\traw_bank\tfiltered_bank\n' \
    > "${STATUS_FILE}"
echo "[queue] output=${OUTPUT_ROOT}"
echo "[queue] tools=${#TOOLS[@]} seeds=${SEEDS_STRING} robust_target=${FILTERED_ENTRIES} max_trials=${MAX_TRIALS}"

failed_jobs=0
completed_jobs=0
for object_name in "${TOOLS[@]}"; do
    for seed in "${SEEDS_ARRAY[@]}"; do
        job_name="${object_name}_seed${seed}"
        job_dir="${OUTPUT_ROOT}/${job_name}"
        raw_bank="${job_dir}/raw.json"
        filtered_bank="${job_dir}/robust.json"
        mkdir -p "${job_dir}/trials"
        partial_banks=()
        raw_count=0
        robust_count=0
        succeeded=0
        completed_trials=0

        for ((trial=1; trial<=MAX_TRIALS; trial++)); do
            completed_trials=${trial}
            trial_seed=$((seed + (trial - 1) * TRIAL_SEED_STRIDE))
            trial_steps=$((ACQUISITION_STEPS + (trial - 1) * TRIAL_STEP_INCREMENT))
            if (( trial_steps > MAX_ACQUISITION_STEPS )); then
                trial_steps=${MAX_ACQUISITION_STEPS}
            fi
            trial_dir="${job_dir}/trials/trial$(printf '%02d' "${trial}")"
            partial_bank="${trial_dir}/partial.json"
            mkdir -p "${trial_dir}"
            rm -f "${partial_bank}"
            echo "[${job_name}] trial=${trial}/${MAX_TRIALS} seed=${trial_seed} steps=${trial_steps}"

            set +e
            python scripts/collect_inhand_grasp_bank.py \
                --headless \
                --checkpoint "${CHECKPOINT}" \
                --policy-config "${POLICY_CONFIG}" \
                --object-name "${object_name}" \
                --output "${partial_bank}" \
                --seed "${trial_seed}" \
                --entries "${TRIAL_REQUESTED_ENTRIES}" \
                --num-envs "${NUM_ENVS}" \
                --acquisition-steps "${trial_steps}" \
                --stable-steps 15 \
                --hold-steps 120 \
                --allow-partial \
                --tactile-entry-fraction "${TACTILE_ENTRY_FRACTION}" \
                2>&1 | tee "${trial_dir}/collect.log"
            set -e

            partial_count="$(bank_count "${partial_bank}" 2>/dev/null)" || partial_count=0
            if (( partial_count <= 0 )); then
                echo "[${job_name}] ERROR: trial ${trial} produced no valid partial bank" >&2
                continue
            fi
            partial_banks+=("${partial_bank}")

            rm -f "${raw_bank}"
            set +e
            python scripts/merge_inhand_grasp_banks.py \
                --output "${raw_bank}" "${partial_banks[@]}" \
                2>&1 | tee "${job_dir}/merge.log"
            merge_status=${PIPESTATUS[0]}
            set -e
            raw_count="$(bank_count "${raw_bank}" 2>/dev/null)" || raw_count=0
            if (( merge_status != 0 || raw_count <= 0 )); then
                echo "[${job_name}] ERROR: cumulative bank validation failed" >&2
                rm -f "${raw_bank}"
                raw_count=0
                continue
            fi
            echo "[${job_name}] partial=${partial_count} cumulative_raw=${raw_count}"

            if (( raw_count < FILTERED_ENTRIES )); then
                continue
            fi
            replay_resets=$((raw_count * REPLAY_PASSES_PER_ENTRY))
            if (( replay_resets < REPLAY_RESETS_MIN )); then
                replay_resets=${REPLAY_RESETS_MIN}
            fi
            rm -f "${filtered_bank}"
            set +e
            python scripts/validate_inhand_grasp_bank.py \
                --headless \
                --grasp-bank "${raw_bank}" \
                --resets "${replay_resets}" \
                --num-envs "${REPLAY_NUM_ENVS}" \
                --hold-steps "${REPLAY_HOLD_STEPS}" \
                --curriculum-stage "${CURRICULUM_STAGE}" \
                --write-filtered-bank "${filtered_bank}" \
                --minimum-filtered-entries "${FILTERED_ENTRIES}" \
                2>&1 | tee "${job_dir}/replay_trial$(printf '%02d' "${trial}").log"
            set -e
            robust_count="$(bank_count "${filtered_bank}" 2>/dev/null)" || robust_count=0
            if (( robust_count >= FILTERED_ENTRIES )); then
                succeeded=1
                echo "[${job_name}] COMPLETE robust=${robust_count} trials=${trial}"
                break
            fi
            echo "[${job_name}] replay target not met; collecting another trial" >&2
        done

        if (( succeeded == 1 )); then
            printf '%s\t%s\tpassed\t%s\t%s\t%s\t%s\t%s\n' \
                "${object_name}" "${seed}" "${completed_trials}" "${raw_count}" \
                "${robust_count}" "${raw_bank}" "${filtered_bank}" >> "${STATUS_FILE}"
            ((completed_jobs += 1))
        else
            printf '%s\t%s\tfailed_max_trials\t%s\t%s\t%s\t%s\t%s\n' \
                "${object_name}" "${seed}" "${completed_trials}" "${raw_count}" \
                "${robust_count}" "${raw_bank}" "${filtered_bank}" >> "${STATUS_FILE}"
            ((failed_jobs += 1))
            echo "[${job_name}] ERROR: robust target ${FILTERED_ENTRIES} not met after ${MAX_TRIALS} trials" >&2
        fi
    done
done

echo "[summary] completed=${completed_jobs} failed=${failed_jobs} status=${STATUS_FILE}"
if (( failed_jobs > 0 )); then
    exit 1
fi
