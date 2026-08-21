#!/usr/bin/env bash
# Build or resume the screwdriver-only procedural grasp bank.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CHECKPOINT="${CHECKPOINT:-${ROOT}/pretrained_policy/model.pth}"
ASSETS_PER_DISTRIBUTION="${ASSETS_PER_DISTRIBUTION:-20}"
GRASPS_PER_ASSET="${GRASPS_PER_ASSET:-4}"
MAX_TRIALS_PER_ASSET="${MAX_TRIALS_PER_ASSET:-256}"
COLLECTION_ENVS="${COLLECTION_ENVS:-2048}"
CACHE_ROOT="${CACHE_ROOT:-${ROOT}/outputs/inhand_adjustment_cache/screwdriver_seed42_n${ASSETS_PER_DISTRIBUTION}}"
GRASP_BANK="${GRASP_BANK:-${CACHE_ROOT}/grasps.json}"
mkdir -p "${CACHE_ROOT}"

if [[ "${REBUILD_CACHE:-0}" == "1" ]]; then
  rm -f "${GRASP_BANK}"
fi

while ! python - "${GRASP_BANK}" "${GRASPS_PER_ASSET}" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
quota = int(sys.argv[2])
if not path.is_file():
    raise SystemExit(1)
payload = json.loads(path.read_text())
assets = payload.get("assets", [])
valid = (
    assets
    and payload.get("procedural", {}).get("tool_types") == ["screwdriver"]
    and all(asset.get("tool_type") == "screwdriver" for asset in assets)
    and all(len(asset.get("entries", [])) >= quota for asset in assets)
)
raise SystemExit(0 if valid else 1)
PY
do
  echo "[grasp-bank] filling screwdriver cache to ${GRASPS_PER_ASSET} grasps per asset"
  python scripts/collect_inhand_grasp_bank.py \
    --procedural \
    --procedural-tool-types screwdriver \
    --resume \
    --output "${GRASP_BANK}" \
    --checkpoint "${CHECKPOINT}" \
    --assets-per-distribution "${ASSETS_PER_DISTRIBUTION}" \
    --grasps-per-asset "${GRASPS_PER_ASSET}" \
    --max-trials-per-asset "${MAX_TRIALS_PER_ASSET}" \
    --num-envs "${COLLECTION_ENVS}" \
    --tactile-entry-fraction 0 \
    --headless
done

python - "${GRASP_BANK}" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
print(f"[grasp-bank] complete: {path.resolve()}")
print(f"  screwdriver: assets={len(payload['assets'])} "
      f"grasps={sum(len(asset['entries']) for asset in payload['assets'])}")
PY
