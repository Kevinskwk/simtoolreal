#!/usr/bin/env bash
# Build or resume the reusable procedural grasp bank used by adjustment tasks.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CHECKPOINT="${CHECKPOINT:-${ROOT}/pretrained_policy/model.pth}"
ASSETS_PER_DISTRIBUTION="${ASSETS_PER_DISTRIBUTION:-20}"
GRASPS_PER_ASSET="${GRASPS_PER_ASSET:-4}"
MAX_TRIALS_PER_ASSET="${MAX_TRIALS_PER_ASSET:-256}"
COLLECTION_ENVS="${COLLECTION_ENVS:-4096}"
CACHE_ROOT="${CACHE_ROOT:-${ROOT}/outputs/inhand_adjustment_cache/procedural_seed42_n${ASSETS_PER_DISTRIBUTION}}"
GRASP_BANK="${GRASP_BANK:-${CACHE_ROOT}/grasps.json}"
mkdir -p "${CACHE_ROOT}"

if [[ "${REBUILD_CACHE:-0}" == "1" ]]; then
  rm -f "${GRASP_BANK}" "${CACHE_ROOT}/scenarios.json"
fi

while ! python - "${GRASP_BANK}" "${GRASPS_PER_ASSET}" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
quota = int(sys.argv[2])
if not path.is_file():
    raise SystemExit(1)
payload = json.loads(path.read_text())
assets = payload.get("assets", [])
raise SystemExit(
    0 if assets and all(len(asset.get("entries", [])) >= quota for asset in assets) else 1
)
PY
do
  echo "[grasp-bank] filling ${GRASP_BANK} to ${GRASPS_PER_ASSET} grasps per asset"
  python scripts/collect_inhand_grasp_bank.py \
    --procedural \
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
import collections, json, pathlib, sys
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
summary = collections.defaultdict(lambda: [0, 0])
for asset in payload["assets"]:
    summary[asset["tool_type"]][0] += 1
    summary[asset["tool_type"]][1] += len(asset["entries"])
print(f"[grasp-bank] complete: {path.resolve()}")
for category in sorted(summary):
    assets, grasps = summary[category]
    print(f"  {category}: assets={assets} grasps={grasps}")
PY
