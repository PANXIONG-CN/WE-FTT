#!/usr/bin/env bash
set -euo pipefail

WEFTT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PY="${WEFTT_DIR}/.venv/bin/python"
EVAL="${WEFTT_DIR}/evaluation_protocol/datasets/mbt_eval_samples_v1.parquet"
CACHE="${WEFTT_DIR}/evaluation_protocol/era5/cache/daily"

"${PY}" "${WEFTT_DIR}/evaluation_protocol/era5/download_from_cds.py" \
  --eval_parquet "${EVAL}" \
  --use_split "all" \
  --zone_type 0 \
  --out_daily_dir "${CACHE}" \
  --cleanup_nc \
  --log_file "${WEFTT_DIR}/evaluation_protocol/era5/results/era5_download_all_v1.log" \
  --out_meta_json "${WEFTT_DIR}/evaluation_protocol/era5/results/era5_download_all_v1.meta.json"

"${PY}" "${WEFTT_DIR}/evaluation_protocol/era5/ocean_residualize.py" \
  --eval_parquet "${EVAL}" \
  --era5_daily_cache_dir "${CACHE}" \
  --alpha 1.0 \
  --use_weights \
  --log_file "${WEFTT_DIR}/evaluation_protocol/era5/results/ocean_residualize_v1.log"

