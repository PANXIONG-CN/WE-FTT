#!/usr/bin/env bash
set -euo pipefail

# 一键复现实验（v1）
#
# 使用方式：
# 1) 确保已安装依赖（建议 venv）：WE-FTT/.venv
# 2) 设置 AMSR2 原始数据根目录
# 3) 运行本脚本

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WEFTT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

AMSR2_ROOT="${AMSR2_ROOT:-/mnt/hdd_4tb_data/ArchivedWorks/MBTpaper_raw_data/AMSR2}"
PY="${PY:-${WEFTT_ROOT}/.venv/bin/python}"

RUN_ERA5="${RUN_ERA5:-0}"           # 设为 1 则执行 T6
ERA5_DOWNLOAD="${ERA5_DOWNLOAD:-0}" # 设为 1 则通过 CDS 下载 ERA5（需 ~/.cdsapirc）
ERA5_CLEANUP_NC="${ERA5_CLEANUP_NC:-0}" # 设为 1 则下载后清理每月 NetCDF（节省空间）

ARGS=(
  "${PY}" "${SCRIPT_DIR}/run_evaluation.py"
  --python "${PY}"
  --amsr2_root "${AMSR2_ROOT}"
  --control_dates_per_event 2
  --pixels_per_event_day 200
  --pre_days 20 --post_days 10
  --placebo_repeats 100
  --placebo_pixels_per_event_day 100
  --use_weights
)

if [[ "${RUN_ERA5}" == "1" ]]; then
  ARGS+=(--run_era5)
  if [[ "${ERA5_DOWNLOAD}" == "1" ]]; then
    ARGS+=(--era5_download)
    if [[ "${ERA5_CLEANUP_NC}" == "1" ]]; then
      ARGS+=(--era5_cleanup_nc)
    fi
  fi
fi

"${ARGS[@]}"
