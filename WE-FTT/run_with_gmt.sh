#!/usr/bin/env bash
# PyGMT运行包装脚本（自动定位含PyGMT的conda环境，并修复C++库路径）
# - 目标：解决 libstdc++/GDAL 等C++库版本冲突，并确保使用安装了PyGMT的Python
# - 用法：
#     ./run_with_gmt.sh supplement_experiments/exp2/scripts/generate_fig_s2.py
#     ./run_with_gmt.sh -c "import pygmt; print(pygmt.__version__)"

set -euo pipefail

DEBUG_MODE="${RUN_WITH_GMT_DEBUG:-0}"
log_debug() {
  if [[ "${DEBUG_MODE}" == "1" ]]; then
    echo "[run_with_gmt] $*" >&2
  fi
}

if [[ $# -lt 1 ]]; then
  echo "用法: $0 <python脚本或 -c '代码'> [参数...]" >&2
  exit 2
fi

# 查找可用的 conda 命令（可能是 mamba 的 shim）
CONDA_BIN="${CONDA_EXE:-}"
if [[ -z "${CONDA_BIN}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BIN="$(command -v conda)"
  elif command -v mamba >/dev/null 2>&1; then
    CONDA_BIN="$(command -v mamba)"
  else
    CONDA_BIN=""
  fi
fi

# 解析可用的conda前缀列表（包含已激活前缀与envs）
ACTIVE_PREFIX="${CONDA_PREFIX:-}"
BASE_PREFIX=""
ENV_PREFIXES=()
if [[ -n "${CONDA_BIN}" ]]; then
  # 获取 json 信息（尽量不依赖 jq）
  JSON_INFO="$(${CONDA_BIN} info --json 2>/dev/null || true)"
  if [[ -n "${JSON_INFO}" ]]; then
    export CONDA_INFO_JSON="${JSON_INFO}"
    ACTIVE_PREFIX="${ACTIVE_PREFIX:-$(python - <<'PY'
import json, os
data = json.loads(os.environ.get('CONDA_INFO_JSON', '{}'))
print(data.get('active_prefix') or '')
PY
)}"
    BASE_PREFIX="$(python - <<'PY'
import json, os
data = json.loads(os.environ.get('CONDA_INFO_JSON', '{}'))
print(data.get('root_prefix') or data.get('conda_prefix') or '')
PY
)"
    # envs 列表
    mapfile -t ENV_PREFIXES < <(python - <<'PY'
import json, os
data = json.loads(os.environ.get('CONDA_INFO_JSON', '{}'))
for p in data.get('envs') or []:
    print(p)
PY
)
    unset CONDA_INFO_JSON
    log_debug "已检测到的conda环境: ${ENV_PREFIXES[*]:-}" 
  fi
fi

# 组装候选前缀：支持自定义优先级
declare -a CANDIDATES=()
add_candidate() {
  local prefix="$1"
  [[ -n "$prefix" && -d "$prefix" ]] || return
  for existing in "${CANDIDATES[@]}"; do
    [[ "$existing" == "$prefix" ]] && return
  done
  CANDIDATES+=("$prefix")
  log_debug "添加候选环境: $prefix"
}

# 1. 显式指定的前缀（PYGMT_PREFIX）
if [[ -n "${PYGMT_PREFIX:-}" ]]; then
  add_candidate "${PYGMT_PREFIX}"
fi

# 2. 明确的环境名称（PYGMT_ENV），默认首选 gpytorch
DESIRED_ENV_NAME="${PYGMT_ENV:-gpytorch}"
if [[ -n "${DESIRED_ENV_NAME}" ]]; then
  for p in "${ENV_PREFIXES[@]:-}"; do
    if [[ "$(basename "$p")" == "${DESIRED_ENV_NAME}" ]]; then
      add_candidate "$p"
      break
    fi
  done
fi

# 3. 当前激活的前缀
if [[ -n "${ACTIVE_PREFIX}" ]]; then add_candidate "${ACTIVE_PREFIX}"; fi

# 4. 所有已知环境
for p in "${ENV_PREFIXES[@]:-}"; do add_candidate "$p"; done

# 5. base/root 前缀
if [[ -n "${BASE_PREFIX}" ]]; then add_candidate "${BASE_PREFIX}"; fi

# 6. 兜底：使用当前 CONDA_PREFIX
if [[ ${#CANDIDATES[@]} -eq 0 && -n "${CONDA_PREFIX:-}" ]]; then
  add_candidate "${CONDA_PREFIX}"
fi

log_debug "候选环境顺序: ${CANDIDATES[*]:-}"

# 在候选前缀中查找已安装 PyGMT 的 python
SELECTED_PREFIX=""
SELECTED_PY=""
for prefix in "${CANDIDATES[@]:-}"; do
  [[ -d "$prefix" ]] || continue
  PY_BIN="$prefix/bin/python"
  [[ -x "$PY_BIN" ]] || continue
  CANDIDATE_LD_PATH="$prefix/lib:$prefix/lib64:${LD_LIBRARY_PATH:-}"
  if LD_LIBRARY_PATH="$CANDIDATE_LD_PATH" "$PY_BIN" - <<'PY' >/dev/null 2>&1
import importlib.util
import sys
sys.exit(0) if importlib.util.find_spec('pygmt') else sys.exit(1)
PY
  then
    SELECTED_PREFIX="$prefix"
    SELECTED_PY="$PY_BIN"
    break
  fi
done

# 如果未找到携带PyGMT的环境，则退回当前python
if [[ -z "${SELECTED_PY}" ]]; then
  SELECTED_PREFIX="${ACTIVE_PREFIX:-${CONDA_PREFIX:-}}"
  SELECTED_PY="$(command -v python)"
fi

log_debug "最终选定的Python: ${SELECTED_PY:-system} (prefix: ${SELECTED_PREFIX:-system})"

# 设置库路径（优先使用选中前缀）
if [[ -n "${SELECTED_PREFIX}" ]]; then
  export LD_LIBRARY_PATH="${SELECTED_PREFIX}/lib:${SELECTED_PREFIX}/lib64:${LD_LIBRARY_PATH:-}"
  export PATH="${SELECTED_PREFIX}/bin:${PATH}"
fi

# 执行
exec "${SELECTED_PY}" "$@"
