#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${WSL_DISTRO_NAME:-}" ]]; then
  echo "ERROR: This script must run inside WSL (Linux shell), not Windows PowerShell."
  exit 1
fi

if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
  echo "WARN: Conda environment is not active. Run: conda activate <your_env>"
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

if [[ -z "${OPENAI_API_KEY:-}" && -n "${DEEPSEEK_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="${DEEPSEEK_API_KEY}"
fi
if [[ -z "${OPENAI_BASE_URL:-}" && -n "${DEEPSEEK_API_KEY:-}" ]]; then
  export OPENAI_BASE_URL="${DEEPSEEK_BASE_URL:-https://api.deepseek.com}"
fi

export GALLIUM_DRIVER="${GALLIUM_DRIVER:-d3d12}"
export MESA_D3D12_DEFAULT_ADAPTER_NAME="${MESA_D3D12_DEFAULT_ADAPTER_NAME:-NVIDIA}"

resolve_python_bin() {
  local configured_bin="${1:-}"
  local fallback_bin="${2:-}"
  local label="${3:-python}"
  if [[ -n "${configured_bin}" ]]; then
    if [[ -x "${configured_bin}" ]]; then
      echo "${configured_bin}"
      return 0
    fi
    echo "ERROR: ${label} interpreter not executable: ${configured_bin}" >&2
    return 1
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  if [[ -n "${fallback_bin}" && -x "${fallback_bin}" ]]; then
    echo "${fallback_bin}"
    return 0
  fi
  echo "ERROR: python not found. Set ${label} interpreter in .env or activate conda env." >&2
  return 1
}

PYTHON_BIN="$(resolve_python_bin "${ADA_SEMNAV_PYTHON_BIN:-}" "" "ADA_SEMNAV_PYTHON_BIN")"
if [[ -n "${ADA_SEMNAV_PYTHON_BIN:-}" ]]; then
  echo "INFO: Using configured interpreter from ADA_SEMNAV_PYTHON_BIN: ${PYTHON_BIN}"
else
  echo "INFO: Using resolved interpreter: ${PYTHON_BIN}"
fi

usage() {
  cat <<'USAGE'
Usage: ./run.sh <command> [args]
Usage: ./run.sh [--timeout-profile auto|quick|normal|heavy] [--timeout-sec N] [--no-timeout] <command> [args]

Active Commands:
  env-check            Verify PixNav + Habitat environment wiring
  analyze-results      Auto-generate experiment result tables (E1-E7)
  plot-budget-curve    Generate budget-performance curve figure

Notes:
  - This launcher defaults to GALLIUM_DRIVER=d3d12 under WSL.
  - Optional interpreter overrides in `.env`:
      ADA_SEMNAV_PYTHON_BIN=/path/to/your/conda/envs/recnav/bin/python
  - Timeout controls:
      RUN_TIMEOUT_PROFILE=auto|quick|normal|heavy   (default: auto)
      RUN_TIMEOUT_SEC=<seconds>                     (explicit override)
      RUN_DISABLE_TIMEOUT=1                         (disable timeout guard)
  - For Habitat commands, it appends:
      --override habitat.simulator.habitat_sim_v0.gpu_device_id=${HABITAT_GPU_DEVICE_ID}
    (default HABITAT_GPU_DEVICE_ID=-1; set 0 to use first GPU)
USAGE
}

timeout_profile="${RUN_TIMEOUT_PROFILE:-auto}"
timeout_sec_override="${RUN_TIMEOUT_SEC:-}"
disable_timeout="${RUN_DISABLE_TIMEOUT:-0}"
cmd=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --timeout-profile)
      timeout_profile="${2:-}"
      shift 2
      ;;
    --timeout-sec)
      timeout_sec_override="${2:-}"
      shift 2
      ;;
    --no-timeout)
      disable_timeout=1
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      cmd="$1"
      shift
      break
      ;;
  esac
done
if [[ -z "$cmd" ]]; then
  usage
  exit 1
fi

habitat_gpu_device_id="${HABITAT_GPU_DEVICE_ID:--1}"
if [[ -n "${WSL_DISTRO_NAME:-}" && "${habitat_gpu_device_id}" != "-1" ]]; then
  if command -v eglinfo >/dev/null 2>&1; then
    egl_brief="$(eglinfo -B 2>/dev/null || true)"
    if [[ "$egl_brief" == *"EGL vendor string: Mesa Project"* ]]; then
      echo "WARN: WSL EGL vendor is Mesa (D3D12 path), CUDA-EGL mapping is unavailable; forcing HABITAT_GPU_DEVICE_ID=-1."
      habitat_gpu_device_id="-1"
    fi
  fi
fi
default_habitat_override=(--override "habitat.simulator.habitat_sim_v0.gpu_device_id=${habitat_gpu_device_id}")

parse_flag_int_value() {
  local flag="$1"
  local default_value="$2"
  shift 2
  local args=("$@")
  local idx=0
  while [[ $idx -lt ${#args[@]} ]]; do
    if [[ "${args[$idx]}" == "$flag" ]]; then
      local next_idx=$((idx + 1))
      if [[ $next_idx -lt ${#args[@]} ]]; then
        echo "${args[$next_idx]}"
        return 0
      fi
    fi
    idx=$((idx + 1))
  done
  echo "${default_value}"
}

default_timeout_for_command() {
  local target_cmd="$1"
  shift
  local args=("$@")
  case "$target_cmd" in
    check-dataset|llm-smoke)
      echo "300"
      ;;
    env-check|smoke-test)
      echo "900"
      ;;
    audit-protocol|audit-metadata|audit-canonical|audit-unresolved|audit-stage6|audit-stage8|analyze-logs|summarize-q3|summarize-stage8)
      echo "1200"
      ;;
    build-case-pack|build-deadlock-pack|build-hardcase-pack|analyze-hardcase|analyze-recovery)
      echo "3600"
      ;;
    deadlock-suite)
      echo "14400"
      ;;
    eval-main-habitat)
      local episodes
      episodes="$(parse_flag_int_value "--episodes" "20" "${args[@]}")"
      if [[ "$episodes" =~ ^[0-9]+$ ]]; then
        if (( episodes <= 5 )); then
          echo "900"
        elif (( episodes <= 50 )); then
          echo "5400"
        elif (( episodes <= 200 )); then
          echo "21600"
        else
          echo "43200"
        fi
      else
        echo "5400"
      fi
      ;;
    *)
      echo "3600"
      ;;
  esac
}

resolve_timeout_seconds() {
  local target_cmd="$1"
  shift
  local args=("$@")
  if [[ "${disable_timeout}" == "1" ]]; then
    echo "0"
    return 0
  fi
  if [[ -n "${timeout_sec_override}" ]]; then
    echo "${timeout_sec_override}"
    return 0
  fi
  local base
  base="$(default_timeout_for_command "$target_cmd" "${args[@]}")"
  case "${timeout_profile}" in
    auto|normal|"")
      echo "${base}"
      ;;
    quick)
      if (( base > 900 )); then
        echo "900"
      else
        echo "${base}"
      fi
      ;;
    heavy)
      local doubled
      doubled=$((base * 2))
      if (( doubled > 86400 )); then
        doubled=86400
      fi
      echo "${doubled}"
      ;;
    *)
      echo "ERROR: invalid --timeout-profile: ${timeout_profile}" >&2
      exit 1
      ;;
  esac
}

effective_timeout_sec="$(resolve_timeout_seconds "$cmd" "$@")"
if [[ "${disable_timeout}" == "1" ]]; then
  echo "INFO: Timeout guard disabled for command=${cmd}"
elif [[ "${effective_timeout_sec}" =~ ^[0-9]+$ ]] && (( effective_timeout_sec > 0 )); then
  echo "INFO: Timeout guard for command=${cmd}: ${effective_timeout_sec}s (profile=${timeout_profile})"
fi

run_python_entry() {
  local python_bin="$1"
  local script_path="$2"
  shift 2
  if [[ "${disable_timeout}" == "1" || "${effective_timeout_sec}" == "0" ]]; then
    set +e
    "$python_bin" "$script_path" "$@"
    local rc_no_timeout=$?
    set -e
    return $rc_no_timeout
  fi
  if command -v timeout >/dev/null 2>&1; then
    set +e
    timeout --foreground "${effective_timeout_sec}s" "$python_bin" "$script_path" "$@"
    local rc=$?
    set -e
    if [[ $rc -eq 124 ]]; then
      echo "ERROR: ${cmd} timed out after ${effective_timeout_sec}s."
      echo "HINT: retry with --timeout-sec <N> or --timeout-profile heavy."
    fi
    return $rc
  fi
  echo "WARN: timeout command not found; running without timeout guard."
  set +e
  "$python_bin" "$script_path" "$@"
  local rc_no_guard=$?
  set -e
  return $rc_no_guard
}

case "$cmd" in
  env-check)
    run_python_entry "$PYTHON_BIN" scripts/tools/stage0_env_check.py "$@"
    ;;
  analyze-results)
    run_python_entry "$PYTHON_BIN" scripts/pixnav/analyze_results.py "$@"
    ;;
  plot-budget-curve)
    run_python_entry "$PYTHON_BIN" scripts/pixnav/plot_budget_curve.py "$@"
    ;;
  *)
    echo "ERROR: Unknown command '$cmd'"
    usage
    exit 1
    ;;
esac
