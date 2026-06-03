#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/pixnav${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
  set +a
fi

PYTHON_BIN="${ADA_SEMNAV_PYTHON_BIN:-python}"

usage() {
  cat <<'USAGE'
Usage: ./run.sh <command> [args]

Commands:
  check-dataset    Validate HM3D/ObjectNav file layout
  env-check        Verify Habitat, PixNav, checkpoints, and imports
  smoke-test       Run a small Habitat reset/step smoke test
  eval-pixnav      Run scripts/pixnav/run_pixnav_host_eval.py
  analyze-results  Summarize result CSV files
  plot-budget      Plot budget curve from result CSV files
  llm-smoke        Test OpenAI-compatible LLM connectivity
  mcnemar          Run McNemar exact test on paired result CSVs
USAGE
}

cmd="${1:-}"
if [[ -z "$cmd" ]]; then
  usage
  exit 1
fi
shift

case "$cmd" in
  check-dataset)
    "$PYTHON_BIN" scripts/core/check_dataset.py "$@"
    ;;
  env-check)
    "$PYTHON_BIN" scripts/tools/stage0_env_check.py "$@"
    ;;
  smoke-test)
    "$PYTHON_BIN" scripts/core/smoke_test.py "$@"
    ;;
  eval-pixnav)
    "$PYTHON_BIN" scripts/pixnav/run_pixnav_host_eval.py "$@"
    ;;
  analyze-results)
    "$PYTHON_BIN" scripts/pixnav/analyze_results.py "$@"
    ;;
  plot-budget)
    "$PYTHON_BIN" scripts/pixnav/plot_budget_curve.py "$@"
    ;;
  llm-smoke)
    "$PYTHON_BIN" scripts/tools/llm_smoke.py "$@"
    ;;
  mcnemar)
    "$PYTHON_BIN" scripts/tools/mcnemar_test.py "$@"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "ERROR: unknown command: $cmd" >&2
    usage
    exit 1
    ;;
esac
