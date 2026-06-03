#!/usr/bin/env bash
set -euo pipefail

export EPISODE_SEED="${EPISODE_SEED:-42}"
export PROXIMITY_STOP="${PROXIMITY_STOP:-0.5}"
export GATE_GOAL_HYSTERESIS="${GATE_GOAL_HYSTERESIS:-0}"
export GATE_LATE_BUDGET_RESERVE="${GATE_LATE_BUDGET_RESERVE:-0}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUDGETS=(100 150 200 250 300 500)
SEED="${SEED:-42}"
EPISODES="${EPISODES:-100}"

for budget in "${BUDGETS[@]}"; do
  bash "${SCRIPT_DIR}/run_experiment.sh" f0 "${SEED}" "${EPISODES}" "${budget}"
done

for budget in "${BUDGETS[@]}"; do
  export PLANNER_TRACE_SOURCE="${REPO_ROOT}/artifacts/runs/experiments/f0_s${budget}/seed_${SEED}/planner_trace.jsonl"
  if [[ ! -f "${PLANNER_TRACE_SOURCE}" ]]; then
    echo "ERROR: missing baseline trace: ${PLANNER_TRACE_SOURCE}" >&2
    exit 1
  fi
  bash "${SCRIPT_DIR}/run_experiment.sh" adarec_locked "${SEED}" "${EPISODES}" "${budget}"
done
