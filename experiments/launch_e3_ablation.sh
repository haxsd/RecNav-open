#!/usr/bin/env bash
set -euo pipefail

export EPISODE_SEED="${EPISODE_SEED:-42}"
export PROXIMITY_STOP="${PROXIMITY_STOP:-0.5}"
export GATE_GOAL_HYSTERESIS="${GATE_GOAL_HYSTERESIS:-0}"
export GATE_LATE_BUDGET_RESERVE="${GATE_LATE_BUDGET_RESERVE:-0}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SEED="${SEED:-42}"
EPISODES="${EPISODES:-100}"
BUDGET="${BUDGET:-200}"

export PLANNER_TRACE_SOURCE="${REPO_ROOT}/artifacts/runs/experiments/f0_s${BUDGET}/seed_${SEED}/planner_trace.jsonl"
if [[ ! -f "${PLANNER_TRACE_SOURCE}" ]]; then
  echo "ERROR: missing baseline trace. Run launch_e2_budget_curve.sh or f0 first." >&2
  exit 1
fi

bash "${SCRIPT_DIR}/run_experiment.sh" adarec_minus_i2 "${SEED}" "${EPISODES}" "${BUDGET}"
bash "${SCRIPT_DIR}/run_experiment.sh" adarec_minus_i3 "${SEED}" "${EPISODES}" "${BUDGET}"
bash "${SCRIPT_DIR}/run_experiment.sh" adarec_minus_i4 "${SEED}" "${EPISODES}" "${BUDGET}"
