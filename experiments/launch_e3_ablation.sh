#!/usr/bin/env bash
set -euo pipefail
# E3: Component Ablation (Table 2)
# All use locked-trace on s200, replaying f0_s200 planner trace.
# Full AdaRec already available from E2.

export EPISODE_SEED=42
export PROXIMITY_STOP=0.5
export GATE_GOAL_HYSTERESIS=0
export GATE_LATE_BUDGET_RESERVE=0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PLANNER_TRACE_SOURCE="${REPO_ROOT}/artifacts/runs/experiments/f0_s200/seed_42/planner_trace.jsonl"

echo "=========================================="
echo "### E3-1: -Gate (fixed_interval k=2) ###"
echo "=========================================="
bash "${SCRIPT_DIR}/run_experiment.sh" adarec_minus_i2 42 100 200
echo "-Gate DONE"

echo "=========================================="
echo "### E3-2: -Memory (disabled + dtg) ###"
echo "=========================================="
bash "${SCRIPT_DIR}/run_experiment.sh" adarec_minus_i3 42 100 200
echo "-Memory DONE"

echo "=========================================="
echo "### E3-3: -Recovery (random + no CL) ###"
echo "=========================================="
bash "${SCRIPT_DIR}/run_experiment.sh" adarec_minus_i4 42 100 200
echo "-Recovery DONE"

echo "=== E3 ABLATION COMPLETE ==="
