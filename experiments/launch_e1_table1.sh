#!/usr/bin/env bash
set -euo pipefail
# E1: Recovery Strategy Comparison (Table 1)
# All use locked-trace on s200, replaying f0_s200 planner trace.
# F0 and AdaRec already available from E2.

export EPISODE_SEED=42
export PROXIMITY_STOP=0.5
export GATE_GOAL_HYSTERESIS=0
export GATE_LATE_BUDGET_RESERVE=0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PLANNER_TRACE_SOURCE="${REPO_ROOT}/artifacts/runs/experiments/f0_s200/seed_42/planner_trace.jsonl"

echo "=========================================="
echo "### E1-1: + Random Walk (s200) ###"
echo "=========================================="
bash "${SCRIPT_DIR}/run_experiment.sh" random_walk_locked 42 100 200
echo "Random Walk DONE"

echo "=========================================="
echo "### E1-2: + Frontier Recovery (s200) ###"
echo "=========================================="
bash "${SCRIPT_DIR}/run_experiment.sh" frontier_locked 42 100 200
echo "Frontier DONE"

echo "=========================================="
echo "### E1-3: + LLM Replan (s200) ###"
echo "=========================================="
bash "${SCRIPT_DIR}/run_experiment.sh" llm_replan_locked 42 100 200
echo "LLM Replan DONE"

echo "=== E1 TABLE 1 COMPLETE ==="
