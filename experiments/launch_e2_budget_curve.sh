#!/usr/bin/env bash
set -euo pipefail
# E2: Budget-Performance Curve (Figure 1)
# Run F0 baseline and RecNav (locked-trace) across multiple step budgets.
# Each budget point: 100 episodes, seed=42, PROXIMITY_STOP=0.5.

export EPISODE_SEED=42
export PROXIMITY_STOP=0.5
export GATE_GOAL_HYSTERESIS=0
export GATE_LATE_BUDGET_RESERVE=0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUDGETS=(100 150 200 250 300 500)

echo "=== E2: Budget Curve — F0 Baseline ==="
for B in "${BUDGETS[@]}"; do
    echo ">>> F0 s${B} <<<"
    bash "${SCRIPT_DIR}/run_experiment.sh" f0 42 100 "${B}"
    echo "F0 s${B} DONE"
done

echo ""
echo "=== E2: Budget Curve — RecNav (locked-trace) ==="
for B in "${BUDGETS[@]}"; do
    TRACE="${SCRIPT_DIR}/f0_s${B}/seed_42/planner_trace.jsonl"
    if [[ ! -f "${TRACE}" ]]; then
        echo "WARN: trace not found at ${TRACE}, skipping adarec_locked s${B}"
        continue
    fi
    export PLANNER_TRACE_SOURCE="${TRACE}"
    echo ">>> adarec_locked s${B} <<<"
    bash "${SCRIPT_DIR}/run_experiment.sh" adarec_locked 42 100 "${B}"
    echo "adarec_locked s${B} DONE"
done

echo ""
echo "=== E2 BUDGET CURVE COMPLETE ==="
echo "Run: python scripts/pixnav/plot_budget_curve.py --out figures/budget_curve.pdf"
