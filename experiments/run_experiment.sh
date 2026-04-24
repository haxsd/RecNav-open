#!/usr/bin/env bash
set -euo pipefail
#
# Unified experiment runner for RecNav paper.
# Usage: bash run_experiment.sh <VARIANT> <SEED> [EPISODES] [MAX_STEPS]
#
# Variants (E1 Main Table):
#   f0           — PixNav baseline (no recovery)
#   llm_replan   — + LLM Replan recovery
#   random_escape— + Random displacement escape
#   backtrack    — + Backtrack + constrained_edge recovery
#   adarec       — RecNav full system (ours)
#
# Variants (E3 Component Ablation):
#   adarec_no_memory — AdaRec minus memory
#   adarec_no_gate   — AdaRec minus intelligent gate
#
# Variants (E4 Recovery Ablation):
#   r1_no_backtrack  — constrained_edge, no backtrack
#
# Variants (E5 Gate):
#   gate_always      — legacy_stop_signal gate
#   gate_fixed2      — fixed_interval k=2
#   gate_fixed4      — fixed_interval k=4
#
# Variants (E6 Memory):
#   mem_disabled     — macro_action + dtg ranking, no memory
#   mem_fifo         — macro_action + memory_dtg ranking, FIFO memory
#
# Variants (E7 Sensitivity):
#   sens_fan5        — fan-out 5 directions
#   sens_fan9        — fan-out 9 directions
#   sens_fwd1        — 1 forward step
#   sens_fwd5        — 5 forward steps
#   sens_no_abstain  — abstain disabled

VARIANT="${1:?Usage: $0 <VARIANT> <SEED> [EPISODES] [MAX_STEPS]}"
SEED="${2:?Usage: $0 <VARIANT> <SEED> [EPISODES] [MAX_STEPS]}"
EPISODES="${3:-56}"
MAX_STEPS="${4:-0}"
EPISODE_SEED="${EPISODE_SEED:-${SEED}}"
PROXIMITY_STOP="${PROXIMITY_STOP:-0.0}"
PLANNER_TRACE_SOURCE="${PLANNER_TRACE_SOURCE:-}"
DTG_PROTECT_MARGIN="${DTG_PROTECT_MARGIN:-0.0}"
POST_RECOVERY_REPLAN="${POST_RECOVERY_REPLAN:-skip}"
MACRO_PROBE_STEPS="${MACRO_PROBE_STEPS:-1}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STEP_TAG=""
if [ "${MAX_STEPS}" -gt 0 ]; then
    STEP_TAG="_s${MAX_STEPS}"
fi
OUT_DIR="${REPO_ROOT}/artifacts/runs/experiments/${VARIANT}${STEP_TAG}/seed_${SEED}"
CSV_PATH="${OUT_DIR}/results.csv"
TRACE_PATH="${OUT_DIR}/planner_trace.jsonl"
mkdir -p "${OUT_DIR}"

# Common defaults
RECOVERY_MODE="host_only"
GATE_MODE="legacy_stop_signal"
MEMORY_MODE="disabled"
MACRO_RANKING="dtg"
MACRO_MAX_FAN=13
MACRO_FWD_STEPS=3
MACRO_ABSTAIN=1
CALL_BUDGET=5
BACKTRACK_STEPS=6
RECOVERY_CLOSED_LOOP=1
RECOVERY_MAX_RETRIES=2

case "${VARIANT}" in
    # === E1 Main Table ===
    f0)
        RECOVERY_MODE="host_only"
        GATE_MODE="legacy_stop_signal"
        MEMORY_MODE="disabled"
        ;;
    llm_replan)
        RECOVERY_MODE="naive_llm"
        GATE_MODE="pose_free"
        MEMORY_MODE="disabled"
        ;;
    random_escape)
        RECOVERY_MODE="macro_action"
        GATE_MODE="pose_free"
        MEMORY_MODE="disabled"
        MACRO_RANKING="random"
        ;;
    backtrack)
        RECOVERY_MODE="constrained_edge"
        GATE_MODE="pose_free"
        MEMORY_MODE="skeleton"
        BACKTRACK_STEPS=6
        ;;
    adarec)
        RECOVERY_MODE="macro_action"
        GATE_MODE="pose_free"
        MEMORY_MODE="skeleton"
        MACRO_RANKING="composite"
        ;;
    adarec_locked)
        # Same as adarec; use PLANNER_TRACE_SOURCE env to replay F0 trace
        RECOVERY_MODE="macro_action"
        GATE_MODE="pose_free"
        MEMORY_MODE="skeleton"
        MACRO_RANKING="composite"
        ;;
    adarec_opt_locked)
        # Optimized adarec with P1+P2+P3; use PLANNER_TRACE_SOURCE env
        RECOVERY_MODE="macro_action"
        GATE_MODE="pose_free"
        MEMORY_MODE="skeleton"
        MACRO_RANKING="composite"
        ;;
    adarec_p3_locked)
        # P3-only ablation: probe aligned with exec; use PLANNER_TRACE_SOURCE env
        RECOVERY_MODE="macro_action"
        GATE_MODE="pose_free"
        MEMORY_MODE="skeleton"
        MACRO_RANKING="composite"
        ;;

    # === E1 Recovery Strategy Comparison (Table 1) ===
    random_walk_locked)
        # + Random Walk: macro_action + random ranking, no memory; use PLANNER_TRACE_SOURCE
        RECOVERY_MODE="macro_action"
        GATE_MODE="pose_free"
        MEMORY_MODE="disabled"
        MACRO_RANKING="random"
        ;;
    frontier_locked)
        # + Frontier Recovery: macro_action + frontier_nearest ranking, skeleton memory; use PLANNER_TRACE_SOURCE
        RECOVERY_MODE="macro_action"
        GATE_MODE="pose_free"
        MEMORY_MODE="skeleton"
        MACRO_RANKING="frontier_nearest"
        ;;
    llm_replan_locked)
        # + LLM Replan: naive_llm recovery, no memory; use PLANNER_TRACE_SOURCE
        RECOVERY_MODE="naive_llm"
        GATE_MODE="pose_free"
        MEMORY_MODE="disabled"
        MACRO_RANKING="dtg"
        ;;

    # === E3 Component Ablation (each removes exactly one innovation) ===
    adarec_minus_i2)
        # -I2 (Gate): replace intelligent gate with fixed interval k=2
        RECOVERY_MODE="macro_action"
        GATE_MODE="fixed_interval"
        MEMORY_MODE="skeleton"
        MACRO_RANKING="composite"
        ;;
    adarec_minus_i3)
        # -I3 (Memory): disable memory, fall back to pure DTG ranking
        RECOVERY_MODE="macro_action"
        GATE_MODE="pose_free"
        MEMORY_MODE="disabled"
        MACRO_RANKING="dtg"
        ;;
    adarec_minus_i4)
        # -I4 (Recovery): replace intelligent ranking with random + disable closed-loop
        RECOVERY_MODE="macro_action"
        GATE_MODE="pose_free"
        MEMORY_MODE="skeleton"
        MACRO_RANKING="random"
        RECOVERY_CLOSED_LOOP=0
        ;;

    # === E4 Recovery Ablation ===
    r1_no_backtrack)
        RECOVERY_MODE="constrained_edge"
        GATE_MODE="pose_free"
        MEMORY_MODE="skeleton"
        BACKTRACK_STEPS=0
        ;;

    # === E5 Gate Strategies ===
    gate_always)
        RECOVERY_MODE="macro_action"
        GATE_MODE="legacy_stop_signal"
        MEMORY_MODE="skeleton"
        MACRO_RANKING="composite"
        ;;
    gate_fixed2)
        RECOVERY_MODE="macro_action"
        GATE_MODE="fixed_interval"
        MEMORY_MODE="skeleton"
        MACRO_RANKING="composite"
        ;;
    gate_fixed4)
        RECOVERY_MODE="macro_action"
        GATE_MODE="fixed_interval"
        MEMORY_MODE="skeleton"
        MACRO_RANKING="composite"
        ;;

    # === E6 Memory Ablation ===
    mem_disabled)
        RECOVERY_MODE="macro_action"
        GATE_MODE="pose_free"
        MEMORY_MODE="disabled"
        MACRO_RANKING="dtg"
        ;;
    mem_fifo)
        RECOVERY_MODE="macro_action"
        GATE_MODE="pose_free"
        MEMORY_MODE="fifo"
        MACRO_RANKING="composite"
        ;;

    # === E7 Sensitivity ===
    sens_fan5)
        RECOVERY_MODE="macro_action"
        GATE_MODE="pose_free"
        MEMORY_MODE="skeleton"
        MACRO_RANKING="composite"
        MACRO_MAX_FAN=5
        ;;
    sens_fan9)
        RECOVERY_MODE="macro_action"
        GATE_MODE="pose_free"
        MEMORY_MODE="skeleton"
        MACRO_RANKING="composite"
        MACRO_MAX_FAN=9
        ;;
    sens_fwd1)
        RECOVERY_MODE="macro_action"
        GATE_MODE="pose_free"
        MEMORY_MODE="skeleton"
        MACRO_RANKING="composite"
        MACRO_FWD_STEPS=1
        ;;
    sens_fwd5)
        RECOVERY_MODE="macro_action"
        GATE_MODE="pose_free"
        MEMORY_MODE="skeleton"
        MACRO_RANKING="composite"
        MACRO_FWD_STEPS=5
        ;;
    sens_no_abstain)
        RECOVERY_MODE="macro_action"
        GATE_MODE="pose_free"
        MEMORY_MODE="skeleton"
        MACRO_RANKING="composite"
        MACRO_ABSTAIN=0
        ;;
    *)
        echo "ERROR: Unknown variant '${VARIANT}'"
        exit 1
        ;;
esac

# Gate interval parameter for fixed_interval mode
GATE_INTERVAL_ARG=""
if [ "${VARIANT}" = "gate_fixed2" ] || [ "${VARIANT}" = "adarec_minus_i2" ]; then
    GATE_INTERVAL_ARG="--gate_interval_events=2"
elif [ "${VARIANT}" = "gate_fixed4" ]; then
    GATE_INTERVAL_ARG="--gate_interval_events=4"
fi

MAX_STEPS_ARG=""
if [ "${MAX_STEPS}" -gt 0 ]; then
    MAX_STEPS_ARG="--max_episode_steps=${MAX_STEPS}"
fi

echo "=========================================="
echo " ${VARIANT} — ${EPISODES}ep × seed=${SEED}"
[ "${MAX_STEPS}" -gt 0 ] && echo " max_steps=${MAX_STEPS}"
echo " recovery=${RECOVERY_MODE} gate=${GATE_MODE} memory=${MEMORY_MODE} ranking=${MACRO_RANKING}"
echo " episode_seed=${EPISODE_SEED} proximity_stop=${PROXIMITY_STOP}"
[ -n "${PLANNER_TRACE_SOURCE}" ] && echo " replay_trace=${PLANNER_TRACE_SOURCE}"
echo " Started: $(date -Iseconds)"
echo "=========================================="

cd "${REPO_ROOT}"
# Activate your conda environment before running, e.g.:
#   source /path/to/miniconda3/etc/profile.d/conda.sh && conda activate recnav
if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
  echo "WARN: No conda env active. Please activate your environment first."
fi

set -a
source "${REPO_ROOT}/.env"
set +a

python "${REPO_ROOT}/scripts/pixnav/run_pixnav_host_eval.py" \
  --eval_episodes="${EPISODES}" \
  --seed="${SEED}" \
  --output_dir="${OUT_DIR}" \
  --csv_path="${CSV_PATH}" \
  --export-telemetry \
  --planner_trace_path="${PLANNER_TRACE_SOURCE:-${TRACE_PATH}}" \
  ${PLANNER_TRACE_SOURCE:+--replay_planner_trace} \
  --recovery_mode="${RECOVERY_MODE}" \
  --recovery_macro_max_fan="${MACRO_MAX_FAN}" \
  --recovery_macro_forward_steps="${MACRO_FWD_STEPS}" \
  --recovery_macro_abstain="${MACRO_ABSTAIN}" \
  --recovery_macro_ranking="${MACRO_RANKING}" \
  --recovery_closed_loop="${RECOVERY_CLOSED_LOOP}" \
  --recovery_max_retries="${RECOVERY_MAX_RETRIES}" \
  --recovery_backtrack_steps="${BACKTRACK_STEPS}" \
  --gate_mode="${GATE_MODE}" \
  --gate_min_active_rules="${GATE_MIN_ACTIVE_RULES:-2}" \
  --gate_novelty_threshold="${GATE_NOVELTY_THRESHOLD:-0.30}" \
  --gate_goal_hysteresis_window="${GATE_GOAL_HYSTERESIS:-2}" \
  --gate_late_budget_reserve="${GATE_LATE_BUDGET_RESERVE:-20}" \
  --memory_mode="${MEMORY_MODE}" \
  --planner_replan_policy=always \
  --call_budget="${CALL_BUDGET}" \
  --episode_seed="${EPISODE_SEED}" \
  --proximity_stop_distance="${PROXIMITY_STOP}" \
  --gate_dtg_protect_margin="${DTG_PROTECT_MARGIN}" \
  --post_recovery_replan="${POST_RECOVERY_REPLAN}" \
  --recovery_macro_probe_steps="${MACRO_PROBE_STEPS}" \
  ${MAX_STEPS_ARG} \
  ${GATE_INTERVAL_ARG}

echo ""
echo "=========================================="
echo " ${VARIANT} DONE — seed=${SEED}"
echo " CSV: ${CSV_PATH}"
echo " Finished: $(date -Iseconds)"
echo "=========================================="
