#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash experiments/run_experiment.sh <variant> <seed> [episodes] [max_steps]
#
# Core variants:
#   f0                 PixNav baseline
#   adarec_locked      full RecNav, usually with PLANNER_TRACE_SOURCE set
#   random_walk_locked random recovery baseline
#   frontier_locked    frontier recovery baseline
#   llm_replan_locked  LLM replanning baseline
#   adarec_minus_i2    gate ablation
#   adarec_minus_i3    memory ablation
#   adarec_minus_i4    recovery ablation

VARIANT="${1:?Usage: $0 <variant> <seed> [episodes] [max_steps]}"
SEED="${2:?Usage: $0 <variant> <seed> [episodes] [max_steps]}"
EPISODES="${3:-100}"
MAX_STEPS="${4:-200}"

EPISODE_SEED="${EPISODE_SEED:-${SEED}}"
PROXIMITY_STOP="${PROXIMITY_STOP:-0.5}"
POST_RECOVERY_REPLAN="${POST_RECOVERY_REPLAN:-skip}"
MACRO_PROBE_STEPS="${MACRO_PROBE_STEPS:-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

PYTHON_BIN="${ADA_SEMNAV_PYTHON_BIN:-python}"

STEP_TAG="_s${MAX_STEPS}"
OUT_DIR="${REPO_ROOT}/artifacts/runs/experiments/${VARIANT}${STEP_TAG}/seed_${SEED}"
CSV_PATH="${OUT_DIR}/results.csv"
TRACE_PATH="${OUT_DIR}/planner_trace.jsonl"
mkdir -p "${OUT_DIR}"

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
GATE_INTERVAL_ARG=""

case "${VARIANT}" in
  f0)
    RECOVERY_MODE="host_only"
    GATE_MODE="legacy_stop_signal"
    MEMORY_MODE="disabled"
    ;;
  adarec_locked)
    RECOVERY_MODE="macro_action"
    GATE_MODE="pose_free"
    MEMORY_MODE="skeleton"
    MACRO_RANKING="composite"
    ;;
  random_walk_locked)
    RECOVERY_MODE="macro_action"
    GATE_MODE="pose_free"
    MEMORY_MODE="disabled"
    MACRO_RANKING="random"
    ;;
  frontier_locked)
    RECOVERY_MODE="macro_action"
    GATE_MODE="pose_free"
    MEMORY_MODE="skeleton"
    MACRO_RANKING="frontier_nearest"
    ;;
  llm_replan_locked)
    RECOVERY_MODE="naive_llm"
    GATE_MODE="pose_free"
    MEMORY_MODE="disabled"
    ;;
  adarec_minus_i2)
    RECOVERY_MODE="macro_action"
    GATE_MODE="fixed_interval"
    MEMORY_MODE="skeleton"
    MACRO_RANKING="composite"
    GATE_INTERVAL_ARG="--gate_interval_events=2"
    ;;
  adarec_minus_i3)
    RECOVERY_MODE="macro_action"
    GATE_MODE="pose_free"
    MEMORY_MODE="disabled"
    MACRO_RANKING="dtg"
    ;;
  adarec_minus_i4)
    RECOVERY_MODE="macro_action"
    GATE_MODE="pose_free"
    MEMORY_MODE="skeleton"
    MACRO_RANKING="random"
    RECOVERY_CLOSED_LOOP=0
    ;;
  *)
    echo "ERROR: unknown variant: ${VARIANT}" >&2
    exit 1
    ;;
esac

TRACE_ARG=()
if [[ -n "${PLANNER_TRACE_SOURCE:-}" ]]; then
  TRACE_ARG=(--planner_trace_path="${PLANNER_TRACE_SOURCE}" --replay_planner_trace)
else
  TRACE_ARG=(--planner_trace_path="${TRACE_PATH}")
fi

echo "variant=${VARIANT} seed=${SEED} episodes=${EPISODES} max_steps=${MAX_STEPS}"
echo "output=${OUT_DIR}"
echo "recovery=${RECOVERY_MODE} gate=${GATE_MODE} memory=${MEMORY_MODE} ranking=${MACRO_RANKING}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" scripts/pixnav/run_pixnav_host_eval.py \
  --eval_episodes="${EPISODES}" \
  --seed="${SEED}" \
  --output_dir="${OUT_DIR}" \
  --csv_path="${CSV_PATH}" \
  --export-telemetry \
  "${TRACE_ARG[@]}" \
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
  --gate_goal_hysteresis_window="${GATE_GOAL_HYSTERESIS:-0}" \
  --gate_late_budget_reserve="${GATE_LATE_BUDGET_RESERVE:-0}" \
  --memory_mode="${MEMORY_MODE}" \
  --planner_replan_policy=always \
  --call_budget="${CALL_BUDGET}" \
  --episode_seed="${EPISODE_SEED}" \
  --proximity_stop_distance="${PROXIMITY_STOP}" \
  --post_recovery_replan="${POST_RECOVERY_REPLAN}" \
  --recovery_macro_probe_steps="${MACRO_PROBE_STEPS}" \
  --max_episode_steps="${MAX_STEPS}" \
  ${GATE_INTERVAL_ARG}
